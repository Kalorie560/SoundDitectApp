# 警告を最初に抑制（importの前に実行）
import warnings
import os
warnings.filterwarnings("ignore")
# torch.classes警告を抑制（Streamlit互換性問題）
warnings.filterwarnings("ignore", ".*torch._classes.*")
warnings.filterwarnings("ignore", ".*torch.*")
warnings.filterwarnings("ignore", ".*Examining the path of torch.classes.*")
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)
os.environ['PYTHONWARNINGS'] = 'ignore'
# PyTorch classes警告を完全に抑制
os.environ['PYTHONHASHSEED'] = 'ignore'

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchaudio
import librosa
from pathlib import Path
import yaml
import time
import logging
from typing import Tuple, Optional
import tempfile
import threading
from datetime import datetime

# Simple recorderモジュールをインポート（エラーハンドリング付き）
try:
    from simple_recorder import SimpleAudioRecorder
    RECORDER_AVAILABLE = True
    RECORDER_ERROR = None
except ImportError as e:
    # sounddeviceが利用できない場合のフォールバック
    from simple_recorder_fallback import FallbackAudioRecorder as SimpleAudioRecorder
    RECORDER_AVAILABLE = False
    RECORDER_ERROR = str(e)
except Exception as e:
    from simple_recorder_fallback import FallbackAudioRecorder as SimpleAudioRecorder
    RECORDER_AVAILABLE = False
    RECORDER_ERROR = str(e)

# Streamlitページ設定
st.set_page_config(
    page_title="音声検出アプリ - Simple Recording",
    page_icon="🎵",
    layout="wide"
)

# ログ設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# デフォルト設定（reference/config.yamlに合わせた構造）
DEFAULT_CONFIG = {
    'audio': {'sample_rate': 44100},
    'model': {
        'input_length': 44100, 
        'num_classes': 2,
        'cnn_layers': [
            {'filters': 64, 'kernel_size': 3, 'stride': 1, 'padding': 'same'},
            {'filters': 128, 'kernel_size': 3, 'stride': 2, 'padding': 1},
            {'filters': 256, 'kernel_size': 3, 'stride': 2, 'padding': 1}
        ],
        'attention': {
            'hidden_dim': 256,
            'num_heads': 8
        },
        'fully_connected': [
            {'units': 512, 'dropout': 0.3},
            {'units': 256, 'dropout': 0.3}
        ]
    }
}

# Multi-head attention layer for audio feature processing
class AttentionLayer(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_heads: int):
        super(AttentionLayer, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        self.query = nn.Linear(input_dim, hidden_dim)
        self.key = nn.Linear(input_dim, hidden_dim)
        self.value = nn.Linear(input_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x):
        batch_size, seq_len, input_dim = x.size()
        
        # Compute queries, keys, values
        Q = self.query(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.key(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.value(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.head_dim)
        attention_weights = torch.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        attended = torch.matmul(attention_weights, V)
        attended = attended.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.hidden_dim
        )
        
        output = self.output(attended)
        return output

# 音声異常検知モデル（1D-CNN + Attention機構）
class SoundAnomalyDetector(nn.Module):
    def __init__(self, config: dict):
        super(SoundAnomalyDetector, self).__init__()
        self.config = config
        
        # CNN layers
        cnn_layers = []
        input_channels = 1
        
        for layer_config in config['model']['cnn_layers']:
            # Handle "same" padding by calculating appropriate padding
            padding = layer_config['padding']
            if padding == 'same':
                # For "same" padding in PyTorch, use kernel_size//2
                padding = layer_config['kernel_size'] // 2
            
            cnn_layers.extend([
                nn.Conv1d(
                    input_channels, 
                    layer_config['filters'],
                    kernel_size=layer_config['kernel_size'],
                    stride=layer_config['stride'],
                    padding=padding
                ),
                nn.BatchNorm1d(layer_config['filters']),
                nn.ReLU(),
                nn.MaxPool1d(2)
            ])
            input_channels = layer_config['filters']
        
        self.cnn = nn.Sequential(*cnn_layers)
        
        # Calculate CNN output size
        self.cnn_output_size = self._get_cnn_output_size(config['model']['input_length'])
        
        # Attention layer
        attention_config = config['model']['attention']
        self.attention = AttentionLayer(
            input_channels,
            attention_config['hidden_dim'],
            attention_config['num_heads']
        )
        
        # Fully connected layers
        fc_layers = []
        fc_input_size = attention_config['hidden_dim']
        
        for fc_config in config['model']['fully_connected']:
            fc_layers.extend([
                nn.Linear(fc_input_size, fc_config['units']),
                nn.ReLU(),
                nn.Dropout(fc_config['dropout'])
            ])
            fc_input_size = fc_config['units']
        
        # Output layer
        fc_layers.append(nn.Linear(fc_input_size, config['model']['num_classes']))
        self.classifier = nn.Sequential(*fc_layers)
        
        # Global average pooling
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
    
    def _get_cnn_output_size(self, input_length: int) -> int:
        # Create dummy input to calculate output size
        x = torch.randn(1, 1, input_length)
        with torch.no_grad():
            x = self.cnn(x)
        return x.size(-1)
    
    def forward(self, x):
        # Input shape: (batch_size, input_length)
        # Add channel dimension: (batch_size, 1, input_length)
        x = x.unsqueeze(1)
        
        # CNN feature extraction
        x = self.cnn(x)  # (batch_size, channels, reduced_length)
        
        # Prepare for attention: (batch_size, seq_len, features)
        x = x.transpose(1, 2)
        
        # Apply attention
        x = self.attention(x)
        
        # Global average pooling
        x = x.transpose(1, 2)  # Back to (batch_size, features, seq_len)
        x = self.global_avg_pool(x).squeeze(-1)  # (batch_size, features)
        
        # Classification
        output = self.classifier(x)
        
        return output

# 設定ファイル読み込み
def load_config():
    """reference/config.yamlがあれば読み込み、なければデフォルト設定を使用"""
    config_path = Path('reference/config.yaml')
    if config_path.exists():
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                yaml_config = yaml.safe_load(f)
            logger.info("✅ reference/config.yamlを読み込みました")
            return yaml_config
        except Exception as e:
            logger.warning(f"設定ファイル読み込みエラー: {e}")
            logger.info("デフォルト設定を使用します")
    else:
        logger.info("reference/config.yamlが見つかりません。デフォルト設定を使用します")
    
    return DEFAULT_CONFIG

# モデル読み込み（アップロードされたファイルまたはデフォルト）
def load_model(model_file=None):
    # 設定を読み込み（reference/config.yamlがあれば優先使用）
    config = load_config()
    model = SoundAnomalyDetector(config)
    
    if model_file is not None:
        try:
            # アップロードされたファイルからモデル読み込み
            import io
            model_data = torch.load(io.BytesIO(model_file.read()), map_location='cpu')
            model.load_state_dict(model_data)
            logger.info("アップロードされたモデルを読み込みました")
            st.success("✅ カスタムモデルを読み込みました")
        except Exception as e:
            logger.error(f"モデル読み込みエラー: {e}")
            st.error(f"モデル読み込みに失敗しました: {e}")
            st.info("ベースラインモデルを使用します")
    else:
        # referenceフォルダのbest_model.pthをチェック
        reference_model_path = Path('reference/best_model.pth')
        default_model_path = Path('models/best_model.pth')
        
        model_loaded = False
        
        # まずreferenceフォルダを試す
        if reference_model_path.exists():
            try:
                state_dict = torch.load(reference_model_path, map_location='cpu')
                model.load_state_dict(state_dict)
                logger.info("reference フォルダの訓練済みモデルを読み込みました")
                st.success("✅ referenceフォルダのモデルを読み込みました")
                model_loaded = True
            except Exception as e:
                logger.warning(f"referenceモデル読み込み失敗: {e}")
        
        # 次にmodelsフォルダを試す
        if not model_loaded and default_model_path.exists():
            try:
                state_dict = torch.load(default_model_path, map_location='cpu')
                model.load_state_dict(state_dict)
                logger.info("models フォルダの訓練済みモデルを読み込みました")
                st.info("📁 modelsフォルダのモデルを使用中")
                model_loaded = True
            except Exception as e:
                logger.warning(f"デフォルトモデル読み込み失敗: {e}")
        
        if not model_loaded:
            # ベースラインモデルの初期化
            logger.info("ベースラインモデルを初期化します")
            st.info("🤖 ベースラインモデルを使用します")
            _initialize_baseline_model(model)
    
    model.eval()
    return model

# ベースラインモデルの初期化
def _initialize_baseline_model(model):
    try:
        for name, param in model.named_parameters():
            if 'weight' in name:
                if len(param.shape) > 1:
                    torch.nn.init.xavier_uniform_(param)
                else:
                    torch.nn.init.uniform_(param, -0.1, 0.1)
            elif 'bias' in name:
                torch.nn.init.constant_(param, 0)
        logger.info("ベースラインモデルの重みを初期化しました")
    except Exception as e:
        logger.warning(f"ベースライン初期化失敗: {e}")

# 音声前処理
def preprocess_audio(audio_data, sample_rate=44100):
    # 正規化
    audio_data = audio_data.astype(np.float32)
    if np.max(np.abs(audio_data)) > 0:
        audio_data = audio_data / np.max(np.abs(audio_data))
    
    # 1秒間（44100サンプル）に調整
    if len(audio_data) > sample_rate:
        audio_data = audio_data[:sample_rate]
    elif len(audio_data) < sample_rate:
        audio_data = np.pad(audio_data, (0, sample_rate - len(audio_data)), mode='constant')
    
    return audio_data

# 音声分析
def analyze_audio(audio_data, model, sample_rate=44100):
    # 1秒毎に分割
    segments = []
    predictions = []
    confidences = []
    
    total_seconds = len(audio_data) // sample_rate
    
    for i in range(total_seconds):
        start_idx = i * sample_rate
        end_idx = start_idx + sample_rate
        segment = audio_data[start_idx:end_idx]
        
        # 前処理
        processed_segment = preprocess_audio(segment, sample_rate)
        segments.append(processed_segment)
        
        # モデル予測
        with torch.no_grad():
            input_tensor = torch.tensor(processed_segment).unsqueeze(0)
            output = model(input_tensor)
            probabilities = torch.softmax(output, dim=1)
            
            confidence, prediction = torch.max(probabilities, 1)
            predictions.append(prediction.item())
            confidences.append(confidence.item())
    
    return segments, predictions, confidences

# 結果プロット（信頼度付き）
def plot_results(audio_data, predictions, sample_rate=44100, confidences=None):
    fig, ax = plt.subplots(figsize=(15, 6))
    
    # 時間軸
    time_axis = np.linspace(0, len(audio_data) / sample_rate, len(audio_data))
    
    # 波形プロット
    ax.plot(time_axis, audio_data, color='blue', alpha=0.7, linewidth=0.5)
    
    # OK/NG区間の背景色
    for i, pred in enumerate(predictions):
        start_time = i
        end_time = i + 1
        color = 'lightgreen' if pred == 0 else 'lightcoral'
        label = 'OK' if pred == 0 else 'NG'
        
        # 信頼度に応じてアルファ値を調整
        if confidences and i < len(confidences):
            alpha = 0.2 + (confidences[i] * 0.4)  # 0.2-0.6の範囲
            confidence_text = f" ({confidences[i]:.2f})"
        else:
            alpha = 0.3
            confidence_text = ""
        
        ax.axvspan(start_time, end_time, alpha=alpha, color=color)
        
        # 中央にラベル表示（信頼度付き）
        mid_time = start_time + 0.5
        ax.text(mid_time, max(audio_data) * 0.8, f"{label}{confidence_text}", 
                ha='center', va='center', fontsize=10, fontweight='bold')
        
        # 信頼度バーを下部に表示
        if confidences and i < len(confidences):
            bar_height = min(audio_data) * 0.1 * confidences[i]
            ax.axvspan(start_time, end_time, ymin=0, ymax=0.05, 
                      alpha=0.8, color='darkblue')
    
    ax.set_xlabel('時間 (秒)', fontsize=12)
    ax.set_ylabel('振幅', fontsize=12)
    ax.set_title('音声波形と異常検知結果（信頼度付き）', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # 凡例
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='lightgreen', alpha=0.5, label='OK (正常)'),
        Patch(facecolor='lightcoral', alpha=0.5, label='NG (異常)'),
        Patch(facecolor='darkblue', alpha=0.8, label='信頼度バー')
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout()
    return fig

# WAVファイル処理関数
def process_wav_file(file_path: str, model, sample_rate: int = 44100):
    """
    WAVファイルを読み込んでモデルで処理
    
    Args:
        file_path: WAVファイルパス
        model: 音声検知モデル
        sample_rate: サンプリング周波数
        
    Returns:
        tuple: (音声データ, 予測結果, 信頼度)
    """
    try:
        # WAVファイル読み込み
        audio_data, sr = SimpleAudioRecorder.load_wav_file(file_path)
        
        if len(audio_data) == 0:
            return None, None, None
        
        # サンプリング周波数変換（必要に応じて）
        if sr != sample_rate:
            audio_data = librosa.resample(audio_data, orig_sr=sr, target_sr=sample_rate)
        
        # 音声分析
        segments, predictions, confidences = analyze_audio(audio_data, model, sample_rate)
        
        return audio_data, predictions, confidences
        
    except Exception as e:
        logger.error(f"WAVファイル処理エラー: {e}")
        return None, None, None

# メイン処理
def main():
    st.title("🎵 音声異常検知アプリ - Simple Recording")
    st.markdown("**シンプルな録音機能でWAVファイルを保存し、1秒毎にOK/NGを判定します**")
    
    # サイドバー設定
    st.sidebar.header("⚙️ 設定")
    
    # オーディオデバイス情報表示
    with st.sidebar.expander("🎤 オーディオデバイス情報"):
        try:
            devices = SimpleAudioRecorder.get_available_devices()
            if devices:
                st.success(f"✅ {len(devices)}個のデバイスが利用可能")
                for device in devices:
                    st.write(f"• {device['name']} ({device['channels']}ch)")
            else:
                st.warning("⚠️ 入力デバイスが見つかりません")
                st.info("""**デバイスが見つからない理由:**
                • マイクが接続されていない
                • 権限設定が必要
                • システム音声ドライバーの問題
                • 他のアプリがマイクを占有""")
                st.markdown("**解決方法:**")
                st.code("python install_dependencies.py")
        except Exception as e:
            st.error(f"❌ デバイス情報取得エラー: {e}")
            logger.error(f"Audio device query error: {e}")
    
    # モデルアップロード
    st.sidebar.subheader("🤖 モデル設定")
    uploaded_model = st.sidebar.file_uploader(
        "モデルファイル (.pth) をアップロード", 
        type=['pth'],
        help="訓練済みPyTorchモデルファイルをアップロードしてください"
    )
    
    sample_rate = st.sidebar.selectbox("サンプリング周波数", [22050, 44100], index=1)
    
    # モデル読み込み
    if 'model' not in st.session_state or uploaded_model:
        with st.spinner("モデルを読み込み中..."):
            st.session_state.model = load_model(uploaded_model)
    
    # 録音設定
    st.sidebar.subheader("📹 録音設定")
    recording_duration = st.sidebar.slider("録音時間 (秒)", 1, 30, 5)
    
    # 音声録音セクション
    st.header("🎤 音声録音 (WAV保存)")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("**Simple Recorderを使用したWAV録音機能**")
        
        # 録音機能の利用可否を表示
        if not RECORDER_AVAILABLE:
            st.warning("⚠️ 録音機能が利用できません")
            st.info(f"**エラー詳細**: {RECORDER_ERROR}")
            
            # システム依存関係の解決方法を表示
            with st.expander("🔧 録音機能を有効にする方法", expanded=True):
                st.markdown("""
                録音機能を有効にするには、以下の手順を実行してください：
                
                **1. システム依存関係のインストール**
                ```bash
                # インストールスクリプトを実行
                python install_dependencies.py
                ```
                
                **2. 手動インストール（上記が失敗した場合）**
                
                **Google Colab環境の場合:**
                ```bash
                !apt-get update -qq
                !apt-get install -y portaudio19-dev python3-pyaudio alsa-utils
                !pip install sounddevice>=0.4.0
                ```
                
                **Linux環境の場合:**
                ```bash
                sudo apt-get update
                sudo apt-get install -y portaudio19-dev python3-pyaudio alsa-utils
                pip install sounddevice>=0.4.0
                ```
                
                **3. 代替手段**
                現在は**WAVファイルアップロード機能**をご利用ください（下記参照）
                """)
            
            # ファイルアップロードボタンを強調表示
            st.markdown("---")
            st.markdown("### 📂 代替手段: WAVファイルをアップロード")
            if st.button("📤 WAVファイルアップロード機能に移動", use_container_width=True):
                st.rerun()
            st.markdown("---")
        else:
            st.success("✅ 録音機能が利用可能です")
        
        # 録音制御
        if 'recorder' not in st.session_state:
            st.session_state.recorder = SimpleAudioRecorder(sample_rate=sample_rate)
        
        recorder = st.session_state.recorder
        
        # 録音状態の初期化
        if 'is_recording' not in st.session_state:
            st.session_state.is_recording = False
        if 'last_recording_data' not in st.session_state:
            st.session_state.last_recording_data = None
        if 'last_wav_file' not in st.session_state:
            st.session_state.last_wav_file = None
        
        # 録音ボタン
        col_rec1, col_rec2, col_rec3 = st.columns(3)
        
        with col_rec1:
            if st.button("🎙️ 録音開始", type="primary", disabled=st.session_state.is_recording):
                try:
                    # 録音ファイル名を生成
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    wav_filename = f"recording_{timestamp}.wav"
                    wav_path = Path("recordings") / wav_filename
                    
                    # recordingsフォルダ作成
                    Path("recordings").mkdir(exist_ok=True)
                    
                    st.session_state.is_recording = True
                    st.session_state.current_wav_file = str(wav_path)
                    
                    # 進捗表示用のプレースホルダー
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    # 録音実行（スレッドコンテキストエラー対策）
                    def record_audio():
                        # Streamlitコンテキスト外で実行するため、直接状態更新は避ける
                        progress_data = {'current': 0, 'total': recording_duration}
                        
                        def progress_callback(current_time, total_time):
                            progress_data['current'] = current_time
                            # UI更新はメインスレッドに委譲（ログのみ記録）
                            logger.info(f"録音進捗: {current_time:.1f}/{total_time:.1f}秒")
                        
                        success = recorder.record_and_save(
                            duration=recording_duration,
                            file_path=str(wav_path),
                            progress_callback=progress_callback
                        )
                        
                        # セッション状態の更新（スレッドセーフ）
                        st.session_state.is_recording = False
                        st.session_state.recording_result = {
                            'success': success,
                            'wav_path': str(wav_path),
                            'wav_filename': wav_filename
                        }
                    
                    # 録音スレッド開始
                    recording_thread = threading.Thread(target=record_audio)
                    recording_thread.daemon = True
                    recording_thread.start()
                    
                    # 進捗表示（メインスレッドで）
                    while st.session_state.is_recording and recording_thread.is_alive():
                        progress_bar.progress(min(time.time() - time.time(), 1.0))
                        status_text.text(f"録音中...")
                        time.sleep(0.1)
                    
                    # 結果処理
                    if 'recording_result' in st.session_state:
                        result = st.session_state.recording_result
                        if result['success']:
                            st.session_state.last_wav_file = result['wav_path']
                            st.success(f"✅ 録音完了！ {result['wav_filename']} に保存されました")
                            
                            # 録音データを読み込み
                            audio_data, sr = SimpleAudioRecorder.load_wav_file(result['wav_path'])
                            st.session_state.last_recording_data = audio_data
                        else:
                            st.error("❌ 録音に失敗しました")
                        
                        # クリーンアップ
                        del st.session_state.recording_result
                    
                    progress_bar.empty()
                    status_text.empty()
                    
                except Exception as e:
                    st.error(f"録音開始エラー: {e}")
                    st.session_state.is_recording = False
        
        with col_rec2:
            if st.button("⏹️ 録音停止", type="secondary", disabled=not st.session_state.is_recording):
                if recorder.is_recording:
                    audio_data = recorder.stop_recording()
                    if len(audio_data) > 0:
                        # WAVファイルに保存
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        wav_filename = f"recording_{timestamp}_manual.wav"
                        wav_path = Path("recordings") / wav_filename
                        Path("recordings").mkdir(exist_ok=True)
                        
                        if recorder.save_to_wav(audio_data, str(wav_path)):
                            st.session_state.last_wav_file = str(wav_path)
                            st.session_state.last_recording_data = audio_data
                            st.success(f"✅ 録音停止・保存完了！ {wav_filename}")
                        else:
                            st.error("❌ WAVファイル保存に失敗しました")
                    else:
                        st.warning("録音データがありません")
                
                st.session_state.is_recording = False
        
        with col_rec3:
            if st.button("🔄 リセット"):
                # 録音停止
                if recorder.is_recording:
                    recorder.stop_recording()
                
                # セッションクリア
                for key in ['last_recording_data', 'last_wav_file', 'is_recording', 
                           'audio_data', 'predictions', 'confidences']:
                    if key in st.session_state:
                        del st.session_state[key]
                
                st.success("リセットしました")
                st.rerun()
        
        # 録音状態表示
        if st.session_state.is_recording:
            st.warning("🔴 録音中...")
        elif st.session_state.last_wav_file:
            st.info(f"📁 最新の録音: {Path(st.session_state.last_wav_file).name}")
        
        # WAVファイル分析ボタン
        if st.session_state.last_wav_file and not st.session_state.is_recording:
            if st.button("🔍 WAVファイル分析", type="primary"):
                with st.spinner("WAVファイルを分析中..."):
                    audio_data, predictions, confidences = process_wav_file(
                        st.session_state.last_wav_file, 
                        st.session_state.model, 
                        sample_rate
                    )
                    
                    if audio_data is not None:
                        # 結果をセッションに保存
                        st.session_state.audio_data = audio_data
                        st.session_state.predictions = predictions
                        st.session_state.confidences = confidences
                        st.session_state.sample_rate = sample_rate
                        st.success("🎯 WAVファイル分析完了！")
                    else:
                        st.error("WAVファイルの分析に失敗しました")
    
    with col2:
        st.markdown("### 📊 録音状況")
        
        if st.session_state.last_recording_data is not None:
            duration = len(st.session_state.last_recording_data) / sample_rate
            st.metric("録音時間", f"{duration:.1f}秒")
            st.metric("データサイズ", f"{len(st.session_state.last_recording_data):,} サンプル")
            st.metric("ファイル", Path(st.session_state.last_wav_file).name if st.session_state.last_wav_file else "なし")
        else:
            st.metric("録音時間", "0.0秒")
            st.metric("データサイズ", "0 サンプル")
            st.metric("ファイル", "なし")
    
    # 保存されたWAVファイル一覧
    st.header("📁 保存されたWAVファイル")
    recordings_path = Path("recordings")
    if recordings_path.exists():
        wav_files = list(recordings_path.glob("*.wav"))
        if wav_files:
            # ファイル選択
            selected_file = st.selectbox(
                "WAVファイルを選択",
                options=[f.name for f in sorted(wav_files, reverse=True)],
                help="分析したいWAVファイルを選択してください"
            )
            
            if selected_file:
                selected_path = recordings_path / selected_file
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    if st.button("🔍 選択ファイル分析"):
                        with st.spinner(f"{selected_file} を分析中..."):
                            audio_data, predictions, confidences = process_wav_file(
                                str(selected_path), 
                                st.session_state.model, 
                                sample_rate
                            )
                            
                            if audio_data is not None:
                                st.session_state.audio_data = audio_data
                                st.session_state.predictions = predictions
                                st.session_state.confidences = confidences
                                st.session_state.sample_rate = sample_rate
                                st.success(f"✅ {selected_file} の分析完了！")
                            else:
                                st.error(f"❌ {selected_file} の分析に失敗しました")
                
                with col2:
                    # ダウンロードボタン
                    if selected_path.exists():
                        with open(selected_path, "rb") as f:
                            st.download_button(
                                label="📥 ダウンロード",
                                data=f.read(),
                                file_name=selected_file,
                                mime="audio/wav"
                            )
                
                with col3:
                    # ファイル削除
                    if st.button("🗑️ ファイル削除", type="secondary"):
                        if selected_path.exists():
                            selected_path.unlink()
                            st.success(f"✅ {selected_file} を削除しました")
                            st.rerun()
        else:
            st.info("保存されたWAVファイルはありません")
    else:
        st.info("recordingsフォルダが見つかりません")
    
    # 音声ファイルアップロード機能
    if not RECORDER_AVAILABLE:
        st.header("📤 音声ファイルアップロード（メイン機能）")
        st.info("🎤 録音機能が利用できないため、WAVファイルアップロード機能をメインとしてご利用ください")
    else:
        st.header("📤 音声ファイルアップロード")
    
    uploaded_audio = st.file_uploader(
        "外部音声ファイルをアップロード", 
        type=['wav', 'mp3', 'flac', 'm4a'],
        help="外部の音声ファイルをアップロードして分析できます。録音機能が利用できない環境では、この機能をメインとしてご利用ください。"
    )
    
    if uploaded_audio:
        with st.spinner("音声ファイルを処理中..."):
            try:
                # 一時ファイルに保存
                with tempfile.NamedTemporaryFile(delete=False, suffix=f'.{uploaded_audio.name.split(".")[-1]}') as tmp_file:
                    tmp_file.write(uploaded_audio.read())
                    tmp_path = tmp_file.name
                
                # librosで音声読み込み
                audio_data, sr = librosa.load(tmp_path, sr=sample_rate, mono=True)
                
                if len(audio_data) > sample_rate:  # 最低1秒必要
                    st.success("✅ 音声ファイル読み込み完了！")
                    
                    # 音声分析
                    with st.spinner("音声を分析中..."):
                        segments, predictions, confidences = analyze_audio(
                            audio_data, st.session_state.model, sample_rate
                        )
                    
                    # 結果をセッションに保存
                    st.session_state.audio_data = audio_data
                    st.session_state.predictions = predictions
                    st.session_state.confidences = confidences
                    st.session_state.sample_rate = sample_rate
                else:
                    st.error("音声ファイルが短すぎます。最低1秒以上の音声が必要です。")
                    
                # 一時ファイル削除
                Path(tmp_path).unlink()
                
            except Exception as e:
                st.error(f"音声ファイル処理エラー: {e}")
    
    # 結果表示
    if 'audio_data' in st.session_state and 'predictions' in st.session_state:
        st.header("📊 分析結果")
        
        # 統計情報
        predictions = st.session_state.predictions
        confidences = st.session_state.confidences
        ok_count = predictions.count(0)
        ng_count = predictions.count(1)
        
        # 信頼度統計
        ok_confidences = [conf for pred, conf in zip(predictions, confidences) if pred == 0]
        ng_confidences = [conf for pred, conf in zip(predictions, confidences) if pred == 1]
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("総時間", f"{len(predictions)}秒")
        with col2:
            st.metric("OK（正常）", f"{ok_count} セグメント", 
                     delta=f"信頼度: {sum(ok_confidences)/len(ok_confidences):.2f}" if ok_confidences else "信頼度: N/A")
        with col3:
            st.metric("NG（異常）", f"{ng_count} セグメント",
                     delta=f"信頼度: {sum(ng_confidences)/len(ng_confidences):.2f}" if ng_confidences else "信頼度: N/A")
        with col4:
            st.metric("平均信頼度", f"{avg_confidence:.3f}")
        
        # 信頼度の色分け表示
        st.subheader("🎯 信頼度サマリー")
        
        confidence_cols = st.columns(len(predictions))
        for i, (pred, conf) in enumerate(zip(predictions, confidences)):
            with confidence_cols[i]:
                status = "✅ OK" if pred == 0 else "❌ NG"
                color = "green" if pred == 0 else "red"
                st.markdown(f"**{i+1}秒目**")
                st.markdown(f"<div style='color: {color}; font-weight: bold;'>{status}</div>", unsafe_allow_html=True)
                st.progress(conf)
                st.caption(f"信頼度: {conf:.3f}")
        
        # 波形グラフ（信頼度付き）
        fig = plot_results(st.session_state.audio_data, predictions, 
                          st.session_state.sample_rate, confidences)
        st.pyplot(fig)
        
        # 詳細結果テーブル
        st.subheader("📋 詳細結果テーブル")
        
        # データフレーム作成
        import pandas as pd
        
        results_data = []
        for i, (pred, conf) in enumerate(zip(predictions, confidences)):
            status_emoji = "✅" if pred == 0 else "❌"
            status_text = "OK (正常)" if pred == 0 else "NG (異常)"
            confidence_level = "高" if conf > 0.8 else "中" if conf > 0.5 else "低"
            
            results_data.append({
                "時刻": f"{i+1}秒目",
                "判定": f"{status_emoji} {status_text}",
                "信頼度": f"{conf:.3f}",
                "信頼度レベル": confidence_level,
                "確信度": f"{conf*100:.1f}%"
            })
        
        results_df = pd.DataFrame(results_data)
        st.dataframe(results_df, use_container_width=True)
    
    # 説明
    st.markdown("---")
    st.markdown("### 📖 使い方")
    st.markdown("""
    #### 🎤 Simple Recording方式:
    1. **録音時間設定**: サイドバーで録音時間を選択
    2. **録音開始**: 「🎙️ 録音開始」ボタンでWAV録音開始
    3. **自動停止**: 設定時間で自動停止・WAV保存
    4. **分析実行**: 「🔍 WAVファイル分析」で解析開始
    
    #### 📁 WAVファイル管理:
    - 録音されたWAVファイルは`recordings/`フォルダに保存
    - ファイル一覧から過去の録音を選択・分析可能
    - ダウンロード・削除機能付き
    
    #### 📤 ファイルアップロード:
    - 外部音声ファイル（WAV, MP3等）のアップロード・分析
    
    **判定について:**
    - 🟢 **OK（正常）**: 正常な音声と判定
    - 🔴 **NG（異常）**: 異常な音声と判定
    
    **改善点:**
    - WebRTCによる複雑な録音処理を排除
    - sounddeviceによる安定した録音
    - WAVファイルへの直接保存
    - ファイル管理機能の追加
    """)

if __name__ == "__main__":
    main()