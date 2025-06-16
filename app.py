import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchaudio
import librosa
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import av
from pathlib import Path
import yaml
import time
import logging
from typing import Tuple, Optional
import warnings
import io
import tempfile

# 警告を抑制
warnings.filterwarnings("ignore")
# torch.classes警告を抑制（Streamlit互換性問題）
warnings.filterwarnings("ignore", ".*torch._classes.*")

# Streamlitページ設定
st.set_page_config(
    page_title="音声検出アプリ",
    page_icon="🎵",
    layout="wide"
)

# ログ設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# デフォルト設定
DEFAULT_CONFIG = {
    'audio': {'sample_rate': 44100},
    'model': {
        'input_length': 44100, 
        'num_classes': 2,
        'cnn_layers': [
            {'filters': 64, 'kernel_size': 3, 'stride': 1, 'padding': 1},
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
            cnn_layers.extend([
                nn.Conv1d(
                    input_channels, 
                    layer_config['filters'],
                    kernel_size=layer_config['kernel_size'],
                    stride=layer_config['stride'],
                    padding=layer_config['padding']
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

# モデル読み込み（アップロードされたファイルまたはデフォルト）
def load_model(model_file=None):
    config = DEFAULT_CONFIG
    model = SoundAnomalyDetector(config)
    
    if model_file is not None:
        try:
            # アップロードされたファイルからモデル読み込み
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

# 結果プロット
def plot_results(audio_data, predictions, sample_rate=44100):
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
        
        ax.axvspan(start_time, end_time, alpha=0.3, color=color)
        
        # 中央にラベル表示
        mid_time = start_time + 0.5
        ax.text(mid_time, max(audio_data) * 0.8, label, 
                ha='center', va='center', fontsize=12, fontweight='bold')
    
    ax.set_xlabel('時間 (秒)', fontsize=12)
    ax.set_ylabel('振幅', fontsize=12)
    ax.set_title('音声波形と異常検知結果', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # 凡例
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='lightgreen', alpha=0.5, label='OK (正常)'),
        Patch(facecolor='lightcoral', alpha=0.5, label='NG (異常)')
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout()
    return fig

# WebRTC音声録音コールバック
class AudioProcessor:
    def __init__(self):
        self.audio_frames = []
        self.recording = False
        
    def recv(self, frame):
        if self.recording:
            sound = frame.to_ndarray()
            self.audio_frames.append(sound)
        return frame

# メイン処理
def main():
    st.title("🎵 音声異常検知アプリ")
    st.markdown("**マイクで録音し、1秒毎にOK/NGを判定して波形上に表示します**")
    
    # サイドバー設定
    st.sidebar.header("⚙️ 設定")
    
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
    
    # WebRTC設定
    RTC_CONFIGURATION = RTCConfiguration({
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    })
    
    # 音声録音セクション
    st.header("🎤 音声録音")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("**下のマイクボタンを押して録音を開始してください**")
        
        # WebRTC音声ストリーミング
        audio_processor = AudioProcessor()
        
        webrtc_ctx = webrtc_streamer(
            key="audio-recorder",
            mode=WebRtcMode.SENDONLY,
            audio_receiver_size=1024,
            rtc_configuration=RTC_CONFIGURATION,
            media_stream_constraints={
                "video": False,
                "audio": {
                    "sampleRate": sample_rate,
                    "channelCount": 1,
                    "echoCancellation": False,
                    "noiseSuppression": False,
                    "autoGainControl": False,
                }
            },
            audio_html_attrs={"autoPlay": True, "controls": False, "muted": False},
        )
        
        if webrtc_ctx.audio_receiver:
            # 録音状態の初期化
            if 'recording' not in st.session_state:
                st.session_state.recording = False
            if 'audio_buffer' not in st.session_state:
                st.session_state.audio_buffer = []
            
            # 録音制御ボタン
            col_rec1, col_rec2 = st.columns(2)
            with col_rec1:
                if st.button("🎙️ 録音開始", type="primary", disabled=st.session_state.recording):
                    st.session_state.recording = True
                    st.session_state.audio_buffer = []
                    st.success("録音を開始しました！")
                    st.rerun()
                    
            with col_rec2:
                if st.button("⏹️ 録音停止", type="secondary", disabled=not st.session_state.recording):
                    st.session_state.recording = False
                    st.info("録音を停止しました")
                    st.rerun()
            
            # 録音状態の表示
            if st.session_state.recording:
                st.warning("🔴 録音中... 停止ボタンを押してください")
            elif st.session_state.audio_buffer:
                try:
                    valid_chunks = [chunk for chunk in st.session_state.audio_buffer if len(chunk) > 0]
                    if valid_chunks:
                        buffer_data = np.concatenate(valid_chunks)
                        buffer_duration = len(buffer_data) / sample_rate
                        st.info(f"📊 録音済み: {buffer_duration:.1f}秒 ({len(valid_chunks)} チャンク)")
                    else:
                        st.info("📊 録音済み: 0.0秒")
                except Exception as e:
                    logger.warning(f"Duration calculation error: {e}")
                    st.info("📊 録音済み: 計算中...")
                
            # 録音中の音声データ処理
            if webrtc_ctx.audio_receiver and st.session_state.get('recording', False):
                try:
                    # より長いタイムアウトで安定したフレーム取得
                    audio_frames = webrtc_ctx.audio_receiver.get_frames(timeout=1.0)
                    if audio_frames:
                        for audio_frame in audio_frames:
                            sound = audio_frame.to_ndarray()
                            
                            # デバッグ情報をログ出力
                            logger.debug(f"Audio frame shape: {sound.shape}, dtype: {sound.dtype}")
                            
                            # 複数チャンネルの場合はモノラルに変換
                            if len(sound.shape) > 1:
                                if sound.shape[1] > 1:  # ステレオ等
                                    sound = np.mean(sound, axis=1)
                                else:  # 既にモノラル（shape: [samples, 1]）
                                    sound = sound.flatten()
                            
                            # データ型をfloat32に変換
                            if sound.dtype != np.float32:
                                sound = sound.astype(np.float32)
                            
                            # 振幅をクリップして異常値を防ぐ
                            sound = np.clip(sound, -1.0, 1.0)
                            
                            # セッションバッファに追加
                            if len(sound) > 0:
                                st.session_state.audio_buffer.append(sound)
                                logger.debug(f"Audio chunk added: {len(sound)} samples")
                            
                except Exception as e:
                    error_msg = str(e).lower()
                    if "timeout" in error_msg:
                        # タイムアウトは正常（データがない時）
                        logger.debug("Audio frame timeout (normal)")
                    else:
                        logger.warning(f"音声フレーム処理エラー: {e}")
                        # 重大なエラーの場合のみ録音を停止
                        if "connection" in error_msg or "stream" in error_msg:
                            st.session_state.recording = False
                            st.error(f"録音接続エラー: {e}")
                            st.rerun()
        
        # 録音データ処理
        audio_buffer_available = bool(st.session_state.get('audio_buffer'))
        analysis_disabled = not audio_buffer_available or st.session_state.get('recording', False)
        
        if st.button("🔍 音声分析", disabled=analysis_disabled):
            if st.session_state.get('audio_buffer'):
                try:
                    # 音声データを結合（改善版）
                    buffer_chunks = st.session_state.audio_buffer
                    valid_chunks = [chunk for chunk in buffer_chunks if len(chunk) > 0]
                    
                    if not valid_chunks:
                        st.error("有効な音声データが見つかりません。")
                        return
                    
                    audio_data = np.concatenate(valid_chunks)
                    logger.info(f"音声データ結合完了: {len(audio_data)} サンプル ({len(audio_data)/sample_rate:.1f}秒)")
                    
                    if len(audio_data) >= sample_rate:  # 最低1秒必要
                        st.success(f"✅ 録音データを確認しました！ ({len(audio_data)/sample_rate:.1f}秒)")
                        
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
                        st.success("🎯 音声分析が完了しました！")
                    else:
                        st.error(f"録音時間が短すぎます（{len(audio_data)/sample_rate:.1f}秒）。最低1秒以上録音してください。")
                        
                except Exception as e:
                    logger.error(f"音声分析エラー: {e}")
                    st.error(f"音声分析中にエラーが発生しました: {e}")
                    # デバッグ情報を表示
                    if st.session_state.get('audio_buffer'):
                        buffer_info = [f"チャンク{i}: {len(chunk)} サンプル" for i, chunk in enumerate(st.session_state.audio_buffer[:5])]
                        st.info(f"バッファ情報（最初の5チャンク）: {', '.join(buffer_info)}")
            else:
                st.error("録音データがありません。先に録音してください。")
    
    with col2:
        st.markdown("### 📊 録音状況")
        
        if st.session_state.get('audio_buffer'):
            try:
                # バッファ内の各チャンクの長さを確認
                buffer_chunks = st.session_state.audio_buffer
                if buffer_chunks:
                    # 空でないチャンクのみを結合
                    valid_chunks = [chunk for chunk in buffer_chunks if len(chunk) > 0]
                    if valid_chunks:
                        buffer_data = np.concatenate(valid_chunks)
                        duration = len(buffer_data) / sample_rate
                        st.metric("録音時間", f"{duration:.1f}秒")
                        st.metric("データサイズ", f"{len(buffer_data):,} サンプル")
                        st.metric("音声チャンク", f"{len(valid_chunks)} 個")
                    else:
                        st.metric("録音時間", "0.0秒")
                        st.metric("データサイズ", "0 サンプル")
                else:
                    st.metric("録音時間", "0.0秒")
            except Exception as e:
                logger.warning(f"バッファ計算エラー: {e}")
                st.metric("録音時間", "計算エラー")
        else:
            st.metric("録音時間", "0.0秒")
            
        if st.button("🔄 リセット"):
            # セッションをクリア
            for key in ['audio_data', 'predictions', 'confidences', 'audio_buffer', 'recording']:
                if key in st.session_state:
                    del st.session_state[key]
            st.success("リセットしました")
            st.rerun()
    
    # 音声ファイルアップロード機能（代替手段）
    st.header("📁 音声ファイルアップロード（代替手段）")
    uploaded_audio = st.file_uploader(
        "音声ファイルをアップロード", 
        type=['wav', 'mp3', 'flac', 'm4a'],
        help="録音の代わりに音声ファイルをアップロードできます"
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
        st.header("📊 結果")
        
        # 統計情報
        predictions = st.session_state.predictions
        ok_count = predictions.count(0)
        ng_count = predictions.count(1)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("総時間", f"{len(predictions)}秒")
        with col2:
            st.metric("OK（正常）", f"{ok_count} セグメント", delta=None)
        with col3:
            st.metric("NG（異常）", f"{ng_count} セグメント", delta=None)
        
        # 波形グラフ
        fig = plot_results(st.session_state.audio_data, predictions, st.session_state.sample_rate)
        st.pyplot(fig)
        
        # 詳細結果
        with st.expander("📋 詳細結果"):
            for i, (pred, conf) in enumerate(zip(predictions, st.session_state.confidences)):
                status = "✅ OK" if pred == 0 else "❌ NG"
                st.write(f"{i+1}秒目: {status} (信頼度: {conf:.3f})")
    
    # 説明
    st.markdown("---")
    st.markdown("### 📖 使い方")
    st.markdown("""
    #### 🎤 音声録音方式:
    1. **マイク許可**: ブラウザのマイク使用許可を与える
    2. **録音開始**: 「🎙️ 録音開始」ボタンを押す
    3. **録音停止**: 「⏹️ 録音停止」ボタンを押す
    4. **分析実行**: 「🔍 音声分析」ボタンで解析開始
    
    #### 📁 ファイルアップロード方式:
    1. **ファイル選択**: 音声ファイル（WAV, MP3等）をアップロード
    2. **自動分析**: ファイルアップロード後に自動で解析開始
    
    #### 🤖 カスタムモデル:
    - サイドバーから訓練済み.pthファイルをアップロード可能
    - モデルなしの場合はベースラインモデルを使用
    
    **判定について:**
    - 🟢 **OK（正常）**: 正常な音声と判定
    - 🔴 **NG（異常）**: 異常な音声と判定
    """)

if __name__ == "__main__":
    main()