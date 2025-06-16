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
import logging
from typing import Tuple, Optional
import tempfile
import copy

# Streamlitページ設定
st.set_page_config(
    page_title="音声異常検知アプリ - ファイルアップロード",
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

# モデルアーキテクチャ自動検出機能
def detect_model_architecture(model_path: Path) -> Optional[dict]:
    """
    保存されたモデルから自動的にアーキテクチャを検出する
    
    Args:
        model_path: モデルファイルのパス
        
    Returns:
        検出されたアーキテクチャに基づく設定辞書、または None
    """
    try:
        # チェックポイントを読み込んでアーキテクチャを検査
        checkpoint = torch.load(model_path, map_location='cpu')
        
        # CNN層の情報をstate_dictから抽出
        cnn_layers = []
        
        # 重みの形状を分析してCNN層を検出
        layer_idx = 0
        while f'cnn.{layer_idx}.weight' in checkpoint:
            weight_shape = checkpoint[f'cnn.{layer_idx}.weight'].shape
            
            # CNN層のパターン: 重みの形状は [out_channels, in_channels, kernel_size]
            if len(weight_shape) == 3:
                filters = weight_shape[0]
                in_channels = weight_shape[1]
                kernel_size = weight_shape[2]
                
                # ストライドとパディングを層パターンから推測（典型的な値）
                stride = 2 if layer_idx > 0 else 1  # 最初の層は通常stride=1、その他はstride=2
                padding = 1 if kernel_size == 3 else 0
                
                cnn_layers.append({
                    'filters': int(filters),
                    'kernel_size': int(kernel_size),
                    'stride': stride,
                    'padding': padding
                })
                
                logger.info(f"   検出されたCNN層 {layer_idx//4}: {filters} フィルタ, カーネル {kernel_size}")
            
            # 次のCNN層にスキップ（各層はweight, bias, batch_norm weight, batch_norm biasを持つ）
            layer_idx += 4
        
        if not cnn_layers:
            logger.warning("チェックポイントからCNN層のアーキテクチャを検出できませんでした")
            return None
        
        # アテンション層の次元を検出
        attention_hidden_dim = 256  # デフォルト
        attention_num_heads = 8     # デフォルト
        
        # アテンション層の重みから hidden_dim を検出
        if 'attention.query.weight' in checkpoint:
            attention_weight_shape = checkpoint['attention.query.weight'].shape
            attention_hidden_dim = attention_weight_shape[0]
            logger.info(f"   検出されたアテンション hidden_dim: {attention_hidden_dim}")
        
        # 全結合層の次元を検出
        fc_layers = []
        fc_idx = 0
        while f'classifier.{fc_idx}.weight' in checkpoint:
            weight_shape = checkpoint[f'classifier.{fc_idx}.weight'].shape
            
            if len(weight_shape) == 2:  # Linear層
                out_features = weight_shape[0]
                in_features = weight_shape[1]
                
                # 最終出力層をスキップ（num_classes）
                next_fc_idx = fc_idx + 3  # ReLUとDropoutをスキップ
                if f'classifier.{next_fc_idx}.weight' in checkpoint:
                    fc_layers.append({
                        'units': int(out_features),
                        'dropout': 0.3  # デフォルトのドロップアウト
                    })
                    logger.info(f"   検出されたFC層 {len(fc_layers)}: {out_features} ユニット")
            
            fc_idx += 3  # ReLUとDropout層をスキップ
        
        # 検出されたアーキテクチャで設定を作成
        detected_config = copy.deepcopy(DEFAULT_CONFIG)  # 元の設定をディープコピー
        
        # 検出された値で更新
        detected_config['model']['cnn_layers'] = cnn_layers
        detected_config['model']['attention']['hidden_dim'] = attention_hidden_dim
        detected_config['model']['attention']['num_heads'] = attention_num_heads
        
        if fc_layers:
            detected_config['model']['fully_connected'] = fc_layers
        
        logger.info(f"✅ モデルアーキテクチャの検出に成功:")
        logger.info(f"   CNN層: {len(cnn_layers)}層、フィルタ数 {[l['filters'] for l in cnn_layers]}")
        logger.info(f"   アテンション: {attention_hidden_dim}次元、{attention_num_heads}ヘッド")
        logger.info(f"   FC層: {len(fc_layers)}層")
        
        return detected_config
        
    except Exception as e:
        logger.warning(f"⚠️ モデルアーキテクチャを検出できませんでした: {e}")
        return None

# モデル読み込み（アップロードされたファイルまたはデフォルト）
def load_model(model_file=None):
    """
    モデルを読み込む（アーキテクチャ自動検出機能付き）
    """
    config = load_config()
    
    if model_file is not None:
        try:
            # アップロードされたファイルからモデル読み込み
            import io
            
            # 一時ファイルに保存してアーキテクチャ検出を実行
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pth') as tmp_file:
                tmp_file.write(model_file.read())
                tmp_path = tmp_file.name
            
            # アーキテクチャ検出を試行
            detected_config = detect_model_architecture(Path(tmp_path))
            
            if detected_config:
                logger.info("🔧 検出されたアーキテクチャでモデルを作成中")
                model = SoundAnomalyDetector(detected_config)
                st.info("🔍 アップロードされたモデルからアーキテクチャを自動検出しました")
            else:
                logger.info("📋 デフォルト設定でモデルを作成中")
                model = SoundAnomalyDetector(config)
            
            # モデルの重みを読み込み
            model_data = torch.load(tmp_path, map_location='cpu')
            model.load_state_dict(model_data)
            
            # 一時ファイルを削除
            Path(tmp_path).unlink()
            
            logger.info("アップロードされたモデルを読み込みました")
            st.success("✅ カスタムモデルを読み込みました")
            
        except Exception as e:
            logger.error(f"モデル読み込みエラー: {e}")
            st.error(f"モデル読み込みに失敗しました: {e}")
            st.info("ベースラインモデルを使用します")
            model = SoundAnomalyDetector(config)
            _initialize_baseline_model(model)
    else:
        # referenceフォルダのモデルファイルをチェック
        model_candidates = [
            Path('reference/best_model.pth'),
            Path('reference/old_best_model.pth'),
            Path('models/best_model.pth')
        ]
        
        model_loaded = False
        
        for model_path in model_candidates:
            if model_path.exists():
                try:
                    # アーキテクチャ検出を試行
                    detected_config = detect_model_architecture(model_path)
                    
                    if detected_config:
                        logger.info(f"🔧 {model_path.name}からアーキテクチャを検出してモデルを作成中")
                        model = SoundAnomalyDetector(detected_config)
                        st.info(f"🔍 {model_path.name}からアーキテクチャを自動検出しました")
                    else:
                        logger.info(f"📋 {model_path.name}に対してデフォルト設定でモデルを作成中")
                        model = SoundAnomalyDetector(config)
                    
                    # モデルの重みを読み込み
                    state_dict = torch.load(model_path, map_location='cpu')
                    model.load_state_dict(state_dict)
                    
                    logger.info(f"{model_path}から訓練済みモデルを読み込みました")
                    st.success(f"✅ {model_path.name}を読み込みました")
                    model_loaded = True
                    break
                    
                except Exception as e:
                    logger.warning(f"{model_path}のモデル読み込み失敗: {e}")
                    continue
        
        if not model_loaded:
            # ベースラインモデルの初期化
            logger.info("ベースラインモデルを初期化します")
            st.info("🤖 ベースラインモデルを使用します")
            model = SoundAnomalyDetector(config)
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

# メイン処理
def main():
    st.title("🎵 音声異常検知アプリ - ファイルアップロード")
    st.markdown("**音声ファイルをアップロードして、1秒毎にOK/NGを判定します**")
    
    # 機能改訂の説明
    st.info("""
    🔄 **アプリが新しくなりました！**
    
    - 📤 **ファイルアップロード専用**: より安定した音声分析体験を提供
    - 🔍 **自動アーキテクチャ検出**: 異なるモデル形状に自動対応
    - 🎯 **シンプルで直感的**: 複雑な録音設定は不要
    """)
    
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
    
    # 音声ファイルアップロード機能
    st.header("📤 音声ファイルアップロード")
    
    uploaded_audio = st.file_uploader(
        "音声ファイルをアップロードしてください", 
        type=['wav', 'mp3', 'flac', 'm4a', 'aac', 'ogg'],
        help="対応フォーマット: WAV, MP3, FLAC, M4A, AAC, OGG"
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
                    st.success(f"✅ 音声ファイル読み込み完了！ ({uploaded_audio.name})")
                    
                    # ファイル情報表示
                    duration = len(audio_data) / sample_rate
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("ファイル名", uploaded_audio.name)
                    with col2:
                        st.metric("再生時間", f"{duration:.1f}秒")
                    with col3:
                        st.metric("サンプリング周波数", f"{sample_rate}Hz")
                    
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
                    
                    st.success("🎯 音声分析完了！")
                    
                else:
                    st.error("音声ファイルが短すぎます。最低1秒以上の音声が必要です。")
                    
                # 一時ファイル削除
                Path(tmp_path).unlink()
                
            except Exception as e:
                st.error(f"音声ファイル処理エラー: {e}")
                logger.error(f"Audio file processing error: {e}")
    else:
        # ファイルがアップロードされていない場合のガイド
        st.markdown("""
        ### 📋 使用方法
        
        1. **上記のボタンをクリック**して音声ファイルを選択
        2. **ファイル形式**: WAV, MP3, FLAC, M4A, AAC, OGG
        3. **ファイルサイズ**: 最大200MB（Streamlitの制限）
        4. **音声長**: 最低1秒以上
        
        ### 💡 対応機能
        
        - 🔍 **自動アーキテクチャ検出**: 異なるモデル次元に自動対応
        - 📊 **詳細分析結果**: 1秒毎の判定と信頼度表示
        - 📈 **視覚的結果**: 波形グラフと色分け表示
        """)
    
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
    st.markdown("### 📖 アプリについて")
    st.markdown("""
    #### 🎯 主な機能:
    - **📤 ファイルアップロード**: 様々な音声形式に対応
    - **🔍 自動アーキテクチャ検出**: 異なるモデル次元に自動適応
    - **📊 詳細分析**: 1秒毎の判定と信頼度表示
    - **📈 視覚化**: 波形グラフと結果の色分け表示
    
    #### 🤖 モデル対応:
    - **reference/best_model.pth**: 最新の訓練済みモデル
    - **reference/old_best_model.pth**: 旧バージョンモデルにも対応
    - **カスタムモデル**: 独自の.pthファイルをアップロード可能
    
    #### 📝 判定について:
    - 🟢 **OK（正常）**: 正常な音声と判定
    - 🔴 **NG（異常）**: 異常な音声と判定
    - 📊 **信頼度**: 0.000-1.000の範囲で判定の確信度を表示
    """)

if __name__ == "__main__":
    main()