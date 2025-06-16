import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchaudio
import librosa
import sounddevice as sd
from pathlib import Path
import yaml
import time
import logging
from typing import Tuple, Optional
import warnings

# 警告を抑制
warnings.filterwarnings("ignore")

# Streamlitページ設定
st.set_page_config(
    page_title="音声検出アプリ",
    page_icon="🎵",
    layout="wide"
)

# ログ設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# config.yamlを読み込み
@st.cache_resource
def load_config():
    try:
        with open('reference/config.yaml', 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    except:
        # デフォルト設定
        return {
            'audio': {'sample_rate': 44100},
            'model': {'input_length': 44100, 'num_classes': 2}
        }

# 音声異常検知モデル（参考フォルダから簡略化）
class SimpleAnomalyDetector(nn.Module):
    def __init__(self, input_length=44100, num_classes=2):
        super().__init__()
        
        # 1D CNN layers
        self.conv1 = nn.Conv1d(1, 32, kernel_size=256, stride=4)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=128, stride=2)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=64, stride=2)
        
        self.pool = nn.MaxPool1d(4)
        self.dropout = nn.Dropout(0.5)
        
        # FC layers
        self.fc1 = nn.Linear(self._get_conv_output_size(input_length), 256)
        self.fc2 = nn.Linear(256, num_classes)
        
    def _get_conv_output_size(self, input_length):
        # 畳み込み層の出力サイズを計算
        x = torch.randn(1, 1, input_length)
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        return x.view(1, -1).size(1)
    
    def forward(self, x):
        # 入力は (batch_size, input_length)
        x = x.unsqueeze(1)  # (batch_size, 1, input_length)
        
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        
        x = x.view(x.size(0), -1)
        x = self.dropout(torch.relu(self.fc1(x)))
        x = self.fc2(x)
        
        return x

# モデル読み込み
@st.cache_resource
def load_model():
    config = load_config()
    model = SimpleAnomalyDetector(
        input_length=config['model']['input_length'],
        num_classes=config['model']['num_classes']
    )
    
    # 訓練済みモデルがあれば読み込み
    model_path = Path('models/best_model.pth')
    if model_path.exists():
        try:
            model.load_state_dict(torch.load(model_path, map_location='cpu'))
            logger.info("訓練済みモデルを読み込みました")
        except:
            logger.warning("訓練済みモデルの読み込みに失敗しました。ベースラインモデルを使用します")
    else:
        logger.info("訓練済みモデルが見つかりません。ベースラインモデルを使用します")
    
    model.eval()
    return model

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

# 音声録音
def record_audio(duration_seconds, sample_rate=44100):
    st.info(f"🎤 {duration_seconds}秒間録音中...")
    
    try:
        # 録音開始
        audio_data = sd.rec(int(duration_seconds * sample_rate), 
                           samplerate=sample_rate, 
                           channels=1, 
                           dtype='float32')
        
        # プログレスバー
        progress_bar = st.progress(0)
        for i in range(int(duration_seconds * 10)):
            time.sleep(0.1)
            progress_bar.progress((i + 1) / (duration_seconds * 10))
        
        sd.wait()  # 録音完了まで待機
        progress_bar.empty()
        
        return audio_data.flatten()
    
    except Exception as e:
        st.error(f"録音エラー: {e}")
        return None

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

# メイン処理
def main():
    st.title("🎵 音声異常検知アプリ")
    st.markdown("**指定した時間マイクで録音し、1秒毎にOK/NGを判定して波形上に表示します**")
    
    # サイドバー設定
    st.sidebar.header("⚙️ 設定")
    duration = st.sidebar.slider("録音時間（秒）", min_value=1, max_value=30, value=5)
    sample_rate = st.sidebar.selectbox("サンプリング周波数", [22050, 44100], index=1)
    
    # モデル読み込み
    if 'model' not in st.session_state:
        with st.spinner("モデルを読み込み中..."):
            st.session_state.model = load_model()
    
    st.success("✅ モデルが準備完了しました！")
    
    # 録音ボタン
    col1, col2 = st.columns([1, 1])
    
    with col1:
        if st.button("🎤 録音開始", type="primary"):
            # 録音実行
            audio_data = record_audio(duration, sample_rate)
            
            if audio_data is not None:
                st.success("✅ 録音完了！")
                
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
    
    with col2:
        if st.button("🔄 リセット"):
            # セッションをクリア
            for key in ['audio_data', 'predictions', 'confidences']:
                if key in st.session_state:
                    del st.session_state[key]
            st.success("リセットしました")
    
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
    1. **録音時間を設定**: サイドバーで録音時間（1-30秒）を選択
    2. **録音開始**: 「🎤 録音開始」ボタンを押してマイクに向かって話す
    3. **結果確認**: 波形グラフで1秒毎のOK/NG判定結果を確認
    4. **リセット**: 新しい録音を行う場合は「🔄 リセット」ボタンを押す
    
    **判定について:**
    - 🟢 **OK（正常）**: 正常な音声と判定
    - 🔴 **NG（異常）**: 異常な音声と判定
    """)

if __name__ == "__main__":
    main()