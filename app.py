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

# モデル読み込み（アップロードされたファイルまたはデフォルト）
def load_model(model_file=None):
    model = SimpleAnomalyDetector(
        input_length=DEFAULT_CONFIG['model']['input_length'],
        num_classes=DEFAULT_CONFIG['model']['num_classes']
    )
    
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
        # デフォルトモデルパスをチェック
        default_model_path = Path('models/best_model.pth')
        if default_model_path.exists():
            try:
                model.load_state_dict(torch.load(default_model_path, map_location='cpu'))
                logger.info("デフォルト訓練済みモデルを読み込みました")
                st.info("📁 デフォルトモデルを使用中")
            except Exception as e:
                logger.warning(f"デフォルトモデル読み込み失敗: {e}")
                st.info("ベースラインモデルを使用します")
        else:
            st.info("ベースラインモデルを使用します")
    
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
            # 録音状態の管理
            if st.button("🎙️ 録音開始", type="primary"):
                st.session_state.recording = True
                st.session_state.audio_buffer = []
                
            if st.button("⏹️ 録音停止", type="secondary"):
                st.session_state.recording = False
                
            # 録音中の音声データ処理
            if webrtc_ctx.audio_receiver and st.session_state.get('recording', False):
                try:
                    audio_frames = webrtc_ctx.audio_receiver.get_frames(timeout=1)
                    for audio_frame in audio_frames:
                        sound = audio_frame.to_ndarray()
                        if 'audio_buffer' not in st.session_state:
                            st.session_state.audio_buffer = []
                        st.session_state.audio_buffer.append(sound.flatten())
                        
                except Exception as e:
                    logger.warning(f"音声フレーム処理エラー: {e}")
        
        # 録音データ処理
        if st.button("🔍 音声分析", disabled=not st.session_state.get('audio_buffer')):
            if st.session_state.get('audio_buffer'):
                # 音声データを結合
                audio_data = np.concatenate(st.session_state.audio_buffer)
                
                if len(audio_data) > sample_rate:  # 最低1秒必要
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
                else:
                    st.error("録音時間が短すぎます。最低1秒以上録音してください。")
            else:
                st.error("録音データがありません。先に録音してください。")
    
    with col2:
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