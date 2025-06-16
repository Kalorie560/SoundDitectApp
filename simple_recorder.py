"""
Simple Audio Recorder Module
シンプルな音声録音モジュール

WAV形式での録音・保存機能を提供
sounddeviceを使用した安定した録音機能
"""

import sounddevice as sd
import numpy as np
import wave
import threading
import time
from pathlib import Path
from typing import Optional, Callable
import logging

logger = logging.getLogger(__name__)

class SimpleAudioRecorder:
    """シンプルな音声録音クラス"""
    
    def __init__(self, sample_rate: int = 44100, channels: int = 1):
        """
        初期化
        
        Args:
            sample_rate: サンプリング周波数 (default: 44100Hz)
            channels: チャンネル数 (default: 1 for mono)
        """
        self.sample_rate = sample_rate
        self.channels = channels
        self.is_recording = False
        self.audio_data = []
        self.recording_thread = None
        self.callback_func = None
        
    def start_recording(self, callback: Optional[Callable] = None) -> bool:
        """
        録音開始
        
        Args:
            callback: 録音中に呼び出されるコールバック関数 (optional)
            
        Returns:
            bool: 録音開始の成功・失敗
        """
        if self.is_recording:
            logger.warning("既に録音中です")
            return False
            
        try:
            # 利用可能なデバイスをチェック
            devices = sd.query_devices()
            logger.info(f"利用可能なオーディオデバイス数: {len(devices)}")
            
            # 録音データをリセット
            self.audio_data = []
            self.callback_func = callback
            self.is_recording = True
            
            # 録音スレッドを開始
            self.recording_thread = threading.Thread(target=self._record_audio)
            self.recording_thread.daemon = True
            self.recording_thread.start()
            
            logger.info(f"録音開始: {self.sample_rate}Hz, {self.channels}ch")
            return True
            
        except Exception as e:
            logger.error(f"録音開始エラー: {e}")
            self.is_recording = False
            return False
    
    def stop_recording(self) -> np.ndarray:
        """
        録音停止
        
        Returns:
            np.ndarray: 録音された音声データ
        """
        if not self.is_recording:
            logger.warning("録音中ではありません")
            return np.array([])
        
        self.is_recording = False
        
        # 録音スレッドの終了を待つ
        if self.recording_thread and self.recording_thread.is_alive():
            self.recording_thread.join(timeout=2.0)
        
        # 音声データを結合
        if self.audio_data:
            audio_array = np.concatenate(self.audio_data)
            logger.info(f"録音停止: {len(audio_array)} サンプル ({len(audio_array)/self.sample_rate:.1f}秒)")
            return audio_array
        else:
            logger.warning("録音データがありません")
            return np.array([])
    
    def _record_audio(self):
        """録音処理（内部メソッド）"""
        try:
            def audio_callback(indata, frames, time, status):
                """sounddeviceコールバック"""
                if status:
                    logger.warning(f"Audio callback status: {status}")
                
                if self.is_recording:
                    # モノラルに変換（必要に応じて）
                    if indata.shape[1] > 1:
                        audio_mono = np.mean(indata, axis=1)
                    else:
                        audio_mono = indata[:, 0]
                    
                    # データ型をfloat32に統一
                    audio_mono = audio_mono.astype(np.float32)
                    
                    # 振幅をクリップ
                    audio_mono = np.clip(audio_mono, -1.0, 1.0)
                    
                    self.audio_data.append(audio_mono)
                    
                    # コールバック関数を呼び出し（存在する場合）
                    if self.callback_func:
                        try:
                            self.callback_func(len(audio_mono), len(self.audio_data))
                        except Exception as e:
                            logger.warning(f"コールバック関数エラー: {e}")
            
            # 録音開始
            with sd.InputStream(
                callback=audio_callback,
                channels=self.channels,
                samplerate=self.sample_rate,
                dtype='float32',
                blocksize=4096  # バッファサイズ
            ):
                while self.is_recording:
                    time.sleep(0.1)  # CPUを占有しないよう待機
                    
        except Exception as e:
            logger.error(f"録音処理エラー: {e}")
            self.is_recording = False
    
    def save_to_wav(self, audio_data: np.ndarray, file_path: str) -> bool:
        """
        音声データをWAVファイルに保存
        
        Args:
            audio_data: 音声データ (numpy array)
            file_path: 保存先ファイルパス
            
        Returns:
            bool: 保存の成功・失敗
        """
        try:
            # パスの作成
            path = Path(file_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            
            # 16bit整数に変換（WAV標準）
            audio_int16 = (audio_data * 32767).astype(np.int16)
            
            # WAVファイル保存
            with wave.open(str(path), 'wb') as wav_file:
                wav_file.setnchannels(self.channels)
                wav_file.setsampwidth(2)  # 16bit = 2bytes
                wav_file.setframerate(self.sample_rate)
                wav_file.writeframes(audio_int16.tobytes())
            
            logger.info(f"WAVファイル保存完了: {file_path} ({len(audio_data)/self.sample_rate:.1f}秒)")
            return True
            
        except Exception as e:
            logger.error(f"WAVファイル保存エラー: {e}")
            return False
    
    def record_and_save(self, duration: float, file_path: str, 
                       progress_callback: Optional[Callable] = None) -> bool:
        """
        指定時間録音してWAVファイルに保存
        
        Args:
            duration: 録音時間（秒）
            file_path: 保存先ファイルパス
            progress_callback: 進捗コールバック関数 (current_time, total_time)
            
        Returns:
            bool: 録音・保存の成功・失敗
        """
        try:
            # 進捗表示用のカウンタ
            progress_counter = {'current': 0.0}
            
            def progress_update(samples_added, total_chunks):
                progress_counter['current'] = min(
                    len(self.audio_data) * 4096 / self.sample_rate, 
                    duration
                )
                if progress_callback:
                    progress_callback(progress_counter['current'], duration)
            
            # 録音開始
            if not self.start_recording(callback=progress_update):
                return False
            
            # 指定時間待機
            start_time = time.time()
            while time.time() - start_time < duration and self.is_recording:
                time.sleep(0.1)
            
            # 録音停止
            audio_data = self.stop_recording()
            
            # WAVファイルに保存
            if len(audio_data) > 0:
                return self.save_to_wav(audio_data, file_path)
            else:
                logger.error("録音データが空です")
                return False
                
        except Exception as e:
            logger.error(f"録音・保存エラー: {e}")
            return False
    
    @staticmethod
    def load_wav_file(file_path: str) -> tuple[np.ndarray, int]:
        """
        WAVファイルを読み込み
        
        Args:
            file_path: WAVファイルパス
            
        Returns:
            tuple: (音声データ, サンプリング周波数)
        """
        try:
            with wave.open(file_path, 'rb') as wav_file:
                # WAVファイル情報取得
                channels = wav_file.getnchannels()
                sample_width = wav_file.getsampwidth()
                sample_rate = wav_file.getframerate()
                frames = wav_file.getnframes()
                
                # 音声データ読み込み
                raw_data = wav_file.readframes(frames)
                
                # numpy配列に変換
                if sample_width == 1:
                    dtype = np.uint8
                elif sample_width == 2:
                    dtype = np.int16
                else:
                    dtype = np.int32
                
                audio_data = np.frombuffer(raw_data, dtype=dtype)
                
                # ステレオの場合はモノラルに変換
                if channels == 2:
                    audio_data = audio_data.reshape(-1, 2)
                    audio_data = np.mean(audio_data, axis=1)
                
                # -1.0 to 1.0の範囲に正規化
                if dtype == np.int16:
                    audio_data = audio_data.astype(np.float32) / 32767.0
                elif dtype == np.uint8:
                    audio_data = (audio_data.astype(np.float32) - 128) / 128.0
                else:
                    audio_data = audio_data.astype(np.float32) / (2**31 - 1)
                
                logger.info(f"WAVファイル読み込み完了: {file_path} ({len(audio_data)/sample_rate:.1f}秒)")
                return audio_data, sample_rate
                
        except Exception as e:
            logger.error(f"WAVファイル読み込みエラー: {e}")
            return np.array([]), 0
    
    @staticmethod
    def get_available_devices():
        """利用可能なオーディオデバイス一覧を取得"""
        try:
            devices = sd.query_devices()
            input_devices = []
            
            for i, device in enumerate(devices):
                if device['max_input_channels'] > 0:
                    input_devices.append({
                        'id': i,
                        'name': device['name'],
                        'channels': device['max_input_channels'],
                        'sample_rate': device['default_samplerate']
                    })
            
            return input_devices
        except Exception as e:
            logger.error(f"デバイス一覧取得エラー: {e}")
            return []


def test_recording():
    """録音機能のテスト"""
    recorder = SimpleAudioRecorder()
    
    print("利用可能なデバイス:")
    devices = SimpleAudioRecorder.get_available_devices()
    for device in devices:
        print(f"  {device['id']}: {device['name']} ({device['channels']}ch)")
    
    print("\n3秒間の録音テストを開始...")
    
    def progress_callback(current, total):
        print(f"\r録音中: {current:.1f}/{total:.1f}秒", end="")
    
    success = recorder.record_and_save(
        duration=3.0,
        file_path="test_recording.wav",
        progress_callback=progress_callback
    )
    
    if success:
        print("\n✅ 録音テスト成功！test_recording.wav に保存されました")
        
        # 読み込みテスト
        audio_data, sample_rate = SimpleAudioRecorder.load_wav_file("test_recording.wav")
        if len(audio_data) > 0:
            print(f"✅ 読み込みテスト成功！ {len(audio_data)} サンプル, {sample_rate}Hz")
        else:
            print("❌ 読み込みテスト失敗")
    else:
        print("\n❌ 録音テスト失敗")


if __name__ == "__main__":
    # ログレベル設定
    logging.basicConfig(level=logging.INFO)
    
    # テスト実行
    test_recording()