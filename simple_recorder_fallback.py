"""
Simple Audio Recorder Fallback Module
シンプルな音声録音モジュール（フォールバック版）

sounddeviceが利用できない環境用の代替実装
ファイルアップロード機能を提供
"""

import numpy as np
import wave
import logging
from pathlib import Path
from typing import Optional, Callable
import io

logger = logging.getLogger(__name__)

class FallbackAudioRecorder:
    """フォールバック音声録音クラス（ファイルアップロード対応）"""
    
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
        logger.info("フォールバックモード: ファイルアップロード機能のみ利用可能")
        
    def start_recording(self, callback: Optional[Callable] = None) -> bool:
        """
        録音開始（フォールバック版では無効）
        
        Returns:
            bool: 常にFalse（録音機能無効）
        """
        logger.warning("録音機能は利用できません。WAVファイルをアップロードしてください。")
        return False
    
    def stop_recording(self) -> np.ndarray:
        """
        録音停止（フォールバック版では空配列を返す）
        
        Returns:
            np.ndarray: 空の配列
        """
        logger.warning("録音機能は利用できません")
        return np.array([])
    
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
        録音・保存（フォールバック版では無効）
        
        Returns:
            bool: 常にFalse（録音機能無効）
        """
        logger.warning("録音機能は利用できません。WAVファイルをアップロードしてください。")
        return False
    
    def load_wav_from_bytes(self, wav_bytes: bytes) -> tuple[np.ndarray, int]:
        """
        アップロードされたWAVバイトデータから音声データを読み込み
        
        Args:
            wav_bytes: WAVファイルのバイトデータ
            
        Returns:
            tuple: (音声データ, サンプリング周波数)
        """
        try:
            # バイトデータからWAVファイルを読み込み
            wav_io = io.BytesIO(wav_bytes)
            
            with wave.open(wav_io, 'rb') as wav_file:
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
                
                logger.info(f"WAVバイトデータ読み込み完了: {len(audio_data)} サンプル, {sample_rate}Hz ({len(audio_data)/sample_rate:.1f}秒)")
                return audio_data, sample_rate
                
        except Exception as e:
            logger.error(f"WAVバイトデータ読み込みエラー: {e}")
            return np.array([]), 0
    
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
        """利用可能なオーディオデバイス一覧（フォールバック版では空リスト）"""
        logger.warning("オーディオデバイス機能は利用できません")
        return []
    
    @staticmethod
    def get_environment_info():
        """環境情報を取得"""
        import platform
        import os
        
        env_info = {
            'platform': platform.system(),
            'platform_version': platform.version(),
            'python_version': platform.python_version(),
            'is_colab': 'COLAB_GPU' in os.environ,
            'sounddevice_available': False
        }
        
        return env_info


def create_recorder_instance():
    """
    環境に応じた録音インスタンスを作成
    
    Returns:
        録音インスタンス（通常版またはフォールバック版）
    """
    try:
        # sounddeviceのインポートを試行
        import sounddevice as sd
        
        # 簡単な動作テスト
        devices = sd.query_devices()
        logger.info(f"sounddevice利用可能: {len(devices)}デバイス検出")
        
        # 通常版のSimpleAudioRecorderを使用
        from simple_recorder import SimpleAudioRecorder
        return SimpleAudioRecorder()
        
    except ImportError as e:
        logger.warning(f"sounddeviceが利用できません: {e}")
        logger.info("フォールバック版を使用します（ファイルアップロードのみ）")
        return FallbackAudioRecorder()
        
    except Exception as e:
        logger.warning(f"sounddevice初期化エラー: {e}")
        logger.info("フォールバック版を使用します（ファイルアップロードのみ）")
        return FallbackAudioRecorder()


if __name__ == "__main__":
    # ログレベル設定
    logging.basicConfig(level=logging.INFO)
    
    # 環境テスト
    recorder = create_recorder_instance()
    
    if isinstance(recorder, FallbackAudioRecorder):
        print("🔄 フォールバックモードで動作中")
        print("📂 WAVファイルアップロード機能のみ利用可能")
        
        # 環境情報表示
        env_info = FallbackAudioRecorder.get_environment_info()
        print("\n📊 環境情報:")
        for key, value in env_info.items():
            print(f"  {key}: {value}")
        
        print("\n💡 録音機能を有効にするには:")
        print("  python install_dependencies.py")
        print("  を実行してください")
    else:
        print("✅ 通常モードで動作中")
        print("🎤 録音機能が利用可能")