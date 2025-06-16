#!/usr/bin/env python3
"""
Fallback機能テストスクリプト
sounddevice依存関係の問題を検証
"""

import sys
import logging

# ログ設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_sounddevice_import():
    """sounddeviceのインポートをテスト"""
    print("🧪 sounddeviceインポートテスト")
    print("-" * 40)
    
    try:
        import sounddevice as sd
        devices = sd.query_devices()
        print(f"✅ sounddevice インポート成功")
        print(f"   検出されたデバイス数: {len(devices)}")
        
        # 入力デバイスの確認
        input_devices = [d for d in devices if d['max_input_channels'] > 0]
        print(f"   入力デバイス数: {len(input_devices)}")
        
        return True, None
        
    except ImportError as e:
        print(f"❌ sounddevice インポートエラー: {e}")
        return False, str(e)
    except Exception as e:
        print(f"❌ sounddevice 初期化エラー: {e}")
        return False, str(e)

def test_recorder_modules():
    """録音モジュールのテスト"""
    print("\n🧪 録音モジュールテスト")
    print("-" * 40)
    
    # simple_recorderのテスト
    print("1. simple_recorder のテスト:")
    try:
        from simple_recorder import SimpleAudioRecorder
        recorder = SimpleAudioRecorder()
        print("   ✅ simple_recorder インポート成功")
        print("   📱 通常の録音機能が利用可能")
        return True, "simple_recorder"
    except ImportError as e:
        print(f"   ❌ simple_recorder インポートエラー: {e}")
    except Exception as e:
        print(f"   ❌ simple_recorder 初期化エラー: {e}")
    
    # fallback版のテスト
    print("\n2. simple_recorder_fallback のテスト:")
    try:
        from simple_recorder_fallback import FallbackAudioRecorder
        recorder = FallbackAudioRecorder()
        print("   ✅ simple_recorder_fallback インポート成功")
        print("   📂 ファイルアップロード機能のみ利用可能")
        return False, "fallback"
    except ImportError as e:
        print(f"   ❌ simple_recorder_fallback インポートエラー: {e}")
        return False, "error"
    except Exception as e:
        print(f"   ❌ simple_recorder_fallback 初期化エラー: {e}")
        return False, "error"

def test_app_import():
    """app.pyのインポートテスト"""
    print("\n🧪 app.pyインポートテスト")
    print("-" * 40)
    
    try:
        # app.pyをインポート（実際のStreamlitアプリは実行しない）
        import app
        print("✅ app.py インポート成功")
        
        # グローバル変数をチェック
        if hasattr(app, 'RECORDER_AVAILABLE'):
            print(f"   録音機能利用可否: {app.RECORDER_AVAILABLE}")
            if not app.RECORDER_AVAILABLE and hasattr(app, 'RECORDER_ERROR'):
                print(f"   エラー詳細: {app.RECORDER_ERROR}")
        
        return True
        
    except Exception as e:
        print(f"❌ app.py インポートエラー: {e}")
        return False

def test_environment():
    """実行環境のテスト"""
    print("\n🧪 実行環境テスト")
    print("-" * 40)
    
    import platform
    import os
    
    env_info = {
        'Platform': platform.system(),
        'Platform Version': platform.version(),
        'Python Version': platform.python_version(),
        'Is Colab': 'COLAB_GPU' in os.environ,
        'Working Directory': os.getcwd()
    }
    
    for key, value in env_info.items():
        print(f"   {key}: {value}")
    
    # 推奨される解決策を表示
    system = platform.system().lower()
    print(f"\n💡 {system} 環境での推奨解決策:")
    
    if 'colab' in str(env_info).lower() or env_info['Is Colab']:
        print("   Google Colab環境が検出されました")
        print("   以下のコマンドを実行してください:")
        print("   !apt-get update -qq")
        print("   !apt-get install -y portaudio19-dev python3-pyaudio alsa-utils")
        print("   !pip install sounddevice>=0.4.0")
    elif system == 'linux':
        print("   Linux環境が検出されました")
        print("   以下のコマンドを実行してください:")
        print("   sudo apt-get update")
        print("   sudo apt-get install -y portaudio19-dev python3-pyaudio alsa-utils")
    elif system == 'darwin':
        print("   macOS環境が検出されました")
        print("   以下のコマンドを実行してください:")
        print("   brew install portaudio")
    elif system == 'windows':
        print("   Windows環境が検出されました")
        print("   通常は追加設定不要です")
    else:
        print("   未知の環境です")

def main():
    """メインテスト実行"""
    print("🎵 SoundDitectApp Fallback機能テスト")
    print("=" * 50)
    
    # 各テストを実行
    sounddevice_ok, error = test_sounddevice_import()
    recorder_ok, recorder_type = test_recorder_modules()
    app_ok = test_app_import()
    test_environment()
    
    # 総合結果
    print("\n📊 テスト結果まとめ")
    print("-" * 40)
    
    if sounddevice_ok:
        print("✅ sounddevice: 正常動作")
        print("🎤 録音機能: 利用可能")
        print("📂 ファイルアップロード: 利用可能")
        print("🚀 推奨アクション: そのままアプリを起動してください")
    else:
        print("❌ sounddevice: 利用不可")
        print("🎤 録音機能: 利用不可")
        print("📂 ファイルアップロード: 利用可能（メイン機能）")
        print("🔧 推奨アクション: 上記の環境別解決策を実行してください")
        print("💡 代替手段: WAVファイルアップロード機能をご利用ください")
    
    if app_ok:
        print("✅ アプリ: 正常起動可能")
        print("🌐 起動コマンド: streamlit run app.py")
    else:
        print("❌ アプリ: 起動に問題がある可能性があります")
    
    print("\n" + "=" * 50)
    print("テスト完了")

if __name__ == "__main__":
    main()