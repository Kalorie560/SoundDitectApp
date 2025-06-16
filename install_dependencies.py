#!/usr/bin/env python3
"""
システム依存関係インストールスクリプト
音声録音機能に必要な依存関係を自動インストール
"""

import subprocess
import sys
import platform
import os

def run_command(command, description):
    """コマンドを実行してエラーハンドリング"""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} 完了")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} 失敗:")
        print(f"   エラー: {e.stderr}")
        return False

def detect_environment():
    """実行環境を検出"""
    if 'COLAB_GPU' in os.environ:
        return 'colab'
    elif platform.system().lower() == 'linux':
        return 'linux'
    elif platform.system().lower() == 'darwin':
        return 'macos'
    elif platform.system().lower() == 'windows':
        return 'windows'
    else:
        return 'unknown'

def install_system_dependencies():
    """システム依存関係をインストール"""
    env = detect_environment()
    print(f"🖥️  検出された環境: {env}")
    
    if env == 'colab':
        print("\n📋 Google Colab環境が検出されました")
        print("以下のコマンドを実行してください：")
        commands = [
            "!apt-get update -qq",
            "!apt-get install -y portaudio19-dev python3-pyaudio alsa-utils",
            "!pip install sounddevice>=0.4.0"
        ]
        for cmd in commands:
            print(f"  {cmd}")
        
        # Colab環境での自動実行を試行
        try:
            success = True
            success &= run_command("apt-get update -qq", "パッケージリスト更新")
            success &= run_command("apt-get install -y portaudio19-dev python3-pyaudio alsa-utils", "音声ライブラリインストール")
            success &= run_command("pip install sounddevice>=0.4.0", "sounddeviceインストール")
            
            if success:
                print("\n✅ Colab環境のセットアップが完了しました！")
                print("📝 注意: Colabでは実際の音声録音はできませんが、WAVファイルのアップロード分析は可能です。")
                return True
            else:
                print("\n⚠️  一部のコマンドが失敗しました。手動で実行してください。")
                return False
                
        except Exception as e:
            print(f"\n⚠️  自動インストールに失敗: {e}")
            print("上記のコマンドを手動で実行してください。")
            return False
    
    elif env == 'linux':
        print("\n📋 Linux環境が検出されました")
        distro_commands = {
            'ubuntu': "sudo apt-get update && sudo apt-get install -y portaudio19-dev python3-pyaudio alsa-utils",
            'debian': "sudo apt-get update && sudo apt-get install -y portaudio19-dev python3-pyaudio alsa-utils",
            'centos': "sudo yum install -y portaudio-devel python3-pyaudio alsa-lib-devel",
            'fedora': "sudo dnf install -y portaudio-devel python3-pyaudio alsa-lib-devel"
        }
        
        print("以下のコマンドをディストリビューションに応じて実行してください：")
        for distro, cmd in distro_commands.items():
            print(f"  {distro.upper()}: {cmd}")
        
        # Ubuntu/Debianで自動実行を試行
        try:
            success = run_command("sudo apt-get update", "パッケージリスト更新")
            success &= run_command("sudo apt-get install -y portaudio19-dev python3-pyaudio alsa-utils", "音声ライブラリインストール")
            
            if success:
                print("\n✅ Linux環境のセットアップが完了しました！")
                return True
            else:
                print("\n⚠️  sudo権限が必要、または対応していないディストリビューションです。")
                print("上記のコマンドを手動で実行してください。")
                return False
                
        except Exception as e:
            print(f"\n⚠️  自動インストールに失敗: {e}")
            return False
    
    elif env == 'macos':
        print("\n📋 macOS環境が検出されました")
        print("以下のコマンドを実行してください：")
        print("  # Homebrewを使用:")
        print("  brew install portaudio")
        print("  # または MacPortsを使用:")
        print("  sudo port install portaudio")
        
        try:
            success = run_command("brew install portaudio", "PortAudioインストール (Homebrew)")
            if success:
                print("\n✅ macOS環境のセットアップが完了しました！")
                return True
            else:
                print("\n⚠️  Homebrewが見つかりません。手動でPortAudioをインストールしてください。")
                return False
        except Exception as e:
            print(f"\n⚠️  自動インストールに失敗: {e}")
            return False
    
    elif env == 'windows':
        print("\n📋 Windows環境が検出されました")
        print("Windows環境ではsounddeviceは通常そのまま動作します。")
        print("問題がある場合は以下を確認してください：")
        print("  1. Visual Studio Build Tools がインストールされているか")
        print("  2. マイクの権限設定が有効になっているか")
        print("  3. 他のアプリケーションがマイクを使用していないか")
        return True
    
    else:
        print(f"\n⚠️  未対応の環境: {env}")
        print("手動でPortAudioライブラリをインストールしてください。")
        return False

def install_python_dependencies():
    """Python依存関係をインストール"""
    print("\n🐍 Python依存関係をインストール中...")
    
    try:
        success = run_command("pip install --upgrade pip", "pipアップグレード")
        success &= run_command("pip install sounddevice>=0.4.0 numpy", "基本依存関係インストール")
        
        if success:
            print("✅ Python依存関係のインストールが完了しました！")
            return True
        else:
            print("❌ Python依存関係のインストールに失敗しました")
            return False
    except Exception as e:
        print(f"❌ インストールエラー: {e}")
        return False

def test_sounddevice():
    """sounddeviceの動作テスト"""
    print("\n🧪 sounddeviceの動作テスト中...")
    
    try:
        import sounddevice as sd
        devices = sd.query_devices()
        print(f"✅ sounddeviceが正常に動作しています！")
        print(f"   検出されたオーディオデバイス数: {len(devices)}")
        
        # 入力デバイスの確認
        input_devices = [d for d in devices if d['max_input_channels'] > 0]
        if input_devices:
            print(f"   利用可能な入力デバイス数: {len(input_devices)}")
            print("   📱 録音機能が使用可能です！")
        else:
            print("   ⚠️  入力デバイスが見つかりません")
            print("   📂 WAVファイルアップロード機能のみ使用可能です")
        
        return True
        
    except ImportError as e:
        print(f"❌ sounddeviceのインポートに失敗: {e}")
        return False
    except Exception as e:
        print(f"❌ sounddeviceテストに失敗: {e}")
        return False

def main():
    """メイン実行関数"""
    print("🎵 SoundDitectApp 依存関係インストーラー")
    print("=" * 50)
    
    # システム依存関係のインストール
    system_success = install_system_dependencies()
    
    # Python依存関係のインストール
    python_success = install_python_dependencies()
    
    # 動作テスト
    if system_success and python_success:
        test_success = test_sounddevice()
        
        if test_success:
            print("\n🎉 すべてのセットアップが完了しました！")
            print("以下のコマンドでアプリを起動できます：")
            print("  streamlit run app.py")
        else:
            print("\n⚠️  セットアップは完了しましたが、音声機能に問題があります")
            print("WAVファイルアップロード機能は使用可能です")
            
    else:
        print("\n❌ セットアップに問題がありました")
        print("README.mdのトラブルシューティングセクションを確認してください")

if __name__ == "__main__":
    main()