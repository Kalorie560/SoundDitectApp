# 🎵 SoundDitectApp

音声異常検知Webアプリケーション - 音声信号のリアルタイムOK/NG判定システム

## 📋 概要

SoundDitectAppは、機械学習を活用したリアルタイム音声異常検知システムです。ユーザーが指定した時間でPCのマイクから録音を行い、1秒毎の音声セグメントに対してOK（正常）/NG（異常）の判定を行い、結果を波形グラフ上に視覚的に表示します。

### 🎯 主な機能

- 🎤 **リアルタイム音声録音**: PCマイクから指定時間での音声録音
- 🧠 **1D-CNN音声解析**: 1D CNNモデルによる1秒毎の音声分析
- 📊 **波形ビジュアライゼーション**: OK/NG結果を波形グラフ上に色分け表示
- 🌐 **日本語インターフェース**: 完全日本語対応のWebUI
- ⚡ **シンプル設計**: 直感的で使いやすいユーザーインターフェース

## 🚀 システム要件

### 最小要件
- **OS**: Windows 10/11, macOS 10.14+, Ubuntu 18.04+
- **Python**: 3.8以上
- **RAM**: 4GB以上
- **マイク**: PCに接続されたマイクデバイス

### 推奨要件
- **RAM**: 8GB以上
- **CPU**: Intel Core i5以上またはAMD Ryzen 5以上
- **ストレージ**: 2GB以上の空き容量

## 📦 インストール

### 1. リポジトリのクローン

```bash
git clone https://github.com/Kalorie560/SoundDitectApp.git
cd SoundDitectApp
```

### 2. 仮想環境の作成（推奨）

```bash
# Python仮想環境を作成
python -m venv venv

# 仮想環境を有効化
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate
```

### 3. 依存関係のインストール

```bash
# pipを最新バージョンにアップグレード
pip install --upgrade pip

# 必要なパッケージをインストール
pip install -r requirements.txt
```

## 🎮 使用方法

### 1. アプリケーションの起動

```bash
streamlit run app.py
```

起動後、ブラウザで `http://localhost:8501` にアクセスします。

### 2. 基本的な使い方

1. **録音時間の設定**
   - サイドバーで録音時間（1-30秒）を選択

2. **音声録音**
   - 「🎤 録音開始」ボタンをクリック
   - マイクに向かって音声を入力
   - プログレスバーで録音進行状況を確認

3. **結果の確認**
   - 自動的に音声分析が実行される
   - 波形グラフでOK/NG結果を視覚的に確認
   - 詳細結果で各セグメントの判定と信頼度を確認

4. **新しい録音**
   - 「🔄 リセット」ボタンで結果をクリア
   - 新しい録音を開始

### 3. 結果の見方

- 🟢 **緑色の背景**: OK（正常音声）と判定されたセグメント
- 🔴 **赤色の背景**: NG（異常音声）と判定されたセグメント
- **統計情報**: 総時間、OKセグメント数、NGセグメント数
- **詳細結果**: 各1秒セグメントの判定結果と信頼度

## 🏗️ プロジェクト構成

```
SoundDitectApp/
├── app.py                 # メインアプリケーション（Streamlit）
├── requirements.txt       # Python依存関係
├── README.md             # このファイル
├── reference/            # 参照用モデル訓練システム
│   ├── README.md        # 訓練システム詳細説明
│   ├── config.yaml      # モデル設定ファイル
│   ├── audio_processor.py  # 音声前処理モジュール
│   └── model_manager.py    # AIモデル管理モジュール
└── models/              # 訓練済みモデル保存先（任意）
    └── best_model.pth   # 訓練済みモデルファイル
```

## 🤖 技術仕様

### モデルアーキテクチャ
- **ベースモデル**: 1D Convolutional Neural Network (CNN)
- **入力**: 44100サンプル（1秒間の音声、44.1kHz）
- **出力**: 2クラス分類（OK: 0, NG: 1）
- **前処理**: 音声正規化、長さ調整

### 技術スタック
- **フロントエンド**: Streamlit
- **機械学習**: PyTorch, torchaudio
- **音声処理**: librosa, sounddevice
- **データ処理**: NumPy, SciPy
- **可視化**: Matplotlib

## 🎓 モデル訓練（上級者向け）

独自のデータでモデルを訓練したい場合は、`reference/`フォルダ内の訓練システムを使用できます。

### 訓練データの準備

1. `data/`フォルダにJSONファイルを配置
2. データ形式：
```json
{
  "waveforms": [
    [0.0, 0.01, -0.01, ...],  // 44100個の音声サンプル
    [0.0, 0.02, 0.01, ...]
  ],
  "labels": ["OK", "NG"],      // 対応するラベル
  "fs": 44100                  // サンプリング周波数
}
```

### 訓練の実行

```bash
# reference フォルダに移動
cd reference

# モデルを訓練
python scripts/train_model.py
```

詳細な訓練手順については `reference/README.md` を参照してください。

## 🔧 トラブルシューティング

### よくある問題と解決方法

#### 1. マイクが認識されない
```bash
# 利用可能なオーディオデバイスを確認
python -c "import sounddevice as sd; print(sd.query_devices())"
```

#### 2. 音声録音でエラーが発生する
- マイクの権限設定を確認
- 他のアプリケーションがマイクを使用していないか確認
- サンプリング周波数を22050Hzに変更してみる

#### 3. モデルが見つからない
- 初回実行時は訓練済みモデルがなくてもベースラインモデルで動作
- より高精度な結果を得るには独自データでの訓練が推奨

#### 4. パッケージのインストールエラー
```bash
# pipのアップグレード
pip install --upgrade pip

# 個別インストール
pip install streamlit torch torchaudio numpy matplotlib librosa sounddevice
```

### システム固有の問題

#### Windows
- Visual Studio Build Toolsが必要な場合があります
- Windows Defender の除外設定を確認

#### macOS
- Xcodeコマンドラインツールのインストールが必要：
```bash
xcode-select --install
```

#### Linux
- システムレベルの音声ライブラリが必要：
```bash
sudo apt-get install portaudio19-dev python3-pyaudio
```

## 🤝 貢献

プロジェクトへの貢献を歓迎します！

1. このリポジトリをフォーク
2. 機能ブランチを作成 (`git checkout -b feature/新機能`)
3. 変更をコミット (`git commit -am '新機能を追加'`)
4. ブランチにプッシュ (`git push origin feature/新機能`)
5. プルリクエストを作成

## 📄 ライセンス

このプロジェクトはMITライセンスの下で公開されています。詳細は[LICENSE](LICENSE)ファイルを参照してください。

## 🙋‍♂️ サポート

### ドキュメント
- [基本的な使い方](#-使用方法)
- [トラブルシューティング](#-トラブルシューティング)
- [モデル訓練システム](reference/README.md)

### 問題の報告
バグ報告や機能要請は[GitHub Issues](https://github.com/Kalorie560/SoundDitectApp/issues)でお願いします。

### お問い合わせ
その他のお問い合わせは[GitHub Discussions](https://github.com/Kalorie560/SoundDitectApp/discussions)をご利用ください。

---

**SoundDitectApp** - シンプルで効果的な音声異常検知ソリューション 🎵✨