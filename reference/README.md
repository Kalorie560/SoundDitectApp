# SoundDitect

音声異常検知AIモデル訓練システム / Sound Anomaly Detection AI Training System

---

## 概要

SoundDitectは、JSONデータファイルから音声異常検知AIモデルを訓練・生成する専用システムです。
1D-CNN + Attention機構を使用した高精度な音声異常検知モデルの学習が可能です。

## 主な機能

- 🧠 **AIモデル訓練**: 1D-CNN + Attention機構による高精度な異常音検知モデル
- 📊 **JSONデータ学習**: dataディレクトリ内のJSONファイルを使用した自動学習
- 📈 **実験管理**: ClearMLによる学習プロセスの記録・管理
- ⚡ **メモリ効率**: 大容量データに対応したストリーミング処理
- 🔧 **設定管理**: config.yamlによる詳細な学習パラメータ設定

## セットアップと実行手順

### 1. 環境準備

```bash
# リポジトリをクローン
git clone https://github.com/Kalorie560/SoundDitect.git
cd SoundDitect

# 仮想環境を作成・有効化
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# pipを最新バージョンにアップグレード（重要）
pip install --upgrade pip

# 依存関係をインストール
pip install -r requirements.txt
```

### 2. ClearML設定（推奨）

```bash
# ClearMLを設定（以下のいずれかの方法を選択）

# 方法1: 自動設定スクリプトを使用（推奨）
python scripts/setup_clearml.py

# 方法2: 手動設定
clearml-init

# 方法3: 環境変数で設定
export CLEARML_WEB_HOST=https://app.clear.ml
export CLEARML_API_HOST=https://api.clear.ml
export CLEARML_FILES_HOST=https://files.clear.ml
export CLEARML_API_ACCESS_KEY=your_access_key_here
export CLEARML_API_SECRET_KEY=your_secret_key_here
```

> 🌟 **ClearMLアカウント**: [https://app.clear.ml](https://app.clear.ml) でアカウント作成後、プロフィールページから認証情報を取得してください。

### 3. 学習データの準備

独自のJSONデータファイルを`./data`ディレクトリに配置してください。

**データ形式要件**:
```json
{
  "waveforms": [
    [0.0, 0.01, -0.01, 0.02, ...],  // 44100個の音声サンプル（1秒分）
    [0.0, 0.0, 0.02, 0.05, ...]
  ],
  "labels": [
    "OK",   // 正常音
    "NG"    // 異常音
  ],
  "fs": 44100,        // サンプリング周波数
  "metric": "RMS"     // 測定指標
}
```

**重要**: システムは新旧両方のデータ形式に対応しています：
- **新形式**: `{"waveforms": [...], "labels": ["OK", "NG"]}` (推奨)
- **旧形式**: `[{"Waveform": [...], "Labels": 0}]` (互換性のため)

### 4. モデル学習

```bash
# モデルを学習（すべての設定はconfig.yamlから読み込まれます）
python scripts/train_model.py
```

## プロジェクト構造

```
SoundDitect/
├── config.yaml              # 設定ファイル
├── requirements.txt          # Python依存関係
├── backend/                 # バックエンドコード
│   ├── audio_processor.py  # 音声前処理
│   └── model_manager.py    # AIモデル管理
├── scripts/                 # ユーティリティスクリプト
│   ├── train_model.py      # モデル学習
│   └── setup_clearml.py    # ClearML設定スクリプト
├── models/                  # 学習済みモデル保存先
├── data/                    # 学習データ
├── logs/                    # ログファイル
└── outputs/                 # 実験結果
```

## システム要件

**CPU要件:**
- **最小**: Intel Core i5-8th gen / AMD Ryzen 5 3600 以上
- **推奨**: Intel Core i7-10th gen / AMD Ryzen 7 4700U 以上  

**GPU要件:**
- **CPU使用時**: GPUは不要（統合グラフィックスで十分）
- **GPU使用時（オプション）**: NVIDIA GTX 1060 / RTX 3050 以上（CUDA対応）

**メモリ要件:**
- **最小**: 4GB RAM
- **推奨**: 8GB RAM 以上

**ストレージ要件:**
- **必要容量**: 2GB以上の空き容量
- **推奨**: SSD（高速なファイルアクセス）

## AIモデルアーキテクチャ

1. **前処理**: バンドパスフィルタ + 正規化
2. **1D-CNN**: 時系列パターンの特徴抽出
3. **Multi-Head Attention**: 重要な時間フレームに注目
4. **分類器**: 正常/異常の2クラス分類

## ClearML実験管理

```bash
# ClearML設定（初回のみ）
clearml-init

# 実験の可視化
clearml-serving --open
```

## カスタマイズ

`config.yaml`ファイルを編集して以下をカスタマイズできます：
- モデルアーキテクチャパラメータ
- 学習設定（エポック数、バッチサイズなど）
- 音声処理設定（サンプリング周波数、フィルタ設定）
- ClearMLプロジェクト設定

## トラブルシューティング

**ModuleNotFoundError: No module named 'yaml'**
- PyYAMLが正しくインストールされていない場合に発生します
- 解決方法:
  ```bash
  pip install --upgrade pip
  pip install pyyaml==6.0.1
  pip install -r requirements.txt
  ```

**モデル学習エラー: 'num_samples should be a positive integer value, but got num_samples=0'**
- **原因**: データ分割問題またはJSONファイルの形式エラー
- **解決方法**:
  1. ./dataディレクトリに正しい形式のJSONファイルを配置
  2. データ形式を確認: `{'waveforms': [[...]], 'labels': ['OK', 'NG'], 'fs': 44100}`

**ClearML SSL接続エラー**
- **対処**: エラーが発生してもモデル学習は継続されます（ClearMLなしで実行）

## ライセンス

This project is licensed under the MIT License - see the LICENSE file for details.