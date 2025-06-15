# SoundDitectApp

1D CNNモデルを使用したリアルタイム音声録音・分類のStreamlitウェブアプリケーションです。

## 機能

- **リアルタイム音声録音**: `streamlit-webrtc`を使用してブラウザからマイク音声をキャプチャ
- **1D CNN分類**: 音声セグメントを1秒ごとにOK (0) またはNG (1) として分類
- **音声処理**: 音声前処理を自動処理（22050 Hzサンプリング、モノラル変換、長さ正規化）
- **可視化**: 時間領域波形グラフに色分けされたセグメントで結果を表示
- **モデルサポート**: 事前学習済みPyTorchモデル（.pthファイル）の読み込み

## インストール

1. リポジトリをクローン:
```bash
git clone https://github.com/Kalorie560/SoundDitectApp.git
cd SoundDitectApp
```

2. 依存関係をインストール:
```bash
pip install -r requirements.txt
```

## 使用方法

1. Streamlitアプリを実行:
```bash
streamlit run app.py
```

2. サイドバーを使用して学習済みモデル（.pthファイル）をアップロード

3. 「Start」をクリックして音声録音を開始

4. マイクに向かって話す - アプリは1秒ごとのチャンクをリアルタイムで処理

5. 「Stop」をクリックして録音を終了し、結果を表示

## モデル要件

アプリケーションは以下の仕様のPyTorchモデルを想定しています：

- **入力形状**: `(batch_size, channels, length) = (1, 1, 22050)`
- **出力**: 二値分類（2クラス: OK=0, NG=1）
- **アーキテクチャ**: `nn.Conv1d`レイヤーを使用した1D CNN
- **音声フォーマット**: 22050 Hzサンプリングレート、モノラルチャンネル、1秒セグメント

### サポートされるモデルアーキテクチャ

アプリケーションは自動的にモデルアーキテクチャを検出し、以下の形式をサポートします：

1. **個別レイヤーアーキテクチャ**: 
   - レイヤーが個別に定義されている（`conv1`, `conv2`, `fc1`, etc.）
   - 従来のモデル定義方式

2. **Sequential アーキテクチャ**:
   - レイヤーが`nn.Sequential`で組織化されている（`cnn.0`, `cnn.1`, `classifier.0`, etc.）
   - よりモジュラーなモデル設計

3. **Attention付きモデル**:
   - Attentionメカニズムを含むモデル（`attention.query`, `attention.key`, etc.）
   - より高度な特徴抽出

### 動的モデル読み込み機能 🆕

- **自動アーキテクチャ適応**: チェックポイントの重みサイズを解析して動的にモデルを構築
- **柔軟なチャンネル数**: 32→64→128 や 64→128→256 など異なるチャンnel構成に自動対応
- **可変カーネルサイズ**: kernel_size=3, 128, 256など様々なサイズに対応
- **分類器入力サイズ自動調整**: チェックポイントから実際の分類器入力次元を抽出して使用
- **プログレッシブ読み込み**: 
  1. 厳密読み込み (strict=True)
  2. 部分読み込み (strict=False) 
  3. キーマッピング変換
  4. レガシー読み込み
- **詳細ログ**: 読み込み過程での詳細な情報表示

## 音声処理

アプリは以下を自動で処理します：
- 22050 Hzへのリサンプリング
- モノラル変換（ステレオ入力の場合）
- 長さ正規化（22050サンプルへのパディングまたは切り詰め）
- 1秒セグメントへのリアルタイムチャンク分割

## 結果の可視化

録音後、アプリは以下を表示します：
- 色分けされた背景の音声波形プロット
- 緑のセグメント: OK分類
- 赤のセグメント: NG分類
- 要約統計（総持続時間、OK/NG数）
- 詳細な秒単位の結果

## 技術詳細

- StreamlitとStreamlit-webrtcで構築
- モデル推論にPyTorchを使用
- torchaudioとnumpyで音声処理
- matplotlibで可視化
- スレッドセーフな音声バッファリングと処理

## トラブルシューティング

### モデル読み込みエラー

**問題**: `Missing key(s) in state_dict`、`Unexpected key(s) in state_dict`、または`size mismatch`エラー

**✅ 自動解決機能**: 
アプリケーションは2025年6月更新で以下の問題を自動的に解決します：

1. **動的アーキテクチャ適応** 🔧
   - チェックポイントの重みを解析して適切なモデル構造を自動構築
   - 異なるチャンネル数（32→64→128 vs 64→128→256）に対応
   - 異なるカーネルサイズ（3 vs 256）に対応
   - 分類器入力次元の自動検出（例: チェックポイント [512, 256] vs 計算値 [512, 5376] の不整合を解決）

2. **プログレッシブ読み込み** 📚
   - 厳密読み込み (strict=True) → 部分読み込み (strict=False) → キーマッピング → レガシー読み込み

3. **Streamlit WebRTC API修正** 🔧
   - `ClientSettings.MediaStreamConstraints.Mode.SENDONLY` → `WebRtcMode.SENDONLY`

**対応状況の確認**: モデル読み込み時にStreamlitアプリで以下のメッセージを確認：
- "Sequential architecture detected" - Sequential形式のモデル
- "Individual layer architecture detected" - 個別レイヤー形式のモデル  
- "Attention mechanism detected" - Attention付きモデル
- "Detected channels: [64, 128, 256], kernel sizes: [3, 3, 3]" - 検出されたアーキテクチャ
- "Detected classifier input size: 256" - 分類器入力次元の自動検出
- "Model loaded successfully with strict loading" - 成功
- "Non-strict loading completed" - 一部パラメータが初期化される可能性

### 音声品質の改善

- 静かな環境での録音を推奨
- マイクとの距離を適切に保つ
- 1秒以上の音声で十分なデータを確保