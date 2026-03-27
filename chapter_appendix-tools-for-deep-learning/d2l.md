# `d2l` API ドキュメント
:label:`sec_d2l`

この節では、`d2l` パッケージに含まれるクラスと関数をアルファベット順に示し、それらが本書のどこで定義されているかを示す。これにより、より詳細な実装や説明を見つけることができる。
[GitHub リポジトリ](https://github.com/d2l-ai/d2l-en/tree/master/d2l) のソースコードも参照のこと。

> 完全な API リファレンス（自動生成されたクラスおよび関数のドキュメント）は、英語版の [d2l.ai API Document](https://d2l.ai/chapter_appendix-tools-for-deep-learning/d2l.html) を参照のこと。

## クラス

| クラス名 | 説明 |
| :--- | :--- |
| `AdditiveAttention` | 加法注意機構 |
| `AddNorm` | 残差接続と層正規化 |
| `AttentionDecoder` | 注意機構付きデコーダの基底クラス |
| `Classifier` | 分類モデルの基底クラス |
| `DataModule` | データモジュールの基底クラス |
| `Decoder` | デコーダの基底クラス |
| `DotProductAttention` | スケーリングドット積注意機構 |
| `Encoder` | エンコーダの基底クラス |
| `EncoderDecoder` | エンコーダ・デコーダモデルの基底クラス |
| `FashionMNIST` | Fashion-MNIST データセット |
| `GRU` | ゲート付き回帰ユニット |
| `HyperParameters` | ハイパーパラメータのユーティリティ基底クラス |
| `LeNet` | LeNet-5 モデル |
| `LinearRegression` | 線形回帰モデル |
| `LinearRegressionScratch` | ゼロから実装した線形回帰 |
| `Module` | すべてのモデルの基底クラス |
| `MTFraEng` | 仏英機械翻訳データセット |
| `MultiHeadAttention` | マルチヘッド注意機構 |
| `PositionalEncoding` | 位置エンコーディング |
| `PositionWiseFFN` | 位置ごとのフィードフォワードネットワーク |
| `ProgressBoard` | 学習進捗の可視化ボード |
| `Residual` | 残差ブロック |
| `ResNeXtBlock` | ResNeXt ブロック |
| `RNN` | 再帰ニューラルネットワーク |
| `RNNLM` | RNN 言語モデル |
| `RNNLMScratch` | ゼロから実装した RNN 言語モデル |
| `RNNScratch` | ゼロから実装した RNN |
| `Seq2Seq` | シーケンスツーシーケンスモデル |
| `Seq2SeqEncoder` | Seq2Seq エンコーダ |
| `SGD` | 確率的勾配降下法 |
| `SoftmaxRegression` | ソフトマックス回帰モデル |
| `SyntheticRegressionData` | 合成回帰データ |
| `TimeMachine` | Time Machine テキストデータセット |
| `Trainer` | モデルの学習ユーティリティ |
| `TransformerEncoder` | Transformer エンコーダ |
| `TransformerEncoderBlock` | Transformer エンコーダブロック |
| `Vocab` | テキストの語彙 |

## 関数

| 関数名 | 説明 |
| :--- | :--- |
| `add_to_class` | クラスにメソッドを動的に追加 |
| `bleu` | BLEU スコアの計算 |
| `check_len` | 引数の長さを検証 |
| `check_shape` | テンソルの形状を検証 |
| `corr2d` | 2次元相互相関演算 |
| `cpu` | CPU デバイスを取得 |
| `gpu` | GPU デバイスを取得 |
| `init_cnn` | CNN の重みを初期化 |
| `init_seq2seq` | Seq2Seq の重みを初期化 |
| `masked_softmax` | マスク付きソフトマックス演算 |
| `num_gpus` | 利用可能な GPU 数を取得 |
| `plot` | グラフの描画 |
| `set_axes` | matplotlib の軸を設定 |
| `set_figsize` | 図のサイズを設定 |
| `show_heatmaps` | ヒートマップを表示 |
| `show_list_len_pair_hist` | リスト長のペアのヒストグラムを表示 |
| `try_all_gpus` | 利用可能な全 GPU を取得、なければ CPU |
| `try_gpu` | GPU を取得、なければ CPU |
| `use_svg_display` | SVG 形式での表示を有効化 |