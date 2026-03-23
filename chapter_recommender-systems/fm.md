# 因子分解機

因子分解機（FM）は、:citet:`Rendle.2010` によって提案された教師ありアルゴリズムで、分類、回帰、ランキングの各タスクに利用できます。登場するとすぐに注目を集め、予測や推薦を行うための人気が高く影響力のある手法となりました。特に、線形回帰モデルと行列分解モデルを一般化したものです。さらに、多項式カーネルを用いたサポートベクターマシンにも似ています。線形回帰や行列分解に対する因子分解機の利点は次のとおりです。(1) $\chi$-項の変数相互作用をモデル化できる。ここで $\chi$ は多項式の次数であり、通常は 2 に設定されます。(2) 因子分解機に関連する高速な最適化アルゴリズムにより、多項式計算の時間計算量を線形計算量まで削減でき、特に高次元の疎な入力に対して非常に効率的です。これらの理由から、因子分解機は現代の広告や製品推薦で広く用いられています。以下で技術的詳細と実装を説明します。


## 2-項因子分解機

形式的には、$x \in \mathbb{R}^d$ を 1 サンプルの特徴ベクトル、$y$ を対応するラベルとします。ラベルは実数値ラベルでも、二値分類の「クリック/非クリック」のようなクラスラベルでもかまいません。次数 2 の因子分解機のモデルは次のように定義されます。

$$
\hat{y}(x) = \mathbf{w}_0 + \sum_{i=1}^d \mathbf{w}_i x_i + \sum_{i=1}^d\sum_{j=i+1}^d \langle\mathbf{v}_i, \mathbf{v}_j\rangle x_i x_j
$$

ここで、$\mathbf{w}_0 \in \mathbb{R}$ は全体のバイアス、$\mathbf{w} \in \mathbb{R}^d$ は $i$ 番目の変数の重み、$\mathbf{V} \in \mathbb{R}^{d\times k}$ は特徴埋め込みを表します。$\mathbf{v}_i$ は $\mathbf{V}$ の $i^\textrm{th}$ 行を表し、$k$ は潜在因子の次元数です。$\langle\cdot, \cdot \rangle$ は 2 つのベクトルの内積です。$\langle \mathbf{v}_i, \mathbf{v}_j \rangle$ は $i^\textrm{th}$ 特徴と $j^\textrm{th}$ 特徴の相互作用をモデル化します。特徴相互作用の中には容易に理解できるものもあり、専門家が設計できます。しかし、それ以外の多くの特徴相互作用はデータの中に隠れており、特定が困難です。そのため、特徴相互作用を自動的にモデル化することで、特徴量設計の労力を大幅に削減できます。最初の 2 項が線形回帰モデルに対応し、最後の項が行列分解モデルの拡張であることは明らかです。特徴 $i$ がアイテムを表し、特徴 $j$ がユーザを表す場合、第 3 項はユーザ埋め込みとアイテム埋め込みの内積そのものです。なお、FM はより高次（次数 > 2）にも一般化できます。ただし、数値安定性が一般化性能を弱める可能性があります。


## 効率的な最適化基準

因子分解機を素朴に最適化すると、すべてのペアワイズ相互作用を計算する必要があるため、計算量は $\mathcal{O}(kd^2)$ になります。この非効率性を解決するために、FM の第 3 項を再整理して計算コストを大幅に削減し、線形時間計算量（$\mathcal{O}(kd)$）にできます。ペアワイズ相互作用項の変形は次のとおりです。

$$
\begin{aligned}
&\sum_{i=1}^d \sum_{j=i+1}^d \langle\mathbf{v}_i, \mathbf{v}_j\rangle x_i x_j \\
 &= \frac{1}{2} \sum_{i=1}^d \sum_{j=1}^d\langle\mathbf{v}_i, \mathbf{v}_j\rangle x_i x_j - \frac{1}{2}\sum_{i=1}^d \langle\mathbf{v}_i, \mathbf{v}_i\rangle x_i x_i \\
 &= \frac{1}{2} \big (\sum_{i=1}^d \sum_{j=1}^d \sum_{l=1}^k\mathbf{v}_{i, l} \mathbf{v}_{j, l} x_i x_j - \sum_{i=1}^d \sum_{l=1}^k \mathbf{v}_{i, l} \mathbf{v}_{i, l} x_i x_i \big)\\
 &=  \frac{1}{2} \sum_{l=1}^k \big ((\sum_{i=1}^d \mathbf{v}_{i, l} x_i) (\sum_{j=1}^d \mathbf{v}_{j, l}x_j) - \sum_{i=1}^d \mathbf{v}_{i, l}^2 x_i^2 \big ) \\
 &= \frac{1}{2} \sum_{l=1}^k \big ((\sum_{i=1}^d \mathbf{v}_{i, l} x_i)^2 - \sum_{i=1}^d \mathbf{v}_{i, l}^2 x_i^2)
 \end{aligned}
$$

この変形により、モデルの計算量は大幅に削減されます。さらに、疎な特徴では非ゼロ要素だけを計算すればよいため、全体の計算量は非ゼロ特徴数に対して線形になります。

FM モデルの学習には、回帰タスクでは MSE 損失、分類タスクではクロスエントロピー損失、ランキングタスクでは BPR 損失を使えます。確率的勾配降下法や Adam などの標準的な最適化手法が利用可能です。

```{.python .input  n=2}
#@tab mxnet
from d2l import mxnet as d2l
from mxnet import init, gluon, np, npx
from mxnet.gluon import nn
import os

npx.set_np()
```

## モデルの実装
以下のコードでは因子分解機を実装します。FM が線形回帰ブロックと効率的な特徴相互作用ブロックから構成されていることが分かります。CTR 予測を分類タスクとして扱うため、最終スコアにシグモイド関数を適用します。

```{.python .input  n=2}
#@tab mxnet
class FM(nn.Block):
    def __init__(self, field_dims, num_factors):
        super(FM, self).__init__()
        num_inputs = int(sum(field_dims))
        self.embedding = nn.Embedding(num_inputs, num_factors)
        self.fc = nn.Embedding(num_inputs, 1)
        self.linear_layer = nn.Dense(1, use_bias=True)

    def forward(self, x):
        square_of_sum = np.sum(self.embedding(x), axis=1) ** 2
        sum_of_square = np.sum(self.embedding(x) ** 2, axis=1)
        x = self.linear_layer(self.fc(x).sum(1)) \
            + 0.5 * (square_of_sum - sum_of_square).sum(1, keepdims=True)
        x = npx.sigmoid(x)
        return x
```

## 広告データセットの読み込み
前節の CTR データラッパーを使ってオンライン広告データセットを読み込みます。

```{.python .input  n=3}
#@tab mxnet
batch_size = 2048
data_dir = d2l.download_extract('ctr')
train_data = d2l.CTRDataset(os.path.join(data_dir, 'train.csv'))
test_data = d2l.CTRDataset(os.path.join(data_dir, 'test.csv'),
                           feat_mapper=train_data.feat_mapper,
                           defaults=train_data.defaults)
train_iter = gluon.data.DataLoader(
    train_data, shuffle=True, last_batch='rollover', batch_size=batch_size,
    num_workers=d2l.get_dataloader_workers())
test_iter = gluon.data.DataLoader(
    test_data, shuffle=False, last_batch='rollover', batch_size=batch_size,
    num_workers=d2l.get_dataloader_workers())
```

## モデルの学習
その後、モデルを学習します。学習率は 0.02、埋め込みサイズはデフォルトで 20 に設定します。モデル学習には `Adam` 最適化手法と `SigmoidBinaryCrossEntropyLoss` 損失を使用します。

```{.python .input  n=5}
#@tab mxnet
devices = d2l.try_all_gpus()
net = FM(train_data.field_dims, num_factors=20)
net.initialize(init.Xavier(), ctx=devices)
lr, num_epochs, optimizer = 0.02, 30, 'adam'
trainer = gluon.Trainer(net.collect_params(), optimizer,
                        {'learning_rate': lr})
loss = gluon.loss.SigmoidBinaryCrossEntropyLoss()
d2l.train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs, devices)
```

## まとめ

* FM は、回帰、分類、ランキングなどさまざまなタスクに適用できる汎用的な枠組みです。
* 特徴相互作用/特徴交差は予測タスクにおいて重要であり、2 項の相互作用は FM により効率的にモデル化できます。

## 演習

* Avazu、MovieLens、Criteo などの他のデータセットで FM を試せますか？
* 埋め込みサイズを変えて性能への影響を確認してください。行列分解の場合と同様の傾向が観察できますか？

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/406)
:end_tab:\n