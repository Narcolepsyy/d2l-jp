# AutoRec: オートエンコーダによるレーティング予測

行列分解モデルはレーティング予測タスクでまずまずの性能を達成しますが、本質的には線形モデルです。そのため、このようなモデルでは、ユーザの嗜好を予測するうえで有用となりうる複雑で非線形かつ入り組んだ関係を捉えることはできません。本節では、非線形ニューラルネットワークによる協調フィルタリングモデルである AutoRec :cite:`Sedhain.Menon.Sanner.ea.2015` を紹介します。これは、オートエンコーダアーキテクチャによって協調フィルタリング（CF）を定式化し、明示的フィードバックに基づいて CF に非線形変換を組み込むことを目指しています。ニューラルネットワークは任意の連続関数を近似できることが証明されており、行列分解の限界を補い、その表現力を高めるのに適しています。

一方で、AutoRec は入力層、隠れ層、再構成（出力）層からなるオートエンコーダと同じ構造を持ちます。オートエンコーダは、入力を隠れた（通常は低次元の）表現へ符号化するために、入力を出力へコピーすることを学習するニューラルネットワークです。AutoRec では、ユーザ/アイテムを明示的に低次元空間へ埋め込む代わりに、相互作用行列の列/行を入力として用い、出力層で相互作用行列を再構成します。

他方で、AutoRec は従来のオートエンコーダとは異なります。隠れ表現を学習するのではなく、AutoRec は出力層の学習/再構成に重点を置きます。部分的に観測された相互作用行列を入力として用い、完成されたレーティング行列を再構成することを目指します。同時に、入力の欠損要素は推薦のために再構成を通じて出力層で補完されます。

AutoRec には、ユーザベースとアイテムベースの 2 つの変種があります。簡潔さのため、ここではアイテムベースの AutoRec のみを紹介します。ユーザベースの AutoRec は同様に導出できます。


## モデル

$\mathbf{R}_{*i}$ をレーティング行列の $i^\textrm{th}$ 列とし、未知のレーティングはデフォルトで 0 に設定されているとします。ニューラルアーキテクチャは次のように定義されます。

$$
h(\mathbf{R}_{*i}) = f(\mathbf{W} \cdot g(\mathbf{V} \mathbf{R}_{*i} + \mu) + b)
$$

ここで $f(\cdot)$ と $g(\cdot)$ は活性化関数、$\mathbf{W}$ と $\mathbf{V}$ は重み行列、$\mu$ と $b$ はバイアスです。$h( \cdot )$ を AutoRec のネットワーク全体を表すものとします。出力 $h(\mathbf{R}_{*i})$ はレーティング行列の $i^\textrm{th}$ 列の再構成です。

次の目的関数は再構成誤差の最小化を目指します。

$$
\underset{\mathbf{W},\mathbf{V},\mu, b}{\mathrm{argmin}} \sum_{i=1}^M{\parallel \mathbf{R}_{*i} - h(\mathbf{R}_{*i})\parallel_{\mathcal{O}}^2} +\lambda(\| \mathbf{W} \|_F^2 + \| \mathbf{V}\|_F^2)
$$

ここで $\| \cdot \|_{\mathcal{O}}$ は観測されたレーティングの寄与のみを考慮することを意味し、すなわち、逆伝播の際には観測された入力に対応する重みのみが更新されます。

```{.python .input  n=3}
#@tab mxnet
from d2l import mxnet as d2l
from mxnet import autograd, gluon, np, npx
from mxnet.gluon import nn
import mxnet as mx

npx.set_np()
```

## モデルの実装

典型的なオートエンコーダはエンコーダとデコーダから構成されます。エンコーダは入力を隠れ表現へ写像し、デコーダは隠れ層を再構成層へ写像します。ここでもこの方法に従い、全結合層を用いてエンコーダとデコーダを作成します。エンコーダの活性化関数はデフォルトで `sigmoid` に設定し、デコーダには活性化関数を適用しません。過学習を抑えるため、エンコード変換の後にドロップアウトを入れます。観測されていない入力の勾配はマスクし、観測されたレーティングのみがモデル学習過程に寄与するようにします。

```{.python .input  n=2}
#@tab mxnet
class AutoRec(nn.Block):
    def __init__(self, num_hidden, num_users, dropout=0.05):
        super(AutoRec, self).__init__()
        self.encoder = nn.Dense(num_hidden, activation='sigmoid',
                                use_bias=True)
        self.decoder = nn.Dense(num_users, use_bias=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input):
        hidden = self.dropout(self.encoder(input))
        pred = self.decoder(hidden)
        if autograd.is_training():  # Mask the gradient during training
            return pred * np.sign(input)
        else:
            return pred
```

## 評価器の再実装

入力と出力が変更されたため、評価関数を再実装する必要がありますが、精度指標としては引き続き RMSE を用います。

```{.python .input  n=3}
#@tab mxnet
def evaluator(network, inter_matrix, test_data, devices):
    scores = []
    for values in inter_matrix:
        feat = gluon.utils.split_and_load(values, devices, even_split=False)
        scores.extend([network(i).asnumpy() for i in feat])
    recons = np.array([item for sublist in scores for item in sublist])
    # Calculate the test RMSE
    rmse = np.sqrt(np.sum(np.square(test_data - np.sign(test_data) * recons))
                   / np.sum(np.sign(test_data)))
    return float(rmse)
```

## モデルの学習と評価

それでは、MovieLens データセットで AutoRec を学習・評価してみましょう。テスト RMSE が行列分解モデルよりも低いことがはっきりと分かり、レーティング予測タスクにおけるニューラルネットワークの有効性が確認できます。

```{.python .input  n=4}
#@tab mxnet
devices = d2l.try_all_gpus()
# Load the MovieLens 100K dataset
df, num_users, num_items = d2l.read_data_ml100k()
train_data, test_data = d2l.split_data_ml100k(df, num_users, num_items)
_, _, _, train_inter_mat = d2l.load_data_ml100k(train_data, num_users,
                                                num_items)
_, _, _, test_inter_mat = d2l.load_data_ml100k(test_data, num_users,
                                               num_items)
train_iter = gluon.data.DataLoader(train_inter_mat, shuffle=True,
                                   last_batch="rollover", batch_size=256,
                                   num_workers=d2l.get_dataloader_workers())
test_iter = gluon.data.DataLoader(np.array(train_inter_mat), shuffle=False,
                                  last_batch="keep", batch_size=1024,
                                  num_workers=d2l.get_dataloader_workers())
# Model initialization, training, and evaluation
net = AutoRec(500, num_users)
net.initialize(ctx=devices, force_reinit=True, init=mx.init.Normal(0.01))
lr, num_epochs, wd, optimizer = 0.002, 25, 1e-5, 'adam'
loss = gluon.loss.L2Loss()
trainer = gluon.Trainer(net.collect_params(), optimizer,
                        {"learning_rate": lr, 'wd': wd})
d2l.train_recsys_rating(net, train_iter, test_iter, loss, trainer, num_epochs,
                        devices, evaluator, inter_mat=test_inter_mat)
```

## まとめ

* オートエンコーダを用いて行列分解アルゴリズムを定式化しつつ、非線形層とドロップアウト正則化を組み込むことができます。
* MovieLens 100K データセットでの実験により、AutoRec は行列分解よりも優れた性能を達成することが示されます。



## 演習

* AutoRec の隠れ次元を変化させ、モデル性能への影響を確認してください。
* 隠れ層をさらに追加してみてください。モデル性能の改善に役立ちますか？
* デコーダとエンコーダの活性化関数のより良い組み合わせを見つけられますか？\n
