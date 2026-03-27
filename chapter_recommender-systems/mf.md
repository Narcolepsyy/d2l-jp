# 行列分解

行列分解 :cite:`Koren.Bell.Volinsky.2009` は、推薦システムの文献で確立されたアルゴリズムです。行列分解モデルの最初のバージョンは、Simon Funk によって有名な[ブログ記事](https://sifter.org/%7Esimon/journal/20061211.html)で提案され、そこで彼は相互作用行列を因子分解するというアイデアを説明しました。その後、2006 年に開催された Netflix コンテストによって広く知られるようになりました。当時、メディアストリーミングおよびビデオレンタル企業である Netflix は、推薦システムの性能を改善するためのコンテストを発表しました。Netflix のベースライン、すなわち Cinematch を 10 パーセント改善できた最良のチームには、100 万米ドルの賞金が与えられるというものでした。そのため、このコンテストは推薦システム研究分野に大きな注目を集めました。その後、グランプリは BellKor's Pragmatic Chaos チーム、すなわち BellKor、Pragmatic Theory、BigChaos の合同チームによって獲得されました（これらのアルゴリズムについては今は気にする必要はありません）。最終スコアはアンサンブル解法（すなわち、多くのアルゴリズムの組み合わせ）の結果でしたが、行列分解アルゴリズムは最終的なブレンドにおいて重要な役割を果たしました。Netflix Grand Prize 解法の技術報告書 :cite:`Toscher.Jahrer.Bell.2009` では、採用されたモデルについて詳細に紹介されています。この節では、行列分解モデルの詳細とその実装について掘り下げます。


## 行列分解モデル

行列分解は協調フィルタリングモデルの一種です。具体的には、このモデルはユーザー-アイテム相互作用行列（たとえば評価行列）を 2 つの低ランク行列の積に分解し、ユーザー-アイテム相互作用の低ランク構造を捉えます。

$\mathbf{R} \in \mathbb{R}^{m \times n}$ を、$m$ 人のユーザーと $n$ 個のアイテムからなる相互作用行列とし、$\mathbf{R}$ の値が明示的な評価を表すとします。ユーザー-アイテム相互作用は、ユーザー潜在行列 $\mathbf{P} \in \mathbb{R}^{m \times k}$ とアイテム潜在行列 $\mathbf{Q} \in \mathbb{R}^{n \times k}$ に分解されます。ここで $k \ll m, n$ は潜在因子の次元です。$\mathbf{P}$ の $u^\textrm{th}$ 行を $\mathbf{p}_u$、$\mathbf{Q}$ の $i^\textrm{th}$ 行を $\mathbf{q}_i$ とします。あるアイテム $i$ に対して、$\mathbf{q}_i$ の各要素は、そのアイテムが映画のジャンルや言語などの特徴をどの程度持っているかを表します。あるユーザー $u$ に対して、$\mathbf{p}_u$ の各要素は、そのユーザーがアイテムの対応する特徴にどの程度関心を持っているかを表します。これらの潜在因子は、前述の例のように明白な次元を表すこともあれば、まったく解釈不能なこともあります。予測評価値は次のように推定できます。

$$\hat{\mathbf{R}} = \mathbf{PQ}^\top$$

ここで $\hat{\mathbf{R}}\in \mathbb{R}^{m \times n}$ は、$\mathbf{R}$ と同じ形状を持つ予測評価行列です。この予測規則の大きな問題の 1 つは、ユーザー/アイテムのバイアスをモデル化できないことです。たとえば、あるユーザーは高い評価を付けがちであったり、あるアイテムは品質が低いために常に低い評価を受けたりします。こうしたバイアスは実世界のアプリケーションでは一般的です。これらのバイアスを捉えるために、ユーザー固有およびアイテム固有のバイアス項が導入されます。具体的には、ユーザー $u$ がアイテム $i$ に与える予測評価は次のように計算されます。

$$
\hat{\mathbf{R}}_{ui} = \mathbf{p}_u\mathbf{q}^\top_i + b_u + b_i
$$

次に、予測評価スコアと実際の評価スコアの平均二乗誤差を最小化することで、行列分解モデルを学習します。目的関数は次のように定義されます。

$$
\underset{\mathbf{P}, \mathbf{Q}, b}{\mathrm{argmin}} \sum_{(u, i) \in \mathcal{K}} \| \mathbf{R}_{ui} -
\hat{\mathbf{R}}_{ui} \|^2 + \lambda (\| \mathbf{P} \|^2_F + \| \mathbf{Q}
\|^2_F + b_u^2 + b_i^2 )
$$

ここで $\lambda$ は正則化率を表します。正則化項 $\lambda (\| \mathbf{P} \|^2_F + \| \mathbf{Q}
\|^2_F + b_u^2 + b_i^2 )$ は、パラメータの大きさにペナルティを課すことで過学習を防ぐために用いられます。$\mathbf{R}_{ui}$ が既知である $(u, i)$ の組は、集合
$\mathcal{K}=\{(u, i) \mid \mathbf{R}_{ui} \textrm{ is known}\}$ に格納されます。モデルパラメータは、確率的勾配降下法や Adam などの最適化アルゴリズムで学習できます。

行列分解モデルの直感的な図を以下に示します。

![Illustration of matrix factorization model](../img/rec-mf.svg)

この節の残りでは、行列分解の実装を説明し、MovieLens データセットでモデルを学習します。

```{.python .input  n=2}
#@tab mxnet
from d2l import mxnet as d2l
from mxnet import autograd, gluon, np, npx
from mxnet.gluon import nn
import mxnet as mx
npx.set_np()
```

## モデルの実装

まず、上で説明した行列分解モデルを実装します。ユーザーとアイテムの潜在因子は `nn.Embedding` で作成できます。`input_dim` はアイテム/ユーザーの数であり、`output_dim` は潜在因子 $k$ の次元です。`output_dim` を 1 に設定することで、`nn.Embedding` を使ってユーザー/アイテムのバイアスも作成できます。`forward` 関数では、ユーザー ID とアイテム ID を使って埋め込みを参照します。

```{.python .input  n=4}
#@tab mxnet
class MF(nn.Block):
    def __init__(self, num_factors, num_users, num_items, **kwargs):
        super(MF, self).__init__(**kwargs)
        self.P = nn.Embedding(input_dim=num_users, output_dim=num_factors)
        self.Q = nn.Embedding(input_dim=num_items, output_dim=num_factors)
        self.user_bias = nn.Embedding(num_users, 1)
        self.item_bias = nn.Embedding(num_items, 1)

    def forward(self, user_id, item_id):
        P_u = self.P(user_id)
        Q_i = self.Q(item_id)
        b_u = self.user_bias(user_id)
        b_i = self.item_bias(item_id)
        outputs = (P_u * Q_i).sum(axis=1) + np.squeeze(b_u) + np.squeeze(b_i)
        return outputs.flatten()
```

## 評価指標

次に、RMSE（root-mean-square error、二乗平均平方根誤差）を実装します。これは、モデルが予測した評価値と実際に観測された評価値（ground truth）との差を測るために一般的に用いられます :cite:`Gunawardana.Shani.2015`。RMSE は次のように定義されます。

$$
\textrm{RMSE} = \sqrt{\frac{1}{|\mathcal{T}|}\sum_{(u, i) \in \mathcal{T}}(\mathbf{R}_{ui} -\hat{\mathbf{R}}_{ui})^2}
$$

ここで $\mathcal{T}$ は、評価対象とするユーザーとアイテムの組からなる集合です。$|\mathcal{T}|$ はこの集合の大きさです。`mx.metric` が提供する RMSE 関数を使うことができます。

```{.python .input  n=3}
#@tab mxnet
def evaluator(net, test_iter, devices):
    rmse = mx.metric.RMSE()  # Get the RMSE
    rmse_list = []
    for idx, (users, items, ratings) in enumerate(test_iter):
        u = gluon.utils.split_and_load(users, devices, even_split=False)
        i = gluon.utils.split_and_load(items, devices, even_split=False)
        r_ui = gluon.utils.split_and_load(ratings, devices, even_split=False)
        r_hat = [net(u, i) for u, i in zip(u, i)]
        rmse.update(labels=r_ui, preds=r_hat)
        rmse_list.append(rmse.get()[1])
    return float(np.mean(np.array(rmse_list)))
```

## モデルの学習と評価


学習関数では、重み減衰付きの $\ell_2$ 損失を採用します。重み減衰の仕組みは、$\ell_2$ 正則化と同じ効果を持ちます。

```{.python .input  n=4}
#@tab mxnet
#@save
def train_recsys_rating(net, train_iter, test_iter, loss, trainer, num_epochs,
                        devices=d2l.try_all_gpus(), evaluator=None,
                        **kwargs):
    timer = d2l.Timer()
    animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0, 2],
                            legend=['train loss', 'test RMSE'])
    for epoch in range(num_epochs):
        metric, l = d2l.Accumulator(3), 0.
        for i, values in enumerate(train_iter):
            timer.start()
            input_data = []
            values = values if isinstance(values, list) else [values]
            for v in values:
                input_data.append(gluon.utils.split_and_load(v, devices))
            train_feat = input_data[:-1] if len(values) > 1 else input_data
            train_label = input_data[-1]
            with autograd.record():
                preds = [net(*t) for t in zip(*train_feat)]
                ls = [loss(p, s) for p, s in zip(preds, train_label)]
            [l.backward() for l in ls]
            l += sum([l.asnumpy() for l in ls]).mean() / len(devices)
            trainer.step(values[0].shape[0])
            metric.add(l, values[0].shape[0], values[0].size)
            timer.stop()
        if len(kwargs) > 0:  # It will be used in section AutoRec
            test_rmse = evaluator(net, test_iter, kwargs['inter_mat'],
                                  devices)
        else:
            test_rmse = evaluator(net, test_iter, devices)
        train_l = l / (i + 1)
        animator.add(epoch + 1, (train_l, test_rmse))
    print(f'train loss {metric[0] / metric[1]:.3f}, '
          f'test RMSE {test_rmse:.3f}')
    print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec '
          f'on {str(devices)}')
```

最後に、すべてをまとめてモデルを学習しましょう。ここでは、潜在因子の次元を 30 に設定します。

```{.python .input  n=5}
#@tab mxnet
devices = d2l.try_all_gpus()
num_users, num_items, train_iter, test_iter = d2l.split_and_load_ml100k(
    test_ratio=0.1, batch_size=512)
net = MF(30, num_users, num_items)
net.initialize(ctx=devices, force_reinit=True, init=mx.init.Normal(0.01))
lr, num_epochs, wd, optimizer = 0.002, 20, 1e-5, 'adam'
loss = gluon.loss.L2Loss()
trainer = gluon.Trainer(net.collect_params(), optimizer,
                        {"learning_rate": lr, 'wd': wd})
train_recsys_rating(net, train_iter, test_iter, loss, trainer, num_epochs,
                    devices, evaluator)
```

以下では、学習済みモデルを使って、あるユーザー（ID 20）があるアイテム（ID 30）に付けるかもしれない評価を予測します。

```{.python .input  n=6}
#@tab mxnet
scores = net(np.array([20], dtype='int', ctx=devices[0]),
             np.array([30], dtype='int', ctx=devices[0]))
scores
```

## 要約

* 行列分解モデルは推薦システムで広く使われています。ユーザーがアイテムに付ける評価を予測するために利用できます。
* 推薦システム向けの行列分解を実装し、学習できます。


## 演習

* 潜在因子の大きさを変えてみましょう。潜在因子の大きさはモデル性能にどのように影響しますか？
* 異なる最適化手法、学習率、重み減衰率を試してみましょう。
* 特定の映画に対する他のユーザーの予測評価スコアを確認してみましょう。
