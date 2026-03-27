# 生成的敵対的ネットワーク
:label:`sec_basic_gan`

本書の大部分では、予測をどのように行うかについて述べてきた。何らかの形で、深層ニューラルネットワークを用いてデータ例からラベルへの写像を学習してきた。この種の学習は識別学習と呼ばれる。たとえば、猫の写真と犬の写真を見分けられるようにしたい、という場合である。分類器と回帰器はどちらも識別学習の例である。そして、バックプロパゲーションで訓練されたニューラルネットワークは、大規模で複雑なデータセットにおける識別学習について、私たちが知っていたことを一変させた。高解像度画像での分類精度は、わずか5〜6年で役に立たない水準から人間並みの水準へと（いくつかの注意点はあるものの）向上した。深層ニューラルネットワークが驚くほど高い性能を発揮する、他のあらゆる識別タスクについて、ここで改めて長々と述べることは控えよう。

しかし、機械学習は識別タスクを解くだけではない。たとえば、大規模なデータセットがあってもラベルがない場合、そのデータの特徴を簡潔に捉えるモデルを学習したいことがある。そのようなモデルがあれば、訓練データの分布に似た合成データ例をサンプリングできる。たとえば、顔写真の大規模なコーパスがあるとき、同じデータセットから来たとしても不自然ではないような、新しい写実的な画像を生成したいかもしれない。この種の学習は生成モデリングと呼ばれる。

最近まで、斬新な写実的画像を合成する方法は存在しなかった。しかし、識別学習における深層ニューラルネットワークの成功が、新たな可能性を切り開いた。ここ3年ほどの大きな潮流の1つは、通常は教師あり学習問題とは考えない問題における課題を克服するために、識別的な深層ネットワークを応用することであった。再帰型ニューラルネットワークによる言語モデルは、識別ネットワーク（次の文字を予測するように訓練されたもの）を用い、訓練後には生成モデルとして機能しうる例である。

2014年、画期的な論文が生成的敵対的ネットワーク（GAN）を導入した :cite:`Goodfellow.Pouget-Abadie.Mirza.ea.2014`。これは、識別モデルの力を活用して優れた生成モデルを得るための、巧妙で新しい方法である。GANの核心にあるのは、偽データと本物のデータを見分けられないなら、そのデータ生成器は優れている、という考え方である。統計学では、これは2標本検定と呼ばれる。すなわち、データセット $X=\{x_1,\ldots, x_n\}$ と $X'=\{x'_1,\ldots, x'_n\}$ が同じ分布から抽出されたかどうかを答えるための検定である。多くの統計学の論文とGANの主な違いは、後者がこの考え方を構成的に用いる点にある。言い換えると、単に「この2つのデータセットは同じ分布から来たようには見えない」と言うモデルを訓練するのではなく、[2標本検定](https://en.wikipedia.org/wiki/Two-sample_hypothesis_testing)を生成モデルへの学習信号として利用するのである。これにより、実データに似たものを生成するまでデータ生成器を改善できる。少なくとも、最先端の深層ニューラルネットワークであっても、それを欺ける必要がある。

![生成的敵対的ネットワーク](../img/gan.svg)
:label:`fig_gan`


GANのアーキテクチャを :numref:`fig_gan` に示す。
見てのとおり、GANのアーキテクチャには2つの要素がある。まず、実物そっくりのデータを生成できる可能性のある装置（たとえば深層ネットワークであるが、実際にはゲームのレンダリングエンジンのような何でもよい）を必要とする。画像を扱うなら画像を生成する必要がある。音声を扱うなら音声系列を生成する必要があり、以下同様である。これを生成器ネットワークと呼ぶ。第2の要素は識別器ネットワークである。これは偽データと本物のデータを互いに見分けようとする。両者は互いに競争関係にある。生成器ネットワークは識別器ネットワークを欺こうとする。その時点で、識別器ネットワークは新しい偽データに適応する。この情報は、今度は生成器ネットワークを改善するために使われ、これが繰り返される。


識別器は、入力 $x$ が本物（実データ由来）か偽物（生成器由来）かを判別する2値分類器である。通常、識別器は入力 $\mathbf x$ に対してスカラー予測 $o\in\mathbb R$ を出力する。たとえば、隠れサイズ1の全結合層を用い、その後にシグモイド関数を適用して予測確率 $D(\mathbf x) = 1/(1+e^{-o})$ を得る。真のデータのラベルを $1$、偽データのラベルを $0$ とする。識別器は交差エントロピー損失を最小化するように訓練する。すなわち、

$$ \min_D \{ - y \log D(\mathbf x) - (1-y)\log(1-D(\mathbf x)) \},$$

生成器については、まずランダム性の源からパラメータ $\mathbf z\in\mathbb R^d$ をサンプルする。たとえば、正規分布 $\mathbf z \sim \mathcal{N} (0, 1)$ である。$\mathbf z$ は潜在変数と呼ばれることがよくある。
そして関数を適用して $\mathbf x'=G(\mathbf z)$ を生成する。生成器の目標は、識別器を欺いて $\mathbf x'=G(\mathbf z)$ を真のデータとして分類させること、すなわち $D( G(\mathbf z)) \approx 1$ とすることである。
言い換えると、与えられた識別器 $D$ に対して、$y=0$ のときの交差エントロピー損失を最大化するように生成器 $G$ のパラメータを更新する。すなわち、

$$ \max_G \{ - (1-y) \log(1-D(G(\mathbf z))) \} = \max_G \{ - \log(1-D(G(\mathbf z))) \}.$$

生成器が完全にうまくいけば、$D(\mathbf x')\approx 1$ となるので、上の損失は0に近くなり、その結果、識別器が十分に進歩するための勾配が小さすぎるという問題が生じる。そこで通常は、次の損失を最小化する。

$$ \min_G \{ - y \log(D(G(\mathbf z))) \} = \min_G \{ - \log(D(G(\mathbf z))) \}, $$

これは、$\mathbf x'=G(\mathbf z)$ を識別器に入力する一方で、ラベル $y=1$ を与えるのと同じである。


要するに、$D$ と $G$ は次の包括的な目的関数を用いた「ミニマックス」ゲームを行っている。

$$\min_D \max_G \{ -E_{x \sim \textrm{Data}} \log D(\mathbf x) - E_{z \sim \textrm{Noise}} \log(1 - D(G(\mathbf z))) \}.$$



GANの応用の多くは画像の文脈にある。デモンストレーションの目的として、まずはもっと単純な分布の当てはめに満足することにしよう。GANを使ってガウス分布のパラメータを推定する、世界で最も非効率な推定器を構築するとどうなるかを示す。始めよう。

```{.python .input}
#@tab mxnet
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import autograd, gluon, init, np, npx
from mxnet.gluon import nn
npx.set_np()
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
import torch
from torch import nn
```

```{.python .input}
#@tab tensorflow
from d2l import tensorflow as d2l
import tensorflow as tf
```

## いくつかの「実データ」を生成する

これは世界で最もつまらない例になるので、単にガウス分布から抽出したデータを生成する。

```{.python .input}
#@tab mxnet, pytorch
X = d2l.normal(0.0, 1, (1000, 2))
A = d2l.tensor([[1, 2], [-0.1, 0.5]])
b = d2l.tensor([1, 2])
data = d2l.matmul(X, A) + b
```

```{.python .input}
#@tab tensorflow
X = d2l.normal((1000, 2), 0.0, 1)
A = d2l.tensor([[1, 2], [-0.1, 0.5]])
b = d2l.tensor([1, 2], tf.float32)
data = d2l.matmul(X, A) + b
```

得られたものを見てみよう。これは、平均 $b$、共分散行列 $A^TA$ をもつガウス分布が、かなり任意な形で平行移動されたものになるはずである。

```{.python .input}
#@tab mxnet, pytorch
d2l.set_figsize()
d2l.plt.scatter(d2l.numpy(data[:100, 0]), d2l.numpy(data[:100, 1]));
print(f'The covariance matrix is\n{d2l.matmul(A.T, A)}')
```

```{.python .input}
#@tab tensorflow
d2l.set_figsize()
d2l.plt.scatter(d2l.numpy(data[:100, 0]), d2l.numpy(data[:100, 1]));
print(f'The covariance matrix is\n{tf.matmul(A, A, transpose_a=True)}')
```

```{.python .input}
#@tab all
batch_size = 8
data_iter = d2l.load_array((data,), batch_size)
```

## 生成器

生成器ネットワークは、可能な限り最も単純なネットワーク、つまり1層の線形モデルにする。これは、線形ネットワークをガウス型のデータ生成器で駆動するからである。したがって、完全に偽装するためのパラメータを学習するだけでよいのである。

```{.python .input}
#@tab mxnet
net_G = nn.Sequential()
net_G.add(nn.Dense(2))
```

```{.python .input}
#@tab pytorch
net_G = nn.Sequential(nn.Linear(2, 2))
```

```{.python .input}
#@tab tensorflow
net_G = tf.keras.layers.Dense(2)
```

## 識別器

識別器については、もう少し識別的にいきよう。少し面白くするために、3層のMLPを使う。

```{.python .input}
#@tab mxnet
net_D = nn.Sequential()
net_D.add(nn.Dense(5, activation='tanh'),
          nn.Dense(3, activation='tanh'),
          nn.Dense(1))
```

```{.python .input}
#@tab pytorch
net_D = nn.Sequential(
    nn.Linear(2, 5), nn.Tanh(),
    nn.Linear(5, 3), nn.Tanh(),
    nn.Linear(3, 1))
```

```{.python .input}
#@tab tensorflow
net_D = tf.keras.models.Sequential([
    tf.keras.layers.Dense(5, activation="tanh", input_shape=(2,)),
    tf.keras.layers.Dense(3, activation="tanh"),
    tf.keras.layers.Dense(1)
])
```

## 訓練

まず、識別器を更新する関数を定義する。

```{.python .input}
#@tab mxnet
#@save
def update_D(X, Z, net_D, net_G, loss, trainer_D):
    """識別器を更新する。"""
    batch_size = X.shape[0]
    ones = np.ones((batch_size,), ctx=X.ctx)
    zeros = np.zeros((batch_size,), ctx=X.ctx)
    with autograd.record():
        real_Y = net_D(X)
        fake_X = net_G(Z)
        # `net_G` の勾配を計算する必要はないので、勾配計算から切り離す。
        fake_Y = net_D(fake_X.detach())
        loss_D = (loss(real_Y, ones) + loss(fake_Y, zeros)) / 2
    loss_D.backward()
    trainer_D.step(batch_size)
    return float(loss_D.sum())
```

```{.python .input}
#@tab pytorch
#@save
def update_D(X, Z, net_D, net_G, loss, trainer_D):
    """識別器を更新する。"""
    batch_size = X.shape[0]
    ones = torch.ones((batch_size,), device=X.device)
    zeros = torch.zeros((batch_size,), device=X.device)
    trainer_D.zero_grad()
    real_Y = net_D(X)
    fake_X = net_G(Z)
    # `net_G` の勾配を計算する必要はないので、勾配計算から切り離す。
    fake_Y = net_D(fake_X.detach())
    loss_D = (loss(real_Y, ones.reshape(real_Y.shape)) +
              loss(fake_Y, zeros.reshape(fake_Y.shape))) / 2
    loss_D.backward()
    trainer_D.step()
    return loss_D
```

```{.python .input}
#@tab tensorflow
#@save
def update_D(X, Z, net_D, net_G, loss, optimizer_D):
    """識別器を更新する。"""
    batch_size = X.shape[0]
    ones = tf.ones((batch_size,)) # 実データに対応するラベル
    zeros = tf.zeros((batch_size,)) # 偽データに対応するラベル
    # `net_G` の勾配を計算する必要はないので、GradientTape の外に置く
    fake_X = net_G(Z)
    with tf.GradientTape() as tape:
        real_Y = net_D(X)
        fake_Y = net_D(fake_X)
        # PyTorch の BCEWithLogitsLoss に合わせるため、損失に batch_size を掛ける
        loss_D = (loss(ones, tf.squeeze(real_Y)) + loss(
            zeros, tf.squeeze(fake_Y))) * batch_size / 2
    grads_D = tape.gradient(loss_D, net_D.trainable_variables)
    optimizer_D.apply_gradients(zip(grads_D, net_D.trainable_variables))
    return loss_D
```

生成器も同様に更新する。ここでは交差エントロピー損失を再利用するが、偽データのラベルを $0$ から $1$ に変更する。

```{.python .input}
#@tab mxnet
#@save
def update_G(Z, net_D, net_G, loss, trainer_G):
    """生成器を更新する。"""
    batch_size = Z.shape[0]
    ones = np.ones((batch_size,), ctx=Z.ctx)
    with autograd.record():
        # 計算を節約するために `update_D` での `fake_X` を再利用してもよい
        fake_X = net_G(Z)
        # `net_D` が変化するので `fake_Y` の再計算が必要
        fake_Y = net_D(fake_X)
        loss_G = loss(fake_Y, ones)
    loss_G.backward()
    trainer_G.step(batch_size)
    return float(loss_G.sum())
```

```{.python .input}
#@tab pytorch
#@save
def update_G(Z, net_D, net_G, loss, trainer_G):
    """生成器を更新する。"""
    batch_size = Z.shape[0]
    ones = torch.ones((batch_size,), device=Z.device)
    trainer_G.zero_grad()
    # 計算を節約するために `update_D` での `fake_X` を再利用してもよい
    fake_X = net_G(Z)
    # `net_D` が変化するので `fake_Y` の再計算が必要
    fake_Y = net_D(fake_X)
    loss_G = loss(fake_Y, ones.reshape(fake_Y.shape))
    loss_G.backward()
    trainer_G.step()
    return loss_G
```

```{.python .input}
#@tab tensorflow
#@save
def update_G(Z, net_D, net_G, loss, optimizer_G):
    """生成器を更新する。"""
    batch_size = Z.shape[0]
    ones = tf.ones((batch_size,))
    with tf.GradientTape() as tape:
        # 計算を節約するために `update_D` での `fake_X` を再利用してもよい
        fake_X = net_G(Z)
        # `net_D` が変化するので `fake_Y` の再計算が必要
        fake_Y = net_D(fake_X)
        # PyTorch の BCEWithLogits loss に合わせるため、損失に batch_size を掛ける
        loss_G = loss(ones, tf.squeeze(fake_Y)) * batch_size
    grads_G = tape.gradient(loss_G, net_G.trainable_variables)
    optimizer_G.apply_gradients(zip(grads_G, net_G.trainable_variables))
    return loss_G
```

識別器と生成器の両方は、交差エントロピー損失を用いた2値ロジスティック回帰を行う。訓練過程を安定させるためにAdamを使う。各反復では、まず識別器を更新し、その後に生成器を更新する。両方の損失と生成例を可視化する。

```{.python .input}
#@tab mxnet
def train(net_D, net_G, data_iter, num_epochs, lr_D, lr_G, latent_dim, data):
    loss = gluon.loss.SigmoidBCELoss()
    net_D.initialize(init=init.Normal(0.02), force_reinit=True)
    net_G.initialize(init=init.Normal(0.02), force_reinit=True)
    trainer_D = gluon.Trainer(net_D.collect_params(),
                              'adam', {'learning_rate': lr_D})
    trainer_G = gluon.Trainer(net_G.collect_params(),
                              'adam', {'learning_rate': lr_G})
    animator = d2l.Animator(xlabel='epoch', ylabel='loss',
                            xlim=[1, num_epochs], nrows=2, figsize=(5, 5),
                            legend=['discriminator', 'generator'])
    animator.fig.subplots_adjust(hspace=0.3)
    for epoch in range(num_epochs):
        # 1エポック訓練する
        timer = d2l.Timer()
        metric = d2l.Accumulator(3)  # loss_D, loss_G, num_examples
        for X in data_iter:
            batch_size = X.shape[0]
            Z = np.random.normal(0, 1, size=(batch_size, latent_dim))
            metric.add(update_D(X, Z, net_D, net_G, loss, trainer_D),
                       update_G(Z, net_D, net_G, loss, trainer_G),
                       batch_size)
        # 生成例を可視化する
        Z = np.random.normal(0, 1, size=(100, latent_dim))
        fake_X = net_G(Z).asnumpy()
        animator.axes[1].cla()
        animator.axes[1].scatter(data[:, 0], data[:, 1])
        animator.axes[1].scatter(fake_X[:, 0], fake_X[:, 1])
        animator.axes[1].legend(['real', 'generated'])
        # 損失を表示する
        loss_D, loss_G = metric[0]/metric[2], metric[1]/metric[2]
        animator.add(epoch + 1, (loss_D, loss_G))
    print(f'loss_D {loss_D:.3f}, loss_G {loss_G:.3f}, '
          f'{metric[2] / timer.stop():.1f} examples/sec')
```

```{.python .input}
#@tab pytorch
def train(net_D, net_G, data_iter, num_epochs, lr_D, lr_G, latent_dim, data):
    loss = nn.BCEWithLogitsLoss(reduction='sum')
    for w in net_D.parameters():
        nn.init.normal_(w, 0, 0.02)
    for w in net_G.parameters():
        nn.init.normal_(w, 0, 0.02)
    trainer_D = torch.optim.Adam(net_D.parameters(), lr=lr_D)
    trainer_G = torch.optim.Adam(net_G.parameters(), lr=lr_G)
    animator = d2l.Animator(xlabel='epoch', ylabel='loss',
                            xlim=[1, num_epochs], nrows=2, figsize=(5, 5),
                            legend=['discriminator', 'generator'])
    animator.fig.subplots_adjust(hspace=0.3)
    for epoch in range(num_epochs):
        # 1エポック訓練する
        timer = d2l.Timer()
        metric = d2l.Accumulator(3)  # loss_D, loss_G, num_examples
        for (X,) in data_iter:
            batch_size = X.shape[0]
            Z = torch.normal(0, 1, size=(batch_size, latent_dim))
            metric.add(update_D(X, Z, net_D, net_G, loss, trainer_D),
                       update_G(Z, net_D, net_G, loss, trainer_G),
                       batch_size)
        # 生成例を可視化する
        Z = torch.normal(0, 1, size=(100, latent_dim))
        fake_X = net_G(Z).detach().numpy()
        animator.axes[1].cla()
        animator.axes[1].scatter(data[:, 0], data[:, 1])
        animator.axes[1].scatter(fake_X[:, 0], fake_X[:, 1])
        animator.axes[1].legend(['real', 'generated'])
        # 損失を表示する
        loss_D, loss_G = metric[0]/metric[2], metric[1]/metric[2]
        animator.add(epoch + 1, (loss_D, loss_G))
    print(f'loss_D {loss_D:.3f}, loss_G {loss_G:.3f}, '
          f'{metric[2] / timer.stop():.1f} examples/sec')
```

```{.python .input}
#@tab tensorflow
def train(net_D, net_G, data_iter, num_epochs, lr_D, lr_G, latent_dim, data):
    loss = tf.keras.losses.BinaryCrossentropy(
        from_logits=True, reduction=tf.keras.losses.Reduction.SUM)
    for w in net_D.trainable_variables:
        w.assign(tf.random.normal(mean=0, stddev=0.02, shape=w.shape))
    for w in net_G.trainable_variables:
        w.assign(tf.random.normal(mean=0, stddev=0.02, shape=w.shape))
    optimizer_D = tf.keras.optimizers.Adam(learning_rate=lr_D)
    optimizer_G = tf.keras.optimizers.Adam(learning_rate=lr_G)
    animator = d2l.Animator(
        xlabel="epoch", ylabel="loss", xlim=[1, num_epochs], nrows=2,
        figsize=(5, 5), legend=["discriminator", "generator"])
    animator.fig.subplots_adjust(hspace=0.3)
    for epoch in range(num_epochs):
        # 1エポック訓練する
        timer = d2l.Timer()
        metric = d2l.Accumulator(3)  # loss_D, loss_G, num_examples
        for (X,) in data_iter:
            batch_size = X.shape[0]
            Z = tf.random.normal(
                mean=0, stddev=1, shape=(batch_size, latent_dim))
            metric.add(update_D(X, Z, net_D, net_G, loss, optimizer_D),
                       update_G(Z, net_D, net_G, loss, optimizer_G),
                       batch_size)
        # 生成例を可視化する
        Z = tf.random.normal(mean=0, stddev=1, shape=(100, latent_dim))
        fake_X = net_G(Z)
        animator.axes[1].cla()
        animator.axes[1].scatter(data[:, 0], data[:, 1])
        animator.axes[1].scatter(fake_X[:, 0], fake_X[:, 1])
        animator.axes[1].legend(["real", "generated"])

        # 損失を表示する
        loss_D, loss_G = metric[0] / metric[2], metric[1] / metric[2]
        animator.add(epoch + 1, (loss_D, loss_G))

    print(f'loss_D {loss_D:.3f}, loss_G {loss_G:.3f}, '
          f'{metric[2] / timer.stop():.1f} examples/sec')
```

ここで、ガウス分布を当てはめるためのハイパーパラメータを指定する。

```{.python .input}
#@tab all
lr_D, lr_G, latent_dim, num_epochs = 0.05, 0.005, 2, 20
train(net_D, net_G, data_iter, num_epochs, lr_D, lr_G,
      latent_dim, d2l.numpy(data[:100]))
```

## まとめ

* 生成的敵対的ネットワーク（GAN）は、生成器と識別器という2つの深層ネットワークから構成される。
* 生成器は、識別器を欺くために、実画像にできるだけ近い画像を生成するように、交差エントロピー損失を最大化する。すなわち、$\max \log(D(\mathbf{x'}))$ である。
* 識別器は、生成画像と真の画像を区別しようとして、交差エントロピー損失を最小化する。すなわち、$\min - y \log D(\mathbf{x}) - (1-y)\log(1-D(\mathbf{x}))$ である。

## 演習

* 生成器が勝つ、すなわち有限サンプル上で識別器が2つの分布を区別できなくなるような平衡は存在するだろうか？
