# ミニバッチ確率的勾配降下法
:label:`sec_minibatch_sgd`

これまで、勾配ベースの学習における2つの極端な方法を見てきました。 :numref:`sec_gd` では、全データセットを使って勾配を計算し、1回の走査ごとにパラメータを更新します。これに対して :numref:`sec_sgd` では、1つの訓練例を一度に処理して前進します。
どちらにもそれぞれ欠点があります。
データが非常に似通っている場合、勾配降下法は特に *データ効率* が良いわけではありません。
一方、確率的勾配降下法は、CPUやGPUがベクトル化の力を十分に活用できないため、特に *計算効率* が良いわけではありません。
このことは、その中間に何かがあるはずだと示唆しています。
実際、これまで議論してきた例では、まさにその中間を使ってきました。

## ベクトル化とキャッシュ

ミニバッチを使うかどうかの判断の核心にあるのは計算効率です。これは、複数のGPUや複数のサーバーへの並列化を考えると最も理解しやすくなります。この場合、少なくとも各GPUに1枚の画像を送る必要があります。1サーバーあたり8 GPU、16サーバーなら、ミニバッチサイズは少なくとも128になります。

単一GPU、あるいはCPUの場合は少し事情が複雑です。これらのデバイスには複数種類のメモリがあり、しばしば複数種類の計算ユニットがあり、それらの間には異なる帯域幅制約があります。
たとえばCPUには少数のレジスタがあり、その後にL1、L2、場合によってはL3キャッシュ（これは異なるプロセッサコア間で共有されます）があります。
これらのキャッシュは、サイズとレイテンシが大きくなるにつれて、同時に帯域幅は小さくなります。
要するに、プロセッサは主記憶インターフェースが供給できる量よりもはるかに多くの演算を実行できるのです。

まず、2GHzで16コア、AVX-512ベクトル化を備えたCPUは、最大で $2 \cdot 10^9 \cdot 16 \cdot 32 = 10^{12}$ バイト/秒を処理できます。GPUの能力はこの数値を容易に100倍上回ります。一方、中級クラスのサーバープロセッサでは100 GB/s程度の帯域幅しかないことがあり、つまりプロセッサに十分なデータを供給するのに必要な量の10分の1未満です。さらに悪いことに、すべてのメモリアクセスが同じではありません。メモリインターフェースは通常64ビット幅以上（たとえばGPUでは最大384ビット）なので、1バイトを読むだけでもはるかに広いアクセスのコストがかかります。

第二に、最初のアクセスにはかなりのオーバーヘッドがあり、連続アクセスは比較的安価です（これはしばしばバースト読み出しと呼ばれます）。複数ソケット、チップレット、その他の構造がある場合のキャッシュなど、考慮すべきことはまだたくさんあります。
より詳しい議論については、この [Wikipedia article](https://en.wikipedia.org/wiki/Cache_hierarchy)
を参照してください。

こうした制約を緩和する方法は、実際にプロセッサへデータを供給できるほど高速なCPUキャッシュの階層を使うことです。これこそが、深層学習におけるバッチ処理の *原動力* です。話を簡単にするために、行列積、たとえば $\mathbf{A} = \mathbf{B}\mathbf{C}$ を考えます。$\mathbf{A}$ を計算する方法はいくつかあります。たとえば、次のような選択肢があります。

1. $\mathbf{A}_{ij} = \mathbf{B}_{i,:} \mathbf{C}_{:,j}$ を計算する、つまり内積を用いて要素ごとに計算する。
1. $\mathbf{A}_{:,j} = \mathbf{B} \mathbf{C}_{:,j}$ を計算する、つまり列ごとに計算する。同様に、$\mathbf{A}$ を1行 $\mathbf{A}_{i,:}$ ずつ計算することもできる。
1. 単純に $\mathbf{A} = \mathbf{B} \mathbf{C}$ を計算する。
1. $\mathbf{B}$ と $\mathbf{C}$ をより小さなブロック行列に分割し、$\mathbf{A}$ を1ブロックずつ計算する。

最初の方法を採ると、要素 $\mathbf{A}_{ij}$ を計算するたびに、1つの行ベクトルと1つの列ベクトルをCPUにコピーする必要があります。さらに悪いことに、行列要素は順番に並んでいるため、2つのベクトルのうち一方については、メモリから読み出す際に多くの離れた場所へアクセスしなければなりません。2番目の方法のほうがはるかに有利です。この方法では、列ベクトル $\mathbf{C}_{:,j}$ をCPUキャッシュに保持したまま、$\mathbf{B}$ を走査し続けることができます。これによりメモリ帯域幅の要求が半分になり、その分アクセスも速くなります。もちろん、3番目の方法が最も望ましいです。しかし残念ながら、多くの行列はキャッシュに完全には収まりません（まさにそれがここで議論していることです）。とはいえ、4番目の方法は実用上有用な代替案を提供します。つまり、行列のブロックをキャッシュに移し、局所的に掛け合わせることができます。最適化されたライブラリがこれを代わりに行ってくれます。これらの操作が実際にどれほど効率的か見てみましょう。

計算効率に加えて、Pythonおよび深層学習フレームワーク自体によるオーバーヘッドもかなり大きいです。コマンドを実行するたびに、PythonインタプリタがMXNetエンジンへ命令を送り、それを計算グラフに挿入してスケジューリング時に処理する必要があることを思い出してください。このようなオーバーヘッドはかなり有害になりえます。要するに、可能な限りベクトル化（および行列）を使うことが強く推奨されます。

```{.python .input}
#@tab mxnet
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import autograd, gluon, init, np, npx
from mxnet.gluon import nn
import time
npx.set_np()

A = np.zeros((256, 256))
B = np.random.normal(0, 1, (256, 256))
C = np.random.normal(0, 1, (256, 256))
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
import numpy as np
import time
import torch
from torch import nn

A = torch.zeros(256, 256)
B = torch.randn(256, 256)
C = torch.randn(256, 256)
```

```{.python .input}
#@tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
import numpy as np
import tensorflow as tf
import time

A = tf.Variable(d2l.zeros((256, 256)))
B = tf.Variable(d2l.normal([256, 256], 0, 1))
C = tf.Variable(d2l.normal([256, 256], 0, 1))
```

この本の残りでは実行時間を頻繁にベンチマークするので、タイマーを定義しておきましょう。

```{.python .input}
#@tab all
class Timer:  #@save
    """複数回の実行時間を記録する。"""
    def __init__(self):
        self.times = []
        self.start()

    def start(self):
        """タイマーを開始する。"""
        self.tik = time.time()

    def stop(self):
        """タイマーを停止し、時間をリストに記録する。"""
        self.times.append(time.time() - self.tik)
        return self.times[-1]

    def avg(self):
        """平均時間を返す。"""
        return sum(self.times) / len(self.times)

    def sum(self):
        """時間の合計を返す。"""
        return sum(self.times)

    def cumsum(self):
        """累積時間を返す。"""
        return np.array(self.times).cumsum().tolist()

timer = Timer()
```

要素ごとの代入では、$\mathbf{B}$ と $\mathbf{C}$ のすべての行と列をそれぞれ走査して、値を $\mathbf{A}$ に代入します。

```{.python .input}
#@tab mxnet
# Compute A = BC one element at a time
timer.start()
for i in range(256):
    for j in range(256):
        A[i, j] = np.dot(B[i, :], C[:, j])
A.wait_to_read()
timer.stop()
```

```{.python .input}
#@tab pytorch
# Compute A = BC one element at a time
timer.start()
for i in range(256):
    for j in range(256):
        A[i, j] = torch.dot(B[i, :], C[:, j])
timer.stop()
```

```{.python .input}
#@tab tensorflow
# Compute A = BC one element at a time
timer.start()
for i in range(256):
    for j in range(256):
        A[i, j].assign(tf.tensordot(B[i, :], C[:, j], axes=1))
timer.stop()
```

より高速な戦略は、列ごとに代入することです。

```{.python .input}
#@tab mxnet
# Compute A = BC one column at a time
timer.start()
for j in range(256):
    A[:, j] = np.dot(B, C[:, j])
A.wait_to_read()
timer.stop()
```

```{.python .input}
#@tab pytorch
# Compute A = BC one column at a time
timer.start()
for j in range(256):
    A[:, j] = torch.mv(B, C[:, j])
timer.stop()
```

```{.python .input}
#@tab tensorflow
timer.start()
for j in range(256):
    A[:, j].assign(tf.tensordot(B, C[:, j], axes=1))
timer.stop()
```

最後に、最も効果的なのは、全体の操作を1つのブロックで実行する方法です。 
任意の2つの行列 $\mathbf{B} \in \mathbb{R}^{m \times n}$ と $\mathbf{C} \in \mathbb{R}^{n \times p}$ の積には、スカラー乗算と加算を別々の演算として数えると（実際には融合されますが）、およそ $2mnp$ 回の浮動小数点演算が必要であることに注意してください。
したがって、$256 \times 256$ の2つの行列を掛けるには
0.03 十億回の浮動小数点演算が必要です。
それぞれの操作速度を見てみましょう。

```{.python .input}
#@tab mxnet
# Compute A = BC in one go
timer.start()
A = np.dot(B, C)
A.wait_to_read()
timer.stop()

gigaflops = [0.03 / i for i in timer.times]
print(f'performance in Gigaflops: element {gigaflops[0]:.3f}, '
      f'column {gigaflops[1]:.3f}, full {gigaflops[2]:.3f}')
```

```{.python .input}
#@tab pytorch
# Compute A = BC in one go
timer.start()
A = torch.mm(B, C)
timer.stop()

gigaflops = [0.03 / i for i in timer.times]
print(f'performance in Gigaflops: element {gigaflops[0]:.3f}, '
      f'column {gigaflops[1]:.3f}, full {gigaflops[2]:.3f}')
```

```{.python .input}
#@tab tensorflow
timer.start()
A.assign(tf.tensordot(B, C, axes=1))
timer.stop()

gigaflops = [0.03 / i for i in timer.times]
print(f'performance in Gigaflops: element {gigaflops[0]:.3f}, '
      f'column {gigaflops[1]:.3f}, full {gigaflops[2]:.3f}')
```

## ミニバッチ

:label:`sec_minibatches`

これまで、パラメータ更新には単一の観測ではなく *ミニバッチ* のデータを読み込むものとして当然視してきました。ここで、その理由を簡単に説明します。単一の観測を処理するには、多数の単一の行列-ベクトル（あるいはベクトル-ベクトル）積を実行する必要があり、これはかなり高コストで、基盤となる深層学習フレームワークにかなりのオーバーヘッドを生じさせます。これは、データにネットワークを適用して評価する場合（しばしば推論と呼ばれます）にも、パラメータ更新のために勾配を計算する場合にも当てはまります。つまり、$\mathbf{w} \leftarrow \mathbf{w} - \eta_t \mathbf{g}_t$ を実行するたびに当てはまり、ここで

$$\mathbf{g}_t = \partial_{\mathbf{w}} f(\mathbf{x}_{t}, \mathbf{w})$$

です。

この操作の *計算* 効率は、複数の観測をまとめたミニバッチに対して一度に適用することで高められます。つまり、単一の観測に対する勾配 $\mathbf{g}_t$ を、小さなバッチに対する次の勾配で置き換えます。

$$\mathbf{g}_t = \partial_{\mathbf{w}} \frac{1}{|\mathcal{B}_t|} \sum_{i \in \mathcal{B}_t} f(\mathbf{x}_{i}, \mathbf{w})$$

これが $\mathbf{g}_t$ の統計的性質にどう影響するか見てみましょう。$\mathbf{x}_t$ もミニバッチ $\mathcal{B}_t$ のすべての要素も訓練セットから一様ランダムに抽出されるので、勾配の期待値は変わりません。一方、分散は大幅に減少します。ミニバッチ勾配は、平均化される $b \stackrel{\textrm{def}}{=} |\mathcal{B}_t|$ 個の独立な勾配から構成されるため、その標準偏差は $b^{-\frac{1}{2}}$ 倍に減少します。これ自体は良いことです。なぜなら、更新が全勾配とより確実に整合することを意味するからです。

素朴に考えると、大きなミニバッチ $\mathcal{B}_t$ を選ぶことが常に望ましいように見えます。ところが、ある点を超えると、標準偏差の追加的な減少は、計算コストの線形増加に比べてごくわずかになります。実際には、良い計算効率を得られ、しかもGPUのメモリに収まる程度に十分大きいミニバッチを選びます。節約効果を示すために、コードを見てみましょう。ここでは同じ行列積を実行しますが、今度は64列ずつの「ミニバッチ」に分割して計算します。

```{.python .input}
#@tab mxnet
timer.start()
for j in range(0, 256, 64):
    A[:, j:j+64] = np.dot(B, C[:, j:j+64])
timer.stop()
print(f'performance in Gigaflops: block {0.03 / timer.times[3]:.3f}')
```

```{.python .input}
#@tab pytorch
timer.start()
for j in range(0, 256, 64):
    A[:, j:j+64] = torch.mm(B, C[:, j:j+64])
timer.stop()
print(f'performance in Gigaflops: block {0.03 / timer.times[3]:.3f}')
```

```{.python .input}
#@tab tensorflow
timer.start()
for j in range(0, 256, 64):
    A[:, j:j+64].assign(tf.tensordot(B, C[:, j:j+64], axes=1))
timer.stop()
print(f'performance in Gigaflops: block {0.03 / timer.times[3]:.3f}')
```

見てわかるように、ミニバッチ上での計算は、実質的に全行列上での計算と同じくらい効率的です。ここで注意が必要です。 :numref:`sec_batch_norm` では、ミニバッチ内の分散の大きさに強く依存する正則化を使いました。ミニバッチサイズを大きくすると分散は小さくなり、それに伴ってバッチ正規化によるノイズ注入の利点も小さくなります。適切な項の再スケーリングと計算方法については、たとえば :citet:`Ioffe.2017` を参照してください。

## データセットの読み込み

ミニバッチがデータからどのように効率よく生成されるか見てみましょう。以下では、異なる航空機の翼の [騒音](https://archive.ics.uci.edu/dataset/291/airfoil+self+noise) をテストするためにNASAが開発したデータセットを使って、これらの最適化アルゴリズムを比較します。便宜上、最初の $1,500$ 例だけを使います。前処理としてデータはホワイトニングされており、つまり平均を引き、各座標の分散を1に再スケーリングしています。

```{.python .input}
#@tab mxnet
#@save
d2l.DATA_HUB['airfoil'] = (d2l.DATA_URL + 'airfoil_self_noise.dat',
                           '76e5be1548fd8222e5074cf0faae75edff8cf93f')

#@save
def get_data_ch11(batch_size=10, n=1500):
    data = np.genfromtxt(d2l.download('airfoil'),
                         dtype=np.float32, delimiter='\t')
    data = (data - data.mean(axis=0)) / data.std(axis=0)
    data_iter = d2l.load_array(
        (data[:n, :-1], data[:n, -1]), batch_size, is_train=True)
    return data_iter, data.shape[1]-1
```

```{.python .input}
#@tab pytorch
#@save
d2l.DATA_HUB['airfoil'] = (d2l.DATA_URL + 'airfoil_self_noise.dat',
                           '76e5be1548fd8222e5074cf0faae75edff8cf93f')

#@save
def get_data_ch11(batch_size=10, n=1500):
    data = np.genfromtxt(d2l.download('airfoil'),
                         dtype=np.float32, delimiter='\t')
    data = torch.from_numpy((data - data.mean(axis=0)) / data.std(axis=0))
    data_iter = d2l.load_array((data[:n, :-1], data[:n, -1]),
                               batch_size, is_train=True)
    return data_iter, data.shape[1]-1
```

```{.python .input}
#@tab tensorflow
#@save
d2l.DATA_HUB['airfoil'] = (d2l.DATA_URL + 'airfoil_self_noise.dat',
                           '76e5be1548fd8222e5074cf0faae75edff8cf93f')

#@save
def get_data_ch11(batch_size=10, n=1500):
    data = np.genfromtxt(d2l.download('airfoil'),
                         dtype=np.float32, delimiter='\t')
    data = (data - data.mean(axis=0)) / data.std(axis=0)
    data_iter = d2l.load_array((data[:n, :-1], data[:n, -1]),
                               batch_size, is_train=True)
    return data_iter, data.shape[1]-1
```

## ゼロからの実装

:numref:`sec_linear_scratch` におけるミニバッチ確率的勾配降下法の実装を思い出してください。以下では、少し一般化した実装を示します。便宜上、後のこの章で導入する他の最適化アルゴリズムと同じ呼び出しシグネチャにしています。具体的には、状態入力 `states` を追加し、ハイパーパラメータを辞書 `hyperparams` に入れます。さらに、訓練関数では各ミニバッチ例の損失を平均するので、最適化アルゴリズム内で勾配をバッチサイズで割る必要はありません。

```{.python .input}
#@tab mxnet
def sgd(params, states, hyperparams):
    for p in params:
        p[:] -= hyperparams['lr'] * p.grad
```

```{.python .input}
#@tab pytorch
def sgd(params, states, hyperparams):
    for p in params:
        p.data.sub_(hyperparams['lr'] * p.grad)
        p.grad.data.zero_()
```

```{.python .input}
#@tab tensorflow
def sgd(params, grads, states, hyperparams):
    for param, grad in zip(params, grads):
        param.assign_sub(hyperparams['lr']*grad)
```

次に、この章で後ほど導入する他の最適化アルゴリズムの利用を容易にするため、汎用的な訓練関数を実装します。これは線形回帰モデルを初期化し、ミニバッチ確率的勾配降下法や、その後に導入する他のアルゴリズムでモデルを訓練するために使えます。

```{.python .input}
#@tab mxnet
#@save
def train_ch11(trainer_fn, states, hyperparams, data_iter,
               feature_dim, num_epochs=2):
    # 初期化
    w = np.random.normal(scale=0.01, size=(feature_dim, 1))
    b = np.zeros(1)
    w.attach_grad()
    b.attach_grad()
    net, loss = lambda X: d2l.linreg(X, w, b), d2l.squared_loss
    # 訓練
    animator = d2l.Animator(xlabel='epoch', ylabel='loss',
                            xlim=[0, num_epochs], ylim=[0.22, 0.35])
    n, timer = 0, d2l.Timer()
    for _ in range(num_epochs):
        for X, y in data_iter:
            with autograd.record():
                l = loss(net(X), y).mean()
            l.backward()
            trainer_fn([w, b], states, hyperparams)
            n += X.shape[0]
            if n % 200 == 0:
                timer.stop()
                animator.add(n/X.shape[0]/len(data_iter),
                             (d2l.evaluate_loss(net, data_iter, loss),))
                timer.start()
    print(f'loss: {animator.Y[0][-1]:.3f}, {timer.sum()/num_epochs:.3f} sec/epoch')
    return timer.cumsum(), animator.Y[0]
```

```{.python .input}
#@tab pytorch
#@save
def train_ch11(trainer_fn, states, hyperparams, data_iter,
               feature_dim, num_epochs=2):
    # 初期化
    w = torch.normal(mean=0.0, std=0.01, size=(feature_dim, 1),
                     requires_grad=True)
    b = torch.zeros((1), requires_grad=True)
    net, loss = lambda X: d2l.linreg(X, w, b), d2l.squared_loss
    # 訓練
    animator = d2l.Animator(xlabel='epoch', ylabel='loss',
                            xlim=[0, num_epochs], ylim=[0.22, 0.35])
    n, timer = 0, d2l.Timer()
    for _ in range(num_epochs):
        for X, y in data_iter:
            l = loss(net(X), y).mean()
            l.backward()
            trainer_fn([w, b], states, hyperparams)
            n += X.shape[0]
            if n % 200 == 0:
                timer.stop()
                animator.add(n/X.shape[0]/len(data_iter),
                             (d2l.evaluate_loss(net, data_iter, loss),))
                timer.start()
    print(f'loss: {animator.Y[0][-1]:.3f}, {timer.sum()/num_epochs:.3f} sec/epoch')
    return timer.cumsum(), animator.Y[0]
```

```{.python .input}
#@tab tensorflow
#@save
def train_ch11(trainer_fn, states, hyperparams, data_iter,
               feature_dim, num_epochs=2):
    # 初期化
    w = tf.Variable(tf.random.normal(shape=(feature_dim, 1),
                                   mean=0, stddev=0.01),trainable=True)
    b = tf.Variable(tf.zeros(1), trainable=True)

    # 訓練
    net, loss = lambda X: d2l.linreg(X, w, b), d2l.squared_loss
    animator = d2l.Animator(xlabel='epoch', ylabel='loss',
                            xlim=[0, num_epochs], ylim=[0.22, 0.35])
    n, timer = 0, d2l.Timer()

    for _ in range(num_epochs):
        for X, y in data_iter:
          with tf.GradientTape() as g:
            l = tf.math.reduce_mean(loss(net(X), y))

          dw, db = g.gradient(l, [w, b])
          trainer_fn([w, b], [dw, db], states, hyperparams)
          n += X.shape[0]
          if n % 200 == 0:
              timer.stop()
              p = n/X.shape[0]
              q = p/tf.data.experimental.cardinality(data_iter).numpy()
              r = (d2l.evaluate_loss(net, data_iter, loss),)
              animator.add(q, r)
              timer.start()
    print(f'loss: {animator.Y[0][-1]:.3f}, {timer.sum()/num_epochs:.3f} sec/epoch')
    return timer.cumsum(), animator.Y[0]
```

バッチ勾配降下法で最適化がどのように進むか見てみましょう。これはミニバッチサイズを1500（つまり例の総数）に設定することで実現できます。その結果、モデルパラメータは各エポックで1回しか更新されません。進展はほとんどありません。実際、6ステップ後には進展が止まります。

```{.python .input}
#@tab all
def train_sgd(lr, batch_size, num_epochs=2):
    data_iter, feature_dim = get_data_ch11(batch_size)
    return train_ch11(
        sgd, None, {'lr': lr}, data_iter, feature_dim, num_epochs)

gd_res = train_sgd(1, 1500, 10)
```

バッチサイズが1に等しいとき、最適化には確率的勾配降下法を使います。実装を簡単にするため、一定の（とはいえ小さな）学習率を選びました。確率的勾配降下法では、例が処理されるたびにモデルパラメータが更新されます。この場合、1エポックあたり1500回の更新に相当します。見てわかるように、目的関数の値の低下は1エポック後に鈍化します。1エポック内で両方の手続きが1500例を処理したにもかかわらず、この実験では確率的勾配降下法のほうが勾配降下法よりも時間を要します。これは、確率的勾配降下法がより頻繁にパラメータを更新し、単一の観測を1つずつ処理するのは効率が悪いためです。

```{.python .input}
#@tab all
sgd_res = train_sgd(0.005, 1)
```

最後に、バッチサイズが100に等しいとき、最適化にはミニバッチ確率的勾配降下法を使います。1エポックあたりに必要な時間は、確率的勾配降下法やバッチ勾配降下法より短くなります。

```{.python .input}
#@tab all
mini1_res = train_sgd(.4, 100)
```

バッチサイズを10に減らすと、各バッチの処理効率が下がるため、各エポックの時間は増加します。

```{.python .input}
#@tab all
mini2_res = train_sgd(.05, 10)
```

これで、先の4つの実験について時間と損失を比較できます。見てわかるように、確率的勾配降下法は処理した例の数という観点ではGDより速く収束しますが、勾配を例ごとに計算するのは効率的ではないため、同じ損失に到達するまでにより多くの時間を要します。ミニバッチ確率的勾配降下法は、収束速度と計算効率のトレードオフをうまく取ることができます。バッチサイズ10は確率的勾配降下法より効率的であり、バッチサイズ100は実行時間の観点ではGDを上回ります。

```{.python .input}
#@tab all
d2l.set_figsize([6, 3])
d2l.plot(*list(map(list, zip(gd_res, sgd_res, mini1_res, mini2_res))),
         'time (sec)', 'loss', xlim=[1e-2, 10],
         legend=['gd', 'sgd', 'batch size=100', 'batch size=10'])
d2l.plt.gca().set_xscale('log')
```

## 簡潔な実装

Gluonでは、`Trainer` クラスを使って最適化アルゴリズムを呼び出せます。これは汎用的な訓練関数を実装するために使われます。これはこの章を通して使います。

```{.python .input}
#@tab mxnet
#@save
def train_concise_ch11(tr_name, hyperparams, data_iter, num_epochs=2):
    # 初期化
    net = nn.Sequential()
    net.add(nn.Dense(1))
    net.initialize(init.Normal(sigma=0.01))
    trainer = gluon.Trainer(net.collect_params(), tr_name, hyperparams)
    loss = gluon.loss.L2Loss()
    animator = d2l.Animator(xlabel='epoch', ylabel='loss',
                            xlim=[0, num_epochs], ylim=[0.22, 0.35])
    n, timer = 0, d2l.Timer()
    for _ in range(num_epochs):
        for X, y in data_iter:
            with autograd.record():
                l = loss(net(X), y)
            l.backward()
            trainer.step(X.shape[0])
            n += X.shape[0]
            if n % 200 == 0:
                timer.stop()
                animator.add(n/X.shape[0]/len(data_iter),
                             (d2l.evaluate_loss(net, data_iter, loss),))
                timer.start()
    print(f'loss: {animator.Y[0][-1]:.3f}, {timer.sum()/num_epochs:.3f} sec/epoch')
```

```{.python .input}
#@tab pytorch
#@save
def train_concise_ch11(trainer_fn, hyperparams, data_iter, num_epochs=4):
    # 初期化
    net = nn.Sequential(nn.Linear(5, 1))
    def init_weights(module):
        if type(module) == nn.Linear:
            torch.nn.init.normal_(module.weight, std=0.01)
    net.apply(init_weights)

    optimizer = trainer_fn(net.parameters(), **hyperparams)
    loss = nn.MSELoss(reduction='none')
    animator = d2l.Animator(xlabel='epoch', ylabel='loss',
                            xlim=[0, num_epochs], ylim=[0.22, 0.35])
    n, timer = 0, d2l.Timer()
    for _ in range(num_epochs):
        for X, y in data_iter:
            optimizer.zero_grad()
            out = net(X)
            y = y.reshape(out.shape)
            l = loss(out, y)
            l.mean().backward()
            optimizer.step()
            n += X.shape[0]
            if n % 200 == 0:
                timer.stop()
                # `MSELoss` computes squared error without the 1/2 factor
                animator.add(n/X.shape[0]/len(data_iter),
                             (d2l.evaluate_loss(net, data_iter, loss) / 2,))
                timer.start()
    print(f'loss: {animator.Y[0][-1]:.3f}, {timer.sum()/num_epochs:.3f} sec/epoch')
```

```{.python .input}
#@tab tensorflow
#@save
def train_concise_ch11(trainer_fn, hyperparams, data_iter, num_epochs=2):
    # 初期化
    net = tf.keras.Sequential()
    net.add(tf.keras.layers.Dense(1,
            kernel_initializer=tf.random_normal_initializer(stddev=0.01)))
    optimizer = trainer_fn(**hyperparams)
    loss = tf.keras.losses.MeanSquaredError()
    animator = d2l.Animator(xlabel='epoch', ylabel='loss',
                            xlim=[0, num_epochs], ylim=[0.22, 0.35])
    n, timer = 0, d2l.Timer()
    for _ in range(num_epochs):
        for X, y in data_iter:
            with tf.GradientTape() as g:
                out = net(X)
                l = loss(y, out)
                params = net.trainable_variables
                grads = g.gradient(l, params)
            optimizer.apply_gradients(zip(grads, params))
            n += X.shape[0]
            if n % 200 == 0:
                timer.stop()
                p = n/X.shape[0]
                q = p/tf.data.experimental.cardinality(data_iter).numpy()
                # `MeanSquaredError` computes squared error without the 1/2
                # factor
                r = (d2l.evaluate_loss(net, data_iter, loss) / 2,)
                animator.add(q, r)
                timer.start()
    print(f'loss: {animator.Y[0][-1]:.3f}, {timer.sum()/num_epochs:.3f} sec/epoch')
```

Gluonを使って最後の実験を繰り返すと、同じ挙動が得られます。

```{.python .input}
#@tab mxnet
data_iter, _ = get_data_ch11(10)
train_concise_ch11('sgd', {'learning_rate': 0.05}, data_iter)
```

```{.python .input}
#@tab pytorch
data_iter, _ = get_data_ch11(10)
trainer = torch.optim.SGD
train_concise_ch11(trainer, {'lr': 0.01}, data_iter)
```

```{.python .input}
#@tab tensorflow
data_iter, _ = get_data_ch11(10)
trainer = tf.keras.optimizers.SGD
train_concise_ch11(trainer, {'learning_rate': 0.05}, data_iter)
```

## まとめ

* ベクトル化により、深層学習フレームワーク由来のオーバーヘッドが減り、CPUやGPUでのメモリ局所性とキャッシュ利用が改善されるため、コードはより効率的になる。
* 確率的勾配降下法に由来する統計効率と、データを大きなバッチでまとめて処理することに由来する計算効率の間にはトレードオフがある。
* ミニバッチ確率的勾配降下法は、計算効率と統計効率の両方の利点を兼ね備えている。
* ミニバッチ確率的勾配降下法では、訓練データのランダムな順列から得られたバッチを処理する（つまり、各観測は各エポックで1回だけ、ただしランダムな順序で処理される）。
* 訓練中に学習率を減衰させることが推奨される。
* 一般に、ミニバッチ確率的勾配降下法は、時計時間で測ったとき、より小さなリスクへ収束するまでの速さにおいて、確率的勾配降下法や勾配降下法より高速である。

## 演習

1. バッチサイズと学習率を変更し、目的関数の値の低下速度と各エポックで消費される時間を観察せよ。
1. MXNetのドキュメントを読み、`Trainer` クラスの `set_learning_rate` 関数を使って、各エポック後にミニバッチ確率的勾配降下法の学習率を前回の1/10に減らせ。
1. ミニバッチ確率的勾配降下法と、実際に訓練セットから *復元抽出* する変種を比較せよ。何が起こるか。
1. 悪意ある魔神が、あなたに知らせずにデータセットを複製したとする（つまり、各観測が2回ずつ現れ、データセットの大きさは元の2倍になるが、誰もそれを教えてくれない）。確率的勾配降下法、ミニバッチ確率的勾配降下法、勾配降下法の挙動はどう変わるか。
