# モメンタム
:label:`sec_momentum`

:numref:`sec_sgd` では、確率的勾配降下法を行うとき、すなわち勾配のノイズを含む変種しか利用できない最適化を行うときに何が起こるかを復習した。特に、ノイズのある勾配では、ノイズに直面したときの学習率の選び方にいっそう注意が必要であることに気づいた。これを急激に下げすぎると、収束が止まってしまう。逆に寛大すぎると、ノイズが最適解から私たちを押し戻し続けるため、十分によい解に収束できない。

## 基礎

この節では、実際によく現れるある種の最適化問題に対して、より効果的な最適化アルゴリズムを探る。


### 漏れのある平均

前節では、計算を高速化する手段としてミニバッチSGDについて議論した。また、勾配を平均することで分散が減るといううれしい副作用もあった。ミニバッチ確率的勾配降下法は次のように計算できる。

$$\mathbf{g}_{t, t-1} = \partial_{\mathbf{w}} \frac{1}{|\mathcal{B}_t|} \sum_{i \in \mathcal{B}_t} f(\mathbf{x}_{i}, \mathbf{w}_{t-1}) = \frac{1}{|\mathcal{B}_t|} \sum_{i \in \mathcal{B}_t} \mathbf{h}_{i, t-1}.
$$

記法を簡単にするため、ここでは $\mathbf{h}_{i, t-1} = \partial_{\mathbf{w}} f(\mathbf{x}_i, \mathbf{w}_{t-1})$ を、時刻 $t-1$ に更新された重みを用いたサンプル $i$ の確率的勾配降下法として用いた。ミニバッチ上で勾配を平均すること以上に、分散低減の効果を活かせたら理想的である。この目的を達成する一つの方法は、勾配計算を「漏れのある平均」に置き換えることである。

$$\mathbf{v}_t = \beta \mathbf{v}_{t-1} + \mathbf{g}_{t, t-1}$$

ここで $\beta \in (0, 1)$ とする。これは実質的に、その時点の勾配を複数の*過去*の勾配で平均したものに置き換えることを意味する。$\mathbf{v}$ は*速度*と呼ばれる。これは、重い球が目的関数の地形を転がり落ちるときに過去の力を積分するのと同様に、過去の勾配を蓄積する。何が起きているかをより詳しく見るために、$\mathbf{v}_t$ を再帰的に展開すると

$$\begin{aligned}
\mathbf{v}_t = \beta^2 \mathbf{v}_{t-2} + \beta \mathbf{g}_{t-1, t-2} + \mathbf{g}_{t, t-1}
= \ldots, = \sum_{\tau = 0}^{t-1} \beta^{\tau} \mathbf{g}_{t-\tau, t-\tau-1}.
\end{aligned}$$

$\beta$ が大きいと長い範囲の平均になり、小さいと勾配法に対するわずかな補正にとどまる。新しい勾配の置き換えは、もはや特定の1回の勾配降下で最急降下方向を指すのではなく、過去の勾配の重み付き平均の方向を指す。これにより、実際にその上で勾配を計算するコストを払わずに、バッチで平均することの利点の大部分を実現できる。この平均化手続きについては、後でさらに詳しく見直す。

上の考え方は、現在では*加速*勾配法として知られる手法、たとえばモメンタム付き勾配法の基礎となった。これらは、最適化問題が悪条件である場合、すなわちある方向では他の方向より進みがずっと遅く、細い峡谷のように見える場合に、はるかに効果的であるという追加の利点を持つ。さらに、連続する勾配を平均することで、より安定した降下方向を得られる。実際、ノイズのない凸問題であっても加速が効くという点は、モメンタムが機能する重要な理由の一つであり、また非常によく効く理由でもある。

予想されるように、その有効性ゆえにモメンタムは深層学習およびそれ以外の最適化でも広く研究されている。詳細な解析と対話的アニメーションについては、たとえば :citet:`Goh.2017` による美しい[解説記事](https://distill.pub/2017/momentum/)を参照されたい。これは :citet:`Polyak.1964` によって提案された。:citet:`Nesterov.2018` には、凸最適化の文脈での詳細な理論的議論がある。深層学習におけるモメンタムが有益であることは、以前から知られている。詳細は、たとえば :citet:`Sutskever.Martens.Dahl.ea.2013` の議論を参照されたい。

### 悪条件の問題

モメンタム法の幾何学的性質をよりよく理解するために、勾配降下法を再び取り上げる。ただし、今回はかなり扱いにくい目的関数を用いる。 :numref:`sec_gd` では $f(\mathbf{x}) = x_1^2 + 2 x_2^2$、すなわち中程度に歪んだ楕円体の目的関数を用いたことを思い出そう。ここでは、この関数を $x_1$ 方向に引き伸ばすことでさらに歪める。

$$f(\mathbf{x}) = 0.1 x_1^2 + 2 x_2^2.$$

前と同様に、$f$ の最小値は $(0, 0)$ にある。この関数は $x_1$ 方向では*非常に*平坦である。では、この新しい関数に対して前と同じように勾配降下法を行うと何が起こるか見てみよう。学習率は $0.4$ とする。

```{.python .input}
#@tab mxnet
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import np, npx
npx.set_np()

eta = 0.4
def f_2d(x1, x2):
    return 0.1 * x1 ** 2 + 2 * x2 ** 2
def gd_2d(x1, x2, s1, s2):
    return (x1 - eta * 0.2 * x1, x2 - eta * 4 * x2, 0, 0)

d2l.show_trace_2d(f_2d, d2l.train_2d(gd_2d))
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
import torch

eta = 0.4
def f_2d(x1, x2):
    return 0.1 * x1 ** 2 + 2 * x2 ** 2
def gd_2d(x1, x2, s1, s2):
    return (x1 - eta * 0.2 * x1, x2 - eta * 4 * x2, 0, 0)

d2l.show_trace_2d(f_2d, d2l.train_2d(gd_2d))
```

```{.python .input}
#@tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
import tensorflow as tf

eta = 0.4
def f_2d(x1, x2):
    return 0.1 * x1 ** 2 + 2 * x2 ** 2
def gd_2d(x1, x2, s1, s2):
    return (x1 - eta * 0.2 * x1, x2 - eta * 4 * x2, 0, 0)

d2l.show_trace_2d(f_2d, d2l.train_2d(gd_2d))
```

構成上、$x_2$ 方向の勾配ははるかに大きく、水平な $x_1$ 方向よりもずっと急速に変化する。したがって、私たちは望ましくない二つの選択肢の間に挟まれる。小さい学習率を選べば $x_2$ 方向で解が発散しないことは保証できるが、$x_1$ 方向での収束は遅くなる。逆に大きい学習率では $x_1$ 方向では速く進むが、$x_2$ 方向で発散する。以下の例は、学習率を $0.4$ から $0.6$ に少し上げただけでも何が起こるかを示している。$x_1$ 方向の収束は改善するが、全体としての解の質はかなり悪くなる。

```{.python .input}
#@tab all
eta = 0.6
d2l.show_trace_2d(f_2d, d2l.train_2d(gd_2d))
```

### モメンタム法

モメンタム法を用いれば、上で述べた勾配降下法の問題を解決できる。上の最適化の軌跡を見ると、過去の勾配を平均するとうまくいきそうだと直感できるかもしれない。実際、$x_1$ 方向では、よく整列した勾配が蓄積されるため、各ステップで進める距離が増える。逆に、勾配が振動する $x_2$ 方向では、振動が互いに打ち消し合うため、集約された勾配によってステップサイズが小さくなる。勾配 $\mathbf{g}_t$ の代わりに $\mathbf{v}_t$ を用いると、次の更新式が得られる。

$$
\begin{aligned}
\mathbf{v}_t &\leftarrow \beta \mathbf{v}_{t-1} + \mathbf{g}_{t, t-1}, \\
\mathbf{x}_t &\leftarrow \mathbf{x}_{t-1} - \eta_t \mathbf{v}_t.
\end{aligned}
$$

$\beta = 0$ のとき、通常の勾配降下法に戻ることに注意しよう。数学的性質をさらに深く掘り下げる前に、まずは実際の挙動を簡単に見てみよう。

```{.python .input}
#@tab all
def momentum_2d(x1, x2, v1, v2):
    v1 = beta * v1 + 0.2 * x1
    v2 = beta * v2 + 4 * x2
    return x1 - eta * v1, x2 - eta * v2, v1, v2

eta, beta = 0.6, 0.5
d2l.show_trace_2d(f_2d, d2l.train_2d(momentum_2d))
```

見てわかるように、以前と同じ学習率でも、モメンタムは依然としてうまく収束する。モメンタムのハイパーパラメータを下げるとどうなるか見てみよう。これを半分の $\beta = 0.25$ にすると、ほとんど収束しない軌跡になる。それでも、モメンタムなしの場合（そのときは解が発散する）よりはかなりよい。

```{.python .input}
#@tab all
eta, beta = 0.6, 0.25
d2l.show_trace_2d(f_2d, d2l.train_2d(momentum_2d))
```

モメンタムは確率的勾配降下法、とくにミニバッチ確率的勾配降下法と組み合わせられることに注意しよう。その場合に変わるのは、勾配 $\mathbf{g}_{t, t-1}$ を $\mathbf{g}_t$ に置き換えることだけである。最後に、便宜上、時刻 $t=0$ で $\mathbf{v}_0 = 0$ と初期化する。では、この漏れのある平均が実際に更新に対して何をしているのか見てみよう。

### 有効サンプル重み

$\mathbf{v}_t = \sum_{\tau = 0}^{t-1} \beta^{\tau} \mathbf{g}_{t-\tau, t-\tau-1}$ を思い出そう。極限では、項は $\sum_{\tau=0}^\infty \beta^\tau = \frac{1}{1-\beta}$ に足し合わさる。言い換えると、勾配降下法や確率的勾配降下法でサイズ $\eta$ の一歩を踏み出す代わりに、サイズ $\frac{\eta}{1-\beta}$ の一歩を踏み出しつつ、同時に、より扱いやすい降下方向を扱っていることになる。これは一度に二つの利点である。$\beta$ の異なる選び方によって重み付けがどう変わるかを示すために、下の図を考えよう。

```{.python .input}
#@tab all
d2l.set_figsize()
betas = [0.95, 0.9, 0.6, 0]
for beta in betas:
    x = d2l.numpy(d2l.arange(40))
    d2l.plt.plot(x, beta ** x, label=f'beta = {beta:.2f}')
d2l.plt.xlabel('time')
d2l.plt.legend();
```

## 実践的な実験

実際にモメンタムがどのように働くか、すなわち適切な最適化器の文脈で使ったときにどうなるかを見てみよう。そのためには、もう少しスケーラブルな実装が必要である。

### ゼロからの実装

（ミニバッチ）確率的勾配降下法と比べると、モメンタム法では補助変数、すなわち速度を保持する必要がある。これは勾配（および最適化問題の変数）と同じ形状を持つ。以下の実装では、これらの変数を `states` と呼ぶ。

```{.python .input}
#@tab mxnet,pytorch
def init_momentum_states(feature_dim):
    v_w = d2l.zeros((feature_dim, 1))
    v_b = d2l.zeros(1)
    return (v_w, v_b)
```

```{.python .input}
#@tab tensorflow
def init_momentum_states(features_dim):
    v_w = tf.Variable(d2l.zeros((features_dim, 1)))
    v_b = tf.Variable(d2l.zeros(1))
    return (v_w, v_b)
```

```{.python .input}
#@tab mxnet
def sgd_momentum(params, states, hyperparams):
    for p, v in zip(params, states):
        v[:] = hyperparams['momentum'] * v + p.grad
        p[:] -= hyperparams['lr'] * v
```

```{.python .input}
#@tab pytorch
def sgd_momentum(params, states, hyperparams):
    for p, v in zip(params, states):
        with torch.no_grad():
            v[:] = hyperparams['momentum'] * v + p.grad
            p[:] -= hyperparams['lr'] * v
        p.grad.data.zero_()
```

```{.python .input}
#@tab tensorflow
def sgd_momentum(params, grads, states, hyperparams):
    for p, v, g in zip(params, states, grads):
            v[:].assign(hyperparams['momentum'] * v + g)
            p[:].assign(p - hyperparams['lr'] * v)
```

実際にこれがどう働くか見てみよう。

```{.python .input}
#@tab all
def train_momentum(lr, momentum, num_epochs=2):
    d2l.train_ch11(sgd_momentum, init_momentum_states(feature_dim),
                   {'lr': lr, 'momentum': momentum}, data_iter,
                   feature_dim, num_epochs)

data_iter, feature_dim = d2l.get_data_ch11(batch_size=10)
train_momentum(0.02, 0.5)
```

モメンタムのハイパーパラメータ `momentum` を 0.9 に増やすと、有効サンプルサイズは $\frac{1}{1 - 0.9} = 10$ とかなり大きくなる。制御しやすくするため、学習率は少し下げて $0.01$ にする。

```{.python .input}
#@tab all
train_momentum(0.01, 0.9)
```

学習率をさらに下げると、滑らかでない最適化問題に関する問題にも対処できる。これを $0.005$ にすると、よい収束特性が得られる。

```{.python .input}
#@tab all
train_momentum(0.005, 0.9)
```

### 簡潔な実装

Gluon では、標準の `sgd` ソルバにすでにモメンタムが組み込まれているので、やることはほとんどない。対応するパラメータを設定すると、非常によく似た軌跡が得られる。

```{.python .input}
#@tab mxnet
d2l.train_concise_ch11('sgd', {'learning_rate': 0.005, 'momentum': 0.9},
                       data_iter)
```

```{.python .input}
#@tab pytorch
trainer = torch.optim.SGD
d2l.train_concise_ch11(trainer, {'lr': 0.005, 'momentum': 0.9}, data_iter)
```

```{.python .input}
#@tab tensorflow
trainer = tf.keras.optimizers.SGD
d2l.train_concise_ch11(trainer, {'learning_rate': 0.005, 'momentum': 0.9},
                       data_iter)
```

## 理論解析

ここまで、$f(x) = 0.1 x_1^2 + 2 x_2^2$ の2次元例はやや作為的に見えた。ここでは、少なくとも凸2次目的関数を最小化する場合には、これは実際に遭遇しうる問題のタイプをかなりよく代表していることを見る。

### 2次凸関数

次の関数を考える。

$$h(\mathbf{x}) = \frac{1}{2} \mathbf{x}^\top \mathbf{Q} \mathbf{x} + \mathbf{x}^\top \mathbf{c} + b.$$

これは一般的な2次関数である。正定値行列 $\mathbf{Q} \succ 0$、すなわち正の固有値を持つ行列に対しては、これは $\mathbf{x}^* = -\mathbf{Q}^{-1} \mathbf{c}$ で最小値 $b - \frac{1}{2} \mathbf{c}^\top \mathbf{Q}^{-1} \mathbf{c}$ をとる。したがって、$h$ は次のように書き換えられる。

$$h(\mathbf{x}) = \frac{1}{2} (\mathbf{x} - \mathbf{Q}^{-1} \mathbf{c})^\top \mathbf{Q} (\mathbf{x} - \mathbf{Q}^{-1} \mathbf{c}) + b - \frac{1}{2} \mathbf{c}^\top \mathbf{Q}^{-1} \mathbf{c}.$$

勾配は $\partial_{\mathbf{x}} h(\mathbf{x}) = \mathbf{Q} (\mathbf{x} - \mathbf{Q}^{-1} \mathbf{c})$ で与えられる。つまり、$\mathbf{x}$ と最小化点の距離に $\mathbf{Q}$ を掛けたものになっている。したがって、速度も $\mathbf{Q} (\mathbf{x}_t - \mathbf{Q}^{-1} \mathbf{c})$ の項の線形結合になる。

$\mathbf{Q}$ は正定値なので、$\mathbf{Q} = \mathbf{O}^\top \boldsymbol{\Lambda} \mathbf{O}$ として固有系に分解できる。ここで $\mathbf{O}$ は直交（回転）行列、$\boldsymbol{\Lambda}$ は正の固有値からなる対角行列である。これにより、$\mathbf{x}$ から $\mathbf{z} \stackrel{\textrm{def}}{=} \mathbf{O} (\mathbf{x} - \mathbf{Q}^{-1} \mathbf{c})$ への変数変換を行うと、はるかに簡単な式が得られる。

$$h(\mathbf{z}) = \frac{1}{2} \mathbf{z}^\top \boldsymbol{\Lambda} \mathbf{z} + b'.$$

ここで $b' = b - \frac{1}{2} \mathbf{c}^\top \mathbf{Q}^{-1} \mathbf{c}$ である。$\mathbf{O}$ は単なる直交行列なので、勾配を意味のある形では変化させない。$\mathbf{z}$ で表すと、勾配降下法は

$$\mathbf{z}_t = \mathbf{z}_{t-1} - \boldsymbol{\Lambda} \mathbf{z}_{t-1} = (\mathbf{I} - \boldsymbol{\Lambda}) \mathbf{z}_{t-1}.$$

この式で重要なのは、勾配降下法が異なる固有空間の間で*混ざらない*ことである。つまり、$\mathbf{Q}$ の固有系で表すと、最適化問題は座標ごとに進む。同様のことは次についても成り立つ。

$$\begin{aligned}
\mathbf{v}_t & = \beta \mathbf{v}_{t-1} + \boldsymbol{\Lambda} \mathbf{z}_{t-1} \\
\mathbf{z}_t & = \mathbf{z}_{t-1} - \eta \left(\beta \mathbf{v}_{t-1} + \boldsymbol{\Lambda} \mathbf{z}_{t-1}\right) \\
    & = (\mathbf{I} - \eta \boldsymbol{\Lambda}) \mathbf{z}_{t-1} - \eta \beta \mathbf{v}_{t-1}.
\end{aligned}$$

これにより、次の定理を証明したことになる。凸2次関数に対するモメンタムあり・なしの勾配降下法は、2次行列の固有ベクトル方向に沿った座標ごとの最適化に分解される。

### スカラー関数

上の結果を踏まえて、関数 $f(x) = \frac{\lambda}{2} x^2$ を最小化するときに何が起こるか見てみよう。勾配降下法では

$$x_{t+1} = x_t - \eta \lambda x_t = (1 - \eta \lambda) x_t.$$

$|1 - \eta \lambda| < 1$ である限り、この最適化は指数的な速度で収束する。なぜなら、$t$ ステップ後には $x_t = (1 - \eta \lambda)^t x_0$ となるからである。これは、$\eta \lambda = 1$ までは学習率 $\eta$ を増やすと収束率が最初は改善することを示している。それを超えると発散し、$\eta \lambda > 2$ では最適化問題は発散する。

```{.python .input}
#@tab all
lambdas = [0.1, 1, 10, 19]
eta = 0.1
d2l.set_figsize((6, 4))
for lam in lambdas:
    t = d2l.numpy(d2l.arange(20))
    d2l.plt.plot(t, (1 - eta * lam) ** t, label=f'lambda = {lam:.2f}')
d2l.plt.xlabel('time')
d2l.plt.legend();
```

モメンタムの場合の収束を解析するために、更新式を2つのスカラー、すなわち $x$ 用と速度 $v$ 用に書き直す。すると次が得られる。

$$
\begin{bmatrix} v_{t+1} \\ x_{t+1} \end{bmatrix} =
\begin{bmatrix} \beta & \lambda \\ -\eta \beta & (1 - \eta \lambda) \end{bmatrix}
\begin{bmatrix} v_{t} \\ x_{t} \end{bmatrix} = \mathbf{R}(\beta, \eta, \lambda) \begin{bmatrix} v_{t} \\ x_{t} \end{bmatrix}.
$$

ここでは、収束挙動を支配する $2 \times 2$ 行列を $\mathbf{R}$ で表した。$t$ ステップ後、初期値 $[v_0, x_0]$ は $\mathbf{R}(\beta, \eta, \lambda)^t [v_0, x_0]$ になる。したがって、収束速度を決めるのは $\mathbf{R}$ の固有値である。優れたアニメーションと詳細な解析については :citet:`Goh.2017` の [Distill の記事](https://distill.pub/2017/momentum/) と :citet:`Flammarion.Bach.2015` を参照されたい。$0 < \eta \lambda < 2 + 2 \beta$ で速度が収束することを示せる。これは、勾配降下法の $0 < \eta \lambda < 2$ と比べて、許容されるパラメータ範囲が広いことを意味する。また一般に、$\beta$ の値が大きいほど望ましいことも示唆している。さらに詳しい内容にはかなりの技術的説明が必要になるため、興味のある読者には原著を参照することを勧める。

## まとめ

* モメンタムは、勾配を過去の勾配の漏れのある平均に置き換える。これにより収束が大幅に加速される。
* ノイズのない勾配降下法にも、（ノイズのある）確率的勾配降下法にも有用である。
* モメンタムは、確率的勾配降下法で起こりやすい最適化過程の停滞を防ぐ。
* 過去のデータを指数的に減衰させて重み付けするため、有効な勾配数は $\frac{1}{1-\beta}$ で与えられる。
* 凸2次問題の場合、これを詳細に明示的に解析できる。
* 実装はかなり簡単だが、追加の状態ベクトル（速度 $\mathbf{v}$）を保存する必要がある。

## 演習

1. モメンタムのハイパーパラメータと学習率の他の組み合わせを使い、異なる実験結果を観察して分析せよ。
1. 複数の固有値を持つ2次問題、すなわち $f(x) = \frac{1}{2} \sum_i \lambda_i x_i^2$、たとえば $\lambda_i = 2^{-i}$ に対して、勾配降下法とモメンタムを試せ。初期値 $x_i = 1$ のときに $x$ の値がどのように減少するかをプロットせよ。
1. $h(\mathbf{x}) = \frac{1}{2} \mathbf{x}^\top \mathbf{Q} \mathbf{x} + \mathbf{x}^\top \mathbf{c} + b$ の最小値と最小化点を導出せよ。
1. モメンタム付き確率的勾配降下法を行うと何が変わるか。モメンタム付きミニバッチ確率的勾配降下法を使うと何が起こるか。パラメータを変えて実験せよ。

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/354)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1070)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/1071)
:end_tab:\n