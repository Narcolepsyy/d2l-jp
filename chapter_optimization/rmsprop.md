# RMSProp
:label:`sec_rmsprop`


:numref:`sec_adagrad` における重要な問題の1つは、学習率が事実上 $\mathcal{O}(t^{-\frac{1}{2}})$ のあらかじめ定められたスケジュールで減少していくことです。これは一般に凸問題には適していますが、深層学習で遭遇するような非凸問題には必ずしも理想的ではありません。それでも、Adagrad の座標ごとの適応性は、前処理器として非常に望ましいものです。

:citet:`Tieleman.Hinton.2012` は、スケジュールによる学習率の調整と座標適応的な学習率を切り離すための簡単な修正として、RMSProp アルゴリズムを提案しました。問題は、Adagrad が勾配 $\mathbf{g}_t$ の二乗を状態ベクトル $\mathbf{s}_t = \mathbf{s}_{t-1} + \mathbf{g}_t^2$ に蓄積していくことです。その結果、正規化がないため $\mathbf{s}_t$ は際限なく増大し続け、アルゴリズムが収束するにつれて本質的には線形に増えていきます。

この問題を修正する1つの方法は、$\mathbf{s}_t / t$ を使うことです。$\mathbf{g}_t$ の分布が適切であれば、これは収束します。残念ながら、この手続きは値の完全な軌跡を記憶するため、極限的な挙動が重要になるまでに非常に長い時間がかかるかもしれません。別の方法として、モメンタム法で用いたのと同じようにリーキー平均を使うことができます。すなわち、あるパラメータ $\gamma > 0$ に対して $\mathbf{s}_t \leftarrow \gamma \mathbf{s}_{t-1} + (1-\gamma) \mathbf{g}_t^2$ とします。他の部分をすべて変えずに残すと RMSProp になります。

## アルゴリズム

式を詳しく書き下してみましょう。

$$\begin{aligned}
    \mathbf{s}_t & \leftarrow \gamma \mathbf{s}_{t-1} + (1 - \gamma) \mathbf{g}_t^2, \\
    \mathbf{x}_t & \leftarrow \mathbf{x}_{t-1} - \frac{\eta}{\sqrt{\mathbf{s}_t + \epsilon}} \odot \mathbf{g}_t.
\end{aligned}$$

定数 $\epsilon > 0$ は通常 $10^{-6}$ に設定され、ゼロ除算や過度に大きなステップサイズを避けるために用いられます。この展開により、学習率 $\eta$ を、各座標ごとに適用されるスケーリングとは独立に制御できるようになります。リーキー平均の観点では、これまでモメンタム法で行ったのと同じ推論を適用できます。$\mathbf{s}_t$ の定義を展開すると

$$
\begin{aligned}
\mathbf{s}_t & = (1 - \gamma) \mathbf{g}_t^2 + \gamma \mathbf{s}_{t-1} \\
& = (1 - \gamma) \left(\mathbf{g}_t^2 + \gamma \mathbf{g}_{t-1}^2 + \gamma^2 \mathbf{g}_{t-2} + \ldots, \right).
\end{aligned}
$$

:numref:`sec_momentum` と同様に、$1 + \gamma + \gamma^2 + \ldots, = \frac{1}{1-\gamma}$ を用います。したがって、重みの総和は $1$ に正規化され、観測の半減期は $\gamma^{-1}$ になります。さまざまな $\gamma$ の選択について、過去40ステップの重みを可視化してみましょう。

```{.python .input}
#@tab mxnet
%matplotlib inline
from d2l import mxnet as d2l
import math
from mxnet import np, npx

npx.set_np()
```

```{.python .input}
#@tab pytorch
from d2l import torch as d2l
import torch
import math
```

```{.python .input}
#@tab tensorflow
from d2l import tensorflow as d2l
import tensorflow as tf
import math
```

```{.python .input}
#@tab all
d2l.set_figsize()
gammas = [0.95, 0.9, 0.8, 0.7]
for gamma in gammas:
    x = d2l.numpy(d2l.arange(40))
    d2l.plt.plot(x, (1-gamma) * gamma ** x, label=f'gamma = {gamma:.2f}')
d2l.plt.xlabel('time');
```

## ゼロからの実装

これまでと同様に、二次関数 $f(\mathbf{x})=0.1x_1^2+2x_2^2$ を用いて RMSProp の軌跡を観察します。 :numref:`sec_adagrad` では、学習率を 0.4 にして Adagrad を使ったとき、学習率が急速に下がりすぎるため、アルゴリズムの後半で変数の移動が非常に遅くなったことを思い出してください。$\eta$ は別々に制御されるので、RMSProp ではこのようなことは起こりません。

```{.python .input}
#@tab all
def rmsprop_2d(x1, x2, s1, s2):
    g1, g2, eps = 0.2 * x1, 4 * x2, 1e-6
    s1 = gamma * s1 + (1 - gamma) * g1 ** 2
    s2 = gamma * s2 + (1 - gamma) * g2 ** 2
    x1 -= eta / math.sqrt(s1 + eps) * g1
    x2 -= eta / math.sqrt(s2 + eps) * g2
    return x1, x2, s1, s2

def f_2d(x1, x2):
    return 0.1 * x1 ** 2 + 2 * x2 ** 2

eta, gamma = 0.4, 0.9
d2l.show_trace_2d(f_2d, d2l.train_2d(rmsprop_2d))
```

次に、深層ネットワークで使うための RMSProp を実装します。こちらも同様に簡単です。

```{.python .input}
#@tab mxnet,pytorch
def init_rmsprop_states(feature_dim):
    s_w = d2l.zeros((feature_dim, 1))
    s_b = d2l.zeros(1)
    return (s_w, s_b)
```

```{.python .input}
#@tab tensorflow
def init_rmsprop_states(feature_dim):
    s_w = tf.Variable(d2l.zeros((feature_dim, 1)))
    s_b = tf.Variable(d2l.zeros(1))
    return (s_w, s_b)
```

```{.python .input}
#@tab mxnet
def rmsprop(params, states, hyperparams):
    gamma, eps = hyperparams['gamma'], 1e-6
    for p, s in zip(params, states):
        s[:] = gamma * s + (1 - gamma) * np.square(p.grad)
        p[:] -= hyperparams['lr'] * p.grad / np.sqrt(s + eps)
```

```{.python .input}
#@tab pytorch
def rmsprop(params, states, hyperparams):
    gamma, eps = hyperparams['gamma'], 1e-6
    for p, s in zip(params, states):
        with torch.no_grad():
            s[:] = gamma * s + (1 - gamma) * torch.square(p.grad)
            p[:] -= hyperparams['lr'] * p.grad / torch.sqrt(s + eps)
        p.grad.data.zero_()
```

```{.python .input}
#@tab tensorflow
def rmsprop(params, grads, states, hyperparams):
    gamma, eps = hyperparams['gamma'], 1e-6
    for p, s, g in zip(params, states, grads):
        s[:].assign(gamma * s + (1 - gamma) * tf.math.square(g))
        p[:].assign(p - hyperparams['lr'] * g / tf.math.sqrt(s + eps))
```

初期学習率を 0.01、重み付け項 $\gamma$ を 0.9 に設定します。つまり、$\mathbf{s}$ は平均して過去 $1/(1-\gamma) = 10$ 個の二乗勾配を集約します。

```{.python .input}
#@tab all
data_iter, feature_dim = d2l.get_data_ch11(batch_size=10)
d2l.train_ch11(rmsprop, init_rmsprop_states(feature_dim),
               {'lr': 0.01, 'gamma': 0.9}, data_iter, feature_dim);
```

## 簡潔な実装

RMSProp はかなり人気のあるアルゴリズムなので、`Trainer` インスタンスでも利用できます。必要なのは、`rmsprop` という名前のアルゴリズムを使ってインスタンス化し、$\gamma$ をパラメータ `gamma1` に割り当てることだけです。

```{.python .input}
#@tab mxnet
d2l.train_concise_ch11('rmsprop', {'learning_rate': 0.01, 'gamma1': 0.9},
                       data_iter)
```

```{.python .input}
#@tab pytorch
trainer = torch.optim.RMSprop
d2l.train_concise_ch11(trainer, {'lr': 0.01, 'alpha': 0.9},
                       data_iter)
```

```{.python .input}
#@tab tensorflow
trainer = tf.keras.optimizers.RMSprop
d2l.train_concise_ch11(trainer, {'learning_rate': 0.01, 'rho': 0.9},
                       data_iter)
```

## まとめ

* RMSProp は Adagrad と非常によく似ており、どちらも勾配の二乗を用いて係数をスケーリングします。
* RMSProp はモメンタムとリーキー平均を共有しています。ただし、RMSProp ではこの手法を係数ごとの前処理器の調整に用います。
* 実際には、学習率は実験者がスケジュール調整する必要があります。
* 係数 $\gamma$ は、各座標のスケールを調整するときにどれだけ長い履歴を考慮するかを決定します。

## 演習

1. $\gamma = 1$ とすると、実験的に何が起こるでしょうか？ なぜでしょうか？
1. 最適化問題を回転させて、$f(\mathbf{x}) = 0.1 (x_1 + x_2)^2 + 2 (x_1 - x_2)^2$ を最小化してみてください。収束はどうなりますか？
1. Fashion-MNIST での学習など、実際の機械学習問題で RMSProp を試してみてください。学習率の調整方法を変えて実験してみましょう。
1. 最適化が進むにつれて $\gamma$ を調整したいと思いますか？ RMSProp はこれにどれくらい敏感でしょうか？

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/356)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1074)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/1075)
:end_tab:\n