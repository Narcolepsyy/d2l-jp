# Adadelta
:label:`sec_adadelta`

Adadelta は AdaGrad (:numref:`sec_adagrad`) のもう一つの変種です。主な違いは、学習率が座標ごとに適応的に変化する度合いを小さくしている点にあります。さらに、伝統的には、変化量そのものを将来の変化の較正に用いるため、学習率を持たない手法だとされています。このアルゴリズムは :citet:`Zeiler.2012` で提案されました。ここまでの前節までのアルゴリズムの議論を踏まえると、かなり素直に理解できます。

## アルゴリズム

要するに、Adadelta は 2 つの状態変数を使います。$\mathbf{s}_t$ は勾配の 2 次モーメントのリーキー平均を保持し、$\Delta\mathbf{x}_t$ はモデル自体のパラメータ変化の 2 次モーメントのリーキー平均を保持します。なお、他の文献や実装との互換性のため、著者らの元の記法と命名をそのまま用いています（モメンタム、Adagrad、RMSProp、Adadelta で同じ目的の変数を表すのに、わざわざ異なるギリシャ文字を使うべき実質的な理由はありません）。

以下に Adadelta の技術的詳細を示します。ここでのパラメータを $\rho$ とすると、 :numref:`sec_rmsprop` と同様に次のリーキー更新を得ます。

$$\begin{aligned}
    \mathbf{s}_t & = \rho \mathbf{s}_{t-1} + (1 - \rho) \mathbf{g}_t^2.
\end{aligned}$$

:numref:`sec_rmsprop` との違いは、再スケールした勾配 $\mathbf{g}_t'$ を用いて更新を行う点です。すなわち、

$$\begin{aligned}
    \mathbf{x}_t  & = \mathbf{x}_{t-1} - \mathbf{g}_t'. \\
\end{aligned}$$

では、再スケールした勾配 $\mathbf{g}_t'$ とは何でしょうか。次のように計算できます。

$$\begin{aligned}
    \mathbf{g}_t' & = \frac{\sqrt{\Delta\mathbf{x}_{t-1} + \epsilon}}{\sqrt{{\mathbf{s}_t + \epsilon}}} \odot \mathbf{g}_t, \\
\end{aligned}$$

ここで $\Delta \mathbf{x}_{t-1}$ は、再スケールした勾配 $\mathbf{g}_t'$ の二乗のリーキー平均です。$\Delta \mathbf{x}_{0}$ を $0$ に初期化し、各ステップで $\mathbf{g}_t'$ を用いて更新します。すなわち、

$$\begin{aligned}
    \Delta \mathbf{x}_t & = \rho \Delta\mathbf{x}_{t-1} + (1 - \rho) {\mathbf{g}_t'}^2,
\end{aligned}$$

また、数値安定性を保つために $\epsilon$（$10^{-5}$ のような小さな値）を加えます。



## 実装

Adadelta では、各変数ごとに 2 つの状態変数 $\mathbf{s}_t$ と $\Delta\mathbf{x}_t$ を保持する必要があります。これにより、次の実装になります。

```{.python .input}
#@tab mxnet
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import np, npx
npx.set_np()

def init_adadelta_states(feature_dim):
    s_w, s_b = d2l.zeros((feature_dim, 1)), d2l.zeros(1)
    delta_w, delta_b = d2l.zeros((feature_dim, 1)), d2l.zeros(1)
    return ((s_w, delta_w), (s_b, delta_b))

def adadelta(params, states, hyperparams):
    rho, eps = hyperparams['rho'], 1e-5
    for p, (s, delta) in zip(params, states):
        # In-place updates via [:]
        s[:] = rho * s + (1 - rho) * np.square(p.grad)
        g = (np.sqrt(delta + eps) / np.sqrt(s + eps)) * p.grad
        p[:] -= g
        delta[:] = rho * delta + (1 - rho) * g * g
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
import torch

def init_adadelta_states(feature_dim):
    s_w, s_b = d2l.zeros((feature_dim, 1)), d2l.zeros(1)
    delta_w, delta_b = d2l.zeros((feature_dim, 1)), d2l.zeros(1)
    return ((s_w, delta_w), (s_b, delta_b))

def adadelta(params, states, hyperparams):
    rho, eps = hyperparams['rho'], 1e-5
    for p, (s, delta) in zip(params, states):
        with torch.no_grad():
            # In-place updates via [:]
            s[:] = rho * s + (1 - rho) * torch.square(p.grad)
            g = (torch.sqrt(delta + eps) / torch.sqrt(s + eps)) * p.grad
            p[:] -= g
            delta[:] = rho * delta + (1 - rho) * g * g
        p.grad.data.zero_()
```

```{.python .input}
#@tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
import tensorflow as tf

def init_adadelta_states(feature_dim):
    s_w = tf.Variable(d2l.zeros((feature_dim, 1)))
    s_b = tf.Variable(d2l.zeros(1))
    delta_w = tf.Variable(d2l.zeros((feature_dim, 1)))
    delta_b = tf.Variable(d2l.zeros(1))
    return ((s_w, delta_w), (s_b, delta_b))

def adadelta(params, grads, states, hyperparams):
    rho, eps = hyperparams['rho'], 1e-5
    for p, (s, delta), grad in zip(params, states, grads):
        s[:].assign(rho * s + (1 - rho) * tf.math.square(grad))
        g = (tf.math.sqrt(delta + eps) / tf.math.sqrt(s + eps)) * grad
        p[:].assign(p - g)
        delta[:].assign(rho * delta + (1 - rho) * g * g)
```

$\rho = 0.9$ を選ぶと、各パラメータ更新の半減期は 10 になります。これはかなりうまく機能する傾向があります。結果は次のようになります。

```{.python .input}
#@tab all
data_iter, feature_dim = d2l.get_data_ch11(batch_size=10)
d2l.train_ch11(adadelta, init_adadelta_states(feature_dim),
               {'rho': 0.9}, data_iter, feature_dim);
```

簡潔な実装としては、高レベル API の Adadelta アルゴリズムをそのまま使います。これにより、より簡潔な呼び出しを 1 行で書けます。

```{.python .input}
#@tab mxnet
d2l.train_concise_ch11('adadelta', {'rho': 0.9}, data_iter)
```

```{.python .input}
#@tab pytorch
trainer = torch.optim.Adadelta
d2l.train_concise_ch11(trainer, {'rho': 0.9}, data_iter)
```

```{.python .input}
#@tab tensorflow
# adadelta is not converging at default learning rate
# but it is converging at lr = 5.0
trainer = tf.keras.optimizers.Adadelta
d2l.train_concise_ch11(trainer, {'learning_rate':5.0, 'rho': 0.9}, data_iter)
```

## まとめ

* Adadelta には学習率パラメータがありません。その代わりに、パラメータの変化率そのものを用いて学習率を適応させます。
* Adadelta では、勾配とパラメータ変化の 2 次モーメントを保存するために 2 つの状態変数が必要です。
* Adadelta はリーキー平均を用いて、適切な統計量の逐次推定を維持します。

## 演習

1. $\rho$ の値を調整してみてください。何が起こりますか？
1. $\mathbf{g}_t'$ を使わずにアルゴリズムを実装する方法を示してください。なぜそれがよい考えなのでしょうか？
1. Adadelta は本当に学習率不要なのでしょうか？ Adadelta を破綻させる最適化問題を見つけられますか？
1. Adadelta と Adagrad、RMS prop を比較し、それらの収束挙動について議論してください。

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/357)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1076)
:end_tab:


:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/1077)
:end_tab:\n