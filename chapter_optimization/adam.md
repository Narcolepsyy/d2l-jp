# Adam
:label:`sec_adam`

この節に入るまでの議論では、効率的な最適化のためのいくつかの手法に出会いました。ここでそれらを詳しく振り返っておきましょう。

* :numref:`sec_sgd` で、たとえば冗長なデータに対する本質的な頑健性のおかげで、最適化問題を解く際には勾配降下法よりも確率的勾配降下法のほうが有効であることを見ました。 
* :numref:`sec_minibatch_sgd` で、1つのミニバッチにより多くの観測を用いるベクトル化によって生じる、さらなる大きな効率向上を見ました。これは、効率的なマルチマシン、マルチGPU、そして全体としての並列処理の鍵です。 
* :numref:`sec_momentum` では、過去の勾配の履歴を集約して収束を加速する仕組みを追加しました。
* :numref:`sec_adagrad` では、座標ごとのスケーリングを用いて、計算効率のよい前処理器を実現しました。 
* :numref:`sec_rmsprop` では、座標ごとのスケーリングを学習率調整から切り離しました。 

Adam :cite:`Kingma.Ba.2014` は、これらすべての手法を1つの効率的な学習アルゴリズムにまとめたものです。予想どおり、これは深層学習で用いる最適化アルゴリズムの中でも、より頑健で有効なものとしてかなり人気のあるアルゴリズムになりました。ただし、問題がないわけではありません。特に :cite:`Reddi.Kale.Kumar.2019` は、分散の制御が不十分なために Adam が発散しうる状況があることを示しています。続く研究で :citet:`Zaheer.Reddi.Sachan.ea.2018` は、これらの問題に対処する Adam の修正版である Yogi を提案しました。これについては後で詳しく述べます。ここではまず Adam アルゴリズムを見ていきましょう。 

## アルゴリズム

Adam の重要な構成要素の1つは、指数加重移動平均（指数移動平均、あるいは leaky averaging とも呼ばれます）を用いて、モーメンタムと勾配の2次モーメントの両方の推定値を得ることです。つまり、次の状態変数を用います。

$$\begin{aligned}
    \mathbf{v}_t & \leftarrow \beta_1 \mathbf{v}_{t-1} + (1 - \beta_1) \mathbf{g}_t, \\
    \mathbf{s}_t & \leftarrow \beta_2 \mathbf{s}_{t-1} + (1 - \beta_2) \mathbf{g}_t^2.
\end{aligned}$$

ここで $\beta_1$ と $\beta_2$ は非負の重み付けパラメータです。一般的な値としては $\beta_1 = 0.9$ と $\beta_2 = 0.999$ がよく使われます。つまり、分散の推定値はモーメンタム項よりも *はるかにゆっくり* と変化します。$\mathbf{v}_0 = \mathbf{s}_0 = 0$ と初期化すると、最初のうちは小さい値にかなり強く偏ることに注意してください。これは、$\sum_{i=0}^{t-1} \beta^i = \frac{1 - \beta^t}{1 - \beta}$ という事実を用いて項を再正規化することで対処できます。対応する正規化済みの状態変数は

$$\hat{\mathbf{v}}_t = \frac{\mathbf{v}_t}{1 - \beta_1^t} \textrm{ and } \hat{\mathbf{s}}_t = \frac{\mathbf{s}_t}{1 - \beta_2^t}.$$

適切な推定値が得られたので、更新式を書けるようになりました。まず、RMSProp と非常によく似た方法で勾配を再スケーリングして、次を得ます。

$$\mathbf{g}_t' = \frac{\eta \hat{\mathbf{v}}_t}{\sqrt{\hat{\mathbf{s}}_t} + \epsilon}.$$

RMSProp と異なり、更新には勾配そのものではなくモーメンタム $\hat{\mathbf{v}}_t$ を使います。さらに、再スケーリングが $\frac{1}{\sqrt{\hat{\mathbf{s}}_t} + \epsilon}$ を用いて行われる点で、見た目に少し違いがあります。前者のほうが実践上わずかにうまくいくと考えられているため、RMSProp からこのように少し変えています。通常は、数値安定性と忠実性のよいトレードオフを得るために $\epsilon = 10^{-6}$ を選びます。 

これで更新を計算するための材料はすべてそろいました。やや拍子抜けするほど単純で、更新は次の形になります。

$$\mathbf{x}_t \leftarrow \mathbf{x}_{t-1} - \mathbf{g}_t'.$$

Adam の設計を見直すと、その着想は明らかです。モーメンタムとスケールが状態変数の中にはっきり現れています。かなり独特な定義のため、項のバイアス補正が必要になります（これは、初期化と更新条件を少し変えれば修正できます）。第2に、RMSProp があるので、2つの項の組み合わせはかなり直接的です。最後に、明示的な学習率 $\eta$ により、収束の問題に対処するためのステップ長を制御できます。 

## 実装 

Adam をゼロから実装するのはそれほど難しくありません。便宜上、時刻ステップのカウンタ $t$ を `hyperparams` 辞書に保存します。それ以外はすべて単純です。

```{.python .input}
#@tab mxnet
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import np, npx
npx.set_np()

def init_adam_states(feature_dim):
    v_w, v_b = d2l.zeros((feature_dim, 1)), d2l.zeros(1)
    s_w, s_b = d2l.zeros((feature_dim, 1)), d2l.zeros(1)
    return ((v_w, s_w), (v_b, s_b))

def adam(params, states, hyperparams):
    beta1, beta2, eps = 0.9, 0.999, 1e-6
    for p, (v, s) in zip(params, states):
        v[:] = beta1 * v + (1 - beta1) * p.grad
        s[:] = beta2 * s + (1 - beta2) * np.square(p.grad)
        v_bias_corr = v / (1 - beta1 ** hyperparams['t'])
        s_bias_corr = s / (1 - beta2 ** hyperparams['t'])
        p[:] -= hyperparams['lr'] * v_bias_corr / (np.sqrt(s_bias_corr) + eps)
    hyperparams['t'] += 1
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
import torch

def init_adam_states(feature_dim):
    v_w, v_b = d2l.zeros((feature_dim, 1)), d2l.zeros(1)
    s_w, s_b = d2l.zeros((feature_dim, 1)), d2l.zeros(1)
    return ((v_w, s_w), (v_b, s_b))

def adam(params, states, hyperparams):
    beta1, beta2, eps = 0.9, 0.999, 1e-6
    for p, (v, s) in zip(params, states):
        with torch.no_grad():
            v[:] = beta1 * v + (1 - beta1) * p.grad
            s[:] = beta2 * s + (1 - beta2) * torch.square(p.grad)
            v_bias_corr = v / (1 - beta1 ** hyperparams['t'])
            s_bias_corr = s / (1 - beta2 ** hyperparams['t'])
            p[:] -= hyperparams['lr'] * v_bias_corr / (torch.sqrt(s_bias_corr)
                                                       + eps)
        p.grad.data.zero_()
    hyperparams['t'] += 1
```

```{.python .input}
#@tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
import tensorflow as tf

def init_adam_states(feature_dim):
    v_w = tf.Variable(d2l.zeros((feature_dim, 1)))
    v_b = tf.Variable(d2l.zeros(1))
    s_w = tf.Variable(d2l.zeros((feature_dim, 1)))
    s_b = tf.Variable(d2l.zeros(1))
    return ((v_w, s_w), (v_b, s_b))

def adam(params, grads, states, hyperparams):
    beta1, beta2, eps = 0.9, 0.999, 1e-6
    for p, (v, s), grad in zip(params, states, grads):
        v[:].assign(beta1 * v  + (1 - beta1) * grad)
        s[:].assign(beta2 * s + (1 - beta2) * tf.math.square(grad))
        v_bias_corr = v / (1 - beta1 ** hyperparams['t'])
        s_bias_corr = s / (1 - beta2 ** hyperparams['t'])
        p[:].assign(p - hyperparams['lr'] * v_bias_corr  
                    / tf.math.sqrt(s_bias_corr) + eps)
```

Adam を使ってモデルを学習する準備が整いました。学習率は $\eta = 0.01$ を使います。

```{.python .input}
#@tab all
data_iter, feature_dim = d2l.get_data_ch11(batch_size=10)
d2l.train_ch11(adam, init_adam_states(feature_dim),
               {'lr': 0.01, 't': 1}, data_iter, feature_dim);
```

`adam` は Gluon の `trainer` 最適化ライブラリの一部として提供されているアルゴリズムの1つなので、より簡潔な実装も容易です。したがって、Gluon での実装に必要なのは設定パラメータを渡すことだけです。

```{.python .input}
#@tab mxnet
d2l.train_concise_ch11('adam', {'learning_rate': 0.01}, data_iter)
```

```{.python .input}
#@tab pytorch
trainer = torch.optim.Adam
d2l.train_concise_ch11(trainer, {'lr': 0.01}, data_iter)
```

```{.python .input}
#@tab tensorflow
trainer = tf.keras.optimizers.Adam
d2l.train_concise_ch11(trainer, {'learning_rate': 0.01}, data_iter)
```

## Yogi

Adam の問題の1つは、$\mathbf{s}_t$ における2次モーメント推定が大きくなりすぎると、凸最適化の設定でさえ収束に失敗しうることです。これに対する修正として、:citet:`Zaheer.Reddi.Sachan.ea.2018` は $\mathbf{s}_t$ の更新（および初期化）を改良したものを提案しました。何が起きているのかを理解するために、Adam の更新を次のように書き直してみましょう。

$$\mathbf{s}_t \leftarrow \mathbf{s}_{t-1} + (1 - \beta_2) \left(\mathbf{g}_t^2 - \mathbf{s}_{t-1}\right).$$

$\mathbf{g}_t^2$ の分散が大きい場合や更新が疎な場合、$\mathbf{s}_t$ は過去の値をあまりにも速く忘れてしまうことがあります。これに対する一つの修正は、$\mathbf{g}_t^2 - \mathbf{s}_{t-1}$ を $\mathbf{g}_t^2 \odot \mathop{\textrm{sgn}}(\mathbf{g}_t^2 - \mathbf{s}_{t-1})$ に置き換えることです。すると、更新の大きさはもはや偏差の大きさに依存しません。これにより Yogi の更新式が得られます。

$$\mathbf{s}_t \leftarrow \mathbf{s}_{t-1} + (1 - \beta_2) \mathbf{g}_t^2 \odot \mathop{\textrm{sgn}}(\mathbf{g}_t^2 - \mathbf{s}_{t-1}).$$

著者らはさらに、モーメンタムを単なる初期の点ごとの推定ではなく、より大きな初期バッチで初期化することを勧めています。ここでは、議論の本質ではなく、またそれがなくても収束は十分良好なので、詳細は省略します。

```{.python .input}
#@tab mxnet
def yogi(params, states, hyperparams):
    beta1, beta2, eps = 0.9, 0.999, 1e-3
    for p, (v, s) in zip(params, states):
        v[:] = beta1 * v + (1 - beta1) * p.grad
        s[:] = s + (1 - beta2) * np.sign(
            np.square(p.grad) - s) * np.square(p.grad)
        v_bias_corr = v / (1 - beta1 ** hyperparams['t'])
        s_bias_corr = s / (1 - beta2 ** hyperparams['t'])
        p[:] -= hyperparams['lr'] * v_bias_corr / (np.sqrt(s_bias_corr) + eps)
    hyperparams['t'] += 1

data_iter, feature_dim = d2l.get_data_ch11(batch_size=10)
d2l.train_ch11(yogi, init_adam_states(feature_dim),
               {'lr': 0.01, 't': 1}, data_iter, feature_dim);
```

```{.python .input}
#@tab pytorch
def yogi(params, states, hyperparams):
    beta1, beta2, eps = 0.9, 0.999, 1e-3
    for p, (v, s) in zip(params, states):
        with torch.no_grad():
            v[:] = beta1 * v + (1 - beta1) * p.grad
            s[:] = s + (1 - beta2) * torch.sign(
                torch.square(p.grad) - s) * torch.square(p.grad)
            v_bias_corr = v / (1 - beta1 ** hyperparams['t'])
            s_bias_corr = s / (1 - beta2 ** hyperparams['t'])
            p[:] -= hyperparams['lr'] * v_bias_corr / (torch.sqrt(s_bias_corr)
                                                       + eps)
        p.grad.data.zero_()
    hyperparams['t'] += 1

data_iter, feature_dim = d2l.get_data_ch11(batch_size=10)
d2l.train_ch11(yogi, init_adam_states(feature_dim),
               {'lr': 0.01, 't': 1}, data_iter, feature_dim);
```

```{.python .input}
#@tab tensorflow
def yogi(params, grads, states, hyperparams):
    beta1, beta2, eps = 0.9, 0.999, 1e-6
    for p, (v, s), grad in zip(params, states, grads):
        v[:].assign(beta1 * v  + (1 - beta1) * grad)
        s[:].assign(s + (1 - beta2) * tf.math.sign(
                   tf.math.square(grad) - s) * tf.math.square(grad))
        v_bias_corr = v / (1 - beta1 ** hyperparams['t'])
        s_bias_corr = s / (1 - beta2 ** hyperparams['t'])
        p[:].assign(p - hyperparams['lr'] * v_bias_corr  
                    / tf.math.sqrt(s_bias_corr) + eps)
    hyperparams['t'] += 1

data_iter, feature_dim = d2l.get_data_ch11(batch_size=10)
d2l.train_ch11(yogi, init_adam_states(feature_dim),
               {'lr': 0.01, 't': 1}, data_iter, feature_dim);
```

## まとめ

* Adam は多くの最適化アルゴリズムの特徴を組み合わせた、かなり頑健な更新規則です。 
* RMSProp を基に作られており、Adam はミニバッチ確率勾配にも EWMA を用います。
* Adam は、モーメンタムと2次モーメントを推定する際の立ち上がりの遅さを調整するためにバイアス補正を使います。 
* 分散の大きい勾配では、収束に問題が生じることがあります。これは、より大きなミニバッチを使うか、$\mathbf{s}_t$ の改良された推定に切り替えることで改善できます。Yogi はそのような代替手法を提供します。 

## 演習

1. 学習率を調整し、実験結果を観察して分析しなさい。
1. バイアス補正を必要としないように、モーメンタムと2次モーメントの更新を書き換えられるか。
1. 収束に伴って学習率 $\eta$ を下げる必要があるのはなぜか。
1. Adam が発散し、Yogi が収束する例を構成してみなさい。\n
