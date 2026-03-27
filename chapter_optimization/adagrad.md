# Adagrad
:label:`sec_adagrad`

まず、まれにしか現れない特徴をもつ学習問題について考えてみよう。


## 疎な特徴と学習率

言語モデルを学習していると想像してほしい。十分な精度を得るには、通常、学習を続けるにつれて学習率を下げていく必要があり、その減衰率はたいてい $\mathcal{O}(t^{-\frac{1}{2}})$ かそれより遅い程度である。ここで、疎な特徴、すなわちごくまれにしか現れない特徴をもつモデルの学習を考える。これは自然言語ではよくあることで、たとえば *learning* よりも *preconditioning* という単語に出会う確率のほうがずっと低いだろう。しかし、計算広告や個人化された協調フィルタリングなど、他の分野でも同様に一般的である。結局のところ、少数の人にしか関心を持たれないものはたくさんある。

まれな特徴に対応するパラメータは、その特徴が現れたときにしか意味のある更新を受けない。学習率が減少していくと、頻出特徴に対応するパラメータは比較的すぐに最適値へ収束する一方で、まれな特徴については、その最適値を決めるのに十分な頻度で観測される前に学習率が小さくなりすぎてしまう、という状況が起こりえる。言い換えると、学習率の減少が、頻出特徴に対しては遅すぎ、まれな特徴に対しては速すぎるのである。

この問題を緩和する一つの手は、特定の特徴を何回見たかを数え、それを学習率調整のための時計として使うことである。つまり、学習率を $\eta = \frac{\eta_0}{\sqrt{t + c}}$ の形で選ぶ代わりに、$\eta_i = \frac{\eta_0}{\sqrt{s(i, t) + c}}$ を使うことができる。ここで $s(i, t)$ は、時刻 $t$ までに観測した特徴 $i$ の非ゼロ回数を数える。これは実際、ほとんど追加コストなしでかなり簡単に実装できる。しかし、データが厳密に疎である場合ではなく、勾配がしばしば非常に小さく、まれに大きくなるようなデータではうまくいかない。そもそも、何を「観測された特徴」とみなすかの境界をどこに引くべきかが明確ではないからである。

:citet:`Duchi.Hazan.Singer.2011` による Adagrad は、このかなり粗いカウンター $s(i, t)$ を、これまでに観測された勾配の二乗の集計に置き換えることでこの問題に対処する。特に、学習率を調整する手段として $s(i, t+1) = s(i, t) + \left(\partial_i f(\mathbf{x})\right)^2$ を用いる。これには二つの利点がある。第一に、勾配が十分に大きいかどうかを判断する必要がなくなる。第二に、勾配の大きさに応じて自動的にスケールが調整される。大きな勾配に頻繁に対応する座標は大きく縮小され、一方で小さな勾配しか持たない座標はより穏やかに扱われる。実際には、これにより計算広告や関連問題に対して非常に有効な最適化手法になる。しかし、ここには Adagrad に本来備わっている追加の利点が隠れており、それは前処理の文脈で理解するのが最もよいだろう。


## 前処理

凸最適化問題は、アルゴリズムの特性を分析するのに適している。というのも、ほとんどの非凸問題では意味のある理論保証を導くのが難しい一方で、*直感* や *洞察* はしばしばそのまま役立つからである。$f(\mathbf{x}) = \frac{1}{2} \mathbf{x}^\top \mathbf{Q} \mathbf{x} + \mathbf{c}^\top \mathbf{x} + b$ を最小化する問題を見てみよう。

:numref:`sec_momentum` で見たように、この問題は固有値分解 $\mathbf{Q} = \mathbf{U}^\top \boldsymbol{\Lambda} \mathbf{U}$ を用いて書き換えることができ、各座標を個別に解ける、はるかに単純な問題になる。

$$f(\mathbf{x}) = \bar{f}(\bar{\mathbf{x}}) = \frac{1}{2} \bar{\mathbf{x}}^\top \boldsymbol{\Lambda} \bar{\mathbf{x}} + \bar{\mathbf{c}}^\top \bar{\mathbf{x}} + b.$$

ここでは $\bar{\mathbf{x}} = \mathbf{U} \mathbf{x}$、したがって $\bar{\mathbf{c}} = \mathbf{U} \mathbf{c}$ を用いた。変換後の問題の最小解は $\bar{\mathbf{x}} = -\boldsymbol{\Lambda}^{-1} \bar{\mathbf{c}}$、最小値は $-\frac{1}{2} \bar{\mathbf{c}}^\top \boldsymbol{\Lambda}^{-1} \bar{\mathbf{c}} + b$ である。$\boldsymbol{\Lambda}$ は $\mathbf{Q}$ の固有値を並べた対角行列なので、これはずっと計算しやすくなる。

$\mathbf{c}$ を少し摂動したとき、$f$ の最小解も少しだけ変わることを期待したいところである。ところが、実際にはそうならない。$\mathbf{c}$ のわずかな変化は $\bar{\mathbf{c}}$ にも同程度のわずかな変化しか与えないが、$f$（および $\bar{f}$）の最小解についてはそうではない。固有値 $\boldsymbol{\Lambda}_i$ が大きいとき、$\bar{x}_i$ と $\bar{f}$ の最小値の変化は小さくなる。逆に、$\boldsymbol{\Lambda}_i$ が小さいときには $\bar{x}_i$ の変化は劇的になりえる。最大固有値と最小固有値の比は、最適化問題の条件数と呼ばれる。

$$\kappa = \frac{\boldsymbol{\Lambda}_1}{\boldsymbol{\Lambda}_d}.$$

条件数 $\kappa$ が大きいと、最適化問題を正確に解くのは困難である。値のダイナミックレンジが大きい場合にも正しく扱えるよう、注意深くする必要がある。この分析から、素朴ではあるものの明らかな疑問が生じる。すべての固有値が $1$ になるように空間を歪めてしまえば、問題を単純に「修正」できないのだろうか。理論上はこれはかなり簡単である。$\mathbf{Q}$ の固有値と固有ベクトルさえあれば、問題を $\mathbf{x}$ から $\mathbf{z} \stackrel{\textrm{def}}{=} \boldsymbol{\Lambda}^{\frac{1}{2}} \mathbf{U} \mathbf{x}$ へと再スケーリングできる。新しい座標系では $\mathbf{x}^\top \mathbf{Q} \mathbf{x}$ は $\|\mathbf{z}\|^2$ に簡約できる。とはいえ、これはかなり実用的でない提案である。一般に、固有値と固有ベクトルを計算することは、実際の問題を解くことよりも *はるかに* 高コストだからである。

固有値を正確に計算するのは高価かもしれないが、それを推測し、多少なりとも近似的に計算するだけでも、何もしないよりはずっとよい場合がある。特に、$\mathbf{Q}$ の対角成分を使って、それに応じて再スケーリングすることができる。これは固有値を計算するよりも *はるかに* 安価である。

$$\tilde{\mathbf{Q}} = \textrm{diag}^{-\frac{1}{2}}(\mathbf{Q}) \mathbf{Q} \textrm{diag}^{-\frac{1}{2}}(\mathbf{Q}).$$

このとき $\tilde{\mathbf{Q}}_{ij} = \mathbf{Q}_{ij} / \sqrt{\mathbf{Q}_{ii} \mathbf{Q}_{jj}}$ であり、特にすべての $i$ について $\tilde{\mathbf{Q}}_{ii} = 1$ である。多くの場合、これにより条件数はかなり改善される。たとえば、先ほど議論したケースでは、問題が軸に整列しているため、この処理だけで問題を完全に解消できる。

残念ながら、ここでも別の問題に直面する。深層学習では通常、目的関数の二階微分にすらアクセスできない。$\mathbf{x} \in \mathbb{R}^d$ に対して、ミニバッチ上であっても二階微分の計算には $\mathcal{O}(d^2)$ の空間と計算量が必要になることがあり、実用的ではない。Adagrad の巧妙なアイデアは、この捉えにくいヘッセ行列の対角成分の代わりに、比較的安価に計算でき、しかも有効な代理量、すなわち勾配そのものの大きさを使うことである。

なぜこれがうまくいくのかを見るために、$\bar{f}(\bar{\mathbf{x}})$ を考えよう。すると

$$\partial_{\bar{\mathbf{x}}} \bar{f}(\bar{\mathbf{x}}) = \boldsymbol{\Lambda} \bar{\mathbf{x}} + \bar{\mathbf{c}} = \boldsymbol{\Lambda} \left(\bar{\mathbf{x}} - \bar{\mathbf{x}}_0\right),$$

ここで $\bar{\mathbf{x}}_0$ は $\bar{f}$ の最小解である。したがって、勾配の大きさは $\boldsymbol{\Lambda}$ と最適解からの距離の両方に依存する。もし $\bar{\mathbf{x}} - \bar{\mathbf{x}}_0$ が変化しないなら、これで十分である。結局のところ、この場合には勾配 $\partial_{\bar{\mathbf{x}}} \bar{f}(\bar{\mathbf{x}})$ の大きさだけで足りる。AdaGrad は確率的勾配降下法なので、最適点においても分散がゼロでない勾配が現れる。その結果、勾配の分散をヘッセ行列のスケールの安価な代理として安全に使うことができる。厳密な解析はこの節の範囲を超える（数ページにわたる内容になる）。詳細は :cite:`Duchi.Hazan.Singer.2011` を参照されたい。


## アルゴリズム

上の議論を形式化しよう。過去の勾配分散を次のように蓄積するために、変数 $\mathbf{s}_t$ を用いる。

$$\begin{aligned}
    \mathbf{g}_t & = \partial_{\mathbf{w}} l(y_t, f(\mathbf{x}_t, \mathbf{w})), \\
    \mathbf{s}_t & = \mathbf{s}_{t-1} + \mathbf{g}_t^2, \\
    \mathbf{w}_t & = \mathbf{w}_{t-1} - \frac{\eta}{\sqrt{\mathbf{s}_t + \epsilon}} \cdot \mathbf{g}_t.
\end{aligned}$$

ここでの演算は座標ごとに適用される。つまり、$\mathbf{v}^2$ の各成分は $v_i^2$ である。同様に、$\frac{1}{\sqrt{v}}$ の各成分は $\frac{1}{\sqrt{v_i}}$ であり、$\mathbf{u} \cdot \mathbf{v}$ の各成分は $u_i v_i$ である。これまでと同様に、$\eta$ は学習率、$\epsilon$ は $0$ で割らないようにするための加算定数である。最後に、$\mathbf{s}_0 = \mathbf{0}$ で初期化する。

モメンタム法と同様に、補助変数を追跡する必要があるが、今回は各座標ごとに個別の学習率を可能にするためである。主な計算コストは通常 $l(y_t, f(\mathbf{x}_t, \mathbf{w}))$ とその導関数の計算であるため、Adagrad のコストは SGD と比べて大きくは増えない。

$\mathbf{s}_t$ に二乗勾配を蓄積するということは、$\mathbf{s}_t$ が本質的に線形の速度で増加することを意味する（実際には、勾配が最初は減少するので、ややそれより遅い）。これにより、座標ごとに調整された $\mathcal{O}(t^{-\frac{1}{2}})$ の学習率が得られる。凸問題に対しては、これはまったく十分である。しかし深層学習では、学習率をもう少しゆっくり下げたい場合がある。そのため、後続の章で議論するいくつかの Adagrad 変種が生まれた。ここではまず、二次の凸問題でどのように振る舞うかを見てみよう。先ほどと同じ問題を使う。

$$f(\mathbf{x}) = 0.1 x_1^2 + 2 x_2^2.$$

以前と同じ学習率、すなわち $\eta = 0.4$ を用いて Adagrad を実装する。見てわかるように、独立変数の反復軌跡はより滑らかになる。しかし、$\boldsymbol{s}_t$ の累積効果により学習率は継続的に減衰するため、反復の後半では独立変数の動きはそれほど大きくない。

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
%matplotlib inline
from d2l import torch as d2l
import math
import torch
```

```{.python .input}
#@tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
import math
import tensorflow as tf
```

```{.python .input}
#@tab all
def adagrad_2d(x1, x2, s1, s2):
    eps = 1e-6
    g1, g2 = 0.2 * x1, 4 * x2
    s1 += g1 ** 2
    s2 += g2 ** 2
    x1 -= eta / math.sqrt(s1 + eps) * g1
    x2 -= eta / math.sqrt(s2 + eps) * g2
    return x1, x2, s1, s2

def f_2d(x1, x2):
    return 0.1 * x1 ** 2 + 2 * x2 ** 2

eta = 0.4
d2l.show_trace_2d(f_2d, d2l.train_2d(adagrad_2d))
```

学習率を $2$ に増やすと、よりよい挙動が見られる。これは、ノイズのない場合であっても学習率の減少がかなり急激になりうることを示しており、パラメータが適切に収束するように注意する必要があることがわかる。

```{.python .input}
#@tab all
eta = 2
d2l.show_trace_2d(f_2d, d2l.train_2d(adagrad_2d))
```

## ゼロからの実装

モメンタム法と同様に、Adagrad でもパラメータと同じ形状の状態変数を保持する必要がある。

```{.python .input}
#@tab mxnet
def init_adagrad_states(feature_dim):
    s_w = d2l.zeros((feature_dim, 1))
    s_b = d2l.zeros(1)
    return (s_w, s_b)

def adagrad(params, states, hyperparams):
    eps = 1e-6
    for p, s in zip(params, states):
        s[:] += np.square(p.grad)
        p[:] -= hyperparams['lr'] * p.grad / np.sqrt(s + eps)
```

```{.python .input}
#@tab pytorch
def init_adagrad_states(feature_dim):
    s_w = d2l.zeros((feature_dim, 1))
    s_b = d2l.zeros(1)
    return (s_w, s_b)

def adagrad(params, states, hyperparams):
    eps = 1e-6
    for p, s in zip(params, states):
        with torch.no_grad():
            s[:] += torch.square(p.grad)
            p[:] -= hyperparams['lr'] * p.grad / torch.sqrt(s + eps)
        p.grad.data.zero_()
```

```{.python .input}
#@tab tensorflow
def init_adagrad_states(feature_dim):
    s_w = tf.Variable(d2l.zeros((feature_dim, 1)))
    s_b = tf.Variable(d2l.zeros(1))
    return (s_w, s_b)

def adagrad(params, grads, states, hyperparams):
    eps = 1e-6
    for p, s, g in zip(params, states, grads):
        s[:].assign(s + tf.math.square(g))
        p[:].assign(p - hyperparams['lr'] * g / tf.math.sqrt(s + eps))
```

:numref:`sec_minibatch_sgd` の実験と比べて、モデルの学習にはより大きな学習率を使う。

```{.python .input}
#@tab all
data_iter, feature_dim = d2l.get_data_ch11(batch_size=10)
d2l.train_ch11(adagrad, init_adagrad_states(feature_dim),
               {'lr': 0.1}, data_iter, feature_dim);
```

## 簡潔な実装

アルゴリズム `adagrad` の `Trainer` インスタンスを使えば、Gluon で Adagrad アルゴリズムを呼び出せる。

```{.python .input}
#@tab mxnet
d2l.train_concise_ch11('adagrad', {'learning_rate': 0.1}, data_iter)
```

```{.python .input}
#@tab pytorch
trainer = torch.optim.Adagrad
d2l.train_concise_ch11(trainer, {'lr': 0.1}, data_iter)
```

```{.python .input}
#@tab tensorflow
trainer = tf.keras.optimizers.Adagrad
d2l.train_concise_ch11(trainer, {'learning_rate' : 0.1}, data_iter)
```

## まとめ

* Adagrad は、座標ごとに動的に学習率を下げる。
* 勾配の大きさを、進捗の速さを調整する手段として用いる。勾配が大きい座標は、より小さな学習率で補正される。
* 深層学習の問題では、メモリと計算量の制約のため、正確な二階微分の計算は通常不可能である。勾配は有用な代理量になりえる。
* 最適化問題の構造がかなり不均一な場合、Adagrad はその歪みを緩和するのに役立つ。
* Adagrad は、まれにしか現れない項に対して学習率をよりゆっくり下げる必要がある疎な特徴に特に有効である。
* 深層学習の問題では、Adagrad は学習率を下げるのが強すぎる場合がある。これを緩和する方法については :numref:`sec_adam` で議論する。

## 演習

1. 直交行列 $\mathbf{U}$ とベクトル $\mathbf{c}$ に対して、次が成り立つことを証明せよ: $\|\mathbf{c} - \mathbf{\delta}\|_2 = \|\mathbf{U} \mathbf{c} - \mathbf{U} \mathbf{\delta}\|_2$. なぜこれは、直交変数変換の後でも摂動の大きさが変わらないことを意味するのか。
1. $f(\mathbf{x}) = 0.1 x_1^2 + 2 x_2^2$ に対して、また目的関数を 45 度回転させた $f(\mathbf{x}) = 0.1 (x_1 + x_2)^2 + 2 (x_1 - x_2)^2$ に対して Adagrad を試してみよ。挙動は異なるか。
1. 行列 $\mathbf{M}$ の固有値 $\lambda_i$ が少なくとも一つの $j$ の選び方について $|\lambda_i - \mathbf{M}_{jj}| \leq \sum_{k \neq j} |\mathbf{M}_{jk}|$ を満たすことを述べる [Gerschgorin の円定理](https://en.wikipedia.org/wiki/Gershgorin_circle_theorem) を証明せよ。
1. Gerschgorin の定理は、対角前処理された行列 $\textrm{diag}^{-\frac{1}{2}}(\mathbf{M}) \mathbf{M} \textrm{diag}^{-\frac{1}{2}}(\mathbf{M})$ の固有値について何を教えてくれるか。
1. Fashion-MNIST に適用した :numref:`sec_lenet` のような適切な深層ネットワークで Adagrad を試してみよ。
1. 学習率の減衰をより緩やかにするには、Adagrad をどのように修正する必要があるか。
