# 確率的勾配降下法
:label:`sec_sgd`

前の章では、学習手順の中で確率的勾配降下法を使い続けてきましたが、なぜそれがうまく働くのかについては説明していませんでした。
それを少し明らかにするために、
:numref:`sec_gd` では勾配降下法の基本原理だけを説明しました。
この節では、
*確率的勾配降下法* についてさらに詳しく議論します。

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

## 確率的勾配更新

深層学習では、目的関数は通常、訓練データセット中の各サンプルに対する損失関数の平均です。
$n$ 個のサンプルからなる訓練データセットが与えられたとき、
$\mathbf{x}$ をパラメータベクトルとして、インデックス $i$ の訓練サンプルに関する損失関数を $f_i(\mathbf{x})$ とします。
すると、目的関数は

$$f(\mathbf{x}) = \frac{1}{n} \sum_{i = 1}^n f_i(\mathbf{x}).$$

となります。

$\mathbf{x}$ における目的関数の勾配は次のように計算されます。

$$\nabla f(\mathbf{x}) = \frac{1}{n} \sum_{i = 1}^n \nabla f_i(\mathbf{x}).$$

勾配降下法を用いると、各独立変数の反復ごとの計算コストは $\mathcal{O}(n)$ であり、$n$ に対して線形に増加します。したがって、訓練データセットが大きいほど、各反復における勾配降下法のコストは高くなります。

確率的勾配降下法（SGD）は、各反復の計算コストを削減します。確率的勾配降下法の各反復では、データサンプルのインデックス $i\in\{1,\ldots, n\}$ を一様にランダムサンプリングし、勾配 $\nabla f_i(\mathbf{x})$ を計算して $\mathbf{x}$ を更新します。

$$\mathbf{x} \leftarrow \mathbf{x} - \eta \nabla f_i(\mathbf{x}),$$

ここで $\eta$ は学習率です。各反復の計算コストが、勾配降下法の $\mathcal{O}(n)$ から定数の $\mathcal{O}(1)$ に下がることがわかります。さらに、確率的勾配 $\nabla f_i(\mathbf{x})$ は全勾配 $\nabla f(\mathbf{x})$ の不偏推定量であることを強調しておきます。なぜなら

$$\mathbb{E}_i \nabla f_i(\mathbf{x}) = \frac{1}{n} \sum_{i = 1}^n \nabla f_i(\mathbf{x}) = \nabla f(\mathbf{x}).$$

だからです。

これは、平均的には確率的勾配が勾配のよい推定値であることを意味します。

ここでは、平均 0、分散 1 のランダムノイズを勾配に加えて確率的勾配降下法を模擬し、勾配降下法と比較してみます。

```{.python .input}
#@tab all
def f(x1, x2):  # Objective function
    return x1 ** 2 + 2 * x2 ** 2

def f_grad(x1, x2):  # Gradient of the objective function
    return 2 * x1, 4 * x2
```

```{.python .input}
#@tab mxnet
def sgd(x1, x2, s1, s2, f_grad):
    g1, g2 = f_grad(x1, x2)
    # Simulate noisy gradient
    g1 += d2l.normal(0.0, 1, (1,))
    g2 += d2l.normal(0.0, 1, (1,))
    eta_t = eta * lr()
    return (x1 - eta_t * g1, x2 - eta_t * g2, 0, 0)
```

```{.python .input}
#@tab pytorch
def sgd(x1, x2, s1, s2, f_grad):
    g1, g2 = f_grad(x1, x2)
    # Simulate noisy gradient
    g1 += torch.normal(0.0, 1, (1,)).item()
    g2 += torch.normal(0.0, 1, (1,)).item()
    eta_t = eta * lr()
    return (x1 - eta_t * g1, x2 - eta_t * g2, 0, 0)
```

```{.python .input}
#@tab tensorflow
def sgd(x1, x2, s1, s2, f_grad):
    g1, g2 = f_grad(x1, x2)
    # Simulate noisy gradient
    g1 += d2l.normal([1], 0.0, 1)
    g2 += d2l.normal([1], 0.0, 1)
    eta_t = eta * lr()
    return (x1 - eta_t * g1, x2 - eta_t * g2, 0, 0)
```

```{.python .input}
#@tab all
def constant_lr():
    return 1

eta = 0.1
lr = constant_lr  # Constant learning rate
d2l.show_trace_2d(f, d2l.train_2d(sgd, steps=50, f_grad=f_grad))
```

見てわかるように、確率的勾配降下法における変数の軌跡は、 :numref:`sec_gd` で見た勾配降下法の軌跡よりもはるかにノイズが大きくなっています。これは勾配の確率的な性質によるものです。つまり、最小値付近に到達しても、$\eta \nabla f_i(\mathbf{x})$ によってその場で注入される不確実性の影響をなお受け続けます。50 ステップ後でも品質はまだあまり良くありません。さらに悪いことに、ステップ数を増やしても改善しません（これを確認するために、より多くのステップ数で試してみることを勧めます）。そこで残る唯一の選択肢は、学習率 $\eta$ を変えることです。しかし、これを小さくしすぎると、最初のうちは意味のある進展が得られません。逆に大きすぎると、上で見たように良い解は得られません。こうした相反する目標を解決する唯一の方法は、最適化が進むにつれて学習率を*動的に*減少させることです。

これが、`sgd` ステップ関数に学習率関数 `lr` を追加している理由でもあります。上の例では、対応する `lr` 関数を定数に設定しているため、学習率スケジューリングの機能は使われていません。

## 動的な学習率

$\eta$ を時間依存の学習率 $\eta(t)$ に置き換えると、最適化アルゴリズムの収束制御はより複雑になります。特に、$\eta$ をどれくらい速く減衰させるべきかを決める必要があります。減衰が速すぎると、最適化を途中で打ち切ってしまいます。遅すぎると、最適化に時間を浪費します。以下は、時間とともに $\eta$ を調整するために使われる基本的な戦略のいくつかです（より高度な戦略は後で議論します）。

$$
\begin{aligned}
    \eta(t) & = \eta_i \textrm{ if } t_i \leq t \leq t_{i+1}  && \textrm{piecewise constant} \\
    \eta(t) & = \eta_0 \cdot e^{-\lambda t} && \textrm{exponential decay} \\
    \eta(t) & = \eta_0 \cdot (\beta t + 1)^{-\alpha} && \textrm{polynomial decay}
\end{aligned}
$$

最初の *区分定数* の場合では、たとえば最適化の進展が停滞したときに学習率を下げます。これは深層ネットワークの学習でよく使われる戦略です。別の方法として、*指数減衰* によってもっと積極的に下げることもできます。残念ながら、これはしばしばアルゴリズムが収束する前に早期停止してしまいます。よく使われる選択肢は、$\alpha = 0.5$ の *多項式減衰* です。凸最適化の場合、この減衰率が良好に振る舞うことを示す証明が数多くあります。

指数減衰が実際にはどのように見えるかを見てみましょう。

```{.python .input}
#@tab all
def exponential_lr():
    # Global variable that is defined outside this function and updated inside
    global t
    t += 1
    return math.exp(-0.1 * t)

t = 1
lr = exponential_lr
d2l.show_trace_2d(f, d2l.train_2d(sgd, steps=1000, f_grad=f_grad))
```

予想どおり、パラメータの分散は大幅に減少しています。しかしその代償として、最適解 $\mathbf{x} = (0, 0)$ に収束できていません。1000 回の反復後でも、最適解からはまだ非常に遠いままです。実際、このアルゴリズムはまったく収束していません。一方、学習率がステップ数の平方根の逆数で減衰する多項式減衰を使うと、50 ステップ後には収束が改善します。

```{.python .input}
#@tab all
def polynomial_lr():
    # Global variable that is defined outside this function and updated inside
    global t
    t += 1
    return (1 + 0.1 * t) ** (-0.5)

t = 1
lr = polynomial_lr
d2l.show_trace_2d(f, d2l.train_2d(sgd, steps=50, f_grad=f_grad))
```

学習率の設定方法には、さらに多くの選択肢があります。たとえば、小さな値から始めて、その後すばやく増やし、さらにゆっくりと再び減らすこともできます。あるいは、小さい学習率と大きい学習率を交互に使うことさえできます。このようなスケジュールには多様なものがあります。ここでは、包括的な理論解析が可能な学習率スケジュール、すなわち凸設定における学習率に焦点を当てます。一般の非凸問題では、意味のある収束保証を得るのは非常に難しいです。というのも、一般には非線形な非凸問題の最小化は NP 困難だからです。概説としては、Tibshirani 2015 の優れた [lecture notes](https://www.stat.cmu.edu/%7Eryantibs/convexopt-F15/lectures/26-nonconvex.pdf) を参照してください。



## 凸目的関数に対する収束解析

凸目的関数に対する確率的勾配降下法の以下の収束解析は任意選択であり、主として問題に対する直感を深めるためのものです。
ここでは、最も簡単な証明の一つ :cite:`Nesterov.Vial.2000` に限定します。
目的関数が特に良い性質を持つ場合など、はるかに高度な証明技法も存在します。


目的関数 $f(\boldsymbol{\xi}, \mathbf{x})$ が、すべての $\boldsymbol{\xi}$ に対して $\mathbf{x}$ に関して凸であると仮定します。
より具体的には、次の確率的勾配降下法の更新を考えます。

$$\mathbf{x}_{t+1} = \mathbf{x}_{t} - \eta_t \partial_\mathbf{x} f(\boldsymbol{\xi}_t, \mathbf{x}),$$

ここで $f(\boldsymbol{\xi}_t, \mathbf{x})$
は、ステップ $t$ においてある分布から引かれた訓練サンプル $\boldsymbol{\xi}_t$
に関する目的関数であり、$\mathbf{x}$ はモデルパラメータです。
次で定義される

$$R(\mathbf{x}) = E_{\boldsymbol{\xi}}[f(\boldsymbol{\xi}, \mathbf{x})]$$

を期待リスク、その $\mathbf{x}$ に関する最小値を $R^*$ とします。最後に、$\mathbf{x}^*$ を最小化解とします（$\mathbf{x}$ が定義される領域内に存在すると仮定します）。このとき、時刻 $t$ における現在のパラメータ $\mathbf{x}_t$ とリスク最小化解 $\mathbf{x}^*$ との距離を追跡し、時間とともに改善するかどうかを調べることができます。

$$\begin{aligned}    &\|\mathbf{x}_{t+1} - \mathbf{x}^*\|^2 \\ =& \|\mathbf{x}_{t} - \eta_t \partial_\mathbf{x} f(\boldsymbol{\xi}_t, \mathbf{x}) - \mathbf{x}^*\|^2 \\    =& \|\mathbf{x}_{t} - \mathbf{x}^*\|^2 + \eta_t^2 \|\partial_\mathbf{x} f(\boldsymbol{\xi}_t, \mathbf{x})\|^2 - 2 \eta_t    \left\langle \mathbf{x}_t - \mathbf{x}^*, \partial_\mathbf{x} f(\boldsymbol{\xi}_t, \mathbf{x})\right\rangle.   \end{aligned}$$
:eqlabel:`eq_sgd-xt+1-xstar`

確率的勾配 $\partial_\mathbf{x} f(\boldsymbol{\xi}_t, \mathbf{x})$ の $\ell_2$ ノルムがある定数 $L$ によって抑えられると仮定すると、

$$\eta_t^2 \|\partial_\mathbf{x} f(\boldsymbol{\xi}_t, \mathbf{x})\|^2 \leq \eta_t^2 L^2.$$
:eqlabel:`eq_sgd-L`


ここで主に関心があるのは、$\mathbf{x}_t$ と $\mathbf{x}^*$ の距離が*期待値の意味で*どう変化するかです。実際、特定のステップ列に対しては、遭遇する $\boldsymbol{\xi}_t$ によって距離が増加することも十分ありえます。したがって、内積を上から抑える必要があります。
任意の凸関数 $f$ について
$f(\mathbf{y}) \geq f(\mathbf{x}) + \langle f'(\mathbf{x}), \mathbf{y} - \mathbf{x} \rangle$
がすべての $\mathbf{x}$ と $\mathbf{y}$ に対して成り立つので、
凸性より

$$f(\boldsymbol{\xi}_t, \mathbf{x}^*) \geq f(\boldsymbol{\xi}_t, \mathbf{x}_t) + \left\langle \mathbf{x}^* - \mathbf{x}_t, \partial_{\mathbf{x}} f(\boldsymbol{\xi}_t, \mathbf{x}_t) \right\rangle.$$
:eqlabel:`eq_sgd-f-xi-xstar`

不等式 :eqref:`eq_sgd-L` と :eqref:`eq_sgd-f-xi-xstar` の両方を :eqref:`eq_sgd-xt+1-xstar` に代入すると、時刻 $t+1$ におけるパラメータ間の距離について次のような上界が得られます。

$$\|\mathbf{x}_{t} - \mathbf{x}^*\|^2 - \|\mathbf{x}_{t+1} - \mathbf{x}^*\|^2 \geq 2 \eta_t (f(\boldsymbol{\xi}_t, \mathbf{x}_t) - f(\boldsymbol{\xi}_t, \mathbf{x}^*)) - \eta_t^2 L^2.$$
:eqlabel:`eqref_sgd-xt-diff`

これは、現在の損失と最適損失の差が $\eta_t L^2/2$ を上回る限り、進展があることを意味します。この差はゼロに収束するはずなので、学習率 $\eta_t$ も*消失*する必要があります。

次に、:eqref:`eqref_sgd-xt-diff` の期待値を取ります。すると

$$E\left[\|\mathbf{x}_{t} - \mathbf{x}^*\|^2\right] - E\left[\|\mathbf{x}_{t+1} - \mathbf{x}^*\|^2\right] \geq 2 \eta_t [E[R(\mathbf{x}_t)] - R^*] -  \eta_t^2 L^2.$$

最後のステップでは、$t \in \{1, \ldots, T\}$ に対する不等式を総和します。和は望遠和になり、下側の項を捨てると

$$\|\mathbf{x}_1 - \mathbf{x}^*\|^2 \geq 2 \left (\sum_{t=1}^T   \eta_t \right) [E[R(\mathbf{x}_t)] - R^*] - L^2 \sum_{t=1}^T \eta_t^2.$$
:eqlabel:`eq_sgd-x1-xstar`

$\mathbf{x}_1$ は与えられているので、期待値を外せることに注意してください。最後に

$$\bar{\mathbf{x}} \stackrel{\textrm{def}}{=} \frac{\sum_{t=1}^T \eta_t \mathbf{x}_t}{\sum_{t=1}^T \eta_t}.$$

を定義します。

$$E\left(\frac{\sum_{t=1}^T \eta_t R(\mathbf{x}_t)}{\sum_{t=1}^T \eta_t}\right) = \frac{\sum_{t=1}^T \eta_t E[R(\mathbf{x}_t)]}{\sum_{t=1}^T \eta_t} = E[R(\mathbf{x}_t)],$$

であり、Jensen の不等式（:eqref:`eq_jensens-inequality` において $i=t$, $\alpha_i = \eta_t/\sum_{t=1}^T \eta_t$ と置く）と $R$ の凸性より、$E[R(\mathbf{x}_t)] \geq E[R(\bar{\mathbf{x}})]$ が従うので、

$$\sum_{t=1}^T \eta_t E[R(\mathbf{x}_t)] \geq \sum_{t=1}^T \eta_t  E\left[R(\bar{\mathbf{x}})\right].$$

これを不等式 :eqref:`eq_sgd-x1-xstar` に代入すると、次の上界が得られます。

$$
\left[E[\bar{\mathbf{x}}]\right] - R^* \leq \frac{r^2 + L^2 \sum_{t=1}^T \eta_t^2}{2 \sum_{t=1}^T \eta_t},
$$

ここで $r^2 \stackrel{\textrm{def}}{=} \|\mathbf{x}_1 - \mathbf{x}^*\|^2$ は、初期パラメータの選択と最終結果との距離の上界です。要するに、収束速度は、確率的勾配のノルムがどのように抑えられているか（$L$）と、初期パラメータ値が最適値からどれだけ離れているか（$r$）に依存します。なお、この上界は $\mathbf{x}_T$ ではなく $\bar{\mathbf{x}}$ に関するものであることに注意してください。これは、$\bar{\mathbf{x}}$ が最適化経路を平滑化したものだからです。
$r, L, T$ が既知であれば、学習率 $\eta = r/(L \sqrt{T})$ を選べます。すると上界は $rL/\sqrt{T}$ になります。つまり、最適解へ $\mathcal{O}(1/\sqrt{T})$ の速度で収束します。





## 確率的勾配と有限サンプル

ここまで、確率的勾配降下法について少し大ざっぱに扱ってきました。通常は分布 $p(x, y)$ から、ラベル $y_i$ を伴うインスタンス $x_i$ をサンプリングし、それを何らかの形でモデルパラメータの更新に使うと仮定してきました。特に有限サンプルサイズの場合には、離散分布 $p(x, y) = \frac{1}{n} \sum_{i=1}^n \delta_{x_i}(x) \delta_{y_i}(y)$
が、ある関数 $\delta_{x_i}$ と $\delta_{y_i}$
に対して確率的勾配降下法を実行することを可能にすると単純に論じてきました。

しかし、実際にはそうしていたわけではありません。この節の玩具例では、もともと確率的でない勾配にノイズを加えていただけであり、つまり $(x_i, y_i)$ の組があるかのように振る舞っていました。ここではそれが正当化できることがわかります（詳細な議論は演習を参照してください）。より問題なのは、これまでの議論では明らかにそうしていなかったことです。代わりに、すべてのインスタンスを*ちょうど一度ずつ*走査していました。なぜこれが望ましいのかを見るために、逆に、離散分布から $n$ 個の観測を*復元抽出*する場合を考えましょう。要素 $i$ をランダムに選ぶ確率は $1/n$ です。したがって、それを*少なくとも一度*選ぶ確率は

$$P(\textrm{choose~} i) = 1 - P(\textrm{omit~} i) = 1 - (1-1/n)^n \approx 1-e^{-1} \approx 0.63.$$

同様の考え方から、あるサンプル（すなわち訓練例）を*ちょうど一度*選ぶ確率は

$${n \choose 1} \frac{1}{n} \left(1-\frac{1}{n}\right)^{n-1} = \frac{n}{n-1} \left(1-\frac{1}{n}\right)^{n} \approx e^{-1} \approx 0.37.$$

復元抽出は、*非復元抽出* と比べて分散を増やし、データ効率を下げます。したがって、実際には後者を行います（そしてこれが本書全体でのデフォルトの選択です）。最後に、訓練データセットを繰り返し走査するときには、毎回*異なる*ランダム順序で走査されることに注意してください。


## まとめ

* 凸問題では、広い範囲の学習率に対して、確率的勾配降下法が最適解に収束することを証明できます。
* 深層学習では、一般にはそうではありません。しかし、凸問題の解析は、最適化にどう取り組むべきかについて有用な洞察を与えてくれます。すなわち、学習率を徐々に下げること、ただし下げすぎないことです。
* 学習率が小さすぎても大きすぎても問題が起こります。実際には、適切な学習率は複数回の実験を経て初めて見つかることが多いです。
* 訓練データセットのサンプル数が多いほど、勾配降下法の各反復の計算コストは高くなるため、そのような場合には確率的勾配降下法が好まれます。
* 非凸の場合、チェックすべき局所最小値の数が指数的になりうるため、確率的勾配降下法に対する最適性保証は一般には得られません。




## 演習

1. 確率的勾配降下法について、さまざまな学習率スケジュールとさまざまな反復回数を試してみなさい。特に、最適解 $(0, 0)$ からの距離を反復回数の関数としてプロットしなさい。
1. 関数 $f(x_1, x_2) = x_1^2 + 2 x_2^2$ に対して、勾配に正規ノイズを加えることが、$\mathbf{x}$ が正規分布から引かれるときの損失関数 $f(\mathbf{x}, \mathbf{w}) = (x_1 - w_1)^2 + 2 (x_2 - w_2)^2$ を最小化することと等価であることを証明しなさい。
1. $\{(x_1, y_1), \ldots, (x_n, y_n)\}$ から復元抽出する場合と非復元抽出する場合で、確率的勾配降下法の収束を比較しなさい。
1. ある勾配（あるいはそれに対応するある座標）が他のすべての勾配より一貫して大きい場合、確率的勾配降下法ソルバをどのように変更しますか？
1. $f(x) = x^2 (1 + \sin x)$ と仮定します。$f$ には局所最小値がいくつありますか？ それを最小化するために、すべての局所最小値を評価する必要があるように $f$ を変更できますか？

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/352)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/497)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/1067)
:end_tab:\n