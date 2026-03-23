# 多変数微積分
:label:`sec_multivariable_calculus`

単一変数関数の導関数についてかなり深く理解できたので、今度は、潜在的に何十億もの重みをもつ損失関数を考えていた元の問いに戻りましょう。

## 高次元での微分
:numref:`sec_single_variable_calculus` が教えてくれるのは、他のすべてを固定したままこれら何十億もの重みのうちの1つだけを変えたとき、何が起こるかが分かるということです！  これは単なる1変数関数にほかなりませんから、次のように書ける。

$$L(w_1+\epsilon_1, w_2, \ldots, w_N) \approx L(w_1, w_2, \ldots, w_N) + \epsilon_1 \frac{d}{dw_1} L(w_1, w_2, \ldots, w_N).$$
:eqlabel:`eq_part_der`

他の変数を固定したまま1つの変数についてとる導関数を *偏微分* と呼び、:eqref:`eq_part_der` の導関数には $\frac{\partial}{\partial w_1}$ という記法を用いる。

では、これを使って $w_2$ を少しだけ $w_2 + \epsilon_2$ に変えてみよう。

$$
\begin{aligned}
L(w_1+\epsilon_1, w_2+\epsilon_2, \ldots, w_N) & \approx L(w_1, w_2+\epsilon_2, \ldots, w_N) + \epsilon_1 \frac{\partial}{\partial w_1} L(w_1, w_2+\epsilon_2, \ldots, w_N+\epsilon_N) \\
& \approx L(w_1, w_2, \ldots, w_N) \\
& \quad + \epsilon_2\frac{\partial}{\partial w_2} L(w_1, w_2, \ldots, w_N) \\
& \quad + \epsilon_1 \frac{\partial}{\partial w_1} L(w_1, w_2, \ldots, w_N) \\
& \quad + \epsilon_1\epsilon_2\frac{\partial}{\partial w_2}\frac{\partial}{\partial w_1} L(w_1, w_2, \ldots, w_N) \\
& \approx L(w_1, w_2, \ldots, w_N) \\
& \quad + \epsilon_2\frac{\partial}{\partial w_2} L(w_1, w_2, \ldots, w_N) \\
& \quad + \epsilon_1 \frac{\partial}{\partial w_1} L(w_1, w_2, \ldots, w_N).
\end{aligned}
$$

ここでも、$\epsilon_1\epsilon_2$ は高次の項であり、前節で $\epsilon^{2}$ を捨てたのと同じように無視できる、という考え方を使いた。:eqref:`eq_part_der` で見たことも同様である。  このように続けていくと、次のように書ける。

$$
L(w_1+\epsilon_1, w_2+\epsilon_2, \ldots, w_N+\epsilon_N) \approx L(w_1, w_2, \ldots, w_N) + \sum_i \epsilon_i \frac{\partial}{\partial w_i} L(w_1, w_2, \ldots, w_N).
$$

これは一見ごちゃごちゃして見えますが、右辺の和がちょうど内積の形になっていることに気づけば、もっと見慣れた形にできる。そこで

$$
\boldsymbol{\epsilon} = [\epsilon_1, \ldots, \epsilon_N]^\top \; \textrm{and} \;
\nabla_{\mathbf{x}} L = \left[\frac{\partial L}{\partial x_1}, \ldots, \frac{\partial L}{\partial x_N}\right]^\top,
$$

とおくと、

$$L(\mathbf{w} + \boldsymbol{\epsilon}) \approx L(\mathbf{w}) + \boldsymbol{\epsilon}\cdot \nabla_{\mathbf{w}} L(\mathbf{w}).$$
:eqlabel:`eq_nabla_use`

ベクトル $\nabla_{\mathbf{w}} L$ を $L$ の *勾配* と呼ぶ。

式 :eqref:`eq_nabla_use` は少し立ち止まって考える価値がある。  これは1次元で見たものとまったく同じ形式で、ただすべてをベクトルと内積に置き換えただけである。  これにより、入力にどのような摂動を与えたときに関数 $L$ がどのように変化するかを近似的に知ることができる。  次節で見るように、これは勾配に含まれる情報を使って、学習が幾何学的にどのように行えるかを理解するうえで重要な道具になる。

しかしその前に、この近似が実際にどう働くかを例で見てみよう。  次の関数を扱っているとする。

$$
f(x, y) = \log(e^x + e^y) \textrm{ with gradient } \nabla f (x, y) = \left[\frac{e^x}{e^x+e^y}, \frac{e^y}{e^x+e^y}\right].
$$

$(0, \log(2))$ のような点を見ると、

$$
f(x, y) = \log(3) \textrm{ with gradient } \nabla f (x, y) = \left[\frac{1}{3}, \frac{2}{3}\right].
$$

したがって、$(\epsilon_1, \log(2) + \epsilon_2)$ における $f$ を近似したいなら、:eqref:`eq_nabla_use` の具体例として次が得られるはずである。

$$
f(\epsilon_1, \log(2) + \epsilon_2) \approx \log(3) + \frac{1}{3}\epsilon_1 + \frac{2}{3}\epsilon_2.
$$

これをコードで確かめて、近似がどれくらい良いか見てみよう。

```{.python .input}
#@tab mxnet
%matplotlib inline
from d2l import mxnet as d2l
from IPython import display
from mpl_toolkits import mplot3d
from mxnet import autograd, np, npx
npx.set_np()

def f(x, y):
    return np.log(np.exp(x) + np.exp(y))
def grad_f(x, y):
    return np.array([np.exp(x) / (np.exp(x) + np.exp(y)),
                     np.exp(y) / (np.exp(x) + np.exp(y))])

epsilon = np.array([0.01, -0.03])
grad_approx = f(0, np.log(2)) + epsilon.dot(grad_f(0, np.log(2)))
true_value = f(0 + epsilon[0], np.log(2) + epsilon[1])
f'approximation: {grad_approx}, true Value: {true_value}'
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
from IPython import display
from mpl_toolkits import mplot3d
import torch
import numpy as np

def f(x, y):
    return torch.log(torch.exp(x) + torch.exp(y))
def grad_f(x, y):
    return torch.tensor([torch.exp(x) / (torch.exp(x) + torch.exp(y)),
                     torch.exp(y) / (torch.exp(x) + torch.exp(y))])

epsilon = torch.tensor([0.01, -0.03])
grad_approx = f(torch.tensor([0.]), torch.log(
    torch.tensor([2.]))) + epsilon.dot(
    grad_f(torch.tensor([0.]), torch.log(torch.tensor(2.))))
true_value = f(torch.tensor([0.]) + epsilon[0], torch.log(
    torch.tensor([2.])) + epsilon[1])
f'approximation: {grad_approx}, true Value: {true_value}'
```

```{.python .input}
#@tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
from IPython import display
from mpl_toolkits import mplot3d
import tensorflow as tf
import numpy as np

def f(x, y):
    return tf.math.log(tf.exp(x) + tf.exp(y))
def grad_f(x, y):
    return tf.constant([(tf.exp(x) / (tf.exp(x) + tf.exp(y))).numpy(),
                        (tf.exp(y) / (tf.exp(x) + tf.exp(y))).numpy()])

epsilon = tf.constant([0.01, -0.03])
grad_approx = f(tf.constant([0.]), tf.math.log(
    tf.constant([2.]))) + tf.tensordot(
    epsilon, grad_f(tf.constant([0.]), tf.math.log(tf.constant(2.))), axes=1)
true_value = f(tf.constant([0.]) + epsilon[0], tf.math.log(
    tf.constant([2.])) + epsilon[1])
f'approximation: {grad_approx}, true Value: {true_value}'
```

## 勾配と勾配降下法の幾何学
:eqref:`eq_nabla_use` の式をもう一度考えましょう。

$$
L(\mathbf{w} + \boldsymbol{\epsilon}) \approx L(\mathbf{w}) + \boldsymbol{\epsilon}\cdot \nabla_{\mathbf{w}} L(\mathbf{w}).
$$

これを使って損失 $L$ を最小化したいとする。  まず、 :numref:`sec_autograd` で最初に説明した勾配降下法を幾何学的に理解しよう。  やることは次の通りである。

1. 初期パラメータ $\mathbf{w}$ をランダムに選ぶ。
2. $\mathbf{w}$ において $L$ を最も急速に減少させる方向 $\mathbf{v}$ を見つける。
3. その方向に少し進む: $\mathbf{w} \rightarrow \mathbf{w} + \epsilon\mathbf{v}$。
4. 繰り返す。

正確なやり方が分からないのは、第2段階でベクトル $\mathbf{v}$ をどう計算するかだけである。  このような方向を *最急降下方向* と呼ぶ。  :numref:`sec_geometry-linear-algebraic-ops` での内積の幾何学的理解を使うと、:eqref:`eq_nabla_use` は次のように書き換えられる。

$$
L(\mathbf{w} + \mathbf{v}) \approx L(\mathbf{w}) + \mathbf{v}\cdot \nabla_{\mathbf{w}} L(\mathbf{w}) = L(\mathbf{w}) + \|\nabla_{\mathbf{w}} L(\mathbf{w})\|\cos(\theta).
$$

ここでは、便宜上、方向ベクトルの長さを1にしてあり、$\theta$ は $\mathbf{v}$ と $\nabla_{\mathbf{w}} L(\mathbf{w})$ のなす角である。  $L$ をできるだけ急速に減少させる方向を見つけたいなら、この式をできるだけ負にしたいわけである。  この式に方向の選び方が入ってくるのは $\cos(\theta)$ を通してだけなので、この余弦をできるだけ負にしたいことになる。  ここで余弦の形を思い出すと、$\cos(\theta) = -1$ にすれば最も負にでき、言い換えれば勾配と選んだ方向のなす角を $\pi$ ラジアン、つまり $180$ 度にすればよいことが分かる。  これを達成する唯一の方法は、まったく逆向きに進むことである。  つまり、$\mathbf{v}$ を $\nabla_{\mathbf{w}} L(\mathbf{w})$ と正反対の方向に向ければよいのです！

ここで、機械学習における最も重要な数学的概念の1つにたどり着きる。最急降下方向は $-\nabla_{\mathbf{w}}L(\mathbf{w})$ の方向を向きる。  したがって、先ほどの直感的なアルゴリズムは次のように書き換えられる。

1. 初期パラメータ $\mathbf{w}$ をランダムに選ぶ。
2. $\nabla_{\mathbf{w}} L(\mathbf{w})$ を計算する。
3. その反対方向に少し進む: $\mathbf{w} \leftarrow \mathbf{w} - \epsilon\nabla_{\mathbf{w}} L(\mathbf{w})$。
4. 繰り返す。


この基本アルゴリズムは多くの研究者によってさまざまに修正・拡張されてきましたが、核となる考え方はどれも同じである。  勾配を使って損失をできるだけ急速に減少させる方向を見つけ、その方向に一歩進むようにパラメータを更新するのである。

## 数学的最適化についての注意
本書を通して、私たちは数値最適化手法に焦点を当てる。深層学習で出会う関数はどれも複雑すぎて、明示的に最小化することができないからである。

しかし、上で得た幾何学的理解が、関数を直接最適化することについて何を教えてくれるのかを考えるのは有益である。

ある関数 $L(\mathbf{x})$ を最小にする $\mathbf{x}_0$ を見つけたいとする。  さらに、誰かがある値を与えて、それが $L$ を最小にする値だと言ったとしよう。  その答えがもっともらしいかどうかを確認する方法はあるでしょうか？

もう一度 :eqref:`eq_nabla_use` を考える。
$$
L(\mathbf{x}_0 + \boldsymbol{\epsilon}) \approx L(\mathbf{x}_0) + \boldsymbol{\epsilon}\cdot \nabla_{\mathbf{x}} L(\mathbf{x}_0).
$$

もし勾配がゼロでなければ、$-\epsilon \nabla_{\mathbf{x}} L(\mathbf{x}_0)$ の方向に一歩進むことで、より小さい $L$ の値を見つけられることが分かる。  したがって、本当に最小値にいるなら、これは起こりえません！  ここから、$\mathbf{x}_0$ が最小値なら $\nabla_{\mathbf{x}} L(\mathbf{x}_0) = 0$ でなければならないと結論できる。  $\nabla_{\mathbf{x}} L(\mathbf{x}_0) = 0$ を満たす点を *臨界点* と呼ぶ。

これは便利である。というのも、まれな場合ではあるが、勾配がゼロになる点をすべて明示的に見つけ、その中で最小の値を与えるものを探せることがあるからである。

具体例として、次の関数を考える。
$$
f(x) = 3x^4 - 4x^3 -12x^2.
$$

この関数の導関数は
$$
\frac{df}{dx} = 12x^3 - 12x^2 -24x = 12x(x-2)(x+1).
$$

最小値の候補となる場所は $x = -1, 0, 2$ だけであり、それぞれの関数値は $-5,0, -32$ である。したがって、$x = 2$ のときにこの関数が最小になると結論できる。  簡単なプロットでこれを確認できる。

```{.python .input}
#@tab mxnet
x = np.arange(-2, 3, 0.01)
f = (3 * x**4) - (4 * x**3) - (12 * x**2)

d2l.plot(x, f, 'x', 'f(x)')
```

```{.python .input}
#@tab pytorch
x = torch.arange(-2, 3, 0.01)
f = (3 * x**4) - (4 * x**3) - (12 * x**2)

d2l.plot(x, f, 'x', 'f(x)')
```

```{.python .input}
#@tab tensorflow
x = tf.range(-2, 3, 0.01)
f = (3 * x**4) - (4 * x**3) - (12 * x**2)

d2l.plot(x, f, 'x', 'f(x)')
```

これは、理論的に扱う場合でも数値的に扱う場合でも知っておくべき重要な事実を示している。最小化（あるいは最大化）できる可能性がある点は勾配がゼロである必要があるが、勾配がゼロの点がすべて真の *大域的* 最小値（あるいは最大値）とは限りない。

## 多変数の連鎖律
4変数 ($w, x, y, z$) の関数があり、それを多くの項を合成して作れるとしよう。

$$\begin{aligned}f(u, v) & = (u+v)^{2} \\u(a, b) & = (a+b)^{2}, \qquad v(a, b) = (a-b)^{2}, \\a(w, x, y, z) & = (w+x+y+z)^{2}, \qquad b(w, x, y, z) = (w+x-y-z)^2.\end{aligned}$$
:eqlabel:`eq_multi_func_def`

このような連鎖した方程式はニューラルネットワークを扱うときによく現れるので、そのような関数の勾配をどう計算するかを理解することが重要である。  どの変数がどの変数に直接関係しているかを見ると、 :numref:`fig_chain-1` にこの関係の視覚的な手がかりが見えてきる。

![上の関数関係。ノードは値を表し、辺は関数依存を示す。](../img/chain-net1.svg)
:label:`fig_chain-1`

:eqref:`eq_multi_func_def` のすべてをそのまま合成して

$$
f(w, x, y, z) = \left(\left((w+x+y+z)^2+(w+x-y-z)^2\right)^2+\left((w+x+y+z)^2-(w+x-y-z)^2\right)^2\right)^2.
$$

と書くこともできる。

その後、単変数の導関数だけを使って微分することもできるが、そうするとすぐに項が大量に出てきて、その多くが重複していることに気づくでしょう！  実際、たとえば次のようになる。

$$
\begin{aligned}
\frac{\partial f}{\partial w} & = 2 \left(2 \left(2 (w + x + y + z) - 2 (w + x - y - z)\right) \left((w + x + y + z)^{2}- (w + x - y - z)^{2}\right) + \right.\\
& \left. \quad 2 \left(2 (w + x - y - z) + 2 (w + x + y + z)\right) \left((w + x - y - z)^{2}+ (w + x + y + z)^{2}\right)\right) \times \\
& \quad \left(\left((w + x + y + z)^{2}- (w + x - y - z)^2\right)^{2}+ \left((w + x - y - z)^{2}+ (w + x + y + z)^{2}\right)^{2}\right).
\end{aligned}
$$

もし $\frac{\partial f}{\partial x}$ も計算したいなら、同様の式がまた現れ、多くの重複項、しかも2つの導関数の間で *共有された* 重複項がたくさん出てきる。  これは膨大な無駄な計算を意味し、もしこのようなやり方で微分を計算しなければならなかったら、深層学習革命は始まる前に止まっていたでしょう！


問題を分解しよう。  まず、$a$ を少し変えたときに $f$ がどう変わるかを理解することから始める。つまり、$w, x, y, z$ は存在しないものとして考える。  勾配を初めて扱ったときと同じように考えましょう。  $a$ に小さな量 $\epsilon$ を加えてみる。

$$
\begin{aligned}
& f(u(a+\epsilon, b), v(a+\epsilon, b)) \\
\approx & f\left(u(a, b) + \epsilon\frac{\partial u}{\partial a}(a, b), v(a, b) + \epsilon\frac{\partial v}{\partial a}(a, b)\right) \\
\approx & f(u(a, b), v(a, b)) + \epsilon\left[\frac{\partial f}{\partial u}(u(a, b), v(a, b))\frac{\partial u}{\partial a}(a, b) + \frac{\partial f}{\partial v}(u(a, b), v(a, b))\frac{\partial v}{\partial a}(a, b)\right].
\end{aligned}
$$

1行目は偏微分の定義から、2行目は勾配の定義から従う。  $\frac{\partial f}{\partial u}(u(a, b), v(a, b))$ のように、各導関数をどこで評価するかを正確に追うのは記法上やや煩雑なので、しばしば次のように簡潔に書く。

$$
\frac{\partial f}{\partial a} = \frac{\partial f}{\partial u}\frac{\partial u}{\partial a}+\frac{\partial f}{\partial v}\frac{\partial v}{\partial a}.
$$

この過程の意味を考えると理解しやすくなる。  私たちは $f(u(a, b), v(a, b))$ のような関数が、$a$ の変化によってどのように値を変えるかを理解しようとしている。  これが起こる経路は2つあり、$a \rightarrow u \rightarrow f$ の経路と $a \rightarrow v \rightarrow f$ の経路である。  これら2つの寄与は連鎖律によってそれぞれ $\frac{\partial w}{\partial u} \cdot \frac{\partial u}{\partial x}$ と $\frac{\partial w}{\partial v} \cdot \frac{\partial v}{\partial x}$ として計算でき、それらを足し合わせる。

:numref:`fig_chain-2` に示すように、右側の関数が左側でつながっているものに依存する、別の関数ネットワークがあると想像する。

![連鎖律のもう少し繊細な例。](../img/chain-net2.svg)
:label:`fig_chain-2`

$\frac{\partial f}{\partial y}$ のようなものを計算するには、$y$ から $f$ へのすべての経路（この場合は3本）について和をとる必要があり、次のようになる。

$$
\frac{\partial f}{\partial y} = \frac{\partial f}{\partial a} \frac{\partial a}{\partial u} \frac{\partial u}{\partial y} + \frac{\partial f}{\partial u} \frac{\partial u}{\partial y} + \frac{\partial f}{\partial b} \frac{\partial b}{\partial v} \frac{\partial v}{\partial y}.
$$

このように連鎖律を理解しておくことは、勾配がネットワークを通ってどのように流れるか、そして LSTM (:numref:`sec_lstm`) や残差層 (:numref:`sec_resnet`) のようなさまざまなアーキテクチャ上の選択が、勾配の流れを制御することで学習過程にどのように役立つかを理解するうえで大きな助けになる。

## 逆伝播アルゴリズム

前節の :eqref:`eq_multi_func_def` の例に戻りましょう。そこでは

$$
\begin{aligned}
f(u, v) & = (u+v)^{2} \\
u(a, b) & = (a+b)^{2}, \qquad v(a, b) = (a-b)^{2}, \\
a(w, x, y, z) & = (w+x+y+z)^{2}, \qquad b(w, x, y, z) = (w+x-y-z)^2.
\end{aligned}
$$

たとえば $\frac{\partial f}{\partial w}$ を計算したいなら、多変数の連鎖律を適用して次のようにできる。

$$
\begin{aligned}
\frac{\partial f}{\partial w} & = \frac{\partial f}{\partial u}\frac{\partial u}{\partial w} + \frac{\partial f}{\partial v}\frac{\partial v}{\partial w}, \\
\frac{\partial u}{\partial w} & = \frac{\partial u}{\partial a}\frac{\partial a}{\partial w}+\frac{\partial u}{\partial b}\frac{\partial b}{\partial w}, \\
\frac{\partial v}{\partial w} & = \frac{\partial v}{\partial a}\frac{\partial a}{\partial w}+\frac{\partial v}{\partial b}\frac{\partial b}{\partial w}.
\end{aligned}
$$

この分解を使って $\frac{\partial f}{\partial w}$ を計算してみよう。  ここで必要なのは、さまざまな1段階の偏微分だけだと分かる。

$$
\begin{aligned}
\frac{\partial f}{\partial u} = 2(u+v), & \quad\frac{\partial f}{\partial v} = 2(u+v), \\
\frac{\partial u}{\partial a} = 2(a+b), & \quad\frac{\partial u}{\partial b} = 2(a+b), \\
\frac{\partial v}{\partial a} = 2(a-b), & \quad\frac{\partial v}{\partial b} = -2(a-b), \\
\frac{\partial a}{\partial w} = 2(w+x+y+z), & \quad\frac{\partial b}{\partial w} = 2(w+x-y-z).
\end{aligned}
$$

これをコードに書くと、かなり扱いやすい式になる。

```{.python .input}
#@tab all
# Compute the value of the function from inputs to outputs
w, x, y, z = -1, 0, -2, 1
a, b = (w + x + y + z)**2, (w + x - y - z)**2
u, v = (a + b)**2, (a - b)**2
f = (u + v)**2
print(f'    f at {w}, {x}, {y}, {z} is {f}')

# Compute the single step partials
df_du, df_dv = 2*(u + v), 2*(u + v)
du_da, du_db, dv_da, dv_db = 2*(a + b), 2*(a + b), 2*(a - b), -2*(a - b)
da_dw, db_dw = 2*(w + x + y + z), 2*(w + x - y - z)

# Compute the final result from inputs to outputs
du_dw, dv_dw = du_da*da_dw + du_db*db_dw, dv_da*da_dw + dv_db*db_dw
df_dw = df_du*du_dw + df_dv*dv_dw
print(f'df/dw at {w}, {x}, {y}, {z} is {df_dw}')
```

しかし、これでも $\frac{\partial f}{\partial x}$ のようなものを簡単に計算できるわけではない。  その理由は、連鎖律を *どのように* 適用するかを選んだからである。  上で行ったことを見ると、可能な限り常に分母に $\partial w$ を残していた。  このようにして、$w$ が他のすべての変数をどう変えるかを見る形で連鎖律を適用したのである。  もしそれが目的なら、それでよいでしょう。  しかし、深層学習での動機を思い出する。私たちは各パラメータが *損失* をどう変えるかを見たいのである。  本質的には、可能な限り常に分子に $\partial f$ を残す形で連鎖律を適用したいのです！

より明示的に言うと、次のように書ける。

$$
\begin{aligned}
\frac{\partial f}{\partial w} & = \frac{\partial f}{\partial a}\frac{\partial a}{\partial w} + \frac{\partial f}{\partial b}\frac{\partial b}{\partial w}, \\
\frac{\partial f}{\partial a} & = \frac{\partial f}{\partial u}\frac{\partial u}{\partial a}+\frac{\partial f}{\partial v}\frac{\partial v}{\partial a}, \\
\frac{\partial f}{\partial b} & = \frac{\partial f}{\partial u}\frac{\partial u}{\partial b}+\frac{\partial f}{\partial v}\frac{\partial v}{\partial b}.
\end{aligned}
$$

この連鎖律の適用では、$\frac{\partial f}{\partial u}, \frac{\partial f}{\partial v}, \frac{\partial f}{\partial a}, \frac{\partial f}{\partial b}, \; \textrm{and} \; \frac{\partial f}{\partial w}$ を明示的に計算することになる。  さらに次の式を加えても何の問題もない。

$$
\begin{aligned}
\frac{\partial f}{\partial x} & = \frac{\partial f}{\partial a}\frac{\partial a}{\partial x} + \frac{\partial f}{\partial b}\frac{\partial b}{\partial x}, \\
\frac{\partial f}{\partial y} & = \frac{\partial f}{\partial a}\frac{\partial a}{\partial y}+\frac{\partial f}{\partial b}\frac{\partial b}{\partial y}, \\
\frac{\partial f}{\partial z} & = \frac{\partial f}{\partial a}\frac{\partial a}{\partial z}+\frac{\partial f}{\partial b}\frac{\partial b}{\partial z}.
\end{aligned}
$$

そして、ネットワーク全体のどのノードを変えたときにも $f$ がどう変わるかを追跡できる。  実装してみよう。

```{.python .input}
#@tab all
# Compute the value of the function from inputs to outputs
w, x, y, z = -1, 0, -2, 1
a, b = (w + x + y + z)**2, (w + x - y - z)**2
u, v = (a + b)**2, (a - b)**2
f = (u + v)**2
print(f'f at {w}, {x}, {y}, {z} is {f}')

# Compute the derivative using the decomposition above
# First compute the single step partials
df_du, df_dv = 2*(u + v), 2*(u + v)
du_da, du_db, dv_da, dv_db = 2*(a + b), 2*(a + b), 2*(a - b), -2*(a - b)
da_dw, db_dw = 2*(w + x + y + z), 2*(w + x - y - z)
da_dx, db_dx = 2*(w + x + y + z), 2*(w + x - y - z)
da_dy, db_dy = 2*(w + x + y + z), -2*(w + x - y - z)
da_dz, db_dz = 2*(w + x + y + z), -2*(w + x - y - z)

# Now compute how f changes when we change any value from output to input
df_da, df_db = df_du*du_da + df_dv*dv_da, df_du*du_db + df_dv*dv_db
df_dw, df_dx = df_da*da_dw + df_db*db_dw, df_da*da_dx + df_db*db_dx
df_dy, df_dz = df_da*da_dy + df_db*db_dy, df_da*da_dz + df_db*db_dz

print(f'df/dw at {w}, {x}, {y}, {z} is {df_dw}')
print(f'df/dx at {w}, {x}, {y}, {z} is {df_dx}')
print(f'df/dy at {w}, {x}, {y}, {z} is {df_dy}')
print(f'df/dz at {w}, {x}, {y}, {z} is {df_dz}')
```

入力から出力へではなく、出力から入力へ向かって導関数を計算していること（上の最初のコード片で行ったように）が、このアルゴリズムに *逆伝播* という名前を与えている。  ここには2つの段階がある。
1. 関数の値と1段階の偏微分を前から後ろへ計算する。  上では分けて書いていませんが、これは1つの *順伝播* にまとめられる。
2. $f$ の勾配を後ろから前へ計算する。  これを *逆伝播* と呼ぶ。

これは、ネットワーク内のすべての重みに対する損失の勾配を1回の計算で求められるようにするために、あらゆる深層学習アルゴリズムが実装しているものそのものである。  このような分解があるというのは驚くべき事実である。

これがどのようにまとめられるかを見るために、この例を少し見てみよう。

```{.python .input}
#@tab mxnet
# Initialize as ndarrays, then attach gradients
w, x, y, z = np.array(-1), np.array(0), np.array(-2), np.array(1)

w.attach_grad()
x.attach_grad()
y.attach_grad()
z.attach_grad()

# Do the computation like usual, tracking gradients
with autograd.record():
    a, b = (w + x + y + z)**2, (w + x - y - z)**2
    u, v = (a + b)**2, (a - b)**2
    f = (u + v)**2

# Execute backward pass
f.backward()

print(f'df/dw at {w}, {x}, {y}, {z} is {w.grad}')
print(f'df/dx at {w}, {x}, {y}, {z} is {x.grad}')
print(f'df/dy at {w}, {x}, {y}, {z} is {y.grad}')
print(f'df/dz at {w}, {x}, {y}, {z} is {z.grad}')
```

```{.python .input}
#@tab pytorch
# Initialize as ndarrays, then attach gradients
w = torch.tensor([-1.], requires_grad=True)
x = torch.tensor([0.], requires_grad=True)
y = torch.tensor([-2.], requires_grad=True)
z = torch.tensor([1.], requires_grad=True)
# Do the computation like usual, tracking gradients
a, b = (w + x + y + z)**2, (w + x - y - z)**2
u, v = (a + b)**2, (a - b)**2
f = (u + v)**2

# Execute backward pass
f.backward()

print(f'df/dw at {w.data.item()}, {x.data.item()}, {y.data.item()}, '
      f'{z.data.item()} is {w.grad.data.item()}')
print(f'df/dx at {w.data.item()}, {x.data.item()}, {y.data.item()}, '
      f'{z.data.item()} is {x.grad.data.item()}')
print(f'df/dy at {w.data.item()}, {x.data.item()}, {y.data.item()}, '
      f'{z.data.item()} is {y.grad.data.item()}')
print(f'df/dz at {w.data.item()}, {x.data.item()}, {y.data.item()}, '
      f'{z.data.item()} is {z.grad.data.item()}')
```

```{.python .input}
#@tab tensorflow
# Initialize as ndarrays, then attach gradients
w = tf.Variable(tf.constant([-1.]))
x = tf.Variable(tf.constant([0.]))
y = tf.Variable(tf.constant([-2.]))
z = tf.Variable(tf.constant([1.]))
# Do the computation like usual, tracking gradients
with tf.GradientTape(persistent=True) as t:
    a, b = (w + x + y + z)**2, (w + x - y - z)**2
    u, v = (a + b)**2, (a - b)**2
    f = (u + v)**2

# Execute backward pass
w_grad = t.gradient(f, w).numpy()
x_grad = t.gradient(f, x).numpy()
y_grad = t.gradient(f, y).numpy()
z_grad = t.gradient(f, z).numpy()

print(f'df/dw at {w.numpy()}, {x.numpy()}, {y.numpy()}, '
      f'{z.numpy()} is {w_grad}')
print(f'df/dx at {w.numpy()}, {x.numpy()}, {y.numpy()}, '
      f'{z.numpy()} is {x_grad}')
print(f'df/dy at {w.numpy()}, {x.numpy()}, {y.numpy()}, '
      f'{z.numpy()} is {y_grad}')
print(f'df/dz at {w.numpy()}, {x.numpy()}, {y.numpy()}, '
      f'{z.numpy()} is {z_grad}')
```

上で行ったことはすべて、`f.backwards()` を呼び出すだけで自動的に実行できる。


## ヘッシアン
単変数微積分と同様に、勾配だけを使うよりも関数をよりよく近似するために、高階導関数を考えることは有用である。

複数変数の関数の高階導関数を扱うとき、すぐに直面する問題が1つある。それは、その数が非常に多いことである。  $n$ 変数の関数 $f(x_1, \ldots, x_n)$ があるとき、2階導関数は $n^{2}$ 個取れる。すなわち、$i$ と $j$ の任意の組に対して

$$
\frac{d^2f}{dx_idx_j} = \frac{d}{dx_i}\left(\frac{d}{dx_j}f\right).
$$

これは伝統的に *ヘッシアン* と呼ばれる行列にまとめられる。

$$\mathbf{H}_f = \begin{bmatrix} \frac{d^2f}{dx_1dx_1} & \cdots & \frac{d^2f}{dx_1dx_n} \\ \vdots & \ddots & \vdots \\ \frac{d^2f}{dx_ndx_1} & \cdots & \frac{d^2f}{dx_ndx_n} \\ \end{bmatrix}.$$
:eqlabel:`eq_hess_def`

この行列の各要素が独立というわけではない。  実際、*混合偏導関数*（複数の変数に関する偏導関数）が存在し連続である限り、任意の $i$ と $j$ について

$$
\frac{d^2f}{dx_idx_j} = \frac{d^2f}{dx_jdx_i}.
$$

これは、まず $x_i$ の方向に関数を摂動し、その後 $x_j$ で摂動した結果と、先に $x_j$ で摂動し、その後 $x_i$ で摂動した結果を比較することで従う。どちらの順序でも、$f$ の出力に対する最終的な変化は同じになるからである。

単変数の場合と同様に、これらの導関数を使うと、ある点の近くで関数がどう振る舞うかをはるかによく把握できる。  特に、単変数で見たように、点 $\mathbf{x}_0$ の近くで最もよく当てはまる2次式を見つけるのに使える。

例を見てみよう。  $f(x_1, x_2) = a + b_1x_1 + b_2x_2 + c_{11}x_1^{2} + c_{12}x_1x_2 + c_{22}x_2^{2}$ とする。  これは2変数の2次式の一般形である。  関数値、その勾配、そして :eqref:`eq_hess_def` のヘッシアンを、すべて原点で見てみると、

$$
\begin{aligned}
f(0,0) & = a, \\
\nabla f (0,0) & = \begin{bmatrix}b_1 \\ b_2\end{bmatrix}, \\
\mathbf{H} f (0,0) & = \begin{bmatrix}2 c_{11} & c_{12} \\ c_{12} & 2c_{22}\end{bmatrix},
\end{aligned}
$$

元の多項式は次のようにして復元できる。

$$
f(\mathbf{x}) = f(0) + \nabla f (0) \cdot \mathbf{x} + \frac{1}{2}\mathbf{x}^\top \mathbf{H} f (0) \mathbf{x}.
$$

一般に、任意の点 $\mathbf{x}_0$ でこの展開を計算すると、

$$
f(\mathbf{x}) = f(\mathbf{x}_0) + \nabla f (\mathbf{x}_0) \cdot (\mathbf{x}-\mathbf{x}_0) + \frac{1}{2}(\mathbf{x}-\mathbf{x}_0)^\top \mathbf{H} f (\mathbf{x}_0) (\mathbf{x}-\mathbf{x}_0).
$$

これは任意次元の入力に対して成り立ち、ある点での関数の最良の2次近似を与える。  例として、次の関数をプロットしてみよう。

$$
f(x, y) = xe^{-x^2-y^2}.
$$

勾配とヘッシアンは
$$
\nabla f(x, y) = e^{-x^2-y^2}\begin{pmatrix}1-2x^2 \\ -2xy\end{pmatrix} \; \textrm{and} \; \mathbf{H}f(x, y) = e^{-x^2-y^2}\begin{pmatrix} 4x^3 - 6x & 4x^2y - 2y \\ 4x^2y-2y &4xy^2-2x\end{pmatrix}.
$$

したがって、少し計算すると、$[-1,0]^\top$ における近似2次式は

$$
f(x, y) \approx e^{-1}\left(-1 - (x+1) +(x+1)^2+y^2\right).
$$

```{.python .input}
#@tab mxnet
# Construct grid and compute function
x, y = np.meshgrid(np.linspace(-2, 2, 101),
                   np.linspace(-2, 2, 101), indexing='ij')
z = x*np.exp(- x**2 - y**2)

# Compute approximating quadratic with gradient and Hessian at (1, 0)
w = np.exp(-1)*(-1 - (x + 1) + (x + 1)**2 + y**2)

# Plot function
ax = d2l.plt.figure().add_subplot(111, projection='3d')
ax.plot_wireframe(x.asnumpy(), y.asnumpy(), z.asnumpy(),
                  **{'rstride': 10, 'cstride': 10})
ax.plot_wireframe(x.asnumpy(), y.asnumpy(), w.asnumpy(),
                  **{'rstride': 10, 'cstride': 10}, color='purple')
d2l.plt.xlabel('x')
d2l.plt.ylabel('y')
d2l.set_figsize()
ax.set_xlim(-2, 2)
ax.set_ylim(-2, 2)
ax.set_zlim(-1, 1)
ax.dist = 12
```

```{.python .input}
#@tab pytorch
# Construct grid and compute function
x, y = torch.meshgrid(torch.linspace(-2, 2, 101),
                   torch.linspace(-2, 2, 101))

z = x*torch.exp(- x**2 - y**2)

# Compute approximating quadratic with gradient and Hessian at (1, 0)
w = torch.exp(torch.tensor([-1.]))*(-1 - (x + 1) + 2 * (x + 1)**2 + 2 * y**2)

# Plot function
ax = d2l.plt.figure().add_subplot(111, projection='3d')
ax.plot_wireframe(x.numpy(), y.numpy(), z.numpy(),
                  **{'rstride': 10, 'cstride': 10})
ax.plot_wireframe(x.numpy(), y.numpy(), w.numpy(),
                  **{'rstride': 10, 'cstride': 10}, color='purple')
d2l.plt.xlabel('x')
d2l.plt.ylabel('y')
d2l.set_figsize()
ax.set_xlim(-2, 2)
ax.set_ylim(-2, 2)
ax.set_zlim(-1, 1)
ax.dist = 12
```

```{.python .input}
#@tab tensorflow
# Construct grid and compute function
x, y = tf.meshgrid(tf.linspace(-2., 2., 101),
                   tf.linspace(-2., 2., 101))

z = x*tf.exp(- x**2 - y**2)

# Compute approximating quadratic with gradient and Hessian at (1, 0)
w = tf.exp(tf.constant([-1.]))*(-1 - (x + 1) + 2 * (x + 1)**2 + 2 * y**2)

# Plot function
ax = d2l.plt.figure().add_subplot(111, projection='3d')
ax.plot_wireframe(x.numpy(), y.numpy(), z.numpy(),
                  **{'rstride': 10, 'cstride': 10})
ax.plot_wireframe(x.numpy(), y.numpy(), w.numpy(),
                  **{'rstride': 10, 'cstride': 10}, color='purple')
d2l.plt.xlabel('x')
d2l.plt.ylabel('y')
d2l.set_figsize()
ax.set_xlim(-2, 2)
ax.set_ylim(-2, 2)
ax.set_zlim(-1, 1)
ax.dist = 12
```

これは :numref:`sec_gd` で議論するニュートン法の基礎を成する。そこでは、最もよく当てはまる2次式を反復的に求め、その2次式を正確に最小化する。

## 少しの行列微積分
行列を含む関数の導関数は、実は特に扱いやすいことが分かる。  この節は記法が重くなるので、最初の読みでは飛ばしても構いませんが、行列を含む関数の導関数が、特に深層学習で中心的な役割を果たす行列演算を考えると、最初に思うよりずっとすっきりしていることを知っておくのは有益である。

まず例から始めましょう。  固定された列ベクトル $\boldsymbol{\beta}$ があり、積の関数 $f(\mathbf{x}) = \boldsymbol{\beta}^\top\mathbf{x}$ を考えて、$\mathbf{x}$ を変えたときに内積がどう変わるかを理解したいとする。

機械学習で行列の導関数を扱うときに役立つ記法として、*分母配置の行列微分* と呼ばれるものがある。これは、微分の分母にあるベクトル、行列、あるいはテンソルの形に合わせて偏微分を並べるものである。  この場合、次のように書く。

$$
\frac{df}{d\mathbf{x}} = \begin{bmatrix}
\frac{df}{dx_1} \\
\vdots \\
\frac{df}{dx_n}
\end{bmatrix},
$$

ここでは列ベクトル $\mathbf{x}$ の形に合わせている。

関数を成分ごとに書くと、

$$
f(\mathbf{x}) = \sum_{i = 1}^{n} \beta_ix_i = \beta_1x_1 + \cdots + \beta_nx_n.
$$

ここで、たとえば $\beta_1$ に関する偏微分をとると、最初の項だけが $\beta_1$ に $x_1$ を掛けたものなので、それ以外はすべてゼロである。したがって

$$
\frac{df}{dx_1} = \beta_1,
$$

より一般には

$$
\frac{df}{dx_i} = \beta_i.
$$

これを行列としてまとめ直すと、

$$
\frac{df}{d\mathbf{x}} = \begin{bmatrix}
\frac{df}{dx_1} \\
\vdots \\
\frac{df}{dx_n}
\end{bmatrix} = \begin{bmatrix}
\beta_1 \\
\vdots \\
\beta_n
\end{bmatrix} = \boldsymbol{\beta}.
$$

これは、この節で何度も出てくる行列微積分のいくつかの特徴を示している。

* まず、計算はかなり複雑になる。
* 次に、最終結果は途中の過程よりずっとすっきりしており、常に単変数の場合と似た形になる。  この場合、$\frac{d}{dx}(bx) = b$ と $\frac{d}{d\mathbf{x}} (\boldsymbol{\beta}^\top\mathbf{x}) = \boldsymbol{\beta}$ が似ていることに注意する。
* 第3に、転置がどこからともなく現れることがよくある。  その根本的な理由は、分母の形に合わせるという約束にある。したがって行列を掛け合わせるときには、元の項の形に戻すために転置を取る必要があるのである。

直感をさらに深めるために、もう少し難しい計算をしてみよう。  列ベクトル $\mathbf{x}$ と正方行列 $A$ があり、次を計算したいとする。

$$\frac{d}{d\mathbf{x}}(\mathbf{x}^\top A \mathbf{x}).$$
:eqlabel:`eq_mat_goal_1`

扱いやすい記法にするため、アインシュタイン記法を使ってこの問題を考えましょう。  この場合、関数は

$$
\mathbf{x}^\top A \mathbf{x} = x_ia_{ij}x_j.
$$

と書ける。

導関数を計算するには、各 $k$ について

$$
\frac{d}{dx_k}(\mathbf{x}^\top A \mathbf{x}) = \frac{d}{dx_k}x_ia_{ij}x_j.
$$

の値を理解する必要がある。

積の法則より、これは

$$
\frac{d}{dx_k}x_ia_{ij}x_j = \frac{dx_i}{dx_k}a_{ij}x_j + x_ia_{ij}\frac{dx_j}{dx_k}.
$$

となる。  $\frac{dx_i}{dx_k}$ のような項は、$i=k$ のとき1、それ以外では0であることは難しくない。  つまり、$i$ と $k$ が異なる項はこの和から消え、最初の和に残るのは $i=k$ のものだけである。  同じ理屈が第2項にも当てはまり、そこでは $j=k$ が必要である。  これにより

$$
\frac{d}{dx_k}x_ia_{ij}x_j = a_{kj}x_j + x_ia_{ik}.
$$

となる。

ここで、アインシュタイン記法における添字の名前は任意である。$i$ と $j$ が異なるという事実は、この計算では本質的ではないので、添字を付け替えて両方とも $i$ を使うようにすると、

$$
\frac{d}{dx_k}x_ia_{ij}x_j = a_{ki}x_i + x_ia_{ik} = (a_{ki} + a_{ik})x_i.
$$

となる。

ここから先に進むには少し練習が必要である。  この結果を行列演算の形で表してみよう。  $a_{ki} + a_{ik}$ は $\mathbf{A} + \mathbf{A}^\top$ の $k, i$ 成分である。  したがって

$$
\frac{d}{dx_k}x_ia_{ij}x_j = [\mathbf{A} + \mathbf{A}^\top]_{ki}x_i.
$$

となる。

同様に、この項は行列 $\mathbf{A} + \mathbf{A}^\top$ とベクトル $\mathbf{x}$ の積なので、

$$
\left[\frac{d}{d\mathbf{x}}(\mathbf{x}^\top A \mathbf{x})\right]_k = \frac{d}{dx_k}x_ia_{ij}x_j = [(\mathbf{A} + \mathbf{A}^\top)\mathbf{x}]_k.
$$

と分かる。

したがって、:eqref:`eq_mat_goal_1` で求めた導関数の $k$ 番目の成分は、右辺ベクトルの $k$ 番目の成分と同じであり、両者は等しいことが分かる。  よって

$$
\frac{d}{d\mathbf{x}}(\mathbf{x}^\top A \mathbf{x}) = (\mathbf{A} + \mathbf{A}^\top)\mathbf{x}.
$$

これは前の例よりかなり多くの作業を要しましたが、最終結果は短いものである。  さらに、通常の単変数微分について次を考えてみてください。

$$
\frac{d}{dx}(xax) = \frac{dx}{dx}ax + xa\frac{dx}{dx} = (a+a)x.
$$

同様に $\frac{d}{dx}(ax^2) = 2ax = (a+a)x$ である。  やはり、単変数の場合に似た結果が得られますが、そこに転置が1つ加わっている。

ここまで来ると、このパターンはかなり怪しく見えてくるはずである。なぜそうなるのか考えてみよう。  このような行列微分をするとき、まず得られる式は別の行列式だと仮定してみる。つまり、行列とその転置の積や和で書ける式である。  そのような式が存在するなら、すべての行列に対して成り立たなければならない。  特に、$1 \times 1$ 行列に対しても成り立つ必要がある。その場合、行列積は単なる数の積、行列和は単なる和であり、転置は何の役にも立ちません！  つまり、どんな式であれ、*必ず* 単変数の式と一致しなければならないのである。  したがって、少し練習すれば、対応する単変数の式がどうなるべきかを知っているだけで、行列の導関数をしばしば推測できるのです！

これを試してみよう。  $\mathbf{X}$ が $n \times m$ 行列、$\mathbf{U}$ が $n \times r$、$\mathbf{V}$ が $r \times m$ だとする。  次を計算してみよう。

$$\frac{d}{d\mathbf{V}} \|\mathbf{X} - \mathbf{U}\mathbf{V}\|_2^{2} = \;?$$
:eqlabel:`eq_mat_goal_2`

この計算は行列分解と呼ばれる分野で重要である。  しかしここでは、単に計算すべき導関数にすぎない。  これが $1\times1$ 行列ならどうなるか想像してみよう。  その場合、次の式になる。

$$
\frac{d}{dv} (x-uv)^{2}= -2(x-uv)u,
$$

これはごく標準的な導関数である。  これを行列式に戻そうとすると、

$$
\frac{d}{d\mathbf{V}} \|\mathbf{X} - \mathbf{U}\mathbf{V}\|_2^{2}= -2(\mathbf{X} - \mathbf{U}\mathbf{V})\mathbf{U}.
$$

となる。

しかし、これを見ると少しおかしいことに気づく。  $\mathbf{X}$ は $n \times m$ であり、$\mathbf{U}\mathbf{V}$ も同じなので、行列 $2(\mathbf{X} - \mathbf{U}\mathbf{V})$ は $n \times m$ である。  一方、$\mathbf{U}$ は $n \times r$ であり、$n \times m$ 行列と $n \times r$ 行列は次元が合わないので掛けられません！

私たちが求めたいのは $\frac{d}{d\mathbf{V}}$ であり、これは $\mathbf{V}$ と同じ形、つまり $r \times m$ である。  したがって、どうにかして $n \times m$ 行列と $n \times r$ 行列を掛け合わせて（おそらく転置を使って）$r \times m$ を得る必要がある。  これは $U^\top$ を $(\mathbf{X} - \mathbf{U}\mathbf{V})$ に掛けることで実現できる。  したがって、:eqref:`eq_mat_goal_2` の解は

$$
\frac{d}{d\mathbf{V}} \|\mathbf{X} - \mathbf{U}\mathbf{V}\|_2^{2}= -2\mathbf{U}^\top(\mathbf{X} - \mathbf{U}\mathbf{V}).
$$

だと推測できる。

これが正しいことを示すには、詳細な計算を省くわけにはいきない。  この経験則が正しいとすでに信じているなら、この導出は飛ばして構いない。  次を計算するには

$$
\frac{d}{d\mathbf{V}} \|\mathbf{X} - \mathbf{U}\mathbf{V}\|_2^2,
$$

各 $a$ と $b$ について

$$
\frac{d}{dv_{ab}} \|\mathbf{X} - \mathbf{U}\mathbf{V}\|_2^{2}= \frac{d}{dv_{ab}} \sum_{i, j}\left(x_{ij} - \sum_k u_{ik}v_{kj}\right)^2.
$$

を求めなければならない。

$\frac{d}{dv_{ab}}$ に関しては $\mathbf{X}$ と $\mathbf{U}$ のすべての要素が定数であることを思い出すと、微分を和の中に押し込み、2乗に連鎖律を適用して

$$
\frac{d}{dv_{ab}} \|\mathbf{X} - \mathbf{U}\mathbf{V}\|_2^{2}= \sum_{i, j}2\left(x_{ij} - \sum_k u_{ik}v_{kj}\right)\left(-\sum_k u_{ik}\frac{dv_{kj}}{dv_{ab}} \right).
$$

となる。

前の導出と同様に、$\frac{dv_{kj}}{dv_{ab}}$ は $k=a$ かつ $j=b$ のときにのみ非ゼロであることに注意できる。  どちらか一方でも満たさなければ、その項はゼロなので自由に捨てられる。  すると

$$
\frac{d}{dv_{ab}} \|\mathbf{X} - \mathbf{U}\mathbf{V}\|_2^{2}= -2\sum_{i}\left(x_{ib} - \sum_k u_{ik}v_{kb}\right)u_{ia}.
$$

となる。

ここで重要な微妙さは、$k=a$ という条件は内側の和の中には現れないということである。なぜなら、その $k$ は内側の項の中で和をとっているダミー変数だからである。  記法をよりきれいにした例として、なぜ

$$
\frac{d}{dx_1} \left(\sum_i x_i \right)^{2}= 2\left(\sum_i x_i \right)
$$

となるのかを考えてみてください。

ここから、和の成分を識別し始められる。  まず、

$$
\sum_k u_{ik}v_{kb} = [\mathbf{U}\mathbf{V}]_{ib}.
$$

したがって、和の中の内側の式全体は

$$
x_{ib} - \sum_k u_{ik}v_{kb} = [\mathbf{X}-\mathbf{U}\mathbf{V}]_{ib}.
$$

となる。

これにより、導関数を次のように書ける。

$$
\frac{d}{dv_{ab}} \|\mathbf{X} - \mathbf{U}\mathbf{V}\|_2^{2}= -2\sum_{i}[\mathbf{X}-\mathbf{U}\mathbf{V}]_{ib}u_{ia}.
$$

これを行列の $a, b$ 成分の形にしたいので、前の例と同じ技法を使って行列式にまとめるには、$u_{ia}$ の添字の順序を入れ替える必要がある。  $u_{ia} = [\mathbf{U}^\top]_{ai}$ に気づけば、

$$
\frac{d}{dv_{ab}} \|\mathbf{X} - \mathbf{U}\mathbf{V}\|_2^{2}= -2\sum_{i} [\mathbf{U}^\top]_{ai}[\mathbf{X}-\mathbf{U}\mathbf{V}]_{ib}.
$$

と書ける。

これは行列積なので、次のように結論できる。

$$
\frac{d}{dv_{ab}} \|\mathbf{X} - \mathbf{U}\mathbf{V}\|_2^{2}= -2[\mathbf{U}^\top(\mathbf{X}-\mathbf{U}\mathbf{V})]_{ab}.
$$

したがって、:eqref:`eq_mat_goal_2` の解は

$$
\frac{d}{d\mathbf{V}} \|\mathbf{X} - \mathbf{U}\mathbf{V}\|_2^{2}= -2\mathbf{U}^\top(\mathbf{X} - \mathbf{U}\mathbf{V}).
$$

と書ける。

これは上で推測した解と一致しています！

ここで、「これまで学んだ微積分の規則をすべてそのまま行列版で書けないのだろうか？  まだ機械的なことは明らかなのだから、さっさと済ませればよいのでは？」と思うのはもっともである。  実際、そのような規則はあり、 :cite:`Petersen.Pedersen.ea.2008` に優れた要約がある。  しかし、単一の値に比べて行列演算の組み合わせ方が非常に多いため、行列の導関数の規則は単変数の場合よりはるかに多くなる。  多くの場合、添字を使って扱うか、適切な場合には自動微分に任せるのが最善である。

## まとめ

* 高次元では、1次元の導関数と同じ役割を果たす勾配を定義できる。  これにより、入力に任意の小さな変化を与えたときに多変数関数がどう変わるかを調べられる。
* 逆伝播アルゴリズムは、多変数の連鎖律を整理して、多くの偏微分を効率よく計算する方法とみなせる。
* 行列微積分を使うと、行列式の導関数を簡潔に書ける。

## 演習
1. 列ベクトル $\boldsymbol{\beta}$ が与えられたとき、$f(\mathbf{x}) = \boldsymbol{\beta}^\top\mathbf{x}$ と $g(\mathbf{x}) = \mathbf{x}^\top\boldsymbol{\beta}$ の両方の導関数を求めよ。  なぜ同じ答えになるのか。
2. $\mathbf{v}$ を $n$ 次元ベクトルとする。$\frac{\partial}{\partial\mathbf{v}}\|\mathbf{v}\|_2$ は何か。
3. $L(x, y) = \log(e^x + e^y)$ とする。  勾配を求めよ。  勾配の成分の和はいくつか。
4. $f(x, y) = x^2y + xy^2$ とする。臨界点が $(0,0)$ のみであることを示せ。  また、$f(x, x)$ を考えることで、$(0,0)$ が最大値か最小値か、それともどちらでもないかを判定せよ。
5. $f(\mathbf{x}) = g(\mathbf{x}) + h(\mathbf{x})$ を最小化しているとする。  $\nabla f = 0$ の条件を $g$ と $h$ の観点から幾何学的にどう解釈できるか。


:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/413)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1090)
:end_tab:


:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/1091)
:end_tab:\n