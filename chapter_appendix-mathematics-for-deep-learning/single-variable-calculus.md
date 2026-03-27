# 単変数微積分
:label:`sec_single_variable_calculus`

:numref:`sec_calculus` では、微分積分学の基本要素を見た。この節では、微積分の基礎と、それを機械学習の文脈でどのように理解し適用できるかを、さらに深く掘り下げる。

## 微分積分学
微分積分学は本質的に、関数が小さな変化の下でどのように振る舞うかを研究する学問である。これが深層学習にとってなぜ核心的なのかを見るために、例を考えてみよう。

便利のため、重みが単一のベクトル $\mathbf{w} = (w_1, \ldots, w_n)$ に連結されている深層ニューラルネットワークがあるとする。訓練データセットが与えられたとき、このデータセット上でのニューラルネットワークの損失を考える。これを $\mathcal{L}(\mathbf{w})$ と書くことにする。

この関数は非常に複雑で、与えられたアーキテクチャのすべての可能なモデルがこのデータセット上でどの程度性能を発揮するかを符号化しているため、どの重みの集合 $\mathbf{w}$ が損失を最小化するのかを見極めるのはほとんど不可能である。したがって実際には、重みを *ランダムに* 初期化し、その後、損失をできるだけ速く減少させる方向へ小さな一歩を反復的に進めることがよくある。

すると問題は、一見するとそれほど簡単ではないものになる。すなわち、重みを最も速く減少させる方向をどう見つけるか、である。これを掘り下げるために、まず単一の重みだけの場合、すなわち単一の実数値 $x$ に対して $L(\mathbf{w}) = L(x)$ となる場合を考えよう。

$x$ を取り、それを小さく $x + \epsilon$ に変えたときに何が起こるかを理解してみよう。具体的な値を思い浮かべたいなら、$\epsilon = 0.0000001$ のような数を考えてみる。何が起こるかを視覚化するために、例として関数 $f(x) = \sin(x^x)$ を $[0, 3]$ 上で描いてみよう。

```{.python .input}
#@tab mxnet
%matplotlib inline
from d2l import mxnet as d2l
from IPython import display
from mxnet import np, npx
npx.set_np()

# Plot a function in a normal range
x_big = np.arange(0.01, 3.01, 0.01)
ys = np.sin(x_big**x_big)
d2l.plot(x_big, ys, 'x', 'f(x)')
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
from IPython import display
import torch
torch.pi = torch.acos(torch.zeros(1)).item() * 2  # Define pi in torch

# Plot a function in a normal range
x_big = torch.arange(0.01, 3.01, 0.01)
ys = torch.sin(x_big**x_big)
d2l.plot(x_big, ys, 'x', 'f(x)')
```

```{.python .input}
#@tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
from IPython import display
import tensorflow as tf
tf.pi = tf.acos(tf.zeros(1)).numpy() * 2  # Define pi in TensorFlow

# Plot a function in a normal range
x_big = tf.range(0.01, 3.01, 0.01)
ys = tf.sin(x_big**x_big)
d2l.plot(x_big, ys, 'x', 'f(x)')
```

この大きなスケールでは、関数の振る舞いは単純ではない。しかし、範囲を $[1.75,2.25]$ のようにもっと小さくすると、グラフがずっと単純になることがわかる。

```{.python .input}
#@tab mxnet
# Plot a the same function in a tiny range
x_med = np.arange(1.75, 2.25, 0.001)
ys = np.sin(x_med**x_med)
d2l.plot(x_med, ys, 'x', 'f(x)')
```

```{.python .input}
#@tab pytorch
# Plot a the same function in a tiny range
x_med = torch.arange(1.75, 2.25, 0.001)
ys = torch.sin(x_med**x_med)
d2l.plot(x_med, ys, 'x', 'f(x)')
```

```{.python .input}
#@tab tensorflow
# Plot a the same function in a tiny range
x_med = tf.range(1.75, 2.25, 0.001)
ys = tf.sin(x_med**x_med)
d2l.plot(x_med, ys, 'x', 'f(x)')
```

さらに極端に、非常に小さな区間まで拡大すると、振る舞いはさらに単純になり、ただの直線になる。

```{.python .input}
#@tab mxnet
# Plot a the same function in a tiny range
x_small = np.arange(2.0, 2.01, 0.0001)
ys = np.sin(x_small**x_small)
d2l.plot(x_small, ys, 'x', 'f(x)')
```

```{.python .input}
#@tab pytorch
# Plot a the same function in a tiny range
x_small = torch.arange(2.0, 2.01, 0.0001)
ys = torch.sin(x_small**x_small)
d2l.plot(x_small, ys, 'x', 'f(x)')
```

```{.python .input}
#@tab tensorflow
# Plot a the same function in a tiny range
x_small = tf.range(2.0, 2.01, 0.0001)
ys = tf.sin(x_small**x_small)
d2l.plot(x_small, ys, 'x', 'f(x)')
```

これが単変数微積分の重要な観察である。よく知られた関数の振る舞いは、十分に小さな範囲では直線で近似できるのである。これは、ほとんどの関数について、$x$ の値を少し動かすと出力 $f(x)$ も少し動くと期待するのが妥当であることを意味する。答えるべき唯一の問いは、「出力の変化は入力の変化に比べてどれくらい大きいのか。半分なのか、2倍なのか？」である。

したがって、関数の入力を小さく変えたときの出力の変化量の比を考えることができる。これを形式的には次のように書ける。

$$
\frac{L(x+\epsilon) - L(x)}{(x+\epsilon) - x} = \frac{L(x+\epsilon) - L(x)}{\epsilon}.
$$

これは、コードで試してみるにはもう十分である。たとえば、$L(x) = x^{2} + 1701(x-4)^3$ だとわかっているとしよう。このとき、$x = 4$ におけるこの値の大きさを次のように調べられる。

```{.python .input}
#@tab all
# Define our function
def L(x):
    return x**2 + 1701*(x-4)**3

# Print the difference divided by epsilon for several epsilon
for epsilon in [0.1, 0.001, 0.0001, 0.00001]:
    print(f'epsilon = {epsilon:.5f} -> {(L(4+epsilon) - L(4)) / epsilon:.5f}')
```

ここで注意深く見ると、この数の出力が怪しいほど $8$ に近いことに気づくだろう。実際、$\epsilon$ を小さくすると、値は次第に $8$ に近づいていきる。したがって、求める値（入力の変化が出力をどれだけ変えるかの度合い）は、$x=4$ において $8$ であると正しく結論できる。数学者はこの事実を次のように表する。

$$
\lim_{\epsilon \rightarrow 0}\frac{L(4+\epsilon) - L(4)}{\epsilon} = 8.
$$

少し歴史的な寄り道をすると、ニューラルネットワーク研究の最初の数十年間、科学者たちはこのアルゴリズム（*有限差分法*）を使って、損失関数が小さな摂動の下でどう変化するかを評価していた。つまり、重みを変えて損失がどう変わるかを見るのである。これは計算効率が悪く、1つの変数の1つの変化が損失にどう影響するかを見るために損失関数を2回評価する必要がある。これを数千個程度のパラメータに対して行おうとすると、データセット全体に対するネットワークの評価が数千回も必要になる。1986年になってようやく、:citet:`Rumelhart.Hinton.Williams.ea.1988` で導入された *誤差逆伝播アルゴリズム* により、重みの *任意の* 変化をまとめて考えたときに損失がどう変わるかを、データセットに対するネットワークの1回の予測と同じ計算時間で求める方法が与えられた。

先ほどの例に戻ると、この値 $8$ は $x$ の値によって異なるので、$x$ の関数として定義するのが自然である。より形式的には、この値に依存する変化率を *導関数* と呼び、次のように書く。

$$\frac{df}{dx}(x) = \lim_{\epsilon \rightarrow 0}\frac{f(x+\epsilon) - f(x)}{\epsilon}.$$
:eqlabel:`eq_der_def`

文献によって、導関数の表記はさまざまである。たとえば、以下の表記はすべて同じものを表している。

$$
\frac{df}{dx} = \frac{d}{dx}f = f' = \nabla_xf = D_xf = f_x.
$$

多くの著者は1つの表記を選んで使い続けるが、それすら必ずしも一貫しているとは限りない。これらすべてに慣れておくのがよいだろう。本書では、複雑な式の導関数を取りたい場合を除いて、$\frac{df}{dx}$ という表記を使う。その場合には、次のような式を書くために $\frac{d}{dx}f$ を使う。
$$
\frac{d}{dx}\left[x^4+\cos\left(\frac{x^2+1}{2x-1}\right)\right].
$$

しばしば、導関数の定義 :eqref:`eq_der_def` をもう一度ほどいて、$x$ を少し変えたときに関数がどう変わるかを直感的に見ると役立つ。

$$\begin{aligned} \frac{df}{dx}(x) = \lim_{\epsilon \rightarrow 0}\frac{f(x+\epsilon) - f(x)}{\epsilon} & \implies \frac{df}{dx}(x) \approx \frac{f(x+\epsilon) - f(x)}{\epsilon} \\ & \implies \epsilon \frac{df}{dx}(x) \approx f(x+\epsilon) - f(x) \\ & \implies f(x+\epsilon) \approx f(x) + \epsilon \frac{df}{dx}(x). \end{aligned}$$
:eqlabel:`eq_small_change`

最後の式は明示的に強調する価値がある。これは、任意の関数について、入力を少し変えると、出力はその小さな変化量に導関数を掛けた分だけ変わることを示している。

このように、導関数は、入力の変化に対して出力がどれだけ変化するかを教えてくれるスケーリング係数として理解できる。

## 微積分の規則
:label:`sec_derivative_table`

ここからは、明示的な関数の導関数をどう計算するかを理解する課題に移る。微積分を完全に厳密に扱うなら、すべてを第一原理から導くことになる。ここではその誘惑に乗るのではなく、よく出てくる規則を理解することにする。

### よく使う導関数
:numref:`sec_calculus` で見たように、導関数を計算するときには、しばしば一連の規則を使って計算をいくつかの基本関数に還元できる。参照しやすいように、ここで繰り返しておきる。

* **定数の導関数。** $\frac{d}{dx}c = 0$。
* **線形関数の導関数。** $\frac{d}{dx}(ax) = a$。
* **べき乗則。** $\frac{d}{dx}x^n = nx^{n-1}$。
* **指数関数の導関数。** $\frac{d}{dx}e^x = e^x$。
* **対数関数の導関数。** $\frac{d}{dx}\log(x) = \frac{1}{x}$。

### 導関数の規則
必要な導関数をすべて個別に計算して表に保存しなければならないとしたら、微分積分学はほとんど不可能だろう。上の導関数を一般化し、たとえば $f(x) = \log\left(1+(x-1)^{10}\right)$ の導関数を求めるような、より複雑な導関数を計算できるのは数学の恩恵である。 :numref:`sec_calculus` で述べたように、その鍵は、関数をさまざまな方法で組み合わせたときに何が起こるか、特に和、積、合成を規則化することである。

* **和の法則。** $\frac{d}{dx}\left(g(x) + h(x)\right) = \frac{dg}{dx}(x) + \frac{dh}{dx}(x)$。
* **積の法則。** $\frac{d}{dx}\left(g(x)\cdot h(x)\right) = g(x)\frac{dh}{dx}(x) + \frac{dg}{dx}(x)h(x)$。
* **連鎖律。** $\frac{d}{dx}g(h(x)) = \frac{dg}{dh}(h(x))\cdot \frac{dh}{dx}(x)$。

:eqref:`eq_small_change` を使って、これらの規則をどう理解できるか見てみよう。和の法則については、次の推論を考える。

$$
\begin{aligned}
f(x+\epsilon) & = g(x+\epsilon) + h(x+\epsilon) \\
& \approx g(x) + \epsilon \frac{dg}{dx}(x) + h(x) + \epsilon \frac{dh}{dx}(x) \\
& = g(x) + h(x) + \epsilon\left(\frac{dg}{dx}(x) + \frac{dh}{dx}(x)\right) \\
& = f(x) + \epsilon\left(\frac{dg}{dx}(x) + \frac{dh}{dx}(x)\right).
\end{aligned}
$$

この結果を、$f(x+\epsilon) \approx f(x) + \epsilon \frac{df}{dx}(x)$ という事実と比べると、望みどおり $\frac{df}{dx}(x) = \frac{dg}{dx}(x) + \frac{dh}{dx}(x)$ であることがわかる。ここでの直感は、入力 $x$ を変えると、$g$ と $h$ がそれぞれ $\frac{dg}{dx}(x)$ と $\frac{dh}{dx}(x)$ だけ出力の変化に寄与する、というものである。


積はより微妙で、これらの式を扱うための新しい観察が必要になる。まずはこれまでと同様に :eqref:`eq_small_change` を使って始めよう。

$$
\begin{aligned}
f(x+\epsilon) & = g(x+\epsilon)\cdot h(x+\epsilon) \\
& \approx \left(g(x) + \epsilon \frac{dg}{dx}(x)\right)\cdot\left(h(x) + \epsilon \frac{dh}{dx}(x)\right) \\
& = g(x)\cdot h(x) + \epsilon\left(g(x)\frac{dh}{dx}(x) + \frac{dg}{dx}(x)h(x)\right) + \epsilon^2\frac{dg}{dx}(x)\frac{dh}{dx}(x) \\
& = f(x) + \epsilon\left(g(x)\frac{dh}{dx}(x) + \frac{dg}{dx}(x)h(x)\right) + \epsilon^2\frac{dg}{dx}(x)\frac{dh}{dx}(x). \\
\end{aligned}
$$


これは上で行った計算に似ているし、実際、答え（$\frac{df}{dx}(x) = g(x)\frac{dh}{dx}(x) + \frac{dg}{dx}(x)h(x)$）が $\epsilon$ の隣に現れている。しかし、$\epsilon^{2}$ の大きさを持つ項があるのが問題である。これを *高次項* と呼ぶ。なぜなら、$\epsilon^2$ のべきは $\epsilon^1$ のべきより高いからである。後の節で、これらを追跡したい場合があることを見るが、今は $\epsilon = 0.0000001$ なら $\epsilon^{2}= 0.0000000000001$ であり、はるかに小さいことに注意する。$\epsilon \rightarrow 0$ とすると、高次項は安全に無視できる。この付録では一般的な慣習として、2つの項が高次項を除いて等しいことを示すのに "$\approx$" を使う。ただし、より厳密にしたいなら、差分商を調べればよいだろう。

$$
\frac{f(x+\epsilon) - f(x)}{\epsilon} = g(x)\frac{dh}{dx}(x) + \frac{dg}{dx}(x)h(x) + \epsilon \frac{dg}{dx}(x)\frac{dh}{dx}(x),
$$

そして $\epsilon \rightarrow 0$ とすると、右辺の最後の項も 0 に収束することがわかる。

最後に、連鎖律についても、再び :eqref:`eq_small_change` を使って進めると、

$$
\begin{aligned}
f(x+\epsilon) & = g(h(x+\epsilon)) \\
& \approx g\left(h(x) + \epsilon \frac{dh}{dx}(x)\right) \\
& \approx g(h(x)) + \epsilon \frac{dh}{dx}(x) \frac{dg}{dh}(h(x))\\
& = f(x) + \epsilon \frac{dg}{dh}(h(x))\frac{dh}{dx}(x),
\end{aligned}
$$

となる。ここで2行目では、関数 $g$ の入力が（$h(x)$ から）微小量 $\epsilon \frac{dh}{dx}(x)$ だけずれたものとして見ている。

これらの規則により、実質的にどんな式でも計算できる柔軟な道具立てが得られる。たとえば、

$$
\begin{aligned}
\frac{d}{dx}\left[\log\left(1+(x-1)^{10}\right)\right] & = \left(1+(x-1)^{10}\right)^{-1}\frac{d}{dx}\left[1+(x-1)^{10}\right]\\
& = \left(1+(x-1)^{10}\right)^{-1}\left(\frac{d}{dx}[1] + \frac{d}{dx}[(x-1)^{10}]\right) \\
& = \left(1+(x-1)^{10}\right)^{-1}\left(0 + 10(x-1)^9\frac{d}{dx}[x-1]\right) \\
& = 10\left(1+(x-1)^{10}\right)^{-1}(x-1)^9 \\
& = \frac{10(x-1)^9}{1+(x-1)^{10}}.
\end{aligned}
$$

各行では次の規則を使っている。

1. 連鎖律と対数関数の導関数。
2. 和の法則。
3. 定数の導関数、連鎖律、べき乗則。
4. 和の法則、線形関数の導関数、定数の導関数。

この例を通して、2つのことが明らかになるはずである。

1. 和、積、定数、べき乗、指数関数、対数関数を使って書ける任意の関数は、これらの規則に従うことで機械的に導関数を計算できる。
2. 人間がこれらの規則に従うのは面倒で、誤りも起こりやすい！

ありがたいことに、この2つの事実を合わせると、先へ進む道が見えてきる。これは機械化のための完璧な候補である。実際、この節の後半で再び取り上げる誤差逆伝播は、まさにそれなのである。

### 線形近似
導関数を扱うとき、上で使った近似を幾何学的に解釈するとしばしば有用である。特に、次の式に注意する。

$$
f(x+\epsilon) \approx f(x) + \epsilon \frac{df}{dx}(x),
$$

これは、点 $(x, f(x))$ を通り、傾きが $\frac{df}{dx}(x)$ の直線で $f$ の値を近似している。このように、導関数は関数 $f$ の線形近似を与えると言える。以下にその例を示す。

```{.python .input}
#@tab mxnet
# Compute sin
xs = np.arange(-np.pi, np.pi, 0.01)
plots = [np.sin(xs)]

# Compute some linear approximations. Use d(sin(x)) / dx = cos(x)
for x0 in [-1.5, 0, 2]:
    plots.append(np.sin(x0) + (xs - x0) * np.cos(x0))

d2l.plot(xs, plots, 'x', 'f(x)', ylim=[-1.5, 1.5])
```

```{.python .input}
#@tab pytorch
# Compute sin
xs = torch.arange(-torch.pi, torch.pi, 0.01)
plots = [torch.sin(xs)]

# Compute some linear approximations. Use d(sin(x))/dx = cos(x)
for x0 in [-1.5, 0.0, 2.0]:
    plots.append(torch.sin(torch.tensor(x0)) + (xs - x0) *
                 torch.cos(torch.tensor(x0)))

d2l.plot(xs, plots, 'x', 'f(x)', ylim=[-1.5, 1.5])
```

```{.python .input}
#@tab tensorflow
# Compute sin
xs = tf.range(-tf.pi, tf.pi, 0.01)
plots = [tf.sin(xs)]

# Compute some linear approximations. Use d(sin(x))/dx = cos(x)
for x0 in [-1.5, 0.0, 2.0]:
    plots.append(tf.sin(tf.constant(x0)) + (xs - x0) *
                 tf.cos(tf.constant(x0)))

d2l.plot(xs, plots, 'x', 'f(x)', ylim=[-1.5, 1.5])
```

### 高階導関数

ここで、表面的には奇妙に見えるかもしれないことをしてみよう。関数 $f$ を取り、その導関数 $\frac{df}{dx}$ を計算する。これにより、任意の点での $f$ の変化率が得られる。

しかし、導関数 $\frac{df}{dx}$ 自体も関数と見なせるので、$\frac{df}{dx}$ の導関数を計算して $\frac{d^2f}{dx^2} = \frac{df}{dx}\left(\frac{df}{dx}\right)$ を得ることを妨げるものはない。これを $f$ の2階導関数と呼ぶ。この関数は、$f$ の変化率の変化率、言い換えれば、変化率がどのように変化しているかを表する。導関数は何回でも適用でき、$n$ 階導関数と呼ばれるものが得られる。記法をすっきりさせるため、$n$ 階導関数を次のように表する。

$$
f^{(n)}(x) = \frac{d^{n}f}{dx^{n}} = \left(\frac{d}{dx}\right)^{n} f.
$$

これがなぜ有用な概念なのかを理解してみよう。以下では、$f^{(2)}(x)$、$f^{(1)}(x)$、および $f(x)$ を可視化する。

まず、2階導関数 $f^{(2)}(x)$ が正の定数である場合を考える。これは、1階導関数の傾きが正であることを意味する。その結果、1階導関数 $f^{(1)}(x)$ は最初は負で、ある点で0になり、最後には正になる。これは元の関数 $f$ の傾きを示しているので、関数 $f$ 自体は減少し、平らになり、その後増加する。言い換えれば、関数 $f$ は上に曲がっており、 :numref:`fig_positive-second` に示すように1つの最小値を持つ。

![2階導関数が正の定数だと仮定すると、1階導関数は増加し、それにより関数自体は最小値を持つ。](../img/posSecDer.svg)
:label:`fig_positive-second`


次に、2階導関数が負の定数である場合、それは1階導関数が減少していることを意味する。これは、1階導関数が最初は正で、ある点で0になり、その後負になることを示す。したがって、関数 $f$ 自体は増加し、平らになり、その後減少する。言い換えれば、関数 $f$ は下に曲がっており、 :numref:`fig_negative-second` に示すように1つの最大値を持つ。

![2階導関数が負の定数だと仮定すると、1階導関数は減少し、それにより関数自体は最大値を持つ。](../img/negSecDer.svg)
:label:`fig_negative-second`


第三に、2階導関数が常に0であれば、1階導関数は決して変化せず、一定です！ これは、$f$ が一定の速度で増加（または減少）し、$f$ 自体が :numref:`fig_zero-second` に示すように直線であることを意味する。

![2階導関数が0だと仮定すると、1階導関数は一定であり、それにより関数自体は直線になる。](../img/zeroSecDer.svg)
:label:`fig_zero-second`

要するに、2階導関数は関数 $f$ の曲がり方を表していると解釈できる。正の2階導関数は上向きの曲がりをもたらし、負の2階導関数は下向きの曲がりを意味し、0の2階導関数は関数 $f$ がまったく曲がっていないことを意味する。

これをもう一歩進めよう。関数 $g(x) = ax^{2}+ bx + c$ を考える。このとき、

$$
\begin{aligned}
\frac{dg}{dx}(x) & = 2ax + b \\
\frac{d^2g}{dx^2}(x) & = 2a.
\end{aligned}
$$

もし元の関数 $f(x)$ があるなら、最初の2つの導関数を計算し、それらに一致するような $a, b, c$ の値を見つけることができる。前の節で、1階導関数が直線による最良近似を与えることを見たが、同様に、この構成は2次式による最良近似を与える。これを $f(x) = \sin(x)$ で可視化してみよう。

```{.python .input}
#@tab mxnet
# Compute sin
xs = np.arange(-np.pi, np.pi, 0.01)
plots = [np.sin(xs)]

# Compute some quadratic approximations. Use d(sin(x)) / dx = cos(x)
for x0 in [-1.5, 0, 2]:
    plots.append(np.sin(x0) + (xs - x0) * np.cos(x0) -
                              (xs - x0)**2 * np.sin(x0) / 2)

d2l.plot(xs, plots, 'x', 'f(x)', ylim=[-1.5, 1.5])
```

```{.python .input}
#@tab pytorch
# Compute sin
xs = torch.arange(-torch.pi, torch.pi, 0.01)
plots = [torch.sin(xs)]

# Compute some quadratic approximations. Use d(sin(x)) / dx = cos(x)
for x0 in [-1.5, 0.0, 2.0]:
    plots.append(torch.sin(torch.tensor(x0)) + (xs - x0) *
                 torch.cos(torch.tensor(x0)) - (xs - x0)**2 *
                 torch.sin(torch.tensor(x0)) / 2)

d2l.plot(xs, plots, 'x', 'f(x)', ylim=[-1.5, 1.5])
```

```{.python .input}
#@tab tensorflow
# Compute sin
xs = tf.range(-tf.pi, tf.pi, 0.01)
plots = [tf.sin(xs)]

# Compute some quadratic approximations. Use d(sin(x)) / dx = cos(x)
for x0 in [-1.5, 0.0, 2.0]:
    plots.append(tf.sin(tf.constant(x0)) + (xs - x0) *
                 tf.cos(tf.constant(x0)) - (xs - x0)**2 *
                 tf.sin(tf.constant(x0)) / 2)

d2l.plot(xs, plots, 'x', 'f(x)', ylim=[-1.5, 1.5])
```

この考え方は次の節で *テイラー級数* に拡張する。

### テイラー級数


*テイラー級数* は、点 $x_0$ における最初の $n$ 個の導関数の値、すなわち $\left\{ f(x_0), f^{(1)}(x_0), f^{(2)}(x_0), \ldots, f^{(n)}(x_0) \right\}$ が与えられたときに、関数 $f(x)$ を近似する方法を与える。考え方は、$x_0$ において与えられたすべての導関数に一致する $n$ 次多項式を見つけることである。

前節で $n=2$ の場合を見たが、少し代数計算をすると、これは次のようになる。

$$
f(x) \approx \frac{1}{2}\frac{d^2f}{dx^2}(x_0)(x-x_0)^{2}+ \frac{df}{dx}(x_0)(x-x_0) + f(x_0).
$$

上で見たように、分母の $2$ は $x^2$ を2回微分したときに得られる $2$ を打ち消すためにあり、他の項はすべて0になる。同じ論理が1階導関数と関数値そのものにも当てはまる。

この論理を $n=3$ まで進めると、次を得る。

$$
f(x) \approx \frac{\frac{d^3f}{dx^3}(x_0)}{6}(x-x_0)^3 + \frac{\frac{d^2f}{dx^2}(x_0)}{2}(x-x_0)^{2}+ \frac{df}{dx}(x_0)(x-x_0) + f(x_0).
$$

ここで $6 = 3 \times 2 = 3!$ は、$x^3$ を3回微分したときに前に現れる定数に由来する。


さらに、次のようにして $n$ 次多項式を得ることができる。

$$
P_n(x) = \sum_{i = 0}^{n} \frac{f^{(i)}(x_0)}{i!}(x-x_0)^{i}.
$$

ここでの記法

$$
f^{(n)}(x) = \frac{d^{n}f}{dx^{n}} = \left(\frac{d}{dx}\right)^{n} f.
$$


実際、$P_n(x)$ は関数 $f(x)$ に対する最良の $n$ 次多項式近似と見なせる。

上の近似の誤差を最後まで掘り下げることはしないが、無限極限について触れておく価値はある。この場合、$\cos(x)$ や $e^{x}$ のような、うまく振る舞う関数（実解析関数として知られる）については、無限個の項を書き下して、まったく同じ関数を近似できる。

$$
f(x) = \sum_{n = 0}^\infty \frac{f^{(n)}(x_0)}{n!}(x-x_0)^{n}.
$$

例として $f(x) = e^{x}$ を考えよう。$e^{x}$ は自分自身の導関数なので、$f^{(n)}(x) = e^{x}$ である。したがって、$x_0 = 0$ でテイラー級数を取ることで、$e^{x}$ は次のように再構成できる。

$$
e^{x} = \sum_{n = 0}^\infty \frac{x^{n}}{n!} = 1 + x + \frac{x^2}{2} + \frac{x^3}{6} + \cdots.
$$

これがコードではどうなるかを見て、テイラー近似の次数を上げると目的の関数 $e^x$ にどんどん近づくことを観察しよう。

```{.python .input}
#@tab mxnet
# Compute the exponential function
xs = np.arange(0, 3, 0.01)
ys = np.exp(xs)

# Compute a few Taylor series approximations
P1 = 1 + xs
P2 = 1 + xs + xs**2 / 2
P5 = 1 + xs + xs**2 / 2 + xs**3 / 6 + xs**4 / 24 + xs**5 / 120

d2l.plot(xs, [ys, P1, P2, P5], 'x', 'f(x)', legend=[
    "Exponential", "Degree 1 Taylor Series", "Degree 2 Taylor Series",
    "Degree 5 Taylor Series"])
```

```{.python .input}
#@tab pytorch
# Compute the exponential function
xs = torch.arange(0, 3, 0.01)
ys = torch.exp(xs)

# Compute a few Taylor series approximations
P1 = 1 + xs
P2 = 1 + xs + xs**2 / 2
P5 = 1 + xs + xs**2 / 2 + xs**3 / 6 + xs**4 / 24 + xs**5 / 120

d2l.plot(xs, [ys, P1, P2, P5], 'x', 'f(x)', legend=[
    "Exponential", "Degree 1 Taylor Series", "Degree 2 Taylor Series",
    "Degree 5 Taylor Series"])
```

```{.python .input}
#@tab tensorflow
# Compute the exponential function
xs = tf.range(0, 3, 0.01)
ys = tf.exp(xs)

# Compute a few Taylor series approximations
P1 = 1 + xs
P2 = 1 + xs + xs**2 / 2
P5 = 1 + xs + xs**2 / 2 + xs**3 / 6 + xs**4 / 24 + xs**5 / 120

d2l.plot(xs, [ys, P1, P2, P5], 'x', 'f(x)', legend=[
    "Exponential", "Degree 1 Taylor Series", "Degree 2 Taylor Series",
    "Degree 5 Taylor Series"])
```

テイラー級数には主に2つの応用がある。

1. *理論的応用*: 複雑すぎる関数を理解しようとするとき、テイラー級数を使うと、それを直接扱える多項式に変換できる。

2. *数値的応用*: $e^{x}$ や $\cos(x)$ のような関数は、機械にとって計算が難しいことがある。機械は固定精度で値の表を保持できる（実際そうすることも多い） が、それでも「$\cos(1)$ の1000桁目は何か？」のような問いは残る。テイラー級数は、こうした問いに答えるのにしばしば役立つ。


## まとめ

* 導関数は、入力を少し変えたときに関数がどう変化するかを表すのに使える。
* 基本的な導関数は、導関数の規則を使って組み合わせることで、任意に複雑な導関数を作れる。
* 導関数は繰り返し適用でき、2階以上の導関数を得られる。次数が上がるほど、関数の振る舞いについてより細かな情報が得られる。
* 単一のデータ例における導関数の情報を使うと、テイラー級数から得られる多項式によって、うまく振る舞う関数を近似できる。


## 演習

1. $x^3-4x+1$ の導関数は何か？
2. $\log(\frac{1}{x})$ の導関数は何か？
3. 真か偽か: $f'(x) = 0$ ならば、$f$ は $x$ において最大値または最小値を持つか？
4. $x\ge0$ に対する $f(x) = x\log(x)$ の最小値はどこか（ここで $f(0)$ では $0$ の極限値を取ると仮定する）？
