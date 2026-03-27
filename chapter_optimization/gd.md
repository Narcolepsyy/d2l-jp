# 勾配降下法
:label:`sec_gd`

この節では、*勾配降下法*の基礎となる概念を紹介する。
深層学習で直接使われることはまれであるが、勾配降下法を理解することは確率的勾配降下法アルゴリズムを理解するうえで重要である。
たとえば、学習率が大きすぎるために最適化問題が発散することがある。この現象は勾配降下法でもすでに観察できる。同様に、前処理は勾配降下法でよく使われる手法であり、より高度なアルゴリズムにも引き継がれる。
まずは簡単な特殊ケースから始めよう。


## 1次元の勾配降下法

1次元の勾配降下法は、勾配降下アルゴリズムがなぜ目的関数の値を減少させうるのかを説明するのに最適な例である。ある連続微分可能な実数値関数 $f: \mathbb{R} \rightarrow \mathbb{R}$ を考える。テイラー展開を用いると

$$f(x + \epsilon) = f(x) + \epsilon f'(x) + \mathcal{O}(\epsilon^2).$$
:eqlabel:`gd-taylor`

すなわち、1次近似では $f(x+\epsilon)$ は点 $x$ における関数値 $f(x)$ と1階導関数 $f'(x)$ によって与えられる。小さい $\epsilon$ に対して、負の勾配方向へ進めば $f$ は減少すると考えるのは自然である。簡単のため、固定ステップサイズ $\eta > 0$ を選び、$\epsilon = -\eta f'(x)$ とする。これを上のテイラー展開に代入すると

$$f(x - \eta f'(x)) = f(x) - \eta f'^2(x) + \mathcal{O}(\eta^2 f'^2(x)).$$
:eqlabel:`gd-taylor-2`

導関数 $f'(x) \neq 0$ が消えないなら、$\eta f'^2(x)>0$ なので進展がある。さらに、$\eta$ を十分小さく選べば高次の項は無視できる。したがって

$$f(x - \eta f'(x)) \lessapprox f(x).$$

これは、$x$ を反復更新する際に

$$x \leftarrow x - \eta f'(x)$$

を用いれば、関数 $f(x)$ の値が下がる可能性があることを意味する。したがって、勾配降下法ではまず初期値 $x$ と定数 $\eta > 0$ を選び、それらを使って停止条件に達するまで $x$ を継続的に反復更新する。たとえば、勾配の大きさ $|f'(x)|$ が十分小さくなったとき、あるいは反復回数がある値に達したときである。

簡単のため、目的関数 $f(x)=x^2$ を選んで、勾配降下法の実装方法を示す。$x=0$ が $f(x)$ を最小化する解であることは分かっているが、それでもこの単純な関数を使って $x$ がどのように変化するかを観察する。

```{.python .input}
#@tab mxnet
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import np, npx
npx.set_np()
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
import numpy as np
import torch
```

```{.python .input}
#@tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
import numpy as np
import tensorflow as tf
```

```{.python .input}
#@tab all
def f(x):  # 目的関数
    return x ** 2

def f_grad(x):  # 目的関数の勾配（導関数）
    return 2 * x
```

次に、初期値として $x=10$ を用い、$\eta=0.2$ と仮定する。勾配降下法で $x$ を10回反復すると、最終的に $x$ の値が最適解に近づくことが分かる。

```{.python .input}
#@tab all
def gd(eta, f_grad):
    x = 10.0
    results = [x]
    for i in range(10):
        x -= eta * f_grad(x)
        results.append(float(x))
    print(f'epoch 10, x: {x:f}')
    return results

results = gd(0.2, f_grad)
```

$x$ に関する最適化の進行は次のように描画できる。

```{.python .input}
#@tab all
def show_trace(results, f):
    n = max(abs(min(results)), abs(max(results)))
    f_line = d2l.arange(-n, n, 0.01)
    d2l.set_figsize()
    d2l.plot([f_line, results], [[f(x) for x in f_line], [
        f(x) for x in results]], 'x', 'f(x)', fmts=['-', '-o'])

show_trace(results, f)
```

### 学習率
:label:`subsec_gd-learningrate`

学習率 $\eta$ はアルゴリズム設計者が設定できる。学習率が小さすぎると、$x$ の更新が非常に遅くなり、よりよい解を得るために多くの反復が必要になる。このような場合に何が起こるかを見るため、同じ最適化問題で $\eta = 0.05$ の進行を考える。見て分かるように、10ステップ後でもまだ最適解からかなり遠いままである。

```{.python .input}
#@tab all
show_trace(gd(0.05, f_grad), f)
```

逆に、学習率が大きすぎると、$\left|\eta f'(x)\right|$ が1次テイラー展開の範囲を超えて大きくなりすぎることがある。つまり、:eqref:`gd-taylor-2` の項 $\mathcal{O}(\eta^2 f'^2(x))$ が無視できなくなる可能性がある。この場合、$x$ の反復が $f(x)$ の値を下げられるとは保証できない。たとえば、学習率を $\eta=1.1$ にすると、$x$ は最適解 $x=0$ を飛び越えてしまい、徐々に発散する。

```{.python .input}
#@tab all
show_trace(gd(1.1, f_grad), f)
```

### 局所最小値

非凸関数で何が起こるかを示すため、定数 $c$ に対して $f(x) = x \cdot \cos(cx)$ を考える。この関数は無限個の局所最小値を持つ。学習率の選び方や問題の条件の良さによって、さまざまな解のうちのどれかに落ち着くことがある。以下の例は、（現実的ではない）高すぎる学習率が悪い局所最小値へ導くことを示している。

```{.python .input}
#@tab all
c = d2l.tensor(0.15 * np.pi)

def f(x):  # 目的関数
    return x * d2l.cos(c * x)

def f_grad(x):  # 目的関数の勾配
    return d2l.cos(c * x) - c * x * d2l.sin(c * x)

show_trace(gd(2, f_grad), f)
```

## 多変数の勾配降下法

1変数の場合の直感がつかめたので、今度は $\mathbf{x} = [x_1, x_2, \ldots, x_d]^\top$ の場合を考えよう。つまり、目的関数 $f: \mathbb{R}^d \to \mathbb{R}$ はベクトルをスカラーに写像する。それに対応して、勾配も多変数になる。これは $d$ 個の偏導関数からなるベクトルである。

$$\nabla f(\mathbf{x}) = \bigg[\frac{\partial f(\mathbf{x})}{\partial x_1}, \frac{\partial f(\mathbf{x})}{\partial x_2}, \ldots, \frac{\partial f(\mathbf{x})}{\partial x_d}\bigg]^\top.$$

勾配の各偏導関数成分 $\partial f(\mathbf{x})/\partial x_i$ は、入力 $x_i$ に関して点 $\mathbf{x}$ での $f$ の変化率を表す。1変数の場合と同様に、多変数関数に対する対応するテイラー近似を使えば、何をすべきかの手がかりが得られる。特に、

$$f(\mathbf{x} + \boldsymbol{\epsilon}) = f(\mathbf{x}) + \mathbf{\boldsymbol{\epsilon}}^\top \nabla f(\mathbf{x}) + \mathcal{O}(\|\boldsymbol{\epsilon}\|^2).$$
:eqlabel:`gd-multi-taylor`

言い換えると、$\boldsymbol{\epsilon}$ に関する2次の項までを見ると、最急降下方向は負の勾配 $-\nabla f(\mathbf{x})$ で与えられる。適切な学習率 $\eta > 0$ を選べば、典型的な勾配降下アルゴリズムが得られる。

$$\mathbf{x} \leftarrow \mathbf{x} - \eta \nabla f(\mathbf{x}).$$

実際にアルゴリズムがどう振る舞うかを見るため、入力として2次元ベクトル $\mathbf{x} = [x_1, x_2]^\top$ を取り、出力がスカラーである目的関数 $f(\mathbf{x})=x_1^2+2x_2^2$ を構成しよう。勾配は $\nabla f(\mathbf{x}) = [2x_1, 4x_2]^\top$ である。初期位置 $[-5, -2]$ から勾配降下法で $\mathbf{x}$ がたどる軌跡を観察する。

まず、さらに2つの補助関数が必要である。1つ目は更新関数を使って初期値に対して20回適用する。2つ目は $\mathbf{x}$ の軌跡を可視化する。

```{.python .input}
#@tab all
def train_2d(trainer, steps=20, f_grad=None):  #@save
    """カスタマイズしたトレーナーで2次元目的関数を最適化する。"""
    # `s1` と `s2` は Momentum, adagrad, RMSProp で使われる内部状態変数
    x1, x2, s1, s2 = -5, -2, 0, 0
    results = [(x1, x2)]
    for i in range(steps):
        if f_grad:
            x1, x2, s1, s2 = trainer(x1, x2, s1, s2, f_grad)
        else:
            x1, x2, s1, s2 = trainer(x1, x2, s1, s2)
        results.append((x1, x2))
    print(f'epoch {i + 1}, x1: {float(x1):f}, x2: {float(x2):f}')
    return results
```

```{.python .input}
#@tab mxnet
def show_trace_2d(f, results):  #@save
    """最適化中の2次元変数の軌跡を表示する。"""
    d2l.set_figsize()
    d2l.plt.plot(*zip(*results), '-o', color='#ff7f0e')
    x1, x2 = d2l.meshgrid(d2l.arange(-55, 1, 1),
                          d2l.arange(-30, 1, 1))
    x1, x2 = x1.asnumpy()*0.1, x2.asnumpy()*0.1
    d2l.plt.contour(x1, x2, f(x1, x2), colors='#1f77b4')
    d2l.plt.xlabel('x1')
    d2l.plt.ylabel('x2')
```

```{.python .input}
#@tab tensorflow
def show_trace_2d(f, results):  #@save
    """最適化中の2次元変数の軌跡を表示する。"""
    d2l.set_figsize()
    d2l.plt.plot(*zip(*results), '-o', color='#ff7f0e')
    x1, x2 = d2l.meshgrid(d2l.arange(-5.5, 1.0, 0.1),
                          d2l.arange(-3.0, 1.0, 0.1))
    d2l.plt.contour(x1, x2, f(x1, x2), colors='#1f77b4')
    d2l.plt.xlabel('x1')
    d2l.plt.ylabel('x2')
```

```{.python .input}
#@tab pytorch
def show_trace_2d(f, results):  #@save
    """最適化中の2次元変数の軌跡を表示する。"""
    d2l.set_figsize()
    d2l.plt.plot(*zip(*results), '-o', color='#ff7f0e')
    x1, x2 = d2l.meshgrid(d2l.arange(-5.5, 1.0, 0.1),
                          d2l.arange(-3.0, 1.0, 0.1), indexing='ij')
    d2l.plt.contour(x1, x2, f(x1, x2), colors='#1f77b4')
    d2l.plt.xlabel('x1')
    d2l.plt.ylabel('x2')
```

次に、学習率 $\eta = 0.1$ のときの最適化変数 $\mathbf{x}$ の軌跡を観察する。20ステップ後には、$\mathbf{x}$ の値が最小値 $[0, 0]$ に近づいていることが分かる。進行はかなり安定しているが、やや遅めである。

```{.python .input}
#@tab all
def f_2d(x1, x2):  # 目的関数
    return x1 ** 2 + 2 * x2 ** 2

def f_2d_grad(x1, x2):  # 目的関数の勾配
    return (2 * x1, 4 * x2)

def gd_2d(x1, x2, s1, s2, f_grad):
    g1, g2 = f_grad(x1, x2)
    return (x1 - eta * g1, x2 - eta * g2, 0, 0)

eta = 0.1
show_trace_2d(f_2d, train_2d(gd_2d, f_grad=f_2d_grad))
```

## 適応的手法

:numref:`subsec_gd-learningrate` で見たように、学習率 $\eta$ を「ちょうどよく」設定するのは難しい。小さすぎると進展がほとんどない。大きすぎると解が振動し、最悪の場合は発散する。学習率を自動的に決められたら、あるいはそもそも学習率を選ぶ必要がなくなったらどうだろうか。
目的関数の値と勾配だけでなく、その*曲率*も見る2次の手法は、この場合に役立つ。これらの手法は計算コストのため深層学習に直接適用することはできないが、以下で述べる多くの望ましい性質を模倣した高度な最適化アルゴリズムを設計するうえで有用な直感を与えてくれる。


### ニュートン法

ある関数 $f: \mathbb{R}^d \rightarrow \mathbb{R}$ のテイラー展開を見直すと、1項目で止める必要はない。実際には

$$f(\mathbf{x} + \boldsymbol{\epsilon}) = f(\mathbf{x}) + \boldsymbol{\epsilon}^\top \nabla f(\mathbf{x}) + \frac{1}{2} \boldsymbol{\epsilon}^\top \nabla^2 f(\mathbf{x}) \boldsymbol{\epsilon} + \mathcal{O}(\|\boldsymbol{\epsilon}\|^3).$$
:eqlabel:`gd-hot-taylor`

煩雑な記法を避けるため、$\mathbf{H} \stackrel{\textrm{def}}{=} \nabla^2 f(\mathbf{x})$ を $f$ のヘッセ行列と定義する。これは $d \times d$ 行列である。$d$ が小さく単純な問題なら、$\mathbf{H}$ は容易に計算できる。一方、深層ニューラルネットワークでは、$\mathcal{O}(d^2)$ 個の要素を保存するコストのため、$\mathbf{H}$ は非常に大きくなりえる。さらに、逆伝播で計算するには高コストすぎる場合もある。ここではそうした点をいったん無視して、どのようなアルゴリズムが得られるかを見てみよう。

そもそも $f$ の最小値では $\nabla f = 0$ が成り立つ。
:numref:`subsec_calculus-grad` の微積分の規則に従い、
:eqref:`gd-hot-taylor` を $\boldsymbol{\epsilon}$ について微分し、高次の項を無視すると

$$\nabla f(\mathbf{x}) + \mathbf{H} \boldsymbol{\epsilon} = 0 \textrm{ and hence }
\boldsymbol{\epsilon} = -\mathbf{H}^{-1} \nabla f(\mathbf{x}).$$

つまり、最適化問題の一部としてヘッセ行列 $\mathbf{H}$ を逆行列化する必要がある。

簡単な例として、$f(x) = \frac{1}{2} x^2$ では $\nabla f(x) = x$ かつ $\mathbf{H} = 1$ である。したがって、どの $x$ に対しても $\epsilon = -x$ が得られる。言い換えると、調整なしで*たった1回*のステップで完全に収束できる。もっとも、これは少し幸運であった。$f(x+\epsilon)= \frac{1}{2} x^2 + \epsilon x + \frac{1}{2} \epsilon^2$ なので、テイラー展開が厳密に一致していたからである。

他の問題ではどうなるか見てみよう。
定数 $c$ に対して凸な双曲線余弦関数 $f(x) = \cosh(cx)$ を考えると、
$x=0$ にある大域的最小値に
数回の反復で到達することが分かる。

```{.python .input}
#@tab all
c = d2l.tensor(0.5)

def f(x):  # 目的関数
    return d2l.cosh(c * x)

def f_grad(x):  # 目的関数の勾配
    return c * d2l.sinh(c * x)

def f_hess(x):  # 目的関数のヘッセ行列
    return c**2 * d2l.cosh(c * x)

def newton(eta=1):
    x = 10.0
    results = [x]
    for i in range(10):
        x -= eta * f_grad(x) / f_hess(x)
        results.append(float(x))
    print('epoch 10, x:', x)
    return results

show_trace(newton(), f)
```

次に、定数 $c$ に対して $f(x) = x \cos(c x)$ のような*非凸*関数を考えよう。ニュートン法ではヘッセ行列で割ることになる点に注意されたい。これは、2階導関数が*負*であると、$f$ の値を*増加*させる方向へ進んでしまう可能性があることを意味する。
これはアルゴリズムの致命的な欠陥である。
実際に何が起こるか見てみよう。

```{.python .input}
#@tab all
c = d2l.tensor(0.15 * np.pi)

def f(x):  # 目的関数
    return x * d2l.cos(c * x)

def f_grad(x):  # 目的関数の勾配
    return d2l.cos(c * x) - c * x * d2l.sin(c * x)

def f_hess(x):  # 目的関数のヘッセ行列
    return - 2 * c * d2l.sin(c * x) - x * c**2 * d2l.cos(c * x)

show_trace(newton(), f)
```

これはひどく失敗した。どうすれば修正できるだろうか。1つの方法は、ヘッセ行列の絶対値を取って「修正」することである。別の戦略は、学習率を再導入することである。これは目的に反するように見えるかもしれないが、実はそうでもない。2次情報があれば、曲率が大きいときには慎重になり、目的関数がより平坦なときには長いステップを取ることができる。
少し小さめの学習率、たとえば $\eta = 0.5$ でこれがどう働くか見てみよう。かなり効率的なアルゴリズムになっていることが分かる。

```{.python .input}
#@tab all
show_trace(newton(0.5), f)
```

### 収束解析

ここでは、ある凸で3回微分可能な目的関数 $f$ について、2階導関数が0でない、すなわち $f'' > 0$ である場合のニュートン法の収束率のみを解析する。多変数の場合の証明は、以下の1次元の議論をそのまま拡張したものなので、直感の助けにならないため省略する。

$x^{(k)}$ を $k^\textrm{th}$ 反復における $x$ の値とし、$e^{(k)} \stackrel{\textrm{def}}{=} x^{(k)} - x^*$ を $k^\textrm{th}$ 反復における最適解からの距離とする。テイラー展開により、条件 $f'(x^*) = 0$ は

$$0 = f'(x^{(k)} - e^{(k)}) = f'(x^{(k)}) - e^{(k)} f''(x^{(k)}) + \frac{1}{2} (e^{(k)})^2 f'''(\xi^{(k)}),$$

と書ける。ここで $\xi^{(k)} \in [x^{(k)} - e^{(k)}, x^{(k)}]$ となるある値である。上の展開を $f''(x^{(k)})$ で割ると

$$e^{(k)} - \frac{f'(x^{(k)})}{f''(x^{(k)})} = \frac{1}{2} (e^{(k)})^2 \frac{f'''(\xi^{(k)})}{f''(x^{(k)})}.$$

更新式 $x^{(k+1)} = x^{(k)} - f'(x^{(k)}) / f''(x^{(k)})$ を思い出されたい。
この更新式を代入し、両辺の絶対値を取ると

$$\left|e^{(k+1)}\right| = \frac{1}{2}(e^{(k)})^2 \frac{\left|f'''(\xi^{(k)})\right|}{f''(x^{(k)})}.$$

したがって、$\left|f'''(\xi^{(k)})\right| / (2f''(x^{(k)})) \leq c$ で抑えられる領域にいるなら、誤差は2乗的に減少し

$$\left|e^{(k+1)}\right| \leq c (e^{(k)})^2.$$

なお、最適化研究者はこれを*線形*収束と呼ぶが、$\left|e^{(k+1)}\right| \leq \alpha \left|e^{(k)}\right|$ のような条件は*定数*収束率と呼ばれる。
この解析にはいくつかの注意点がある。
第一に、急速収束の領域にいつ到達するかについては、実際にはあまり保証がない。分かるのは、その領域に入れば収束が非常に速いということだけである。第二に、この解析では $f$ が高階導関数まで良い性質を持つことが必要である。要するに、$f$ が値の変化の仕方に関して「予想外」の性質を持たないことを保証する必要がある。



### 前処理

完全なヘッセ行列を計算して保存するのが非常に高コストであることは、まったく驚くことではない。したがって、代替手段を見つけることが望ましい。改善の1つの方法が*前処理*である。これはヘッセ行列全体を計算する代わりに、*対角*成分だけを計算する。これにより、次の形の更新アルゴリズムが得られる。

$$\mathbf{x} \leftarrow \mathbf{x} - \eta \textrm{diag}(\mathbf{H})^{-1} \nabla f(\mathbf{x}).$$


これは完全なニュートン法ほど良くはないが、何もしないよりははるかに優れている。
これが有効な理由を理解するために、一方の変数がミリメートル単位の高さを表し、もう一方がキロメートル単位の高さを表す状況を考えてみよう。どちらも自然なスケールはメートルだとすると、パラメータ化の不一致がひどいことになる。幸い、前処理を使えばこれを解消できる。実質的には、勾配降下法における前処理は、$\mathbf{x}$ の各変数（ベクトルの各座標）ごとに異なる学習率を選ぶことに相当する。
後で見るように、前処理は確率的勾配降下法の最適化アルゴリズムにおけるいくつかの革新を支えている。


### ラインサーチ付き勾配降下法

勾配降下法の主要な問題の1つは、目標を飛び越えてしまったり、進展が不十分だったりすることである。この問題の簡単な解決策は、勾配降下法と組み合わせてラインサーチを使うことである。つまり、$\nabla f(\mathbf{x})$ で与えられる方向を使い、その後、$f(\mathbf{x} - \eta \nabla f(\mathbf{x}))$ を最小化する学習率 $\eta$ を二分探索で求める。

このアルゴリズムは高速に収束する（解析と証明については、たとえば :citet:`Boyd.Vandenberghe.2004` を参照）。しかし、深層学習の目的ではこれはあまり現実的ではない。というのも、ラインサーチの各ステップで目的関数をデータセット全体に対して評価する必要があるからである。これはあまりにも高コストである。

## まとめ

* 学習率は重要である。大きすぎると発散し、小さすぎると進展しない。
* 勾配降下法は局所最小値に捕まることがある。
* 高次元では学習率の調整は複雑である。
* 前処理はスケール調整に役立つ。
* ニュートン法は、凸問題で正しく機能し始めるとかなり高速である。
* 非凸問題に対して、何の調整もせずにニュートン法を使うのは危険である。

## 演習

1. 勾配降下法について、さまざまな学習率と目的関数を試してみよ。
1. 区間 $[a, b]$ における凸関数を最小化するためのラインサーチを実装せよ。
    1. 二分探索に導関数は必要か。つまり、$[a, (a+b)/2]$ と $[(a+b)/2, b]$ のどちらを選ぶかを決めるのに必要か。
    1. このアルゴリズムの収束率はどれくらい速いか。
    1. アルゴリズムを実装し、$\log (\exp(x) + \exp(-2x -3))$ の最小化に適用せよ。
1. 勾配降下法が非常に遅くなる $\mathbb{R}^2$ 上の目的関数を設計せよ。ヒント: 座標ごとに異なるスケールを使う。
1. 前処理を用いたニュートン法の軽量版を実装せよ。
    1. 対角ヘッセ行列を前処理として使え。
    1. 実際の（符号付きの）値ではなく、その絶対値を使え。
    1. 上の問題に適用せよ。
1. 上のアルゴリズムをいくつかの目的関数（凸でも非凸でもよい）に適用せよ。座標を $45$ 度回転させると何が起こるか。
