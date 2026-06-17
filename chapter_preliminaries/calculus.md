{.python .input}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow', 'jax'])
```

# 微分積分
:label:`sec_calculus`

円の面積をどのように求めるかは、長いあいだ謎であった。
その後、古代ギリシャの数学者アルキメデスは、
円に内接する多角形の頂点数を次第に増やしていくという
画期的な着想を示した
(:numref:`fig_circle_area`)。
頂点数が $n$ の多角形は、
$n$ 個の三角形に分割できる。
分割を細かくしていくと、
各三角形の高さは半径 $r$ に近づく。
同時に、頂点数が十分大きければ弧とその弦の比は 1 に近づくため、
底辺は $2 \pi r/n$ に近づく。
したがって、多角形の面積は
$n \cdot r \cdot \frac{1}{2} (2 \pi r/n) = \pi r^2$
に近づく。

![極限過程として円の面積を求める。](../img/polygon-circle.svg)
:label:`fig_circle_area`

この極限の考え方は、
*微分学* と *積分学* の両方の基礎にある。
前者は、引数を変化させたときに
関数値がどのように変わるかを明らかにする。
これは、損失関数を減少させるために
パラメータを反復的に更新する深層学習の
*最適化問題* においてとりわけ有用である。
最適化とは、モデルを訓練データに適合させる過程であり、微分積分はそのための重要な基礎である。
ただし、最終的な目標は
*これまで見たことのない* データに対しても良い性能を示すことである。
この問題は *汎化* と呼ばれ、
以降の章で重要な主題となる。

```{.python .input}
%%tab mxnet
%matplotlib inline
from d2l import mxnet as d2l
from matplotlib_inline import backend_inline
from mxnet import np, npx
npx.set_np()
```

```{.python .input}
%%tab pytorch
%matplotlib inline
from d2l import torch as d2l
from matplotlib_inline import backend_inline
import numpy as np
```

```{.python .input}
%%tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
from matplotlib_inline import backend_inline
import numpy as np
```

```{.python .input}
%%tab jax
%matplotlib inline
from d2l import jax as d2l
from matplotlib_inline import backend_inline
import numpy as np
```

## 導関数と微分

簡単に言えば、*導関数* とは、引数の変化に対する関数値の変化率である。
導関数は、各パラメータを
微小な量だけ変化させたときに、
損失関数がどれだけ変わるかを示す。
形式的には、スカラーからスカラーへの写像である
関数 $f: \mathbb{R} \rightarrow \mathbb{R}$ に対して、
[**点 $x$ における $f$ の *導関数* は次のように定義される**]

[**$$f'(x) = \lim_{h \rightarrow 0} \frac{f(x+h) - f(x)}{h}.$$**]
:eqlabel:`eq_derivative`

右辺の式は *極限* と呼ばれ、
ある変数が特定の値に近づくときに
式全体の値がどうなるかを表す。
この極限は、摂動 $h$ と
関数値の変化 $f(x + h) - f(x)$ の比が、
$h$ の大きさを 0 に近づけたときに
どの値へ収束するかを示している。

$f'(x)$ が存在するとき、$f$ は $x$ において
*微分可能* であるという。
また、ある集合、たとえば区間 $[a,b]$ 上のすべての $x$ で
$f'(x)$ が存在するとき、$f$ はその集合上で微分可能であるという。
すべての関数が微分可能とは限らず、
最適化したい多くの関数、たとえば精度や
受信者動作特性曲線下面積（AUC）もその例である。
しかし、損失の導関数を計算することは
深層ニューラルネットワークを訓練するほぼすべてのアルゴリズムにおいて
極めて重要であるため、
しばしば微分可能な *代理* を最適化する。


導関数 $f'(x)$ は、
$x$ に関する $f(x)$ の *瞬間的* な変化率として解釈できる。
例を通して直感を養おう。
[**$u = f(x) = 3x^2-4x$ と定義する。**]

```{.python .input}
%%tab mxnet
def f(x):
    return 3 * x ** 2 - 4 * x
```

```{.python .input}
%%tab pytorch
def f(x):
    return 3 * x ** 2 - 4 * x
```

```{.python .input}
%%tab tensorflow
def f(x):
    return 3 * x ** 2 - 4 * x
```

```{.python .input}
%%tab jax
def f(x):
    return 3 * x ** 2 - 4 * x
```

[**$x=1$ とすると、$\frac{f(x+h) - f(x)}{h}$ は**] [**$h$ が $0$ に近づくにつれて $2$ に近づく**]
ことがわかる。
この実験は数学的証明ほど厳密ではないが、
$f'(1) = 2$ であることを確かめるには十分である。

```{.python .input}
%%tab all
for h in 10.0**np.arange(-1, -6, -1):
    print(f'h={h:.5f}, numerical limit={(f(1+h)-f(1))/h:.5f}')
```

導関数には、いくつかの同値な記法がある。
$y = f(x)$ とすると、次の表現はすべて同じ意味である。

$$f'(x) = y' = \frac{dy}{dx} = \frac{df}{dx} = \frac{d}{dx} f(x) = Df(x) = D_x f(x),$$

ここで、$\frac{d}{dx}$ と $D$ は *微分作用素* である。
以下に、いくつかの基本的な関数の導関数を示す。

$$\begin{aligned} \frac{d}{dx} C & = 0 && \textrm{任意の定数 $C$ に対して} \\ \frac{d}{dx} x^n & = n x^{n-1} && \textrm{$n \neq 0$ のとき} \\ \frac{d}{dx} e^x & = e^x \\ \frac{d}{dx} \ln x & = x^{-1}. \end{aligned}$$

微分可能な関数から構成された関数も、
しばしばそれ自体が微分可能である。
次の規則は、任意の微分可能な関数 $f$ と $g$、
および定数 $C$ を扱う際に有用である。

$$\begin{aligned} \frac{d}{dx} [C f(x)] & = C \frac{d}{dx} f(x) && \textrm{定数倍の法則} \\ \frac{d}{dx} [f(x) + g(x)] & = \frac{d}{dx} f(x) + \frac{d}{dx} g(x) && \textrm{和の法則} \\ \frac{d}{dx} [f(x) g(x)] & = f(x) \frac{d}{dx} g(x) + g(x) \frac{d}{dx} f(x) && \textrm{積の法則} \\ \frac{d}{dx} \frac{f(x)}{g(x)} & = \frac{g(x) \frac{d}{dx} f(x) - f(x) \frac{d}{dx} g(x)}{g^2(x)} && \textrm{商の法則} \end{aligned}$$

これらを用いると、$3 x^2 - 4x$ の導関数は次のように求まる。

$$\frac{d}{dx} [3 x^2 - 4x] = 3 \frac{d}{dx} x^2 - 4 \frac{d}{dx} x = 6x - 4.$$

$x = 1$ を代入すると、確かにこの点での導関数は $2$ になる。
導関数は、ある点における関数の *傾き* を与えることに注意しよう。  

## 可視化ユーティリティ

[**`matplotlib` ライブラリを使って関数の傾きを可視化できる**]。
そのために、いくつかの関数を定義する。
名前が示すとおり、`use_svg_display` は `matplotlib` に
より鮮明な画像を得るため SVG 形式でグラフを出力するよう指示する。
コメント `#@save` は特別な修飾子であり、
任意の関数、クラス、その他のコードブロックを `d2l` パッケージに保存し、
後でコードを繰り返さずに
たとえば `d2l.use_svg_display()` のように呼び出せるようにする。

```{.python .input}
%%tab all
def use_svg_display():  #@save
    """Jupyter でプロットを表示するために svg 形式を使う。"""
    backend_inline.set_matplotlib_formats('svg')
```

`set_figsize` を使うと、図の大きさを設定できる。
`from matplotlib import pyplot as plt` という import 文は
`d2l` パッケージ内で `#@save` によりマークされているので、
`d2l.plt` を呼び出せる。

```{.python .input}
%%tab all
def set_figsize(figsize=(3.5, 2.5)):  #@save
    """matplotlib の図のサイズを設定する。"""
    use_svg_display()
    d2l.plt.rcParams['figure.figsize'] = figsize
```

`set_axes` 関数は、ラベル、範囲、スケールを含む
各種の属性を軸に設定する。

```{.python .input}
%%tab all
#@save
def set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend):
    """matplotlib の軸を設定する。"""
    axes.set_xlabel(xlabel), axes.set_ylabel(ylabel)
    axes.set_xscale(xscale), axes.set_yscale(yscale)
    axes.set_xlim(xlim),     axes.set_ylim(ylim)
    if legend:
        axes.legend(legend)
    axes.grid()
```

これら 3 つの関数を使うと、複数の曲線を重ねて描く `plot` 関数を定義できる。
ここでのコードの大半は、入力のサイズと形状が整合していることを確認するためのものである。

```{.python .input}
%%tab all
#@save
def plot(X, Y=None, xlabel=None, ylabel=None, legend=[], xlim=None,
         ylim=None, xscale='linear', yscale='linear',
         fmts=('-', 'm--', 'g-.', 'r:'), figsize=(3.5, 2.5), axes=None):
    """データ点をプロットする。"""

    def has_one_axis(X):  # X (tensor or list) が 1 軸なら True
        return (hasattr(X, "ndim") and X.ndim == 1 or isinstance(X, list)
                and not hasattr(X[0], "__len__"))
    
    if has_one_axis(X): X = [X]
    if Y is None:
        X, Y = [[]] * len(X), X
    elif has_one_axis(Y):
        Y = [Y]
    if len(X) != len(Y):
        X = X * len(Y)
        
    set_figsize(figsize)
    if axes is None:
        axes = d2l.plt.gca()
    axes.cla()
    for x, y, fmt in zip(X, Y, fmts):
        axes.plot(x,y,fmt) if len(x) else axes.plot(y,fmt)
    set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
```

これで、[**関数 $u = f(x)$ と、$x=1$ におけるその接線 $y = 2x - 3$ を描画できる**]。
ここで係数 $2$ は接線の傾きである。

```{.python .input}
%%tab all
x = np.arange(0, 3, 0.1)
plot(x, [f(x), 2 * x - 3], 'x', 'f(x)', legend=['f(x)', 'Tangent line (x=1)'])
```

## 偏導関数と勾配
:label:`subsec_calculus-grad`

ここまでは、1 変数関数の微分を扱ってきた。
深層学習では、*多変数* 関数も扱う必要がある。
ここでは、そのような *多変数* 関数に対する導関数の概念を簡単に導入する。


$y = f(x_1, x_2, \ldots, x_n)$ を $n$ 変数関数とする。
$i^\textrm{th}$ パラメータ $x_i$ に関する $y$ の *偏導関数* は

$$ \frac{\partial y}{\partial x_i} = \lim_{h \rightarrow 0} \frac{f(x_1, \ldots, x_{i-1}, x_i+h, x_{i+1}, \ldots, x_n) - f(x_1, \ldots, x_i, \ldots, x_n)}{h}.$$


$\frac{\partial y}{\partial x_i}$ を計算するには、
$x_1, \ldots, x_{i-1}, x_{i+1}, \ldots, x_n$ を定数とみなし、
$x_i$ に関する $y$ の導関数を求めればよい。
偏導関数の記法には次のようなものがあり、
いずれも同じ意味である。

$$\frac{\partial y}{\partial x_i} = \frac{\partial f}{\partial x_i} = \partial_{x_i} f = \partial_i f = f_{x_i} = f_i = D_i f = D_{x_i} f.$$

多変数関数のすべての変数に関する偏導関数を
まとめると、関数の *勾配* と呼ばれるベクトルが得られる。
関数 $f: \mathbb{R}^n \rightarrow \mathbb{R}$ の入力が
$n$ 次元ベクトル $\mathbf{x} = [x_1, x_2, \ldots, x_n]^\top$ で、
出力がスカラーであるとする。
$\mathbf{x}$ に関する関数 $f$ の勾配は、
$n$ 個の偏導関数からなるベクトルである。

$$\nabla_{\mathbf{x}} f(\mathbf{x}) = \left[\partial_{x_1} f(\mathbf{x}), \partial_{x_2} f(\mathbf{x}), \ldots
\partial_{x_n} f(\mathbf{x})\right]^\top.$$ 

曖昧さがなければ、
$\nabla_{\mathbf{x}} f(\mathbf{x})$ は通常
$\nabla f(\mathbf{x})$ と書く。
多変数関数を微分する際には、次の規則が有用である。

* すべての $\mathbf{A} \in \mathbb{R}^{m \times n}$ について、$\nabla_{\mathbf{x}} \mathbf{A} \mathbf{x} = \mathbf{A}^\top$ および $\nabla_{\mathbf{x}} \mathbf{x}^\top \mathbf{A}  = \mathbf{A}$ が成り立つ。
* 正方行列 $\mathbf{A} \in \mathbb{R}^{n \times n}$ については、$\nabla_{\mathbf{x}} \mathbf{x}^\top \mathbf{A} \mathbf{x}  = (\mathbf{A} + \mathbf{A}^\top)\mathbf{x}$ が成り立ち、特に
$\nabla_{\mathbf{x}} \|\mathbf{x} \|^2 = \nabla_{\mathbf{x}} \mathbf{x}^\top \mathbf{x} = 2\mathbf{x}$ である。

同様に、任意の行列 $\mathbf{X}$ について、
$\nabla_{\mathbf{X}} \|\mathbf{X} \|_\textrm{F}^2 = 2\mathbf{X}$ が成り立つ。 



## 連鎖律

深層学習で現れる勾配は、しばしば計算が複雑になる。
なぜなら、
深く入れ子になった関数
（関数の中の関数の中の関数……）を扱うからである。
幸い、*連鎖律* がこれを扱うための道具を与える。
まず 1 変数関数に戻り、
$y = f(g(x))$ であり、基礎となる関数
$y=f(u)$ と $u=g(x)$ が
どちらも微分可能であるとする。
連鎖律は次のように述べる。


$$\frac{dy}{dx} = \frac{dy}{du} \frac{du}{dx}.$$



多変数関数に戻ると、
$y = f(\mathbf{u})$ が変数
$u_1, u_2, \ldots, u_m$ をもち、
各 $u_i = g_i(\mathbf{x})$ が変数
$x_1, x_2, \ldots, x_n$ の関数である、
すなわち $\mathbf{u} = g(\mathbf{x})$ とする。
このとき連鎖律は次のように書ける。

$$\frac{\partial y}{\partial x_{i}} = \frac{\partial y}{\partial u_{1}} \frac{\partial u_{1}}{\partial x_{i}} + \frac{\partial y}{\partial u_{2}} \frac{\partial u_{2}}{\partial x_{i}} + \ldots + \frac{\partial y}{\partial u_{m}} \frac{\partial u_{m}}{\partial x_{i}} \ \textrm{ したがって } \ \nabla_{\mathbf{x}} y =  \mathbf{A} \nabla_{\mathbf{u}} y,$$

ここで $\mathbf{A} \in \mathbb{R}^{n \times m}$ は、
ベクトル $\mathbf{u}$ のベクトル $\mathbf{x}$ に関する導関数を並べた *行列* である。
したがって、勾配を評価するには
ベクトルと行列の積を計算する必要がある。
これが、線形代数が深層学習システムを構築するうえで
非常に重要な構成要素である
主な理由の一つである。 



## 議論

ここでは深い話題の表面を少しなぞったにすぎないが、
すでにいくつかの重要な概念が見えている。
第一に、微分の合成規則は機械的に適用できるため、
勾配は *自動的に* 計算できる。
この作業には創造性が不要なので、
認知資源を他の問題に振り向けられる。
第二に、ベクトル値関数の導関数を計算するには、
出力から入力へ向かって変数の依存グラフをたどりながら
行列を掛け合わせる必要がある。
特に、このグラフは関数を評価するときには *順方向* に、
勾配を計算するときには *逆方向* にたどる。
後の章では、連鎖律を適用する計算手順である
逆伝播を正式に導入する。

最適化の観点から見ると、勾配によって
損失を減少させるためにモデルのパラメータを
どのように動かすべきかを決定できる。
そして、この本全体で用いる最適化アルゴリズムの各ステップでは、
勾配の計算が必要になる。

## 演習

1. ここまでは導関数の規則を当然のものとして扱ってきた。 
   定義と極限を用いて、(i) $f(x) = c$, (ii) $f(x) = x^n$, (iii) $f(x) = e^x$, (iv) $f(x) = \log x$ の性質を証明せよ。
1. 同様に、積の法則、和の法則、商の法則を第一原理から証明せよ。 
1. 定数倍の法則が積の法則の特殊な場合として導かれることを証明せよ。 
1. $f(x) = x^x$ の導関数を求めよ。 
1. ある $x$ に対して $f'(x) = 0$ であるとはどういう意味だろうか。 
   そのようなことが成り立つ関数 $f$ と位置 $x$ の例を挙げよ。 
1. 関数 $y = f(x) = x^3 - \frac{1}{x}$ を描画し、$x = 1$ における接線も描画せよ。
1. 関数 
   $f(\mathbf{x}) = 3x_1^2 + 5e^{x_2}$ の勾配を求めよ。
1. 関数 
   $f(\mathbf{x}) = \|\mathbf{x}\|_2$ の勾配は何だろうか。$\mathbf{x} = \mathbf{0}$ のときはどうなるか？
1. $u = f(x, y, z)$ かつ $x = x(a, b)$, $y = y(a, b)$, $z = z(a, b)$ の場合の
   連鎖律を書けるか。
1. 逆写像をもつ関数 $f(x)$ が与えられたとき、
   その逆関数 $f^{-1}(x)$ の導関数を求めよ。 
   ここで $f^{-1}(f(x)) = x$ かつ逆に $f(f^{-1}(y)) = y$ である。 
   ヒント: 導出ではこれらの性質を使え。