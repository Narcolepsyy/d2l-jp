{.python .input}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow', 'jax'])
```

# 微分積分
:label:`sec_calculus`

長い間、円の面積をどのように計算するかは謎のままでした。
その後、古代ギリシャの数学者アルキメデスが、
円の内側に頂点数を増やした多角形を次々と内接させるという
画期的なアイデアを考案しました
(:numref:`fig_circle_area`)。
$n$ 個の頂点をもつ多角形については、
$n$ 個の三角形が得られます。
円をより細かく分割するにつれて、
各三角形の高さは半径 $r$ に近づきます。
同時に、弧と割線の比が頂点数が十分大きいときに 1 に近づくので、
底辺は $2 \pi r/n$ に近づきます。
したがって、多角形の面積は
$n \cdot r \cdot \frac{1}{2} (2 \pi r/n) = \pi r^2$
に近づきます。

![極限過程として円の面積を求める。](../img/polygon-circle.svg)
:label:`fig_circle_area`

この極限過程は、
*微分学* と *積分学* の両方の根底にあります。
前者は、引数を操作することで
関数値をどのように増減させるかを教えてくれます。
これは、損失関数を減らすために
パラメータを繰り返し更新する深層学習で直面する
*最適化問題* において非常に役立ちます。
最適化はモデルを訓練データに適合させるプロセスであり、微分積分学はそのための重要な基礎知識です。
ただし、最終的な目標は
*これまで見たことのない* データで良い性能を発揮することだという点を忘れてはいけません。
この問題は *汎化* と呼ばれ、
他の章で重要な焦点となります。

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

簡単に言えば、*導関数* とは、引数の変化に対する関数の変化率です。
導関数は、各パラメータを
微小な量だけ *増減* させたときに、
損失関数がどれだけ変化するかを示してくれます。
形式的には、スカラーからスカラーへの写像である
関数 $f: \mathbb{R} \rightarrow \mathbb{R}$ に対して、
[**点 $x$ における $f$ の *導関数* は次のように定義されます**]

[**$$f'(x) = \lim_{h \rightarrow 0} \frac{f(x+h) - f(x)}{h}.$$**]
:eqlabel:`eq_derivative`

右辺のこの項は *極限* と呼ばれ、
ある変数が特定の値に近づくときに
式の値がどうなるかを教えてくれます。
この極限は、摂動 $h$ と
関数値の変化 $f(x + h) - f(x)$ の比が、
その大きさを 0 に近づけたときに
どの値に収束するかを示しています。

$f'(x)$ が存在するとき、$f$ は $x$ において
*微分可能* であるといいます。
また、ある集合、たとえば区間 $[a,b]$ 上のすべての $x$ で
$f'(x)$ が存在するとき、$f$ はその集合上で微分可能であるといいます。
すべての関数が微分可能なわけではなく、
最適化したい多くの関数、たとえば精度や
受信者動作特性曲線下面積（AUC）もその例です。
しかし、損失の導関数を計算することは
深層ニューラルネットワークを訓練するほぼすべてのアルゴリズムにおいて
極めて重要なステップであるため、
しばしば微分可能な *代理* を最適化します。


導関数 $f'(x)$ は、
$x$ に関する $f(x)$ の *瞬間的* な変化率として解釈できます。
例を通して直感を養いましょう。
[**$u = f(x) = 3x^2-4x$ と定義します。**]

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

[**$x=1$ とすると、$\frac{f(x+h) - f(x)}{h}$ が**] [**$h$ が $0$ に近づくにつれて $2$ に近づく**]
ことがわかります。
この実験は数学的証明ほど厳密ではありませんが、
確かに $f'(1) = 2$ であることがすぐにわかります。

```{.python .input}
%%tab all
for h in 10.0**np.arange(-1, -6, -1):
    print(f'h={h:.5f}, numerical limit={(f(1+h)-f(1))/h:.5f}')
```

導関数には、いくつか同値な記法があります。
$y = f(x)$ とすると、次の表現はすべて同値です。

$$f'(x) = y' = \frac{dy}{dx} = \frac{df}{dx} = \frac{d}{dx} f(x) = Df(x) = D_x f(x),$$

ここで、$\frac{d}{dx}$ と $D$ は *微分作用素* です。
以下に、いくつかの一般的な関数の導関数を示します。

$$\begin{aligned} \frac{d}{dx} C & = 0 && \textrm{任意の定数 $C$ に対して} \\ \frac{d}{dx} x^n & = n x^{n-1} && \textrm{$n \neq 0$ のとき} \\ \frac{d}{dx} e^x & = e^x \\ \frac{d}{dx} \ln x & = x^{-1}. \end{aligned}$$

微分可能な関数から合成された関数も、
しばしばそれ自体が微分可能です。
次の規則は、任意の微分可能な関数 $f$ と $g$、
および定数 $C$ の合成を扱う際に役立ちます。

$$\begin{aligned} \frac{d}{dx} [C f(x)] & = C \frac{d}{dx} f(x) && \textrm{定数倍の法則} \\ \frac{d}{dx} [f(x) + g(x)] & = \frac{d}{dx} f(x) + \frac{d}{dx} g(x) && \textrm{和の法則} \\ \frac{d}{dx} [f(x) g(x)] & = f(x) \frac{d}{dx} g(x) + g(x) \frac{d}{dx} f(x) && \textrm{積の法則} \\ \frac{d}{dx} \frac{f(x)}{g(x)} & = \frac{g(x) \frac{d}{dx} f(x) - f(x) \frac{d}{dx} g(x)}{g^2(x)} && \textrm{商の法則} \end{aligned}$$

これを用いると、次のようにして $3 x^2 - 4x$ の導関数を求められます。

$$\frac{d}{dx} [3 x^2 - 4x] = 3 \frac{d}{dx} x^2 - 4 \frac{d}{dx} x = 6x - 4.$$

$x = 1$ を代入すると、確かにこの位置での導関数は $2$ になります。
導関数は、ある特定の位置における関数の *傾き* を教えてくれることに注意してください。  

## 可視化ユーティリティ

[**`matplotlib` ライブラリを使って関数の傾きを可視化できます**]。
いくつかの関数を定義する必要があります。
名前が示すように、`use_svg_display` は `matplotlib` に
より鮮明な画像のために SVG 形式でグラフを出力するよう指示します。
コメント `#@save` は特別な修飾子で、
任意の関数、クラス、その他のコードブロックを `d2l` パッケージに保存し、
後でコードを繰り返さずに
たとえば `d2l.use_svg_display()` のように呼び出せるようにします。

```{.python .input}
%%tab all
def use_svg_display():  #@save
    """Jupyter でプロットを表示するために svg 形式を使う。"""
    backend_inline.set_matplotlib_formats('svg')
```

便利なことに、`set_figsize` で図のサイズを設定できます。
`from matplotlib import pyplot as plt` という import 文は
`d2l` パッケージ内で `#@save` によりマークされているので、
`d2l.plt` を呼び出せます。

```{.python .input}
%%tab all
def set_figsize(figsize=(3.5, 2.5)):  #@save
    """matplotlib の図のサイズを設定する。"""
    use_svg_display()
    d2l.plt.rcParams['figure.figsize'] = figsize
```

`set_axes` 関数は、ラベル、範囲、スケールを含む
属性を軸に関連付けることができます。

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

これら 3 つの関数を使うと、複数の曲線を重ねて描く `plot` 関数を定義できます。
ここでのコードの多くは、入力のサイズと形状が一致することを確認しているだけです。

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

これで、[**関数 $u = f(x)$ と、$x=1$ におけるその接線 $y = 2x - 3$ を描画できます**]。
ここで係数 $2$ は接線の傾きです。

```{.python .input}
%%tab all
x = np.arange(0, 3, 0.1)
plot(x, [f(x), 2 * x - 3], 'x', 'f(x)', legend=['f(x)', 'Tangent line (x=1)'])
```

## 偏導関数と勾配
:label:`subsec_calculus-grad`

ここまでは、1 変数だけの関数を微分してきました。
深層学習では、*多変数* の関数も扱う必要があります。
そのような *多変数* 関数に適用される導関数の概念を簡単に導入します。


$y = f(x_1, x_2, \ldots, x_n)$ を $n$ 個の変数をもつ関数とします。
$i^\textrm{th}$ パラメータ $x_i$ に関する $y$ の *偏導関数* は

$$ \frac{\partial y}{\partial x_i} = \lim_{h \rightarrow 0} \frac{f(x_1, \ldots, x_{i-1}, x_i+h, x_{i+1}, \ldots, x_n) - f(x_1, \ldots, x_i, \ldots, x_n)}{h}.$$


$\frac{\partial y}{\partial x_i}$ を計算するには、
$x_1, \ldots, x_{i-1}, x_{i+1}, \ldots, x_n$ を定数として扱い、
$x_i$ に関する $y$ の導関数を計算すればよいです。
偏導関数の記法には次のようなものがあり、
どれも同じ意味です。

$$\frac{\partial y}{\partial x_i} = \frac{\partial f}{\partial x_i} = \partial_{x_i} f = \partial_i f = f_{x_i} = f_i = D_i f = D_{x_i} f.$$

多変数関数のすべての変数に関する偏導関数を
連結すると、関数の *勾配* と呼ばれるベクトルが得られます。
関数 $f: \mathbb{R}^n \rightarrow \mathbb{R}$ の入力が
$n$ 次元ベクトル $\mathbf{x} = [x_1, x_2, \ldots, x_n]^\top$ で、
出力がスカラーであるとします。
$\mathbf{x}$ に関する関数 $f$ の勾配は、
$n$ 個の偏導関数からなるベクトルです。

$$\nabla_{\mathbf{x}} f(\mathbf{x}) = \left[\partial_{x_1} f(\mathbf{x}), \partial_{x_2} f(\mathbf{x}), \ldots
\partial_{x_n} f(\mathbf{x})\right]^\top.$$ 

曖昧さがなければ、
$\nabla_{\mathbf{x}} f(\mathbf{x})$ は通常
$\nabla f(\mathbf{x})$ と書かれます。
多変数関数を微分する際には、次の規則が役立ちます。

* すべての $\mathbf{A} \in \mathbb{R}^{m \times n}$ について、$\nabla_{\mathbf{x}} \mathbf{A} \mathbf{x} = \mathbf{A}^\top$ および $\nabla_{\mathbf{x}} \mathbf{x}^\top \mathbf{A}  = \mathbf{A}$ が成り立ちます。
* 正方行列 $\mathbf{A} \in \mathbb{R}^{n \times n}$ については、$\nabla_{\mathbf{x}} \mathbf{x}^\top \mathbf{A} \mathbf{x}  = (\mathbf{A} + \mathbf{A}^\top)\mathbf{x}$ が成り立ち、特に
$\nabla_{\mathbf{x}} \|\mathbf{x} \|^2 = \nabla_{\mathbf{x}} \mathbf{x}^\top \mathbf{x} = 2\mathbf{x}$ です。

同様に、任意の行列 $\mathbf{X}$ について、
$\nabla_{\mathbf{X}} \|\mathbf{X} \|_\textrm{F}^2 = 2\mathbf{X}$ が成り立ちます。 



## 連鎖律

深層学習では、扱う勾配はしばしば計算が難しくなります。
なぜなら、私たちは
深く入れ子になった関数
（関数の中の関数の中の関数…）を扱っているからです。
幸いなことに、*連鎖律* がこれを処理してくれます。
1 変数関数に戻ると、
$y = f(g(x))$ であり、基礎となる関数
$y=f(u)$ と $u=g(x)$ が
どちらも微分可能だとします。
連鎖律は次のように述べます。


$$\frac{dy}{dx} = \frac{dy}{du} \frac{du}{dx}.$$



多変数関数に戻ると、
$y = f(\mathbf{u})$ が変数
$u_1, u_2, \ldots, u_m$ をもち、
各 $u_i = g_i(\mathbf{x})$ が変数
$x_1, x_2, \ldots, x_n$ をもつ、
すなわち $\mathbf{u} = g(\mathbf{x})$ とします。
このとき連鎖律は次のように述べます。

$$\frac{\partial y}{\partial x_{i}} = \frac{\partial y}{\partial u_{1}} \frac{\partial u_{1}}{\partial x_{i}} + \frac{\partial y}{\partial u_{2}} \frac{\partial u_{2}}{\partial x_{i}} + \ldots + \frac{\partial y}{\partial u_{m}} \frac{\partial u_{m}}{\partial x_{i}} \ \textrm{ したがって } \ \nabla_{\mathbf{x}} y =  \mathbf{A} \nabla_{\mathbf{u}} y,$$

ここで $\mathbf{A} \in \mathbb{R}^{n \times m}$ は、
ベクトル $\mathbf{u}$ のベクトル $\mathbf{x}$ に関する導関数を含む *行列* です。
したがって、勾配を評価するには
ベクトルと行列の積を計算する必要があります。
これが、線形代数が深層学習システムを構築するうえで
非常に重要な構成要素である
主要な理由の一つです。 



## 議論

ここでは深い話題の表面を少しなぞったにすぎませんが、
すでにいくつかの概念が見えてきます。
第一に、微分の合成規則は日常的に適用できるため、
勾配を *自動的に* 計算できます。
この作業には創造性が不要なので、
私たちは他のことに認知資源を集中できます。
第二に、ベクトル値関数の導関数を計算するには、
出力から入力へと変数の依存グラフをたどりながら
行列を掛け合わせる必要があります。
特に、このグラフは関数を評価するときには *順方向* に、
勾配を計算するときには *逆方向* にたどられます。
後の章では、連鎖律を適用する計算手順である
逆伝播を正式に導入します。

最適化の観点から見ると、勾配によって
損失を下げるためにモデルのパラメータを
どのように動かすべきかを決定できます。
そして、この本全体で使う最適化アルゴリズムの各ステップでは、
勾配の計算が必要になります。

## 演習

1. ここまでは導関数の規則を当然のものとして扱ってきました。 
   定義と極限を用いて、(i) $f(x) = c$, (ii) $f(x) = x^n$, (iii) $f(x) = e^x$, (iv) $f(x) = \log x$ の性質を証明してください。
1. 同様に、積の法則、和の法則、商の法則を第一原理から証明してください。 
1. 定数倍の法則が積の法則の特殊な場合として導かれることを証明してください。 
1. $f(x) = x^x$ の導関数を求めてください。 
1. ある $x$ に対して $f'(x) = 0$ であるとはどういう意味でしょうか。 
   そのようなことが成り立つ関数 $f$ と位置 $x$ の例を挙げてください。 
1. 関数 $y = f(x) = x^3 - \frac{1}{x}$ を描画し、$x = 1$ における接線も描画してください。
1. 関数 
   $f(\mathbf{x}) = 3x_1^2 + 5e^{x_2}$ の勾配を求めてください。
1. 関数 
   $f(\mathbf{x}) = \|\mathbf{x}\|_2$ の勾配は何でしょうか。$\mathbf{x} = \mathbf{0}$ のときはどうなりますか？
1. $u = f(x, y, z)$ かつ $x = x(a, b)$, $y = y(a, b)$, $z = z(a, b)$ の場合の
   連鎖律を書けますか？
1. 逆写像をもつ関数 $f(x)$ が与えられたとき、
   その逆関数 $f^{-1}(x)$ の導関数を求めてください。 
   ここで $f^{-1}(f(x)) = x$ かつ逆に $f(f^{-1}(y)) = y$ です。 
   ヒント: 導出ではこれらの性質を使ってください。 
