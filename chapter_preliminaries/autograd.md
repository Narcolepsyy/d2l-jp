{.python .input}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow', 'jax'])
```

# 自動微分
:label:`sec_autograd`

:numref:`sec_calculus` で思い出したように、
導関数を計算することは、
深層ネットワークを学習させるために用いる
すべての最適化アルゴリズムにおいて
極めて重要なステップである。
計算自体は単純であるが、
手作業で求めるのは面倒で誤りも起こりやすく、
モデルが複雑になるほど
こうした問題はさらに大きくなる。

幸いなことに、現代のあらゆる深層学習フレームワークは、*自動微分*（しばしば *autograd* と略される）を提供し、この手間のかかる作業を自動化してくれる。
データを各関数へ順に通していくと、
フレームワークは、各値が他の値にどのように依存しているかを追跡する
*計算グラフ*を構築する。
導関数を計算するには、
自動微分はこのグラフを逆向きにたどり、
連鎖律を適用する。
このように連鎖律を適用する計算アルゴリズムは
*逆伝播*と呼ばれる。

autograd ライブラリは
ここ10年ほどで大きな注目を集めてきたが、
その歴史は長く、
最初期の autograd に関する言及は
半世紀以上前にさかのぼります :cite:`Wengert.1964`。
現代の逆伝播の核となる考え方は
1980年の博士論文にまでさかのぼり :cite:`Speelpenning.1980`、
1980年代後半にさらに発展した :cite:`Griewank.1989`。
逆伝播は勾配を計算するための
標準的な方法になっているが、唯一の選択肢ではない。
たとえば、Julia プログラミング言語では
順伝播が用いられている :cite:`Revels.Lubin.Papamarkou.2016`。
方法を探る前に、
まずは autograd パッケージを使いこなしよう。

```{.python .input}
%%tab mxnet
from mxnet import autograd, np, npx
npx.set_np()
```

```{.python .input}
%%tab pytorch
import torch
```

```{.python .input}
%%tab tensorflow
import tensorflow as tf
```

```{.python .input}
%%tab jax
from jax import numpy as jnp
```

## 単純な関数

[**関数 $y = 2\mathbf{x}^{\top}\mathbf{x}$ を、
列ベクトル $\mathbf{x}$ に関して微分したい**]
としよう。
まず、`x` に初期値を割り当てる。

```{.python .input  n=1}
%%tab mxnet
x = np.arange(4.0)
x
```

```{.python .input  n=7}
%%tab pytorch
x = torch.arange(4.0)
x
```

```{.python .input}
%%tab tensorflow
x = tf.range(4, dtype=tf.float32)
x
```

```{.python .input}
%%tab jax
x = jnp.arange(4.0)
x
```

:begin_tab:`mxnet, pytorch, tensorflow`
[**`y` を $\mathbf{x}$ に関して微分する前に、
その勾配を保存する場所が必要である。**]
一般に、導関数を求めるたびに新しいメモリを割り当てることは避ける。
というのも、深層学習では
同じパラメータに関する導関数を
何度も連続して計算する必要があり、
メモリ不足に陥る危険があるからである。
スカラー値関数のベクトル $\mathbf{x}$ に関する勾配は、
$\mathbf{x}$ と同じ形状をもつベクトル値になる。
:end_tab:

```{.python .input  n=8}
%%tab mxnet
# `attach_grad` を呼び出して、テンソルの勾配用メモリを確保する
x.attach_grad()
# `x` に関する勾配を計算した後は、`grad` 属性を通じて
# その値にアクセスできる。初期値は 0 である
x.grad
```

```{.python .input  n=9}
%%tab pytorch
# x = torch.arange(4.0, requires_grad=True) としてもよい
x.requires_grad_(True)
x.grad  # 勾配の初期値はデフォルトで None
```

```{.python .input}
%%tab tensorflow
x = tf.Variable(x)
```

[**次に、`x` の関数を計算して、その結果を `y` に代入する。**]

```{.python .input  n=10}
%%tab mxnet
# 計算グラフを構築するため、コードは `autograd.record` スコープ内にある
with autograd.record():
    y = 2 * np.dot(x, x)
y
```

```{.python .input  n=11}
%%tab pytorch
y = 2 * torch.dot(x, x)
y
```

```{.python .input}
%%tab tensorflow
# すべての計算をテープに記録する
with tf.GradientTape() as t:
    y = 2 * tf.tensordot(x, x, axes=1)
y
```

```{.python .input}
%%tab jax
y = lambda x: 2 * jnp.dot(x, x)
y(x)
```

:begin_tab:`mxnet`
[**これで `y` を `x` に関して微分できる**]。
`backward` メソッドを呼び出す。
次に、`x` の `grad` 属性を通じて勾配にアクセスできる。
:end_tab:

:begin_tab:`pytorch`
[**これで `y` を `x` に関して微分できる**]。
`backward` メソッドを呼び出す。
次に、`x` の `grad` 属性を通じて勾配にアクセスできる。
:end_tab:

:begin_tab:`tensorflow`
[**これで `y` を `x` に関して微分できる**]。
`gradient` メソッドを呼び出す。
:end_tab:

:begin_tab:`jax`
[**これで `y` を `x` に関して微分できる**]。
`grad` 変換を通して行う。
:end_tab:

```{.python .input}
%%tab mxnet
y.backward()
x.grad
```

```{.python .input  n=12}
%%tab pytorch
y.backward()
x.grad
```

```{.python .input}
%%tab tensorflow
x_grad = t.gradient(y, x)
x_grad
```

```{.python .input}
%%tab jax
from jax import grad
# `grad` 変換は、元の関数の勾配を計算する Python 関数を返す
x_grad = grad(y)(x)
x_grad
```

[**関数 $y = 2\mathbf{x}^{\top}\mathbf{x}$ の $\mathbf{x}$ に関する勾配は
$4\mathbf{x}$ になることはすでに分かっている。**]
これで、自動的に計算された勾配と
期待される結果が一致することを確認できる。

```{.python .input  n=13}
%%tab mxnet
x.grad == 4 * x
```

```{.python .input  n=14}
%%tab pytorch
x.grad == 4 * x
```

```{.python .input}
%%tab tensorflow
x_grad == 4 * x
```

```{.python .input}
%%tab jax
x_grad == 4 * x
```

:begin_tab:`mxnet`
[**では、`x` の別の関数を計算して、その勾配を求めよう。**]
MXNet では、新しい勾配を記録するたびに
勾配バッファがリセットされることに注意しよ。
:end_tab:

:begin_tab:`pytorch`
[**では、`x` の別の関数を計算して、その勾配を求めよう。**]
PyTorch では、新しい勾配を記録しても
勾配バッファは自動的にはリセットされない。
その代わり、新しい勾配は
すでに保存されている勾配に加算される。
この挙動は、
複数の目的関数の和を最適化したいときに便利である。
勾配バッファをリセットするには、
次のように `x.grad.zero_()` を呼び出す。
:end_tab:

:begin_tab:`tensorflow`
[**では、`x` の別の関数を計算して、その勾配を求めよう。**]
TensorFlow では、新しい勾配を記録するたびに
勾配バッファがリセットされることに注意しよ。
:end_tab:

```{.python .input}
%%tab mxnet
with autograd.record():
    y = x.sum()
y.backward()
x.grad  # 新しく計算された勾配で上書きされる
```

```{.python .input  n=20}
%%tab pytorch
x.grad.zero_()  # 勾配をリセットする
y = x.sum()
y.backward()
x.grad
```

```{.python .input}
%%tab tensorflow
with tf.GradientTape() as t:
    y = tf.reduce_sum(x)
t.gradient(y, x)  # 新しく計算された勾配で上書きされる
```

```{.python .input}
%%tab jax
y = lambda x: x.sum()
grad(y)(x)
```

## スカラーでない変数の逆伝播

`y` がベクトルのとき、
`y` の `x` に関する導関数を表す最も自然な表現は、
*ヤコビアン*と呼ばれる行列である。
これは、`y` の各成分について
`x` の各成分に関する偏導関数を含む。
同様に、`y` と `x` がより高次元であれば、
微分の結果はさらに高次のテンソルになることもある。

ヤコビアンは
いくつかの高度な機械学習技法では現れるが、
より一般的には、
`y` の各成分の勾配を
`x` の成分ごとに合計して、
`x` と同じ形状のベクトルを得たいことが多いである。
たとえば、訓練例の *バッチ* ごとに
別々に計算された損失関数の値を表すベクトルを
扱うことがよくある。
この場合、私たちが欲しいのは
[**各例ごとに個別に計算された勾配を足し合わせること**]
だけである。

:begin_tab:`mxnet`
MXNet では、勾配を計算する前に
すべてのテンソルを和でスカラーに縮約することで
この問題に対処する。
言い換えると、ヤコビアン
$\partial_{\mathbf{x}} \mathbf{y}$ を返す代わりに、
和の勾配
$\partial_{\mathbf{x}} \sum_i y_i$ を返す。
:end_tab:

:begin_tab:`pytorch`
深層学習フレームワークは
スカラーでないテンソルの勾配の解釈がそれぞれ異なるため、
PyTorch は混乱を避けるための手順をいくつか用意している。
スカラーでない対象に対して `backward` を呼び出すと、
それをスカラーに縮約する方法を PyTorch に伝えない限りエラーになる。
より形式的には、`backward` が
$\partial_{\mathbf{x}} \mathbf{y}$ ではなく
$\mathbf{v}^\top \partial_{\mathbf{x}} \mathbf{y}$ を計算するようにするための
ベクトル $\mathbf{v}$ を与える必要がある。
この先は少し分かりにくいかもしれないが、
後で明らかになる理由により、
この引数（$\mathbf{v}$ を表すもの）は `gradient` と名付けられている。
より詳しい説明は、Yang Zhang の
[Medium の記事](https://zhang-yang.medium.com/the-gradient-argument-in-pytorchs-backward-function-explained-by-examples-68f266950c29) を参照しよ。
:end_tab:

:begin_tab:`tensorflow`
デフォルトでは、TensorFlow は和の勾配を返す。
言い換えると、ヤコビアン
$\partial_{\mathbf{x}} \mathbf{y}$ を返す代わりに、
和の勾配
$\partial_{\mathbf{x}} \sum_i y_i$ を返す。
:end_tab:

```{.python .input}
%%tab mxnet
with autograd.record():
    y = x * x  
y.backward()
x.grad  # y = sum(x * x) の勾配に等しい
```

```{.python .input}
%%tab pytorch
x.grad.zero_()
y = x * x
y.backward(gradient=torch.ones(len(y)))  # より高速: y.sum().backward()
x.grad
```

```{.python .input}
%%tab tensorflow
with tf.GradientTape() as t:
    y = x * x
t.gradient(y, x)  # y = tf.reduce_sum(x * x) と同じ
```

```{.python .input}
%%tab jax
y = lambda x: x * x
# grad はスカラー出力関数に対してのみ定義される
grad(lambda x: y(x).sum())(x)
```

## 計算の切り離し

ときには、[**いくつかの計算を
記録された計算グラフの外に出したい**]ことがある。
たとえば、入力を使って
補助的な中間項を作るが、
その項については勾配を計算したくないとする。
この場合、最終結果から
それぞれの計算グラフを *切り離す* 必要がある。
次の簡単な例でこれを明確にしよう。
`z = x * y` かつ `y = x * x` であるが、
`y` を介して伝わる影響ではなく、
`z` に対する `x` の *直接的* な影響に注目したいとする。
この場合、`y` と同じ値を持つ新しい変数 `u` を作れるが、
その *来歴*（どのように生成されたか）は消去される。
したがって `u` はグラフ内に祖先を持たず、
勾配は `u` を通って `x` へ流れない。
たとえば、`z = x * u` の勾配を取ると、
結果は `u` になる
（`z = x * x * x` だから `3 * x * x` になると
予想したかもしれないが、そうはならない）。

```{.python .input}
%%tab mxnet
with autograd.record():
    y = x * x
    u = y.detach()
    z = u * x
z.backward()
x.grad == u
```

```{.python .input  n=21}
%%tab pytorch
x.grad.zero_()
y = x * x
u = y.detach()
z = u * x

z.sum().backward()
x.grad == u
```

```{.python .input}
%%tab tensorflow
# 計算グラフを保持するために persistent=True を設定する。
# これにより t.gradient を複数回実行できる
with tf.GradientTape(persistent=True) as t:
    y = x * x
    u = tf.stop_gradient(y)
    z = u * x

x_grad = t.gradient(z, x)
x_grad == u
```

```{.python .input}
%%tab jax
import jax

y = lambda x: x * x
# jax.lax のプリミティブは XLA 演算の Python ラッパーである
u = jax.lax.stop_gradient(y(x))
z = lambda x: u * x

grad(lambda x: z(x).sum())(x) == y(x)
```

この手順は
`z` へ至るグラフから `y` の祖先を
切り離するが、
`y` へ至る計算グラフ自体は
残っているので、`x` に関する `y` の勾配を
計算できる。

```{.python .input}
%%tab mxnet
y.backward()
x.grad == 2 * x
```

```{.python .input}
%%tab pytorch
x.grad.zero_()
y.sum().backward()
x.grad == 2 * x
```

```{.python .input}
%%tab tensorflow
t.gradient(y, x) == 2 * x
```

```{.python .input}
%%tab jax
grad(lambda x: y(x).sum())(x) == 2 * x
```

## 勾配と Python の制御フロー

これまで、入力から出力までの経路が
`z = x * x * x` のような関数によって
明確に定義されている場合を見てきた。
プログラミングでは、結果の計算方法に
もっと自由度がある。
たとえば、補助変数に依存させたり、
中間結果に応じて条件分岐を行ったりできる。
自動微分を使う利点の一つは、
[**たとえ計算グラフを構築するのに
Python の制御フローの迷路を通り抜ける必要があっても**]
（たとえば条件分岐、ループ、任意の関数呼び出し）、
[**最終的に得られる変数の勾配を計算できることである。**]
これを示すために、次のコード片を考えよう。
ここでは `while` ループの反復回数と
`if` 文の評価の両方が
入力 `a` の値に依存している。

```{.python .input}
%%tab mxnet
def f(a):
    b = a * 2
    while np.linalg.norm(b) < 1000:
        b = b * 2
    if b.sum() > 0:
        c = b
    else:
        c = 100 * b
    return c
```

```{.python .input}
%%tab pytorch
def f(a):
    b = a * 2
    while b.norm() < 1000:
        b = b * 2
    if b.sum() > 0:
        c = b
    else:
        c = 100 * b
    return c
```

```{.python .input}
%%tab tensorflow
def f(a):
    b = a * 2
    while tf.norm(b) < 1000:
        b = b * 2
    if tf.reduce_sum(b) > 0:
        c = b
    else:
        c = 100 * b
    return c
```

```{.python .input}
%%tab jax
def f(a):
    b = a * 2
    while jnp.linalg.norm(b) < 1000:
        b = b * 2
    if b.sum() > 0:
        c = b
    else:
        c = 100 * b
    return c
```

以下では、この関数にランダムな値を入力として渡して呼び出す。
入力は確率変数なので、
計算グラフがどのような形になるかは分からない。
しかし、特定の入力に対して `f(a)` を実行するたびに、
特定の計算グラフが実現され、
その後 `backward` を実行できる。

```{.python .input}
%%tab mxnet
a = np.random.normal()
a.attach_grad()
with autograd.record():
    d = f(a)
d.backward()
```

```{.python .input}
%%tab pytorch
a = torch.randn(size=(), requires_grad=True)
d = f(a)
d.backward()
```

```{.python .input}
%%tab tensorflow
a = tf.Variable(tf.random.normal(shape=()))
with tf.GradientTape() as t:
    d = f(a)
d_grad = t.gradient(d, a)
d_grad
```

```{.python .input}
%%tab jax
from jax import random
a = random.normal(random.PRNGKey(1), ())
d = f(a)
d_grad = grad(f)(a)
```

デモのために少し作為的な関数 `f` であるが、
入力への依存は非常に単純である。
それは、区分的に定義されたスケールをもつ
`a` の *線形* 関数である。
したがって、`f(a) / a` は定数成分からなるベクトルであり、
さらに `f(a) / a` は
`a` に関する `f(a)` の勾配と一致するはずである。

```{.python .input}
%%tab mxnet
a.grad == d / a
```

```{.python .input}
%%tab pytorch
a.grad == d / a
```

```{.python .input}
%%tab tensorflow
d_grad == d / a
```

```{.python .input}
%%tab jax
d_grad == d / a
```

動的な制御フローは深層学習で非常に一般的である。
たとえば、テキストを処理するとき、
計算グラフは入力の長さに依存する。
このような場合、自動微分は統計モデリングにとって不可欠である。
というのも、勾配を *a priori* に計算することは不可能だからである。

## 議論

これで、自動微分の力を少し体験できたはずである。
導関数を自動的かつ効率的に計算するライブラリの発展は、
深層学習の実践者にとって
生産性を大きく高めるものであり、
彼らが単純作業ではないことに集中できるようにしてくれた。
さらに、autograd を使えば、
紙と鉛筆での手計算では
膨大な時間がかかりすぎるような巨大なモデルも設計できる。
興味深いことに、私たちは autograd を使ってモデルを
（統計的な意味で）*最適化*するが、
autograd ライブラリ自体の
（計算機科学的な意味での）*最適化*は、
フレームワーク設計者にとって
非常に重要な関心事である
豊かな研究分野である。
ここでは、コンパイラやグラフ操作の技術が用いられ、
最も迅速でメモリ効率のよい方法で結果を計算する。

今のところ、次の基本を覚えておきよう。
(i) 導関数を求めたい変数に勾配を付与する;
(ii) 目的値の計算を記録する;
(iii) 逆伝播関数を実行する;
(iv) 得られた勾配にアクセスする。


## 演習

1. 2階導関数を計算するのが1階導関数よりはるかに高コストなのはなぜですか。
1. 逆伝播の関数を実行した直後に、もう一度実行してみよ。何が起こるか調べよ。
1. `d` を `a` に関して微分する制御フローの例で、変数 `a` をランダムなベクトルや行列に変えたらどうなるだろうか。この時点で、`f(a)` の計算結果はもはやスカラーではない。結果はどうなるか。どう分析すればよいだろうか。
1. $f(x) = \sin(x)$ とする。$f$ のグラフとその導関数 $f'$ のグラフを描いよ。$f'(x) = \cos(x)$ であることは使わず、自動微分を用いて結果を得よ。 
1. $f(x) = ((\log x^2) \cdot \sin x) + x^{-1}$ とする。$x$ から $f(x)$ までの結果をたどる依存グラフを書いよ。 
1. 連鎖律を用いて、前述の関数の導関数 $\frac{df}{dx}$ を計算し、先ほど構築した依存グラフ上の各項に対応付けよ。 
1. グラフと中間導関数の結果が与えられたとき、勾配を計算する方法はいくつかある。$x$ から $f$ へ向かって一度計算し、もう一度 $f$ から $x$ に向かってたどって計算しよ。$x$ から $f$ への経路は一般に *順方向微分* として知られ、$f$ から $x$ への経路は逆方向微分として知られている。 
1. いつ順方向微分を使い、いつ逆方向微分を使うべきだろうか。ヒント: 必要な中間データ量、各ステップの並列化可能性、関係する行列やベクトルのサイズを考えよ。 
