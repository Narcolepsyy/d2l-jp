{.python .input}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow', 'jax'])
```

# 自動微分
:label:`sec_autograd`

:numref:`sec_calculus` で見たように、
導関数の計算は、
深層ネットワークの学習に用いる
あらゆる最適化アルゴリズムにおいて
きわめて重要な段階である。
計算そのものは単純であっても、
手作業で求めるのは煩雑で誤りやすく、
モデルが複雑になるほど
その問題は深刻になる。

幸い、現代の深層学習フレームワークはいずれも
*自動微分*（しばしば *autograd* と略す）を備えており、
この煩雑な作業を自動化する。
データを各関数に順に通すと、
フレームワークは各値が他の値にどう依存するかを追跡する
*計算グラフ*を構築する。
導関数を計算する際には、
自動微分はこのグラフを逆向きにたどり、
連鎖律を適用する。
この連鎖律の適用に基づく計算アルゴリズムを
*逆伝播*と呼ぶ。

autograd ライブラリは
ここ10年ほどで大きな注目を集めてきたが、
その歴史は古く、
最初期の言及は
半世紀以上前にさかのぼる :cite:`Wengert.1964`。
現代的な逆伝播の中核となる考え方は
1980年の博士論文にまでさかのぼり :cite:`Speelpenning.1980`、
1980年代後半にさらに発展した :cite:`Griewank.1989`。
逆伝播は勾配計算の
標準的な方法となっているが、唯一の方法ではない。
たとえば、Julia プログラミング言語では
順方向微分が用いられている :cite:`Revels.Lubin.Papamarkou.2016`。
これらの方法を詳しく見る前に、
まずは autograd パッケージの基本的な使い方を学ぼう。

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
その勾配を格納する場所が必要である。**]
一般に、導関数を求めるたびに新たなメモリを割り当てるのは避ける。
深層学習では
同じパラメータに関する導関数を
何度も連続して計算する必要があり、
メモリ不足を招きかねないからである。
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

[**次に、`x` の関数を計算し、その結果を `y` に代入する。**]

```{.python .input  n=10}
%%tab mxnet
# 計算グラフを構築するため、コードは `autograd.record` スコープ内に置く
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
その後、`x` の `grad` 属性を通じて勾配にアクセスできる。
:end_tab:

:begin_tab:`pytorch`
[**これで `y` を `x` に関して微分できる**]。
`backward` メソッドを呼び出す。
その後、`x` の `grad` 属性を通じて勾配にアクセスできる。
:end_tab:

:begin_tab:`tensorflow`
[**これで `y` を `x` に関して微分できる**]。
`gradient` メソッドを呼び出す。
:end_tab:

:begin_tab:`jax`
[**これで `y` を `x` に関して微分できる**]。
`grad` 変換を用いる。
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
$4\mathbf{x}$ である。**]
したがって、自動的に計算された勾配が
期待どおりの結果と一致することを確認できる。

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
[**次に、`x` の別の関数を計算し、その勾配を求めよう。**]
MXNet では、新たに勾配を記録するたびに
勾配バッファがリセットされることに注意しよう。
:end_tab:

:begin_tab:`pytorch`
[**次に、`x` の別の関数を計算し、その勾配を求めよう。**]
PyTorch では、新たに勾配を記録しても
勾配バッファは自動ではリセットされない。
その代わり、新しい勾配は
すでに保存されている勾配に加算される。
この挙動は、
複数の目的関数の和を最適化したいときに便利である。
勾配バッファをリセットするには、
次のように `x.grad.zero_()` を呼び出す。
:end_tab:

:begin_tab:`tensorflow`
[**次に、`x` の別の関数を計算し、その勾配を求めよう。**]
TensorFlow では、新たに勾配を記録するたびに
勾配バッファがリセットされることに注意しよう。
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

`y` がベクトルであるとき、
`y` の `x` に関する導関数を表す最も自然な表現は、
*ヤコビアン*と呼ばれる行列である。
これは、`y` の各成分について
`x` の各成分に関する偏導関数を並べたものである。
同様に、`y` と `x` がより高次元であれば、
微分結果はさらに高次のテンソルになることもある。

ヤコビアンは
高度な機械学習手法で現れることもあるが、
より一般には、
`y` の各成分の勾配を
`x` の成分ごとに合計し、
`x` と同じ形状のベクトルを得たい場合が多い。
たとえば、訓練例の *バッチ* ごとに
個別に計算された損失関数の値を表すベクトルを
扱うことがよくある。
この場合に必要なのは、
[**各例ごとに個別に計算された勾配を足し合わせること**]
だけである。

:begin_tab:`mxnet`
MXNet では、勾配を計算する前に
すべてのテンソルを和によってスカラーへ縮約することで
この問題に対処する。
言い換えると、ヤコビアン
$\partial_{\mathbf{x}} \mathbf{y}$ を返す代わりに、
和の勾配
$\partial_{\mathbf{x}} \sum_i y_i$ を返す。
:end_tab:

:begin_tab:`pytorch`
深層学習フレームワークごとに
スカラーでないテンソルの勾配の扱いが異なるため、
PyTorch は混乱を避けるための仕組みをいくつか備えている。
スカラーでない対象に対して `backward` を呼び出すと、
それをどのようにスカラーへ縮約するかを PyTorch に伝えない限りエラーになる。
より厳密には、`backward` が
$\partial_{\mathbf{x}} \mathbf{y}$ ではなく
$\mathbf{v}^\top \partial_{\mathbf{x}} \mathbf{y}$ を計算するようにするための
ベクトル $\mathbf{v}$ を与える必要がある。
ここは少し分かりにくいかもしれないが、
後に明らかになる理由から、
この引数（$\mathbf{v}$ を表すもの）は `gradient` と名付けられている。
詳しくは、Yang Zhang による
[Medium の記事](https://zhang-yang.medium.com/the-gradient-argument-in-pytorchs-backward-function-explained-by-examples-68f266950c29) を参照されたい。
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

ときには、[**一部の計算を
記録された計算グラフの外に置きたい**]ことがある。
たとえば、入力を用いて
補助的な中間項を作るが、
その項については勾配を計算したくないとする。
この場合、最終結果から
それぞれの計算グラフを *切り離す* 必要がある。
次の簡単な例でこれを明確にしよう。
`z = x * y` かつ `y = x * x` であるが、
`y` を介して伝わる影響ではなく、
`z` に対する `x` の *直接的* な影響だけに注目したいとする。
このとき、`y` と同じ値をもつ新しい変数 `u` を作れるが、
その *来歴*（どのように生成されたか）は消去される。
したがって `u` はグラフ内に祖先をもたず、
勾配は `u` を通って `x` へ流れない。
たとえば、`z = x * u` の勾配を取ると、
結果は `u` になる
（`z = x * x * x` なので `3 * x * x` になると
予想するかもしれないが、そうはならない）。

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
`z` に至るグラフから `y` の祖先を
切り離すが、
`y` に至る計算グラフそのものは
残っているので、`x` に関する `y` の勾配は
引き続き計算できる。

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

これまでは、入力から出力までの経路が
`z = x * x * x` のような関数によって
明示的に定義されている場合を見てきた。
しかし実際のプログラミングでは、
結果の計算方法に
はるかに大きな自由度がある。
たとえば、補助変数に依存させたり、
中間結果に応じて条件分岐したりできる。
自動微分の利点の一つは、
[**計算グラフの構築に
Python の複雑な制御フローを通る必要があっても**]
（たとえば条件分岐、ループ、任意の関数呼び出し）、
[**最終的に得られた変数の勾配を計算できることにある。**]
これを示すために、次のコード片を考えよう。
ここでは `while` ループの反復回数と
`if` 文の評価結果の両方が
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

以下では、この関数にランダムな値を入力として与えて呼び出す。
入力は確率変数なので、
計算グラフがどのような形になるかは事前には分からない。
しかし、特定の入力に対して `f(a)` を実行するたびに、
具体的な計算グラフが実現され、
その後で `backward` を実行できる。

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

この `f` はデモのためにやや作為的に作った関数であるが、
入力への依存関係はきわめて単純である。
これは、区分的に定義されたスケールをもつ
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
このような場合、自動微分は統計モデリングに不可欠である。
勾配を *a priori* に計算することは不可能だからである。

## 議論

ここまでで、自動微分の力をある程度体感できたはずである。
導関数を自動かつ効率的に計算するライブラリの発展は、
深層学習の実践者の生産性を大きく高め、
単純作業ではない本質的な問題に集中できるようにした。
さらに、autograd を使えば、
紙と鉛筆で手計算していては
膨大な時間を要するような巨大なモデルも設計できる。
興味深いことに、autograd を用いてモデルを
（統計的な意味で）*最適化*する一方で、
autograd ライブラリ自体の
（計算機科学的な意味での）*最適化*も、
フレームワーク設計者にとって
きわめて重要な研究課題である。
そこでは、コンパイラやグラフ操作の技術を用いて、
最も高速かつメモリ効率のよい方法で結果を計算する。

現時点では、次の基本を押さえておけばよい。
(i) 導関数を求めたい変数に勾配を関連付ける;
(ii) 目的値の計算を記録する;
(iii) 逆伝播関数を実行する;
(iv) 得られた勾配にアクセスする。


## 演習

1. 2階導関数の計算が1階導関数よりはるかに高コストなのはなぜか。
1. 逆伝播関数を実行した直後に、もう一度実行してみよ。何が起こるか調べよ。
1. `d` を `a` に関して微分する制御フローの例で、変数 `a` をランダムなベクトルや行列に変えたらどうなるだろうか。この時点では、`f(a)` の計算結果はもはやスカラーではない。結果はどうなるか。どう分析すべきだろうか。
1. $f(x) = \sin(x)$ とする。$f$ のグラフとその導関数 $f'$ のグラフを描け。$f'(x) = \cos(x)$ は使わず、自動微分によって結果を得よ。
1. $f(x) = ((\log x^2) \cdot \sin x) + x^{-1}$ とする。$x$ から $f(x)$ までの結果をたどる依存グラフを書け。
1. 連鎖律を用いて、前述の関数の導関数 $\frac{df}{dx}$ を計算し、先ほど構築した依存グラフ上の各項に対応付けよ。
1. グラフと中間導関数の結果が与えられたとき、勾配を計算する方法はいくつかある。$x$ から $f$ へ向かって一度計算し、さらに $f$ から $x$ に向かってもう一度たどって計算せよ。$x$ から $f$ への経路は一般に *順方向微分* と呼ばれ、$f$ から $x$ への経路は逆方向微分と呼ばれる。
1. いつ順方向微分を使い、いつ逆方向微分を使うべきだろうか。ヒント: 必要な中間データ量、各段階の並列化可能性、関係する行列やベクトルのサイズを考えよ。