{.python .input}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow', 'jax'])
```

# 線形代数
:label:`sec_linear-algebra`

ここまでで、データセットをテンソルに読み込み、
基本的な数学演算でこれらのテンソルを操作できるようになりました。
より洗練されたモデルを構築し始めるには、
線形代数のいくつかの道具も必要になります。
この節では、スカラー演算から始めて行列積へと進みながら、
最も重要な概念をやさしく導入します。

```{.python .input}
%%tab mxnet
from mxnet import np, npx
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

## スカラー


日常的な数学の大部分は、
数を一つずつ操作することから成り立っています。
形式的には、これらの値を *スカラー* と呼びます。
たとえば、パロアルトの気温は
穏やかな華氏 $72$ 度です。
気温を摂氏に変換したければ、
$f$ を $72$ に設定して式
$c = \frac{5}{9}(f - 32)$ を評価します。
この式では、$5$、$9$、$32$ は定数スカラーです。
変数 $c$ と $f$ は一般に未知のスカラーを表します。

スカラーは通常の小文字
（たとえば $x$、$y$、$z$）
で表し、すべての（連続な）
*実数値* スカラーの空間を $\mathbb{R}$ で表します。
手短にするため、*空間* の厳密な定義は省略します。
ここでは、式 $x \in \mathbb{R}$ は
$x$ が実数値スカラーであることを表す形式的な書き方だと覚えておいてください。
記号 $\in$（「〜に属する」と読みます）は、
集合への所属を表します。
たとえば、$x, y \in \{0, 1\}$ は、
$x$ と $y$ が $0$ か $1$ しか取れない変数であることを示します。

[**スカラーは、1つの要素だけを含むテンソルとして実装されます。**]
以下では、2つのスカラーを代入し、
おなじみの加算、乗算、除算、べき乗を行います。

```{.python .input}
%%tab mxnet
x = np.array(3.0)
y = np.array(2.0)

x + y, x * y, x / y, x ** y
```

```{.python .input}
%%tab pytorch
x = torch.tensor(3.0)
y = torch.tensor(2.0)

x + y, x * y, x / y, x**y
```

```{.python .input}
%%tab tensorflow
x = tf.constant(3.0)
y = tf.constant(2.0)

x + y, x * y, x / y, x**y
```

```{.python .input}
%%tab jax
x = jnp.array(3.0)
y = jnp.array(2.0)

x + y, x * y, x / y, x**y
```

## ベクトル

現時点では、[**ベクトルはスカラーの固定長配列だと考えてよいでしょう。**]
コード上の対応物と同様に、
これらのスカラーをベクトルの *要素* と呼びます
（同義語として *エントリ* や *成分* があります）。
ベクトルが実世界のデータセットの例を表すとき、
その値には何らかの現実世界での意味があります。
たとえば、ローンのデフォルトリスクを予測するモデルを学習しているなら、
各申請者をベクトルに対応付け、
その成分は収入、勤務年数、
過去のデフォルト回数などの量に対応するかもしれません。
心臓発作のリスクを研究しているなら、
各ベクトルは患者を表し、
その成分は直近のバイタルサイン、コレステロール値、
1日あたりの運動時間などに対応するかもしれません。
ベクトルは太字の小文字
（たとえば $\mathbf{x}$、$\mathbf{y}$、$\mathbf{z}$）
で表します。

ベクトルは $1^{\textrm{st}}$-order テンソルとして実装されます。
一般に、このようなテンソルはメモリ制約の範囲で任意の長さを持てます。注意: Python では、ほとんどのプログラミング言語と同様に、ベクトルの添字は $0$ から始まります。これは *ゼロ始まりのインデックス付け* とも呼ばれます。一方、線形代数では添字は $1$ から始まります（1始まりのインデックス付け）。

```{.python .input}
%%tab mxnet
x = np.arange(3)
x
```

```{.python .input}
%%tab pytorch
x = torch.arange(3)
x
```

```{.python .input}
%%tab tensorflow
x = tf.range(3)
x
```

```{.python .input}
%%tab jax
x = jnp.arange(3)
x
```

添字を使ってベクトルの要素を参照できます。
たとえば、$x_2$ は $\mathbf{x}$ の2番目の要素を表します。
$x_2$ はスカラーなので、太字にはしません。
デフォルトでは、ベクトルは要素を縦に並べて可視化します。

$$\mathbf{x} =\begin{bmatrix}x_{1}  \\ \vdots  \\x_{n}\end{bmatrix}.$$
:eqlabel:`eq_vec_def`

ここで $x_1, \ldots, x_n$ はベクトルの要素です。
後で、こうした *列ベクトル* と、要素を横に並べた *行ベクトル* を区別します。
[**テンソルの要素にはインデックスでアクセスする**] ことを思い出してください。

```{.python .input}
%%tab all
x[2]
```

ベクトルが $n$ 個の要素を含むことを示すには、
$\mathbf{x} \in \mathbb{R}^n$ と書きます。
形式的には、$n$ をベクトルの *次元数* と呼びます。
[**コードでは、これはテンソルの長さに対応します**]。
Python の組み込み関数 `len` で取得できます。

```{.python .input}
%%tab all
len(x)
```

長さは `shape` 属性でも取得できます。
shape は、各軸に沿ったテンソルの長さを示すタプルです。
[**1つの軸しか持たないテンソルの shape は、1つの要素だけを持ちます。**]

```{.python .input}
%%tab all
x.shape
```

しばしば「次元」という語は、
軸の数と特定の軸に沿った長さの両方を指すために
曖昧に使われます。
この混乱を避けるため、
軸の数を指すときには *階数* を用い、
要素数を指すときには *次元数* を専ら用います。


## 行列

スカラーが $0^{\textrm{th}}$-order テンソルであり、
ベクトルが $1^{\textrm{st}}$-order テンソルであるのと同様に、
行列は $2^{\textrm{nd}}$-order テンソルです。
行列は太字の大文字
（たとえば $\mathbf{X}$、$\mathbf{Y}$、$\mathbf{Z}$）
で表し、コードでは2つの軸を持つテンソルとして表現します。
式 $\mathbf{A} \in \mathbb{R}^{m \times n}$ は、
行列 $\mathbf{A}$ が $m \times n$ 個の実数値スカラーを含み、
$m$ 行 $n$ 列に並んでいることを示します。
$m = n$ のとき、その行列は *正方* 行列と呼びます。
視覚的には、任意の行列を表として示せます。
個々の要素を参照するには、
行と列の両方の添字を付けます。たとえば、
$a_{ij}$ は $\mathbf{A}$ の
$i^{\textrm{th}}$ 行 $j^{\textrm{th}}$ 列に属する値です。

$$\mathbf{A}=\begin{bmatrix} a_{11} & a_{12} & \cdots & a_{1n} \\ a_{21} & a_{22} & \cdots & a_{2n} \\ \vdots & \vdots & \ddots & \vdots \\ a_{m1} & a_{m2} & \cdots & a_{mn} \\ \end{bmatrix}.$$
:eqlabel:`eq_matrix_def`


コードでは、行列 $\mathbf{A} \in \mathbb{R}^{m \times n}$ を
shape が ($m$, $n$) の $2^{\textrm{nd}}$-order テンソルとして表します。
[**任意の適切なサイズの $m \times n$ テンソルを
$m \times n$ 行列に変換できます**]
`reshape` に希望の shape を渡すことで実現できます。

```{.python .input}
%%tab mxnet
A = np.arange(6).reshape(3, 2)
A
```

```{.python .input}
%%tab pytorch
A = torch.arange(6).reshape(3, 2)
A
```

```{.python .input}
%%tab tensorflow
A = tf.reshape(tf.range(6), (3, 2))
A
```

```{.python .input}
%%tab jax
A = jnp.arange(6).reshape(3, 2)
A
```

ときには軸を入れ替えたいことがあります。
行列の行と列を交換した結果を、その *転置* と呼びます。
形式的には、行列 $\mathbf{A}$ の転置を $\mathbf{A}^\top$ で表し、
$\mathbf{B} = \mathbf{A}^\top$ ならば、すべての $i$ と $j$ について
$b_{ij} = a_{ji}$ です。
したがって、$m \times n$ 行列の転置は
$n \times m$ 行列になります。

$$
\mathbf{A}^\top =
\begin{bmatrix}
    a_{11} & a_{21} & \dots  & a_{m1} \\
    a_{12} & a_{22} & \dots  & a_{m2} \\
    \vdots & \vdots & \ddots  & \vdots \\
    a_{1n} & a_{2n} & \dots  & a_{mn}
\end{bmatrix}.
$$

コードでは、任意の[**行列の転置**]を次のように取得できます。

```{.python .input}
%%tab mxnet, pytorch, jax
A.T
```

```{.python .input}
%%tab tensorflow
tf.transpose(A)
```

[**対称行列は、正方行列のうち自分自身の転置に等しいものです:
$\mathbf{A} = \mathbf{A}^\top$.**]
次の行列は対称です。

```{.python .input}
%%tab mxnet
A = np.array([[1, 2, 3], [2, 0, 4], [3, 4, 5]])
A == A.T
```

```{.python .input}
%%tab pytorch
A = torch.tensor([[1, 2, 3], [2, 0, 4], [3, 4, 5]])
A == A.T
```

```{.python .input}
%%tab tensorflow
A = tf.constant([[1, 2, 3], [2, 0, 4], [3, 4, 5]])
A == tf.transpose(A)
```

```{.python .input}
%%tab jax
A = jnp.array([[1, 2, 3], [2, 0, 4], [3, 4, 5]])
A == A.T
```

行列はデータセットを表現するのに便利です。
通常、行は個々の記録に対応し、
列は異なる属性に対応します。



## テンソル

スカラー、ベクトル、行列だけでも
機械学習の道のりをかなり進めますが、
やがてはより高階の [**テンソル**] を扱う必要が出てきます。
テンソルは[**$n^{\textrm{th}}$-order 配列への拡張を
一般的に記述する方法を与えてくれます。**]
*テンソルクラス* のソフトウェアオブジェクトを「テンソル」と呼ぶのは、
それらも任意の数の軸を持てるからです。
数学的対象としての *テンソル* と、
コード上での実装を同じ語で呼ぶのは紛らわしいかもしれませんが、
通常は文脈から意味が明らかです。
一般のテンソルは特別な書体の大文字
（たとえば $\mathsf{X}$、$\mathsf{Y}$、$\mathsf{Z}$）
で表し、そのインデックス付けの仕組み
（たとえば $x_{ijk}$ や $[\mathsf{X}]_{1, 2i-1, 3}$）
は行列のそれから自然に拡張されます。

画像を扱い始めると、テンソルはさらに重要になります。
各画像は、高さ、幅、*チャネル* に対応する軸を持つ
$3^{\textrm{rd}}$-order テンソルとして与えられます。
各空間位置では、各色（赤、緑、青）の強度が
チャネル方向に並べられます。
さらに、画像の集合はコード上では
$4^{\textrm{th}}$-order テンソルとして表され、
個々の画像は第1軸に沿ってインデックス付けされます。
高階テンソルは、ベクトルや行列と同様に、
shape の成分数を増やすことで構成されます。

```{.python .input}
%%tab mxnet
np.arange(24).reshape(2, 3, 4)
```

```{.python .input}
%%tab pytorch
torch.arange(24).reshape(2, 3, 4)
```

```{.python .input}
%%tab tensorflow
tf.reshape(tf.range(24), (2, 3, 4))
```

```{.python .input}
%%tab jax
jnp.arange(24).reshape(2, 3, 4)
```

## テンソル演算の基本的性質

スカラー、ベクトル、行列、
および高階テンソルには、
便利な性質がいくつかあります。
たとえば、要素ごとの演算は、
入力と同じ shape を持つ出力を生成します。

```{.python .input}
%%tab mxnet
A = np.arange(6).reshape(2, 3)
B = A.copy()  # Assign a copy of A to B by allocating new memory
A, A + B
```

```{.python .input}
%%tab pytorch
A = torch.arange(6, dtype=torch.float32).reshape(2, 3)
B = A.clone()  # Assign a copy of A to B by allocating new memory
A, A + B
```

```{.python .input}
%%tab tensorflow
A = tf.reshape(tf.range(6, dtype=tf.float32), (2, 3))
B = A  # No cloning of A to B by allocating new memory
A, A + B
```

```{.python .input}
%%tab jax
A = jnp.arange(6, dtype=jnp.float32).reshape(2, 3)
B = A
A, A + B
```

[**2つの行列の要素ごとの積は *Hadamard 積* と呼ばれます**]（$\odot$ で表します）。
2つの行列 $\mathbf{A}, \mathbf{B} \in \mathbb{R}^{m \times n}$ の
Hadamard 積の各要素は次のように書けます。



$$
\mathbf{A} \odot \mathbf{B} =
\begin{bmatrix}
    a_{11}  b_{11} & a_{12}  b_{12} & \dots  & a_{1n}  b_{1n} \\
    a_{21}  b_{21} & a_{22}  b_{22} & \dots  & a_{2n}  b_{2n} \\
    \vdots & \vdots & \ddots & \vdots \\
    a_{m1}  b_{m1} & a_{m2}  b_{m2} & \dots  & a_{mn}  b_{mn}
\end{bmatrix}.
$$

```{.python .input}
%%tab all
A * B
```

[**スカラーとテンソルの加算や乗算**] は、元のテンソルと同じ shape の結果を生成します。
ここでは、テンソルの各要素にスカラーを加算（または乗算）しています。

```{.python .input}
%%tab mxnet
a = 2
X = np.arange(24).reshape(2, 3, 4)
a + X, (a * X).shape
```

```{.python .input}
%%tab pytorch
a = 2
X = torch.arange(24).reshape(2, 3, 4)
a + X, (a * X).shape
```

```{.python .input}
%%tab tensorflow
a = 2
X = tf.reshape(tf.range(24), (2, 3, 4))
a + X, (a * X).shape
```

```{.python .input}
%%tab jax
a = 2
X = jnp.arange(24).reshape(2, 3, 4)
a + X, (a * X).shape
```

## リダクション
:label:`subsec_lin-alg-reduction`

しばしば、テンソルの要素の [**総和を計算したい**] ことがあります。
長さ $n$ のベクトル $\mathbf{x}$ の要素の和を表すには、
$\sum_{i=1}^n x_i$ と書きます。これには簡単な関数があります。

```{.python .input}
%%tab mxnet
x = np.arange(3)
x, x.sum()
```

```{.python .input}
%%tab pytorch
x = torch.arange(3, dtype=torch.float32)
x, x.sum()
```

```{.python .input}
%%tab tensorflow
x = tf.range(3, dtype=tf.float32)
x, tf.reduce_sum(x)
```

```{.python .input}
%%tab jax
x = jnp.arange(3, dtype=jnp.float32)
x, x.sum()
```

任意の shape のテンソルの要素の [**和を表すには**]、
すべての軸にわたって単純に和を取ればよいです。
たとえば、$m \times n$ 行列 $\mathbf{A}$ の要素の和は
$\sum_{i=1}^{m} \sum_{j=1}^{n} a_{ij}$ と書けます。

```{.python .input}
%%tab mxnet, pytorch, jax
A.shape, A.sum()
```

```{.python .input}
%%tab tensorflow
A.shape, tf.reduce_sum(A)
```

デフォルトでは、sum 関数を呼ぶと
テンソルはすべての軸に沿って *リダクション* され、
最終的にスカラーが得られます。
ライブラリでは、テンソルを
どの軸に沿ってリダクションするかを [**指定することもできます**]。
行（軸0）に沿ってすべての要素を足し合わせるには、
`sum` で `axis=0` を指定します。
入力行列は軸0に沿ってリダクションされて出力ベクトルを生成するため、
この軸は出力の shape から消えます。

```{.python .input}
%%tab mxnet, pytorch, jax
A.shape, A.sum(axis=0).shape
```

```{.python .input}
%%tab tensorflow
A.shape, tf.reduce_sum(A, axis=0).shape
```

`axis=1` を指定すると、列方向（軸1）が、すべての列の要素を足し合わせることでリダクションされます。

```{.python .input}
%%tab mxnet, pytorch, jax
A.shape, A.sum(axis=1).shape
```

```{.python .input}
%%tab tensorflow
A.shape, tf.reduce_sum(A, axis=1).shape
```

行と列の両方に沿って和を取って行列をリダクションすることは、
行列のすべての要素を足し合わせることと同じです。

```{.python .input}
%%tab mxnet, pytorch, jax
A.sum(axis=[0, 1]) == A.sum()  # Same as A.sum()
```

```{.python .input}
%%tab tensorflow
tf.reduce_sum(A, axis=[0, 1]), tf.reduce_sum(A)  # Same as tf.reduce_sum(A)
```

[**関連する量として *平均*、別名 *アベレージ* があります。**]
平均は、和を要素数の総数で割ることで求めます。
平均の計算は非常に頻繁に行われるため、
`sum` と同様に使える専用のライブラリ関数があります。

```{.python .input}
%%tab mxnet, jax
A.mean(), A.sum() / A.size
```

```{.python .input}
%%tab pytorch
A.mean(), A.sum() / A.numel()
```

```{.python .input}
%%tab tensorflow
tf.reduce_mean(A), tf.reduce_sum(A) / tf.size(A).numpy()
```

同様に、平均を計算する関数も
特定の軸に沿ってテンソルをリダクションできます。

```{.python .input}
%%tab mxnet, pytorch, jax
A.mean(axis=0), A.sum(axis=0) / A.shape[0]
```

```{.python .input}
%%tab tensorflow
tf.reduce_mean(A, axis=0), tf.reduce_sum(A, axis=0) / A.shape[0]
```

## 非リダクション和
:label:`subsec_lin-alg-non-reduction`

和や平均を計算する関数を呼ぶときに、
[**軸の数を変えずに保つ**] と便利な場合があります。
これは、ブロードキャスト機構を使いたいときに重要です。

```{.python .input}
%%tab mxnet, pytorch, jax
sum_A = A.sum(axis=1, keepdims=True)
sum_A, sum_A.shape
```

```{.python .input}
%%tab tensorflow
sum_A = tf.reduce_sum(A, axis=1, keepdims=True)
sum_A, sum_A.shape
```

たとえば、`sum_A` は各行を足し合わせた後も2つの軸を保つので、
ブロードキャストを使って [**`A` を `sum_A` で割る**] ことができ、
各行の和が $1$ になる行列を作れます。

```{.python .input}
%%tab all
A / sum_A
```

`A` の要素の累積和をある軸に沿って、
たとえば `axis=0`（行ごと）で計算したければ、
`cumsum` 関数を呼べます。
設計上、この関数は入力テンソルをどの軸に沿ってもリダクションしません。

```{.python .input}
%%tab mxnet, pytorch, jax
A.cumsum(axis=0)
```

```{.python .input}
%%tab tensorflow
tf.cumsum(A, axis=0)
```

## ドット積

ここまでで、要素ごとの演算、和、平均だけを扱ってきました。
もしこれしかできないなら、線形代数が独立した節を持つ価値はありません。
幸い、ここからが面白くなります。
最も基本的な演算の1つがドット積です。
2つのベクトル $\mathbf{x}, \mathbf{y} \in \mathbb{R}^d$ に対して、
その *ドット積* $\mathbf{x}^\top \mathbf{y}$（*内積*、$\langle \mathbf{x}, \mathbf{y}  \rangle$ とも呼ばれます）は、
同じ位置にある要素の積の和です:
$\mathbf{x}^\top \mathbf{y} = \sum_{i=1}^{d} x_i y_i$。

[~~2つのベクトルの *ドット積* は、同じ位置にある要素の積の和である~~]

```{.python .input}
%%tab mxnet
y = np.ones(3)
x, y, np.dot(x, y)
```

```{.python .input}
%%tab pytorch
y = torch.ones(3, dtype = torch.float32)
x, y, torch.dot(x, y)
```

```{.python .input}
%%tab tensorflow
y = tf.ones(3, dtype=tf.float32)
x, y, tf.tensordot(x, y, axes=1)
```

```{.python .input}
%%tab jax
y = jnp.ones(3, dtype = jnp.float32)
x, y, jnp.dot(x, y)
```

同値な見方として、[**2つのベクトルのドット積は、
要素ごとの乗算の後に和を取ることで計算できます:**]

```{.python .input}
%%tab mxnet
np.sum(x * y)
```

```{.python .input}
%%tab pytorch
torch.sum(x * y)
```

```{.python .input}
%%tab tensorflow
tf.reduce_sum(x * y)
```

```{.python .input}
%%tab jax
jnp.sum(x * y)
```

ドット積は幅広い文脈で有用です。
たとえば、ある値の集合をベクトル $\mathbf{x}  \in \mathbb{R}^n$ で表し、
重みの集合を $\mathbf{w} \in \mathbb{R}^n$ で表すと、
重み $\mathbf{w}$ に従った $\mathbf{x}$ の値の重み付き和は
ドット積 $\mathbf{x}^\top \mathbf{w}$ として表せます。
重みが非負で、かつ $1$ に和が等しい、すなわち
$\left(\sum_{i=1}^{n} {w_i} = 1\right)$ とき、
ドット積は *重み付き平均* を表します。
2つのベクトルを単位長に正規化すると、
ドット積はそれらのなす角の余弦を表します。
この節の後半で、この *長さ* の概念を正式に導入します。


## 行列--ベクトル積

ドット積の計算方法がわかったので、
$m \times n$ 行列 $\mathbf{A}$ と
$n$ 次元ベクトル $\mathbf{x}$ の間の *積* を理解し始められます。
まず、行ベクトルの観点から行列を可視化します。

$$\mathbf{A}=
\begin{bmatrix}
\mathbf{a}^\top_{1} \\
\mathbf{a}^\top_{2} \\
\vdots \\
\mathbf{a}^\top_m \\
\end{bmatrix},$$

ここで各 $\mathbf{a}^\top_{i} \in \mathbb{R}^n$ は、
行列 $\mathbf{A}$ の $i^\textrm{th}$ 行を表す行ベクトルです。

[**行列--ベクトル積 $\mathbf{A}\mathbf{x}$ は、
長さ $m$ の列ベクトルにすぎず、
その $i^\textrm{th}$ 要素はドット積
$\mathbf{a}^\top_i \mathbf{x}$ です:**]

$$
\mathbf{A}\mathbf{x}
= \begin{bmatrix}
\mathbf{a}^\top_{1} \\
\mathbf{a}^\top_{2} \\
\vdots \\
\mathbf{a}^\top_m \\
\end{bmatrix}\mathbf{x}
= \begin{bmatrix}
 \mathbf{a}^\top_{1} \mathbf{x}  \\
 \mathbf{a}^\top_{2} \mathbf{x} \\
\vdots\\
 \mathbf{a}^\top_{m} \mathbf{x}\\
\end{bmatrix}.
$$

行列 $\mathbf{A}\in \mathbb{R}^{m \times n}$ による乗算は、
ベクトルを $\mathbb{R}^{n}$ から $\mathbb{R}^{m}$ へ写す変換だと考えられます。
このような変換は非常に有用です。
たとえば、回転は特定の正方行列との乗算として表現できます。
行列--ベクトル積は、前の層の出力から
ニューラルネットワークの各層の出力を計算する際の
主要な計算も表しています。

:begin_tab:`mxnet`
コードで行列--ベクトル積を表すには、
同じ `dot` 関数を使います。
どの演算になるかは引数の型から推論されます。
`A` の列方向の次元（軸1に沿った長さ）が
`x` の次元（長さ）と同じでなければならないことに注意してください。
:end_tab:

:begin_tab:`pytorch`
コードで行列--ベクトル積を表すには、
`mv` 関数を使います。
`A` の列方向の次元（軸1に沿った長さ）が
`x` の次元（長さ）と同じでなければならないことに注意してください。
Python には便利な演算子 `@` があり、
行列--ベクトル積と行列--行列積の両方を
（引数に応じて）実行できます。
したがって `A@x` と書けます。
:end_tab:

:begin_tab:`tensorflow`
コードで行列--ベクトル積を表すには、
`matvec` 関数を使います。
`A` の列方向の次元（軸1に沿った長さ）が
`x` の次元（長さ）と同じでなければならないことに注意してください。
:end_tab:

```{.python .input}
%%tab mxnet
A.shape, x.shape, np.dot(A, x)
```

```{.python .input}
%%tab pytorch
A.shape, x.shape, torch.mv(A, x), A@x
```

```{.python .input}
%%tab tensorflow
A.shape, x.shape, tf.linalg.matvec(A, x)
```

```{.python .input}
%%tab jax
A.shape, x.shape, jnp.matmul(A, x)
```

## 行列--行列積

ドット積と行列--ベクトル積に慣れれば、
*行列--行列積* は簡単に理解できます。

2つの行列
$\mathbf{A} \in \mathbb{R}^{n \times k}$
と $\mathbf{B} \in \mathbb{R}^{k \times m}$ があるとします。

$$\mathbf{A}=\begin{bmatrix}
 a_{11} & a_{12} & \cdots & a_{1k} \\
 a_{21} & a_{22} & \cdots & a_{2k} \\
\vdots & \vdots & \ddots & \vdots \\
 a_{n1} & a_{n2} & \cdots & a_{nk} \\
\end{bmatrix},\quad
\mathbf{B}=\begin{bmatrix}
 b_{11} & b_{12} & \cdots & b_{1m} \\
 b_{21} & b_{22} & \cdots & b_{2m} \\
\vdots & \vdots & \ddots & \vdots \\
 b_{k1} & b_{k2} & \cdots & b_{km} \\
\end{bmatrix}.$$


$\mathbf{A}$ の $i^\textrm{th}$ 行を表す行ベクトルを
$\mathbf{a}^\top_{i} \in \mathbb{R}^k$ とし、
$\mathbf{B}$ の $j^\textrm{th}$ 列を表す列ベクトルを
$\mathbf{b}_{j} \in \mathbb{R}^k$ とします。

$$\mathbf{A}=
\begin{bmatrix}
\mathbf{a}^\top_{1} \\
\mathbf{a}^\top_{2} \\
\vdots \\
\mathbf{a}^\top_n \\
\end{bmatrix},
\quad \mathbf{B}=\begin{bmatrix}
 \mathbf{b}_{1} & \mathbf{b}_{2} & \cdots & \mathbf{b}_{m} \\
\end{bmatrix}.
$$


行列積 $\mathbf{C} \in \mathbb{R}^{n \times m}$ を作るには、
各要素 $c_{ij}$ を
$\mathbf{A}$ の $i^\textrm{th}$ 行と
$\mathbf{B}$ の $j^\textrm{th}$ 列のドット積、
すなわち $\mathbf{a}^\top_i \mathbf{b}_j$ として計算するだけです。

$$\mathbf{C} = \mathbf{AB} = \begin{bmatrix}
\mathbf{a}^\top_{1} \\
\mathbf{a}^\top_{2} \\
\vdots \\
\mathbf{a}^\top_n \\
\end{bmatrix}
\begin{bmatrix}
 \mathbf{b}_{1} & \mathbf{b}_{2} & \cdots & \mathbf{b}_{m} \\
\end{bmatrix}
= \begin{bmatrix}
\mathbf{a}^\top_{1} \mathbf{b}_1 & \mathbf{a}^\top_{1}\mathbf{b}_2& \cdots & \mathbf{a}^\top_{1} \mathbf{b}_m \\
 \mathbf{a}^\top_{2}\mathbf{b}_1 & \mathbf{a}^\top_{2} \mathbf{b}_2 & \cdots & \mathbf{a}^\top_{2} \mathbf{b}_m \\
 \vdots & \vdots & \ddots &\vdots\\
\mathbf{a}^\top_{n} \mathbf{b}_1 & \mathbf{a}^\top_{n}\mathbf{b}_2& \cdots& \mathbf{a}^\top_{n} \mathbf{b}_m
\end{bmatrix}.
$$

[**行列--行列積 $\mathbf{AB}$ は、
$m$ 個の行列--ベクトル積
または $m \times n$ 個のドット積を計算し、
その結果をつなぎ合わせて
$n \times m$ 行列を作るものだと考えられます。**]
次のコード片では、`A` と `B` に対して行列積を行います。
ここで `A` は2行3列の行列で、
`B` は3行4列の行列です。
乗算後、2行4列の行列が得られます。

```{.python .input}
%%tab mxnet
B = np.ones(shape=(3, 4))
np.dot(A, B)
```

```{.python .input}
%%tab pytorch
B = torch.ones(3, 4)
torch.mm(A, B), A@B
```

```{.python .input}
%%tab tensorflow
B = tf.ones((3, 4), tf.float32)
tf.matmul(A, B)
```

```{.python .input}
%%tab jax
B = jnp.ones((3, 4))
jnp.matmul(A, B)
```

*行列--行列積* という用語は、
しばしば単に *行列積* と短縮され、
Hadamard 積と混同してはなりません。


## ノルム
:label:`subsec_lin-algebra-norms`

線形代数で最も有用な演算子のいくつかは *ノルム* です。
直感的には、ベクトルのノルムはそれがどれだけ *大きい* かを教えてくれます。
たとえば、$\ell_2$ ノルムはベクトルの（ユークリッド）長さを測ります。
ここで私たちは、ベクトルの成分の大きさに関する *サイズ* の概念
（次元数ではありません）を使っています。

ノルムは、ベクトルをスカラーに写す関数 $\| \cdot \|$ であり、
次の3つの性質を満たします。

1. 任意のベクトル $\mathbf{x}$ について、ベクトル（のすべての要素）を
   スカラー $\alpha \in \mathbb{R}$ でスケールすると、そのノルムもそれに応じてスケールする:
   $$\|\alpha \mathbf{x}\| = |\alpha| \|\mathbf{x}\|.$$
2. 任意のベクトル $\mathbf{x}$ と $\mathbf{y}$ について、
   ノルムは三角不等式を満たす:
   $$\|\mathbf{x} + \mathbf{y}\| \leq \|\mathbf{x}\| + \|\mathbf{y}\|.$$
3. ベクトルのノルムは非負であり、ベクトルがゼロのときにのみ 0 になる:
   $$\|\mathbf{x}\| > 0 \textrm{ for all } \mathbf{x} \neq 0.$$

多くの関数が有効なノルムであり、異なるノルムは
異なるサイズの概念を表します。
小学校の図形で直角三角形の斜辺を求めるときに学ぶ
ユークリッドノルムは、
ベクトルの要素の二乗和の平方根です。
形式的には、これは [**$\ell_2$ *ノルム***] と呼ばれ、次のように表されます。

[**$$\|\mathbf{x}\|_2 = \sqrt{\sum_{i=1}^n x_i^2}.$$**]

`norm` メソッドは $\ell_2$ ノルムを計算します。

```{.python .input}
%%tab mxnet
u = np.array([3, -4])
np.linalg.norm(u)
```

```{.python .input}
%%tab pytorch
u = torch.tensor([3.0, -4.0])
torch.norm(u)
```

```{.python .input}
%%tab tensorflow
u = tf.constant([3.0, -4.0])
tf.norm(u)
```

```{.python .input}
%%tab jax
u = jnp.array([3.0, -4.0])
jnp.linalg.norm(u)
```

[**$\ell_1$ ノルム**] もよく使われ、
それに対応する尺度はマンハッタン距離と呼ばれます。
定義により、$\ell_1$ ノルムは
ベクトルの要素の絶対値の和です。

[**$$\|\mathbf{x}\|_1 = \sum_{i=1}^n \left|x_i \right|.$$**]

$\ell_2$ ノルムと比べると、外れ値の影響を受けにくいです。
$\ell_1$ ノルムを計算するには、
絶対値と和の演算を組み合わせます。

```{.python .input}
%%tab mxnet
np.abs(u).sum()
```

```{.python .input}
%%tab pytorch
torch.abs(u).sum()
```

```{.python .input}
%%tab tensorflow
tf.reduce_sum(tf.abs(u))
```

```{.python .input}
%%tab jax
jnp.linalg.norm(u, ord=1) # same as jnp.abs(u).sum()
```

$\ell_2$ ノルムと $\ell_1$ ノルムはどちらも、
より一般的な $\ell_p$ *ノルム* の特殊な場合です。

$$\|\mathbf{x}\|_p = \left(\sum_{i=1}^n \left|x_i \right|^p \right)^{1/p}.$$

行列の場合は、事情がより複雑です。
そもそも行列は、個々の要素の集合としても、
ベクトルに作用して別のベクトルへ変換する対象としても見られます。
たとえば、行列--ベクトル積 $\mathbf{X} \mathbf{v}$ が
$\mathbf{v}$ に比べてどれだけ長くなりうるかを問うことができます。
この考え方は、*スペクトル* ノルムと呼ばれるものにつながります。
ここではまず、[**計算がずっと簡単な *フロベニウスノルム* を導入します**]。
これは、行列の要素の二乗和の平方根として定義されます。

[**$$\|\mathbf{X}\|_\textrm{F} = \sqrt{\sum_{i=1}^m \sum_{j=1}^n x_{ij}^2}.$$**]

フロベニウスノルムは、行列の形をしたベクトルに対する
$\ell_2$ ノルムのように振る舞います。
次の関数を呼ぶと、行列のフロベニウスノルムが計算されます。

```{.python .input}
%%tab mxnet
np.linalg.norm(np.ones((4, 9)))
```

```{.python .input}
%%tab pytorch
torch.norm(torch.ones((4, 9)))
```

```{.python .input}
%%tab tensorflow
tf.norm(tf.ones((4, 9)))
```

```{.python .input}
%%tab jax
jnp.linalg.norm(jnp.ones((4, 9)))
```

あまり先走りすぎたくはありませんが、
これらの概念がなぜ有用なのかについての直感は
すでに少し持っておけます。
深層学習では、しばしば最適化問題を解こうとします。
観測データに割り当てられる確率を *最大化* すること、
推薦モデルに関連する収益を *最大化* すること、
予測と正解観測値の間の距離を *最小化* すること、
同じ人物の写真の表現同士の距離を *最小化* しつつ、
異なる人物の写真の表現同士の距離を *最大化* すること。
これらの距離は深層学習アルゴリズムの目的関数を構成し、
しばしばノルムとして表されます。


## 議論

この節では、現代の深層学習のかなりの部分を理解するのに必要な
線形代数をひととおり見てきました。
とはいえ、線形代数にはまだまだ多くの内容があり、
その多くは機械学習に有用です。
たとえば、行列は因子に分解でき、
その分解によって実世界のデータセットに潜む
低次元構造を明らかにできることがあります。
機械学習には、行列分解とその高階テンソルへの一般化を用いて
データセットの構造を発見し、
予測問題を解くことに焦点を当てた
サブフィールドが丸ごと存在します。
しかし、この本の焦点は深層学習です。
そして、実際のデータセットに機械学習を適用して
手を動かした後のほうが、
より多くの数学を学ぶ意欲が高まると私たちは考えています。
そのため、後でさらに数学を導入する余地は残しつつも、
ここでこの節を締めくくります。

もっと線形代数を学びたいなら、
優れた書籍やオンライン資料がたくさんあります。
より発展的な速習コースとしては、
:citet:`Strang.1993`, :citet:`Kolter.2008`, :citet:`Petersen.Pedersen.ea.2008` を参照してください。

要点をまとめると:

* スカラー、ベクトル、行列、テンソルは
  線形代数で使われる基本的な数学的対象であり、
  それぞれ 0、1、2、および任意個の軸を持ちます。
* テンソルは、インデックス付けや `sum`、`mean` などの演算によって、
  指定した軸に沿ってスライスしたりリダクションしたりできます。
* 要素ごとの積は Hadamard 積と呼ばれます。
  これに対して、ドット積、行列--ベクトル積、行列--行列積は
  要素ごとの演算ではなく、一般に入力とは異なる shape を持つ対象を返します。
* Hadamard 積と比べると、行列--行列積は
  計算にかなり時間がかかります（2次時間ではなく3次時間）。
* ノルムはベクトル（または行列）の大きさに関するさまざまな概念を捉え、
  2つのベクトルの差に適用して距離を測るのによく使われます。
* よく使われるベクトルノルムには $\ell_1$ ノルムと $\ell_2$ ノルムがあり、
  よく使われる行列ノルムには *スペクトル* ノルムと *フロベニウス* ノルムがあります。


## 演習

1. 行列の転置の転置は元の行列そのものであることを証明せよ: $(\mathbf{A}^\top)^\top = \mathbf{A}$。
1. 2つの行列 $\mathbf{A}$ と $\mathbf{B}$ について、和と転置が可換であることを示せ: $\mathbf{A}^\top + \mathbf{B}^\top = (\mathbf{A} + \mathbf{B})^\top$。
1. 任意の正方行列 $\mathbf{A}$ について、$\mathbf{A} + \mathbf{A}^\top$ は常に対称か。前の2つの演習の結果だけを使って証明できるか。
1. この節では shape が (2, 3, 4) のテンソル `X` を定義しました。`len(X)` の出力は何でしょうか。コードを実行せずに答えを書き、その後コードで確認してください。
1. 任意の shape のテンソル `X` について、`len(X)` は常に `X` のある軸の長さに対応しますか。その軸はどれですか。
1. `A / A.sum(axis=1)` を実行して何が起こるか見てください。結果を分析できますか。
1. マンハッタンの中心部で2点間を移動するとき、座標、すなわち通りと街路の観点で、どれだけの距離を移動する必要がありますか。斜めに移動できますか。
1. shape が (2, 3, 4) のテンソルを考えます。軸0、1、2 に沿った和の出力の shape はそれぞれ何ですか。
1. 3つ以上の軸を持つテンソルを `linalg.norm` 関数に入力して、その出力を観察してください。この関数は任意の shape のテンソルに対して何を計算しますか。
1. たとえば $\mathbf{A} \in \mathbb{R}^{2^{10} \times 2^{16}}$, $\mathbf{B} \in \mathbb{R}^{2^{16} \times 2^{5}}$, $\mathbf{C} \in \mathbb{R}^{2^{5} \times 2^{14}}$ のような3つの大きな行列を、ガウス乱数で初期化したとします。積 $\mathbf{A} \mathbf{B} \mathbf{C}$ を計算したいとき、$(\mathbf{A} \mathbf{B}) \mathbf{C}$ と $\mathbf{A} (\mathbf{B} \mathbf{C})$ のどちらで計算するかによって、メモリ使用量や速度に違いはありますか。なぜですか。
1. たとえば $\mathbf{A} \in \mathbb{R}^{2^{10} \times 2^{16}}$, $\mathbf{B} \in \mathbb{R}^{2^{16} \times 2^{5}}$, $\mathbf{C} \in \mathbb{R}^{2^{5} \times 2^{16}}$ のような3つの大きな行列を考えます。$\mathbf{A} \mathbf{B}$ と $\mathbf{A} \mathbf{C}^\top$ のどちらを計算するかによって速度に違いはありますか。なぜですか。もし $\mathbf{C} = \mathbf{B}^\top$ をメモリを複製せずに初期化したら何が変わりますか。なぜですか。
1. たとえば $\mathbf{A}, \mathbf{B}, \mathbf{C} \in \mathbb{R}^{100 \times 200}$ の3つの行列を考えます。$[\mathbf{A}, \mathbf{B}, \mathbf{C}]$ をスタックして3つの軸を持つテンソルを構成してください。次元数はいくつですか。第3軸の第2成分を取り出して $\mathbf{B}$ を復元してください。答えが正しいことを確認してください。

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/30)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/31)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/196)
:end_tab:

:begin_tab:`jax`
[Discussions](https://discuss.d2l.ai/t/17968)
:end_tab:\n