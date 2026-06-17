```{.python .input}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow', 'jax'])
```

# 線形代数
:label:`sec_linear-algebra`

ここまでで、データセットをテンソルとして読み込み、
基本的な数学演算でそれらを操作できるようになった。
さらに洗練されたモデルを構築するには、
線形代数のいくつかの道具も必要になる。
この節では、スカラー演算から始めて行列積へと進みながら、
最も重要な概念を段階的に導入する。

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


数学における基本的な演算の多くは、
個々の数値を操作することから成る。
形式的には、こうした値を *スカラー* と呼ぶ。
たとえば、パロアルトの気温が
華氏 $72$ 度だとする。
これを摂氏に変換するには、
$f$ に $72$ を代入して
$c = \frac{5}{9}(f - 32)$ を計算すればよい。
この式では、$5$、$9$、$32$ は定数スカラーである。
変数 $c$ と $f$ は一般に未知のスカラーを表す。

スカラーは通常の小文字
（たとえば $x$、$y$、$z$）
で表し、すべての（連続な）
*実数値* スカラーの空間を $\mathbb{R}$ で表す。
簡潔さのため、*空間* の厳密な定義は省く。
ここでは、式 $x \in \mathbb{R}$ は
$x$ が実数値スカラーであることを示す形式的な記法だと考えればよい。
記号 $\in$ は集合への所属を表す。
たとえば、$x, y \in \{0, 1\}$ は、
$x$ と $y$ が $0$ または $1$ のみを取ることを意味する。

[**スカラーは、1つの要素だけを持つテンソルとして実装する。**]
以下では、2つのスカラーを代入し、
よく知られた加算、乗算、除算、べき乗を行う。

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

ここでは、[**ベクトルをスカラーの固定長配列と考えればよい。**]
通常の配列と同様に、
これらのスカラーをベクトルの *要素* と呼ぶ
（*エントリ* や *成分* とも呼ぶ）。
ベクトルが現実のデータセット中の例を表すとき、
その値は何らかの現実的な意味を持つ。
たとえば、ローンの債務不履行リスクを予測するモデルを学習するなら、
各申請者を1つのベクトルに対応付け、
その成分は収入、勤続年数、
過去の債務不履行回数などに対応するかもしれない。
心臓発作のリスクを研究するなら、
各ベクトルは患者を表し、
その成分は最近のバイタルサイン、コレステロール値、
1日あたりの運動時間などに対応するかもしれない。
ベクトルは太字の小文字
（たとえば $\mathbf{x}$、$\mathbf{y}$、$\mathbf{z}$）
で表す。

ベクトルは $1^{\textrm{st}}$-order テンソルとして実装する。
一般に、このようなテンソルはメモリの制約が許す限り任意の長さを持てる。
注意として、Python では他の多くのプログラミング言語と同様に、
ベクトルの添字は $0$ から始まる。
これは *ゼロ始まりのインデックス付け* と呼ばれる。
一方、線形代数では添字を $1$ から始めるのが普通である
（1始まりのインデックス付け）。

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

添字を使ってベクトルの要素を参照できる。
たとえば、$x_2$ は $\mathbf{x}$ の2番目の要素を表す。
$x_2$ はスカラーなので、太字にはしない。
通常、ベクトルは要素を縦に並べて表す。

$$\mathbf{x} =\begin{bmatrix}x_{1}  \\ \vdots  \\x_{n}\end{bmatrix}.$$
:eqlabel:`eq_vec_def`

ここで $x_1, \ldots, x_n$ はベクトルの要素である。
後で、このような *列ベクトル* と、
要素を横に並べた *行ベクトル* を区別する。
[**テンソルの要素にはインデックスでアクセスする**] ことを思い出そう。

```{.python .input}
%%tab all
x[2]
```

ベクトルが $n$ 個の要素を含むことを示すには、
$\mathbf{x} \in \mathbb{R}^n$ と書く。
形式的には、$n$ をベクトルの *次元数* と呼ぶ。
[**コードでは、これはテンソルの長さに対応する。**]
Python の組み込み関数 `len` で取得できる。

```{.python .input}
%%tab all
len(x)
```

長さは `shape` 属性でも取得できる。
shape は、各軸に沿ったテンソルの長さを表すタプルである。
[**1つの軸しか持たないテンソルの shape は、1つの要素だけを持つ。**]

```{.python .input}
%%tab all
x.shape
```

しばしば「次元」という語は、
軸の数と、特定の軸に沿った長さの両方を指して
曖昧に使われる。
この混乱を避けるため、
軸の数を指すときには *階数* を用い、
要素数を指すときには *次元数* を用いることにする。


## 行列

スカラーが $0^{\textrm{th}}$-order テンソルであり、
ベクトルが $1^{\textrm{st}}$-order テンソルであるのと同様に、
行列は $2^{\textrm{nd}}$-order テンソルである。
行列は太字の大文字
（たとえば $\mathbf{X}$、$\mathbf{Y}$、$\mathbf{Z}$）
で表し、コードでは2つの軸を持つテンソルとして表現する。
式 $\mathbf{A} \in \mathbb{R}^{m \times n}$ は、
行列 $\mathbf{A}$ が $m \times n$ 個の実数値スカラーを含み、
$m$ 行 $n$ 列に配置されていることを意味する。
$m = n$ のとき、その行列を *正方行列* と呼ぶ。
視覚的には、任意の行列を表として表せる。
個々の要素を参照するには、
行と列の両方の添字を付ける。たとえば、
$a_{ij}$ は $\mathbf{A}$ の
$i^{\textrm{th}}$ 行 $j^{\textrm{th}}$ 列の値である。

$$\mathbf{A}=\begin{bmatrix} a_{11} & a_{12} & \cdots & a_{1n} \\ a_{21} & a_{22} & \cdots & a_{2n} \\ \vdots & \vdots & \ddots & \vdots \\ a_{m1} & a_{m2} & \cdots & a_{mn} \\ \end{bmatrix}.$$
:eqlabel:`eq_matrix_def`


コードでは、行列 $\mathbf{A} \in \mathbb{R}^{m \times n}$ を
shape が ($m$, $n$) の $2^{\textrm{nd}}$-order テンソルとして表す。
[**適切なサイズの $m \times n$ テンソルは、
$m \times n$ 行列に変形できる。**]
これは `reshape` に希望する shape を渡せばよい。

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

軸を入れ替えたい場合もある。
行列の行と列を交換した結果を *転置* と呼ぶ。
形式的には、行列 $\mathbf{A}$ の転置を $\mathbf{A}^\top$ で表し、
$\mathbf{B} = \mathbf{A}^\top$ なら、すべての $i$ と $j$ について
$b_{ij} = a_{ji}$ である。
したがって、$m \times n$ 行列の転置は
$n \times m$ 行列になる。

$$
\mathbf{A}^\top =
\begin{bmatrix}
    a_{11} & a_{21} & \dots  & a_{m1} \\
    a_{12} & a_{22} & \dots  & a_{m2} \\
    \vdots & \vdots & \ddots  & \vdots \\
    a_{1n} & a_{2n} & \dots  & a_{mn}
\end{bmatrix}.
$$

コードでは、任意の[**行列の転置**]を次のように得る。

```{.python .input}
%%tab mxnet, pytorch, jax
A.T
```

```{.python .input}
%%tab tensorflow
tf.transpose(A)
```

[**対称行列とは、自身の転置と等しい正方行列である:
$\mathbf{A} = \mathbf{A}^\top$.**]
次の行列は対称である。

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

行列はデータセットの表現にも便利である。
通常、行は個々の記録に対応し、
列は異なる属性に対応する。



## テンソル

スカラー、ベクトル、行列だけでも
機械学習の多くを扱えるが、
やがてはより高階の [**テンソル**] を扱う必要が生じる。
テンソルは[**$n^{\textrm{th}}$-order 配列への拡張を
一般的に記述する方法を与える。**]
ソフトウェア上の *テンソルクラス* のオブジェクトを「テンソル」と呼ぶのは、
それらも任意個の軸を持てるからである。
数学的対象としての *テンソル* と、
コード上の実装を同じ語で呼ぶのは紛らわしいかもしれないが、
通常は文脈から意味が明らかである。
一般のテンソルは特別な書体の大文字
（たとえば $\mathsf{X}$、$\mathsf{Y}$、$\mathsf{Z}$）
で表し、そのインデックス付け
（たとえば $x_{ijk}$ や $[\mathsf{X}]_{1, 2i-1, 3}$）
は行列の場合から自然に拡張される。

画像を扱い始めると、テンソルはさらに重要になる。
各画像は、高さ、幅、*チャネル* に対応する軸を持つ
$3^{\textrm{rd}}$-order テンソルとして表される。
各空間位置では、各色（赤、緑、青）の強度が
チャネル方向に並ぶ。
さらに、画像の集合はコード上では
$4^{\textrm{th}}$-order テンソルとして表され、
個々の画像は第1軸に沿ってインデックス付けされる。
高階テンソルも、ベクトルや行列と同様に、
shape の成分数を増やすことで構成する。

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
いくつかの便利な性質がある。
たとえば、要素ごとの演算は、
入力と同じ shape を持つ出力を生成する。

```{.python .input}
%%tab mxnet
A = np.arange(6).reshape(2, 3)
B = A.copy()  # A のコピーを B に割り当て、新たにメモリを確保する
A, A + B
```

```{.python .input}
%%tab pytorch
A = torch.arange(6, dtype=torch.float32).reshape(2, 3)
B = A.clone()  # A のコピーを B に割り当て、新たにメモリを確保する
A, A + B
```

```{.python .input}
%%tab tensorflow
A = tf.reshape(tf.range(6, dtype=tf.float32), (2, 3))
B = A  # AをBへ新規メモリ割り当てで複製しない
A, A + B
```

```{.python .input}
%%tab jax
A = jnp.arange(6, dtype=jnp.float32).reshape(2, 3)
B = A
A, A + B
```

[**2つの行列の要素ごとの積は *Hadamard 積* と呼ぶ**]（$\odot$ で表す）。
2つの行列 $\mathbf{A}, \mathbf{B} \in \mathbb{R}^{m \times n}$ の
Hadamard 積の各要素は次のように書ける。



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

[**スカラーとテンソルの加算や乗算**] は、元のテンソルと同じ shape の結果を返す。
ここでは、テンソルの各要素にスカラーを加算し、
あるいは乗算している。

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

しばしば、テンソルの要素の [**総和を計算したい**]。
長さ $n$ のベクトル $\mathbf{x}$ の要素の和は、
$\sum_{i=1}^n x_i$ と書ける。これを計算する簡単な関数がある。

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
すべての軸にわたって和を取ればよい。
たとえば、$m \times n$ 行列 $\mathbf{A}$ の要素の和は
$\sum_{i=1}^{m} \sum_{j=1}^{n} a_{ij}$ と書ける。

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
最終的にスカラーが得られる。
ライブラリでは、テンソルを
どの軸に沿ってリダクションするかを [**指定できる**]。
行（軸0）に沿ってすべての要素を足し合わせるには、
`sum` に `axis=0` を指定する。
入力行列は軸0に沿ってリダクションされて出力ベクトルを生成するため、
この軸は出力の shape から消える。

```{.python .input}
%%tab mxnet, pytorch, jax
A.shape, A.sum(axis=0).shape
```

```{.python .input}
%%tab tensorflow
A.shape, tf.reduce_sum(A, axis=0).shape
```

`axis=1` を指定すると、列方向（軸1）がリダクションされ、
各行について列方向の要素を足し合わせる。

```{.python .input}
%%tab mxnet, pytorch, jax
A.shape, A.sum(axis=1).shape
```

```{.python .input}
%%tab tensorflow
A.shape, tf.reduce_sum(A, axis=1).shape
```

行と列の両方に沿って和を取って行列をリダクションすることは、
行列のすべての要素を足し合わせることと同じである。

```{.python .input}
%%tab mxnet, pytorch, jax
A.sum(axis=[0, 1]) == A.sum()  # A.sum()と同じ
```

```{.python .input}
%%tab tensorflow
tf.reduce_sum(A, axis=[0, 1]), tf.reduce_sum(A)  # tf.reduce_sum(A) と同じ
```

[**関連する量として *平均*、すなわち *アベレージ* がある。**]
平均は、和を要素数で割ることで求める。
平均の計算は非常に頻繁に現れるため、
`sum` と同様に使える専用のライブラリ関数がある。

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
特定の軸に沿ってテンソルをリダクションできる。

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

和や平均を計算する関数を呼ぶとき、
[**軸の数を保ったままにする**] と便利な場合がある。
これは、ブロードキャスト機構を使いたいときに重要である。

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
各行の和が $1$ になる行列を作れる。

```{.python .input}
%%tab all
A / sum_A
```

`A` の要素の累積和を、たとえば `axis=0`（行方向）に沿って計算したければ、
`cumsum` 関数を呼べる。
この関数は設計上、入力テンソルをどの軸に沿ってもリダクションしない。

```{.python .input}
%%tab mxnet, pytorch, jax
A.cumsum(axis=0)
```

```{.python .input}
%%tab tensorflow
tf.cumsum(A, axis=0)
```

## ドット積

ここまでで、要素ごとの演算、和、平均だけを扱ってきた。
もしそれだけなら、線形代数が独立した節を持つ必要はない。
幸い、ここからが本題である。
最も基本的な演算の1つがドット積である。
2つのベクトル $\mathbf{x}, \mathbf{y} \in \mathbb{R}^d$ に対して、
その *ドット積* $\mathbf{x}^\top \mathbf{y}$（*内積*、$\langle \mathbf{x}, \mathbf{y}  \rangle$ とも呼ぶ）は、
対応する要素同士の積の和である:
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

同値な見方をすると、[**2つのベクトルのドット積は、
要素ごとに乗算した後で和を取れば計算できる:**]

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

ドット積は幅広い文脈で有用である。
たとえば、ある値の集合をベクトル $\mathbf{x}  \in \mathbb{R}^n$ で表し、
重みの集合を $\mathbf{w} \in \mathbb{R}^n$ で表すと、
重み $\mathbf{w}$ に従った $\mathbf{x}$ の重み付き和は
ドット積 $\mathbf{x}^\top \mathbf{w}$ として表せる。
重みが非負で、かつ和が $1$、すなわち
$\left(\sum_{i=1}^{n} {w_i} = 1\right)$ であるとき、
ドット積は *重み付き平均* を表す。
2つのベクトルを単位長に正規化すると、
ドット積はそれらのなす角の余弦を表す。
この節の後半で、この *長さ* の概念を正式に導入する。


## 行列--ベクトル積

ドット積の計算方法がわかれば、
$m \times n$ 行列 $\mathbf{A}$ と
$n$ 次元ベクトル $\mathbf{x}$ の *積* も理解できる。
まず、行列を行ベクトルの集まりとして見る。

$$\mathbf{A}=
\begin{bmatrix}
\mathbf{a}^\top_{1} \\
\mathbf{a}^\top_{2} \\
\vdots \\
\mathbf{a}^\top_m \\
\end{bmatrix},$$

ここで各 $\mathbf{a}^\top_{i} \in \mathbb{R}^n$ は、
行列 $\mathbf{A}$ の $i^\textrm{th}$ 行を表す行ベクトルである。

[**行列--ベクトル積 $\mathbf{A}\mathbf{x}$ は、
長さ $m$ の列ベクトルであり、
その $i^\textrm{th}$ 要素はドット積
$\mathbf{a}^\top_i \mathbf{x}$ である:**]

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
ベクトルを $\mathbb{R}^{n}$ から $\mathbb{R}^{m}$ へ写す変換とみなせる。
このような変換は非常に有用である。
たとえば、回転は特定の正方行列との乗算として表せる。
行列--ベクトル積は、前の層の出力から
ニューラルネットワークの各層の出力を計算する際の
主要な計算でもある。

:begin_tab:`mxnet`
コードで行列--ベクトル積を表すには、
同じ `dot` 関数を使う。
どの演算になるかは引数の型から推論される。
`A` の列方向の次元（軸1に沿った長さ）が
`x` の次元（長さ）と一致しなければならないことに注意しよう。
:end_tab:

:begin_tab:`pytorch`
コードで行列--ベクトル積を表すには、
`mv` 関数を使う。
`A` の列方向の次元（軸1に沿った長さ）が
`x` の次元（長さ）と一致しなければならないことに注意しよう。
Python には便利な演算子 `@` があり、
行列--ベクトル積と行列--行列積の両方を
（引数に応じて）実行できる。
したがって `A@x` と書ける。
:end_tab:

:begin_tab:`tensorflow`
コードで行列--ベクトル積を表すには、
`matvec` 関数を使う。
`A` の列方向の次元（軸1に沿った長さ）が
`x` の次元（長さ）と一致しなければならないことに注意しよう。
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
*行列--行列積* も容易に理解できる。

2つの行列
$\mathbf{A} \in \mathbb{R}^{n \times k}$
と $\mathbf{B} \in \mathbb{R}^{k \times m}$ があるとする。

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
$\mathbf{b}_{j} \in \mathbb{R}^k$ とする。

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
すなわち $\mathbf{a}^\top_i \mathbf{b}_j$ として計算すればよい。

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
あるいは $m \times n$ 個のドット積を計算し、
その結果を並べて
$n \times m$ 行列を作るものと考えられる。**]
次のコード片では、`A` と `B` に対して行列積を計算する。
ここで `A` は2行3列の行列で、
`B` は3行4列の行列である。
乗算後、2行4列の行列が得られる。

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

*行列--行列積* という語は、
しばしば単に *行列積* と略される。
Hadamard 積と混同してはならない。


## ノルム
:label:`subsec_lin-algebra-norms`

線形代数で最も有用な演算子のいくつかが *ノルム* である。
直感的には、ベクトルのノルムはそれがどれだけ *大きい* かを表す。
たとえば、$\ell_2$ ノルムはベクトルの（ユークリッド）長さを測る。
ここでいう *大きさ* は、成分の規模に関する概念であり、
次元数のことではない。

ノルムは、ベクトルをスカラーに写す関数 $\| \cdot \|$ であり、
次の3つの性質を満たす。

1. 任意のベクトル $\mathbf{x}$ について、ベクトル（のすべての要素）を
   スカラー $\alpha \in \mathbb{R}$ でスケールすると、そのノルムも同じ比率で変化する:
   $$\|\alpha \mathbf{x}\| = |\alpha| \|\mathbf{x}\|.$$
2. 任意のベクトル $\mathbf{x}$ と $\mathbf{y}$ について、
   ノルムは三角不等式を満たす:
   $$\|\mathbf{x} + \mathbf{y}\| \leq \|\mathbf{x}\| + \|\mathbf{y}\|.$$
3. ベクトルのノルムは非負であり、ベクトルがゼロのときに限って 0 になる:
   $$\|\mathbf{x}\| > 0 \textrm{ for all } \mathbf{x} \neq 0.$$

多くの関数が有効なノルムであり、異なるノルムは
異なる大きさの概念を表す。
小学校で直角三角形の斜辺を求めるときに学ぶ
ユークリッドノルムは、
ベクトルの要素の二乗和の平方根である。
形式的には、 [**$\ell_2$ *ノルム***] と呼ばれ、次のように表される。

[**$$\|\mathbf{x}\|_2 = \sqrt{\sum_{i=1}^n x_i^2}.$$**]

`norm` メソッドは $\ell_2$ ノルムを計算する。

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
それに対応する尺度はマンハッタン距離と呼ばれる。
定義より、$\ell_1$ ノルムは
ベクトルの要素の絶対値の和である。

[**$$\|\mathbf{x}\|_1 = \sum_{i=1}^n \left|x_i \right|.$$**]

$\ell_2$ ノルムと比べると、外れ値の影響を受けにくい。
$\ell_1$ ノルムを計算するには、
絶対値と和の演算を組み合わせる。

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
jnp.linalg.norm(u, ord=1) # jnp.abs(u).sum()と同じ
```

$\ell_2$ ノルムと $\ell_1$ ノルムはどちらも、
より一般的な $\ell_p$ *ノルム* の特殊な場合である。

$$\|\mathbf{x}\|_p = \left(\sum_{i=1}^n \left|x_i \right|^p \right)^{1/p}.$$

行列の場合は、事情がやや複雑である。
行列は、個々の要素の集まりとしても、
ベクトルに作用して別のベクトルへ変換する対象としても見られるからである。
たとえば、行列--ベクトル積 $\mathbf{X} \mathbf{v}$ が
$\mathbf{v}$ に比べてどれだけ長くなりうるかを問える。
この考え方は、*スペクトル* ノルムにつながる。
ここではまず、[**計算がはるかに容易な *フロベニウスノルム* を導入する**]。
これは、行列の要素の二乗和の平方根として定義される。

[**$$\|\mathbf{X}\|_\textrm{F} = \sqrt{\sum_{i=1}^m \sum_{j=1}^n x_{ij}^2}.$$**]

フロベニウスノルムは、行列を1つの長いベクトルとみなしたときの
$\ell_2$ ノルムのように振る舞う。
次の関数を呼ぶと、行列のフロベニウスノルムを計算できる。

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

あまり先を急ぎすぎたくはないが、
これらの概念がなぜ有用なのかについての直感は
すでに少し持てる。
深層学習では、しばしば最適化問題を解く。
観測データに割り当てられる確率を *最大化* すること、
推薦モデルに関連する収益を *最大化* すること、
予測と正解観測値の間の距離を *最小化* すること、
同一人物の写真表現同士の距離を *最小化* しつつ、
異なる人物の写真表現同士の距離を *最大化* することなどである。
これらの距離は深層学習アルゴリズムの目的関数を構成し、
しばしばノルムで表される。


## 議論

この節では、現代の深層学習のかなりの部分を理解するのに必要な
線形代数をひととおり見てきた。
とはいえ、線形代数にはまだ多くの内容があり、
その多くは機械学習に有用である。
たとえば、行列は因子に分解でき、
その分解によって実世界のデータセットに潜む
低次元構造が明らかになることがある。
機械学習には、行列分解とその高階テンソルへの一般化を用いて
データセットの構造を発見し、
予測問題を解くことに焦点を当てた
サブフィールドが存在する。
しかし、この本の焦点は深層学習である。
そして、実際のデータセットに機械学習を適用して
手を動かした後のほうが、
さらに多くの数学を学ぶ動機も高まると考えている。
そのため、後でさらに数学を導入する余地を残しつつ、
ここでこの節を締めくくる。

さらに線形代数を学びたいなら、
優れた書籍やオンライン資料が数多くある。
やや発展的な速習コースとしては、
:citet:`Strang.1993`, :citet:`Kolter.2008`, :citet:`Petersen.Pedersen.ea.2008` を参照されたい。

要点をまとめると:

* スカラー、ベクトル、行列、テンソルは
  線形代数で用いる基本的な数学的対象であり、
  それぞれ 0、1、2、および任意個の軸を持つ。
* テンソルは、インデックス付けや `sum`、`mean` などの演算によって、
  指定した軸に沿ってスライスしたりリダクションしたりできる。
* 要素ごとの積は Hadamard 積と呼ばれる。
  これに対して、ドット積、行列--ベクトル積、行列--行列積は
  要素ごとの演算ではなく、一般に入力とは異なる shape を持つ対象を返す。
* Hadamard 積と比べると、行列--行列積は
  計算にかなり時間がかかる（2次時間ではなく3次時間）。
* ノルムはベクトル（または行列）の大きさに関するさまざまな概念を捉え、
  2つのベクトルの差に適用して距離を測るのによく使われる。
* よく使われるベクトルノルムには $\ell_1$ ノルムと $\ell_2$ ノルムがあり、
  よく使われる行列ノルムには *スペクトル* ノルムと *フロベニウス* ノルムがある。


## 演習

1. 行列の転置の転置は元の行列そのものであることを証明せよ: $(\mathbf{A}^\top)^\top = \mathbf{A}$。
1. 2つの行列 $\mathbf{A}$ と $\mathbf{B}$ について、和と転置が可換であることを示せ: $\mathbf{A}^\top + \mathbf{B}^\top = (\mathbf{A} + \mathbf{B})^\top$。
1. 任意の正方行列 $\mathbf{A}$ について、$\mathbf{A} + \mathbf{A}^\top$ は常に対称か。前の2つの演習の結果だけを使って証明できるか。
1. この節では shape が (2, 3, 4) のテンソル `X` を定義した。`len(X)` の出力は何だろうか。コードを実行せずに答えを書き、その後コードで確認しよう。
1. 任意の shape のテンソル `X` について、`len(X)` は常に `X` のある軸の長さに対応するか。その軸はどれであるか。
1. `A / A.sum(axis=1)` を実行して何が起こるか見よ。結果を分析できるか。
1. マンハッタンの中心部で2点間を移動するとき、座標、すなわち通りと街路の観点で、どれだけの距離を移動する必要があるか。斜めに移動できるか。
1. shape が (2, 3, 4) のテンソルを考える。軸0、1、2 に沿った和の出力の shape はそれぞれ何であるか。
1. 3つ以上の軸を持つテンソルを `linalg.norm` 関数に入力して、その出力を観察しよう。この関数は任意の shape のテンソルに対して何を計算するか。
1. たとえば $\mathbf{A} \in \mathbb{R}^{2^{10} \times 2^{16}}$, $\mathbf{B} \in \mathbb{R}^{2^{16} \times 2^{5}}$, $\mathbf{C} \in \mathbb{R}^{2^{5} \times 2^{14}}$ のような3つの大きな行列を、ガウス乱数で初期化したとする。積 $\mathbf{A} \mathbf{B} \mathbf{C}$ を計算したいとき、$(\mathbf{A} \mathbf{B}) \mathbf{C}$ と $\mathbf{A} (\mathbf{B} \mathbf{C})$ のどちらで計算するかによって、メモリ使用量や速度に違いはあるか。なぜであるか。
1. たとえば $\mathbf{A} \in \mathbb{R}^{2^{10} \times 2^{16}}$, $\mathbf{B} \in \mathbb{R}^{2^{16} \times 2^{5}}$, $\mathbf{C} \in \mathbb{R}^{2^{5} \times 2^{16}}$ のような3つの大きな行列を考える。$\mathbf{A} \mathbf{B}$ と $\mathbf{A} \mathbf{C}^\top$ のどちらを計算するかによって速度に違いはあるか。なぜであるか。もし $\mathbf{C} = \mathbf{B}^\top$ をメモリを複製せずに初期化したら何が変わるか。なぜであるか。
1. たとえば $\mathbf{A}, \mathbf{B}, \mathbf{C} \in \mathbb{R}^{100 \times 200}$ の3つの行列を考える。$[\mathbf{A}, \mathbf{B}, \mathbf{C}]$ をスタックして3つの軸を持つテンソルを構成しよう。次元数はいくつであるか。第3軸の第2成分を取り出して $\mathbf{B}$ を復元しよう。答えが正しいことを確認しよう。
```