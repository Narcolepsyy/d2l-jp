# 固有分解
:label:`sec_eigendecompositions`

固有値は、線形代数を学ぶ際にしばしば最も有用な概念の一つである。
しかし、初心者にとってはその重要性を見落としやすいものでもある。
以下では、固有分解を導入し、
それがなぜこれほど重要なのかを少しでも伝えたいと思いる。

次の成分をもつ行列 $A$ があるとする。

$$
\mathbf{A} = \begin{bmatrix}
2 & 0 \\
0 & -1
\end{bmatrix}.
$$

$A$ を任意のベクトル $\mathbf{v} = [x, y]^\top$ に適用すると、
$\mathbf{A}\mathbf{v} = [2x, -y]^\top$ というベクトルが得られる。
これは直感的には、
ベクトルを $x$ 方向に2倍に引き伸ばし、
その後 $y$ 方向に反転させる、と解釈できる。

しかし、*ある種の*ベクトルでは、何かが変わらないまま残る。
すなわち、$[1, 0]^\top$ は $[2, 0]^\top$ に写され、
$[0, 1]^\top$ は $[0, -1]^\top$ に写される。
これらのベクトルは依然として同じ直線上にあり、
変化するのは行列がそれらをそれぞれ $2$ と $-1$ の倍率で伸縮することだけである。
このようなベクトルを*固有ベクトル*、
その伸縮の倍率を*固有値*と呼ぶ。

一般に、ある数 $\lambda$ とベクトル $\mathbf{v}$ が存在して

$$
\mathbf{A}\mathbf{v} = \lambda \mathbf{v}.
$$

となるとき、$\mathbf{v}$ は $A$ の固有ベクトルであり、$\lambda$ は固有値であると言う。

## 固有値の求め方
それらをどのように求めるかを考えてみよう。両辺から $\lambda \mathbf{v}$ を引き、
さらにベクトルをくくり出すと、上式は次と同値であることがわかる。

$$(\mathbf{A} - \lambda \mathbf{I})\mathbf{v} = 0.$$
:eqlabel:`eq_eigvalue_der`

:eqref:`eq_eigvalue_der` が成り立つためには、$(\mathbf{A} - \lambda \mathbf{I})$
がある方向をゼロまで押しつぶす必要がある。
したがってこれは可逆ではなく、ゆえに行列式はゼロである。
したがって、$\det(\mathbf{A}-\lambda \mathbf{I}) = 0$ となる $\lambda$ を求めることで、
*固有値*を見つけることができる。
固有値が求まれば、
$\mathbf{A}\mathbf{v} = \lambda \mathbf{v}$
を解くことで、対応する*固有ベクトル*を求められる。

### 例
もう少し難しい行列で見てみよう。

$$
\mathbf{A} = \begin{bmatrix}
2 & 1\\
2 & 3
\end{bmatrix}.
$$

$\det(\mathbf{A}-\lambda \mathbf{I}) = 0$ を考えると、
これは多項式方程式
$0 = (2-\lambda)(3-\lambda)-2 = (4-\lambda)(1-\lambda)$
と同値であることがわかる。
したがって、固有値は $4$ と $1$ の2つである。
対応するベクトルを求めるには、次を解く必要がある。

$$
\begin{bmatrix}
2 & 1\\
2 & 3
\end{bmatrix}\begin{bmatrix}x \\ y\end{bmatrix} = \begin{bmatrix}x \\ y\end{bmatrix}  \; \textrm{and} \;
\begin{bmatrix}
2 & 1\\
2 & 3
\end{bmatrix}\begin{bmatrix}x \\ y\end{bmatrix}  = \begin{bmatrix}4x \\ 4y\end{bmatrix} .
$$

それぞれ、ベクトル $[1, -1]^\top$ と $[1, 2]^\top$ で解ける。

これを組み込みの `numpy.linalg.eig` ルーチンを使ってコードで確認できる。

```{.python .input}
#@tab mxnet
%matplotlib inline
from d2l import mxnet as d2l
from IPython import display
import numpy as np

np.linalg.eig(np.array([[2, 1], [2, 3]]))
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
from IPython import display
import torch

torch.linalg.eig(torch.tensor([[2, 1], [2, 3]], dtype=torch.float64))
```

```{.python .input}
#@tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
from IPython import display
import tensorflow as tf

tf.linalg.eig(tf.constant([[2, 1], [2, 3]], dtype=tf.float64))
```

`numpy` は固有ベクトルを長さ1に正規化する一方で、
ここでは任意の長さを取っていた。
さらに、符号の選び方も任意である。
しかし、計算されたベクトルは、
同じ固有値に対して手計算で求めたものと平行である。

## 行列の分解
前の例をもう一歩進めてみよう。次を

$$
\mathbf{W} = \begin{bmatrix}
1 & 1 \\
-1 & 2
\end{bmatrix},
$$

行列 $\mathbf{A}$ の固有ベクトルを列にもつ行列とする。さらに

$$
\boldsymbol{\Sigma} = \begin{bmatrix}
1 & 0 \\
0 & 4
\end{bmatrix},
$$

を対応する固有値を対角に並べた行列とする。
すると、固有値と固有ベクトルの定義から

$$
\mathbf{A}\mathbf{W} =\mathbf{W} \boldsymbol{\Sigma} .
$$

が成り立ちる。

行列 $W$ は可逆なので、両辺の右から $W^{-1}$ を掛けると、
次のように書ける。

$$\mathbf{A} = \mathbf{W} \boldsymbol{\Sigma} \mathbf{W}^{-1}.$$
:eqlabel:`eq_eig_decomp`

次節でこれのいくつかの良い帰結を見るが、
今のところは、このような分解は
線形独立な固有ベクトルを十分な数だけ見つけられる限り
（したがって $W$ が可逆である限り）存在する、
ということだけ知っておけば十分である。

## 固有分解に対する演算
:eqref:`eq_eig_decomp` の固有分解の良い点の一つは、
通常出会う多くの演算を固有分解の形で
きれいに書けることである。最初の例として、次を考える。

$$
\mathbf{A}^n = \overbrace{\mathbf{A}\cdots \mathbf{A}}^{\textrm{$n$ times}} = \overbrace{(\mathbf{W}\boldsymbol{\Sigma} \mathbf{W}^{-1})\cdots(\mathbf{W}\boldsymbol{\Sigma} \mathbf{W}^{-1})}^{\textrm{$n$ times}} =  \mathbf{W}\overbrace{\boldsymbol{\Sigma}\cdots\boldsymbol{\Sigma}}^{\textrm{$n$ times}}\mathbf{W}^{-1} = \mathbf{W}\boldsymbol{\Sigma}^n \mathbf{W}^{-1}.
$$

これは、行列の任意の正のべきについて、
固有分解では固有値を同じべきに上げるだけでよいことを示している。
負のべきについても同様に示せるので、
行列を逆にしたいなら、各固有値を逆数にするだけでよいことになる。

$$
\mathbf{A}^{-1} = \mathbf{W}\boldsymbol{\Sigma}^{-1} \mathbf{W}^{-1},
$$

つまり、各固有値を逆にすればよいのである。
これは各固有値がゼロでない限り成り立つので、
可逆であることとゼロ固有値を持たないことは同じだとわかる。

実際、さらに調べると、$\lambda_1, \ldots, \lambda_n$
がある行列の固有値であるとき、その行列の行列式は

$$
\det(\mathbf{A}) = \lambda_1 \cdots \lambda_n,
$$

すなわち全固有値の積になる。
これは直感的にも納得できる。$\mathbf{W}$ がどのような伸縮をしても、
$W^{-1}$ がそれを打ち消すので、最終的に起こる伸縮は
対角行列 $\boldsymbol{\Sigma}$ による掛け算だけである。
そして対角行列は、対角成分の積だけ体積を伸縮させる。

最後に、階数は行列の線形独立な列の最大数だったことを思い出する。
固有分解を詳しく見ると、
階数は $\mathbf{A}$ のゼロでない固有値の数と同じであることがわかる。

例はいくらでも続けられるが、要点はおそらく明らかだろう。
固有分解は多くの線形代数計算を簡単にし、
多くの数値アルゴリズムや線形代数で行う解析の基礎となる
基本的な操作なのである。

## 対称行列の固有分解
上の手順がうまく働くほど十分な数の線形独立な固有ベクトルを
見つけられない場合もある。たとえば次の行列

$$
\mathbf{A} = \begin{bmatrix}
1 & 1 \\
0 & 1
\end{bmatrix},
$$

は、固有ベクトルをただ1つ、すなわち $(1, 0)^\top$ しか持たない。
このような行列を扱うには、
ここで扱える範囲を超えるより高度な手法
（たとえばジョルダン標準形や特異値分解）が必要である。
そのため、しばしば固有ベクトルの完全な集合の存在を保証できる
行列に話を限定する必要がある。

最もよく現れるのは*対称行列*の族で、
これは $\mathbf{A} = \mathbf{A}^\top$ を満たす行列である。
この場合、$W$ を*直交行列*、すなわち列ベクトルがすべて長さ1で互いに直交し、
$\mathbf{W}^\top = \mathbf{W}^{-1}$ を満たす行列として取ることができ、
固有値はすべて実数になる。
したがって、この特別な場合には :eqref:`eq_eig_decomp` を

$$
\mathbf{A} = \mathbf{W}\boldsymbol{\Sigma}\mathbf{W}^\top .
$$

と書ける。

## ガーシュゴリンの円定理
固有値は直感的に理解するのが難しいことがよくある。
任意の行列を与えられても、計算せずに固有値について言えることはあまりない。
しかし、対角成分が大きい場合に、よい近似を簡単に与えてくれる定理が一つある。

$\mathbf{A} = (a_{ij})$ を任意の正方行列（$n\times n$）とする。
$r_i = \sum_{j \neq i} |a_{ij}|$ と定義する。
$\mathcal{D}_i$ を、中心が $a_{ii}$、半径が $r_i$ の複素平面上の円板とする。
すると、$\mathbf{A}$ のすべての固有値はどれか一つの $\mathcal{D}_i$ に含まれる。

これは少し分かりにくいので、例を見てみよう。
次の行列を考える。

$$
\mathbf{A} = \begin{bmatrix}
1.0 & 0.1 & 0.1 & 0.1 \\
0.1 & 3.0 & 0.2 & 0.3 \\
0.1 & 0.2 & 5.0 & 0.5 \\
0.1 & 0.3 & 0.5 & 9.0
\end{bmatrix}.
$$

$r_1 = 0.3$, $r_2 = 0.6$, $r_3 = 0.8$, $r_4 = 0.9$ である。
この行列は対称なので、固有値はすべて実数である。
したがって、固有値は次の範囲のいずれかに入る。

$$[a_{11}-r_1, a_{11}+r_1] = [0.7, 1.3], $$

$$[a_{22}-r_2, a_{22}+r_2] = [2.4, 3.6], $$

$$[a_{33}-r_3, a_{33}+r_3] = [4.2, 5.8], $$

$$[a_{44}-r_4, a_{44}+r_4] = [8.1, 9.9]. $$


数値計算を行うと、
固有値はおよそ $0.99$, $2.97$, $4.95$, $9.08$ であり、
与えられた範囲の中に十分収まっていることがわかる。

```{.python .input}
#@tab mxnet
A = np.array([[1.0, 0.1, 0.1, 0.1],
              [0.1, 3.0, 0.2, 0.3],
              [0.1, 0.2, 5.0, 0.5],
              [0.1, 0.3, 0.5, 9.0]])

v, _ = np.linalg.eig(A)
v
```

```{.python .input}
#@tab pytorch
A = torch.tensor([[1.0, 0.1, 0.1, 0.1],
              [0.1, 3.0, 0.2, 0.3],
              [0.1, 0.2, 5.0, 0.5],
              [0.1, 0.3, 0.5, 9.0]])

v, _ = torch.linalg.eig(A)
v
```

```{.python .input}
#@tab tensorflow
A = tf.constant([[1.0, 0.1, 0.1, 0.1],
                [0.1, 3.0, 0.2, 0.3],
                [0.1, 0.2, 5.0, 0.5],
                [0.1, 0.3, 0.5, 9.0]])

v, _ = tf.linalg.eigh(A)
v
```

このようにして、固有値は近似できる。
そして、対角成分が他のすべての要素より
かなり大きい場合には、その近似はかなり正確になる。

小さなことではあるが、固有分解のような複雑で繊細な話題では、
少しでも直感的な理解を得られるのは良いことである。

## 有用な応用: 反復写像の成長

固有ベクトルが原理的に何であるかを理解したので、
ニューラルネットワークの振る舞いの中心的な問題である
適切な重み初期化に対して、どのように深い理解を与えられるかを見てみよう。

### 長期的な振る舞いとしての固有ベクトル

深層ニューラルネットワークの初期化に関する完全な数学的検討は
本書の範囲を超えるが、ここではそのおもちゃ版を見て、
固有値がこれらのモデルの働きをどう理解する助けになるかを見てみよう。
ご存じのように、ニューラルネットワークは線形変換の層と
非線形演算を交互に挟むことで動作する。
ここでは簡単のため、非線形性はなく、
変換は単一の行列演算 $A$ の反復だと仮定する。
すると、モデルの出力は

$$
\mathbf{v}_{out} = \mathbf{A}\cdot \mathbf{A}\cdots \mathbf{A} \mathbf{v}_{in} = \mathbf{A}^N \mathbf{v}_{in}.
$$

これらのモデルを初期化するとき、$A$ はガウス分布に従う成分をもつランダム行列として取られるので、
そのような行列を一つ作ってみよう。
具体的には、平均0、分散1のガウス分布に従う $5 \times 5$ 行列から始める。

```{.python .input}
#@tab mxnet
np.random.seed(8675309)

k = 5
A = np.random.randn(k, k)
A
```

```{.python .input}
#@tab pytorch
torch.manual_seed(42)

k = 5
A = torch.randn(k, k, dtype=torch.float64)
A
```

```{.python .input}
#@tab tensorflow
k = 5
A = tf.random.normal((k, k), dtype=tf.float64)
A
```

### ランダムデータに対する振る舞い
おもちゃモデルでは簡単のため、
入力するデータベクトル $\mathbf{v}_{in}$
はランダムな5次元ガウスベクトルだと仮定する。
何が起こってほしいかを考えてみよう。
文脈として、画像のような入力データを
その画像が猫の写真である確率のような予測に変換しようとする
一般的な機械学習問題を考えてみる。
もし $\mathbf{A}$ を繰り返し適用するとランダムベクトルが非常に長く伸びるなら、
入力の小さな変化が出力の大きな変化へと増幅される。
つまり、入力画像のごく小さな修正が
まったく異なる予測につながってしまうのである。
これは正しくなさそうです！

逆に、$\mathbf{A}$ がランダムベクトルを短く縮めてしまうなら、
多くの層を通した後でベクトルは実質的に消えてしまい、
出力は入力に依存しなくなる。これも明らかに正しくない！

出力が入力に応じて変化するが、変化しすぎないようにするには、
成長と減衰の狭い境界を歩かなければならない。

ランダムな入力ベクトルに対して行列 $\mathbf{A}$ を繰り返し掛け、
ノルムを追跡すると何が起こるか見てみよう。

```{.python .input}
#@tab mxnet
# Calculate the sequence of norms after repeatedly applying `A`
v_in = np.random.randn(k, 1)

norm_list = [np.linalg.norm(v_in)]
for i in range(1, 100):
    v_in = A.dot(v_in)
    norm_list.append(np.linalg.norm(v_in))

d2l.plot(np.arange(0, 100), norm_list, 'Iteration', 'Value')
```

```{.python .input}
#@tab pytorch
# Calculate the sequence of norms after repeatedly applying `A`
v_in = torch.randn(k, 1, dtype=torch.float64)

norm_list = [torch.norm(v_in).item()]
for i in range(1, 100):
    v_in = A @ v_in
    norm_list.append(torch.norm(v_in).item())

d2l.plot(torch.arange(0, 100), norm_list, 'Iteration', 'Value')
```

```{.python .input}
#@tab tensorflow
# Calculate the sequence of norms after repeatedly applying `A`
v_in = tf.random.normal((k, 1), dtype=tf.float64)

norm_list = [tf.norm(v_in).numpy()]
for i in range(1, 100):
    v_in = tf.matmul(A, v_in)
    norm_list.append(tf.norm(v_in).numpy())

d2l.plot(tf.range(0, 100), norm_list, 'Iteration', 'Value')
```

ノルムが制御不能に増大している！
実際、比の列を取ると、あるパターンが見えてきる。

```{.python .input}
#@tab mxnet
# Compute the scaling factor of the norms
norm_ratio_list = []
for i in range(1, 100):
    norm_ratio_list.append(norm_list[i]/norm_list[i - 1])

d2l.plot(np.arange(1, 100), norm_ratio_list, 'Iteration', 'Ratio')
```

```{.python .input}
#@tab pytorch
# Compute the scaling factor of the norms
norm_ratio_list = []
for i in range(1, 100):
    norm_ratio_list.append(norm_list[i]/norm_list[i - 1])

d2l.plot(torch.arange(1, 100), norm_ratio_list, 'Iteration', 'Ratio')
```

```{.python .input}
#@tab tensorflow
# Compute the scaling factor of the norms
norm_ratio_list = []
for i in range(1, 100):
    norm_ratio_list.append(norm_list[i]/norm_list[i - 1])

d2l.plot(tf.range(1, 100), norm_ratio_list, 'Iteration', 'Ratio')
```

上の計算の最後の部分を見ると、
ランダムベクトルは `1.974459321485[...]` 倍に伸びていることがわかる。
末尾の部分は少し変動するが、
伸長率は安定している。

### 固有ベクトルとの関係

固有ベクトルと固有値は、何かがどれだけ伸びるかに対応することを見てきたが、
それは特定のベクトルと特定の伸長についてでした。
では、$\mathbf{A}$ に対してそれらが何であるかを見てみよう。
ここで少し注意がある。すべてを見るには、
複素数まで行く必要があることがわかる。
複素数は伸長と回転だと考えられる。
複素数のノルム
（実部と虚部の二乗和の平方根）を取ることで、
その伸長率を測れる。ついでに並べ替えよう。

```{.python .input}
#@tab mxnet
# Compute the eigenvalues
eigs = np.linalg.eigvals(A).tolist()
norm_eigs = [np.absolute(x) for x in eigs]
norm_eigs.sort()
print(f'norms of eigenvalues: {norm_eigs}')
```

```{.python .input}
#@tab pytorch
# Compute the eigenvalues
eigs = torch.linalg.eig(A).eigenvalues.tolist()
norm_eigs = [torch.abs(torch.tensor(x)) for x in eigs]
norm_eigs.sort()
print(f'norms of eigenvalues: {norm_eigs}')
```

```{.python .input}
#@tab tensorflow
# Compute the eigenvalues
eigs = tf.linalg.eigh(A)[0].numpy().tolist()
norm_eigs = [tf.abs(tf.constant(x, dtype=tf.float64)) for x in eigs]
norm_eigs.sort()
print(f'norms of eigenvalues: {norm_eigs}')
```

### 観察

ここで少し意外なことが起きているのがわかる。
ランダムベクトルに対して行列 $\mathbf{A}$ を長期的に適用したときの伸長として
先ほど特定したその数は、
*まさに*
（小数点以下13桁まで正確に！）
$\mathbf{A}$ の最大固有値なのである。
これは明らかに偶然ではない！

しかし、これを幾何学的に考えると、納得がいきる。
ランダムベクトルを考えよう。
このランダムベクトルはあらゆる方向を少しずつ向いているので、
特に、$\mathbf{A}$ の最大固有値に対応する固有ベクトルと
少なくとも少しは同じ方向を向いている。
これは非常に重要なので、
*主固有値*および*主固有ベクトル*と呼ばれる。
$\mathbf{A}$ を適用すると、ランダムベクトルは
考えられるすべての方向に伸びるが、
最も強く伸びるのはこの主固有ベクトルに対応する方向である。
つまり、$A$ を適用した後では、
ランダムベクトルはより長くなり、
主固有ベクトルにより近い方向を向くようになる。
行列を何度も適用すると、
主固有ベクトルとの整列はどんどん近づき、
実用上は、ランダムベクトルは主固有ベクトルへと変換されたとみなせるようになる！
実際、このアルゴリズムは、
行列の最大固有値と固有ベクトルを求める
*べき乗法*の基礎になっている。詳細は、たとえば :cite:`Golub.Van-Loan.1996` を参照する。

### 正規化の修正

さて、上の議論から、
ランダムベクトルがまったく伸びたり縮んだりしないことが望ましい、
つまりランダムベクトルが処理全体を通しておおむね同じ大きさを保つことが望ましい、
と結論した。
そのために、行列をこの主固有値で再スケーリングし、
最大固有値が1になるようにする。
この場合に何が起こるか見てみよう。

```{.python .input}
#@tab mxnet
# Rescale the matrix `A`
A /= norm_eigs[-1]

# Do the same experiment again
v_in = np.random.randn(k, 1)

norm_list = [np.linalg.norm(v_in)]
for i in range(1, 100):
    v_in = A.dot(v_in)
    norm_list.append(np.linalg.norm(v_in))

d2l.plot(np.arange(0, 100), norm_list, 'Iteration', 'Value')
```

```{.python .input}
#@tab pytorch
# Rescale the matrix `A`
A /= norm_eigs[-1]

# Do the same experiment again
v_in = torch.randn(k, 1, dtype=torch.float64)

norm_list = [torch.norm(v_in).item()]
for i in range(1, 100):
    v_in = A @ v_in
    norm_list.append(torch.norm(v_in).item())

d2l.plot(torch.arange(0, 100), norm_list, 'Iteration', 'Value')
```

```{.python .input}
#@tab tensorflow
# Rescale the matrix `A`
A /= norm_eigs[-1]

# Do the same experiment again
v_in = tf.random.normal((k, 1), dtype=tf.float64)

norm_list = [tf.norm(v_in).numpy()]
for i in range(1, 100):
    v_in = tf.matmul(A, v_in)
    norm_list.append(tf.norm(v_in).numpy())

d2l.plot(tf.range(0, 100), norm_list, 'Iteration', 'Value')
```

前と同様に連続するノルムの比も描けば、実際に安定していくことがわかる。

```{.python .input}
#@tab mxnet
# Also plot the ratio
norm_ratio_list = []
for i in range(1, 100):
    norm_ratio_list.append(norm_list[i]/norm_list[i-1])

d2l.plot(np.arange(1, 100), norm_ratio_list, 'Iteration', 'Ratio')
```

```{.python .input}
#@tab pytorch
# Also plot the ratio
norm_ratio_list = []
for i in range(1, 100):
    norm_ratio_list.append(norm_list[i]/norm_list[i-1])

d2l.plot(torch.arange(1, 100), norm_ratio_list, 'Iteration', 'Ratio')
```

```{.python .input}
#@tab tensorflow
# Also plot the ratio
norm_ratio_list = []
for i in range(1, 100):
    norm_ratio_list.append(norm_list[i]/norm_list[i-1])

d2l.plot(tf.range(1, 100), norm_ratio_list, 'Iteration', 'Ratio')
```

## 議論

これで、私たちが望んでいたことがまさに起きているのがわかる！
主固有値で行列を正規化すると、
ランダムデータは以前のように爆発せず、
最終的にはある特定の値に落ち着きる。
これらを第一原理から導けるとよいのであるが、
実際にその数学を深く見ると、
独立な平均0、分散1のガウス成分をもつ大きなランダム行列の最大固有値は、
平均するとおよそ $\sqrt{n}$、
この場合は $\sqrt{5} \approx 2.2$ になることがわかる。
これは *円法則* として知られる興味深い事実によるものです :cite:`Ginibre.1965`。
ランダム行列の固有値（およびそれに関連する特異値）と
適切なニューラルネットワーク初期化との関係は、
:citet:`Pennington.Schoenholz.Ganguli.2017` およびその後の研究で
深い結びつきがあることが示されている。

## まとめ
* 固有ベクトルとは、行列によって方向を変えずに伸ばされるベクトルである。
* 固有値とは、行列の作用によって固有ベクトルがどれだけ伸ばされるかを表す量である。
* 行列の固有分解により、多くの演算を固有値に対する演算へと還元できる。
* ガーシュゴリンの円定理は、行列の固有値の近似値を与えてくれる。
* 反復的な行列べきの振る舞いは、主に最大固有値の大きさに依存する。この理解はニューラルネットワーク初期化の理論に多くの応用をもつ。

## 演習
1. 次の行列の固有値と固有ベクトルは何か。
$$
\mathbf{A} = \begin{bmatrix}
2 & 1 \\
1 & 2
\end{bmatrix}?
$$
1. 次の行列の固有値と固有ベクトルは何か。また、前の例と比べてこの例の何が奇妙か。
$$
\mathbf{A} = \begin{bmatrix}
2 & 1 \\
0 & 2
\end{bmatrix}.
$$
1. 固有値を計算せずに、次の行列の最小固有値が $0.5$ より小さい可能性はあるか。*注*: この問題は頭の中で解ける。
$$
\mathbf{A} = \begin{bmatrix}
3.0 & 0.1 & 0.3 & 1.0 \\
0.1 & 1.0 & 0.1 & 0.2 \\
0.3 & 0.1 & 5.0 & 0.0 \\
1.0 & 0.2 & 0.0 & 1.8
\end{bmatrix}.
$$
