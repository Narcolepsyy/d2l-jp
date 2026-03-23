{.python .input}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow', 'jax'])
```

# データ操作
:label:`sec_ndarray`

何かを成し遂げるためには、データを保存し操作する何らかの方法が必要です。一般に、データに対して行うべき重要なことは2つあります。すなわち、(i) データを取得すること、そして (ii) それらがコンピュータの中に入った後に処理することです。データを保存する方法がなければ、データを取得しても意味がありません。そこでまずは、*テンソル*とも呼ばれる $n$ 次元配列に手を動かしてみましょう。NumPy の科学技術計算パッケージをすでに知っているなら、これは簡単でしょう。現代のすべての深層学習フレームワークでは、*テンソルクラス*（MXNet では `ndarray`、PyTorch と TensorFlow では `Tensor`）は NumPy の `ndarray` に似ていますが、いくつかの強力な機能が追加されています。第一に、テンソルクラスは自動微分をサポートします。第二に、NumPy が CPU 上でしか動作しないのに対し、GPU を活用して数値計算を高速化できます。これらの特性により、ニューラルネットワークは実装しやすく、かつ高速に実行できます。



## はじめに

:begin_tab:`mxnet`
まず、MXNet から `np`（`numpy`）モジュールと `npx`（`numpy_extension`）モジュールをインポートします。ここで、`np` モジュールには NumPy でサポートされる関数が含まれ、`npx` モジュールには NumPy 風の環境で深層学習を可能にするために開発された拡張機能群が含まれています。テンソルを使うときは、ほとんど常に `set_np` 関数を呼び出します。これは MXNet の他のコンポーネントによるテンソル処理との互換性のためです。
:end_tab:

:begin_tab:`pytorch`
[**まず、PyTorch ライブラリをインポートします。パッケージ名は `torch` であることに注意してください。**]
:end_tab:

:begin_tab:`tensorflow`
まず `tensorflow` をインポートします。簡潔さのため、実務者はしばしば `tf` という別名を付けます。
:end_tab:

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
import jax
from jax import numpy as jnp
```

[**テンソルは、数値の（場合によっては多次元の）配列を表します。**]  
1次元の場合、すなわちデータに1つの軸しか必要ない場合、テンソルは *ベクトル* と呼ばれます。2つの軸を持つテンソルは *行列* と呼ばれます。$k > 2$ の軸を持つ場合は、特別な名前は使わず、単に $k$-*階テンソル* と呼びます。

:begin_tab:`mxnet`
MXNet には、値をあらかじめ入れた新しいテンソルを作成するためのさまざまな関数があります。たとえば `arange(n)` を呼び出すと、0（含む）から `n`（含まない）までの等間隔の値を持つベクトルを作成できます。デフォルトでは、間隔の大きさは $1$ です。特に指定しない限り、新しいテンソルは主記憶に保存され、CPU ベースの計算に割り当てられます。
:end_tab:

:begin_tab:`pytorch`
PyTorch には、値をあらかじめ入れた新しいテンソルを作成するためのさまざまな関数があります。たとえば `arange(n)` を呼び出すと、0（含む）から `n`（含まない）までの等間隔の値を持つベクトルを作成できます。デフォルトでは、間隔の大きさは $1$ です。特に指定しない限り、新しいテンソルは主記憶に保存され、CPU ベースの計算に割り当てられます。
:end_tab:

:begin_tab:`tensorflow`
TensorFlow には、値をあらかじめ入れた新しいテンソルを作成するためのさまざまな関数があります。たとえば `range(n)` を呼び出すと、0（含む）から `n`（含まない）までの等間隔の値を持つベクトルを作成できます。デフォルトでは、間隔の大きさは $1$ です。特に指定しない限り、新しいテンソルは主記憶に保存され、CPU ベースの計算に割り当てられます。
:end_tab:

```{.python .input}
%%tab mxnet
x = np.arange(12)
x
```

```{.python .input}
%%tab pytorch
x = torch.arange(12, dtype=torch.float32)
x
```

```{.python .input}
%%tab tensorflow
x = tf.range(12, dtype=tf.float32)
x
```

```{.python .input}
%%tab jax
x = jnp.arange(12)
x
```

:begin_tab:`mxnet`
これらの各値はテンソルの *要素* と呼ばれます。テンソル `x` には 12 個の要素があります。テンソル内の要素数の総数は `size` 属性で確認できます。
:end_tab:

:begin_tab:`pytorch`
これらの各値はテンソルの *要素* と呼ばれます。テンソル `x` には 12 個の要素があります。テンソル内の要素数の総数は `numel` メソッドで確認できます。
:end_tab:

:begin_tab:`tensorflow`
これらの各値はテンソルの *要素* と呼ばれます。テンソル `x` には 12 個の要素があります。テンソル内の要素数の総数は `size` 関数で確認できます。
:end_tab:

```{.python .input}
%%tab mxnet, jax
x.size
```

```{.python .input}
%%tab pytorch
x.numel()
```

```{.python .input}
%%tab tensorflow
tf.size(x)
```

（**テンソルの *shape*（各軸に沿った長さ）**）は `shape` 属性を調べることで取得できます。ここではベクトルを扱っているので、`shape` には1つの要素しか含まれず、サイズと同じです。

```{.python .input}
%%tab all
x.shape
```

`reshape` を呼び出すことで、[**サイズや値を変えずにテンソルの形状を変更できます**]。たとえば、形状が (12,) のベクトル `x` を、形状が (3, 4) の行列 `X` に変換できます。この新しいテンソルはすべての要素を保持したまま、それらを行列として再配置します。ベクトルの要素は1行ずつ並んでいるので、`x[3] == X[0, 3]` となることに注意してください。

```{.python .input}
%%tab mxnet, pytorch, jax
X = x.reshape(3, 4)
X
```

```{.python .input}
%%tab tensorflow
X = tf.reshape(x, (3, 4))
X
```

`reshape` に対して各 shape 成分をすべて明示するのは冗長であることに注意してください。テンソルのサイズはすでに分かっているので、他の成分が分かれば shape の1成分を求められます。たとえば、サイズが $n$ で目標の shape が ($h$, $w$) のテンソルでは、$w = n/h$ であることが分かります。shape の成分を自動的に推定させるには、自動推定されるべき成分に `-1` を指定します。ここでは、`x.reshape(3, 4)` の代わりに `x.reshape(-1, 4)` や `x.reshape(3, -1)` と書いても同じです。

実務では、すべて 0 や 1 で初期化されたテンソルを扱うことがよくあります。[**すべての要素が 0 に設定されたテンソルを構築する**]（~~または 1~~）には、`zeros` 関数を使って shape を (2, 3, 4) にします。

```{.python .input}
%%tab mxnet
np.zeros((2, 3, 4))
```

```{.python .input}
%%tab pytorch
torch.zeros((2, 3, 4))
```

```{.python .input}
%%tab tensorflow
tf.zeros((2, 3, 4))
```

```{.python .input}
%%tab jax
jnp.zeros((2, 3, 4))
```

同様に、`ones` を呼び出すことで、すべて 1 のテンソルを作成できます。

```{.python .input}
%%tab mxnet
np.ones((2, 3, 4))
```

```{.python .input}
%%tab pytorch
torch.ones((2, 3, 4))
```

```{.python .input}
%%tab tensorflow
tf.ones((2, 3, 4))
```

```{.python .input}
%%tab jax
jnp.ones((2, 3, 4))
```

しばしば、与えられた確率分布から各要素を[**ランダムに（かつ独立に）サンプリングする**]必要があります。たとえば、ニューラルネットワークのパラメータはしばしばランダムに初期化されます。次のコード片は、平均 0、標準偏差 1 の標準ガウス（正規）分布から要素を抽出したテンソルを作成します。

```{.python .input}
%%tab mxnet
np.random.normal(0, 1, size=(3, 4))
```

```{.python .input}
%%tab pytorch
torch.randn(3, 4)
```

```{.python .input}
%%tab tensorflow
tf.random.normal(shape=[3, 4])
```

```{.python .input}
%%tab jax
# Any call of a random function in JAX requires a key to be
# specified, feeding the same key to a random function will
# always result in the same sample being generated
jax.random.normal(jax.random.PRNGKey(0), (3, 4))
```

最後に、数値リテラルを含む（場合によっては入れ子になった）Python のリストを与えることで、[**各要素の正確な値を指定してテンソルを構築できます**]。ここでは、リストのリストを使って行列を構築しています。最外側のリストが軸 0 に対応し、内側のリストが軸 1 に対応します。

```{.python .input}
%%tab mxnet
np.array([[2, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
```

```{.python .input}
%%tab pytorch
torch.tensor([[2, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
```

```{.python .input}
%%tab tensorflow
tf.constant([[2, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
```

```{.python .input}
%%tab jax
jnp.array([[2, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
```

## インデックス付けとスライシング

Python のリストと同様に、インデックス付け（0 から始まる）によってテンソルの要素にアクセスできます。リストの末尾からの位置に基づいて要素にアクセスするには、負のインデックスを使います。さらに、スライシング（たとえば `X[start:stop]`）によってインデックスの範囲全体にアクセスできます。このとき返される値には最初のインデックス（`start`）は含まれますが、最後のインデックス（`stop`）は含まれません。最後に、$k$-*階テンソル* に対して1つのインデックス（またはスライス）だけを指定した場合、それは軸 0 に沿って適用されます。したがって、次のコードでは、[**`[-1]` は最後の行を選択し、`[1:3]` は2行目と3行目を選択します**]。

```{.python .input}
%%tab all
X[-1], X[1:3]
```

:begin_tab:`mxnet, pytorch`
読み取るだけでなく、（**インデックスを指定して行列の要素を *書き込む* こともできます。**）
:end_tab:

:begin_tab:`tensorflow`
TensorFlow の `Tensors` は不変であり、代入することはできません。TensorFlow の `Variables` は状態を保持する可変コンテナであり、代入をサポートします。TensorFlow では、`Variable` への代入を通じて勾配が逆方向に流れないことに注意してください。

`Variable` 全体に値を代入するだけでなく、インデックスを指定して `Variable` の要素を書き込むこともできます。
:end_tab:

```{.python .input}
%%tab mxnet, pytorch
X[1, 2] = 17
X
```

```{.python .input}
%%tab tensorflow
X_var = tf.Variable(X)
X_var[1, 2].assign(9)
X_var
```

```{.python .input}
%%tab jax
# JAX arrays are immutable. jax.numpy.ndarray.at index
# update operators create a new array with the corresponding
# modifications made
X_new_1 = X.at[1, 2].set(17)
X_new_1
```

複数の要素に同じ値を[**代入したい場合は、代入操作の左辺でインデックス付けを行います。**] たとえば、`[:2, :]` は1行目と2行目にアクセスし、`:` は軸 1（列）に沿ったすべての要素を取ります。ここでは行列についてインデックス付けを説明しましたが、これはベクトルや2次元より高いテンソルにも同様に使えます。

```{.python .input}
%%tab mxnet, pytorch
X[:2, :] = 12
X
```

```{.python .input}
%%tab tensorflow
X_var = tf.Variable(X)
X_var[:2, :].assign(tf.ones(X_var[:2,:].shape, dtype=tf.float32) * 12)
X_var
```

```{.python .input}
%%tab jax
X_new_2 = X_new_1.at[:2, :].set(12)
X_new_2
```

## 演算

テンソルの構築方法と、その要素の読み書き方法が分かったので、さまざまな数学演算を使ってテンソルを操作できるようになります。中でも最も有用なのは *要素ごとの* 演算です。これは、標準的なスカラー演算をテンソルの各要素に適用します。2つのテンソルを入力に取る関数では、要素ごとの演算は対応する要素の各組に標準的な二項演算子を適用します。スカラーをスカラーへ写す任意の関数から、要素ごとの関数を作ることができます。

数学記法では、このような *単項* スカラー演算子（1つの入力を取る）を、シグネチャ
$f: \mathbb{R} \rightarrow \mathbb{R}$
で表します。これは、関数が任意の実数を別の実数へ写すことを意味します。$e^x$ のような単項演算子を含む、ほとんどの標準的な演算子は要素ごとに適用できます。

```{.python .input}
%%tab mxnet
np.exp(x)
```

```{.python .input}
%%tab pytorch
torch.exp(x)
```

```{.python .input}
%%tab tensorflow
tf.exp(x)
```

```{.python .input}
%%tab jax
jnp.exp(x)
```

同様に、実数の組を1つの実数へ写す *二項* スカラー演算子は、シグネチャ
$f: \mathbb{R}, \mathbb{R} \rightarrow \mathbb{R}$
で表します。任意の2つのベクトル $\mathbf{u}$ と $\mathbf{v}$ が *同じ shape* を持ち、二項演算子 $f$ が与えられたとき、各 $i$ について $c_i \gets f(u_i, v_i)$ と置くことで、ベクトル $\mathbf{c} = F(\mathbf{u},\mathbf{v})$ を生成できます。ここで、$c_i, u_i, v_i$ はそれぞれベクトル $\mathbf{c}, \mathbf{u}, \mathbf{v}$ の $i$-*番目* の要素です。ここでは、スカラー関数を要素ごとのベクトル演算へ *持ち上げる* ことで、ベクトル値の
$F: \mathbb{R}^d, \mathbb{R}^d \rightarrow \mathbb{R}^d$
を得ています。加算（`+`）、減算（`-`）、乗算（`*`）、除算（`/`）、べき乗（`**`）といった一般的な標準算術演算子は、任意の shape を持つ同じ shape のテンソルに対して、すべて *持ち上げられ*、要素ごとの演算として使えるようになっています。

```{.python .input}
%%tab mxnet
x = np.array([1, 2, 4, 8])
y = np.array([2, 2, 2, 2])
x + y, x - y, x * y, x / y, x ** y
```

```{.python .input}
%%tab pytorch
x = torch.tensor([1.0, 2, 4, 8])
y = torch.tensor([2, 2, 2, 2])
x + y, x - y, x * y, x / y, x ** y
```

```{.python .input}
%%tab tensorflow
x = tf.constant([1.0, 2, 4, 8])
y = tf.constant([2.0, 2, 2, 2])
x + y, x - y, x * y, x / y, x ** y
```

```{.python .input}
%%tab jax
x = jnp.array([1.0, 2, 4, 8])
y = jnp.array([2, 2, 2, 2])
x + y, x - y, x * y, x / y, x ** y
```

要素ごとの計算に加えて、内積や行列積のような線形代数演算も実行できます。これらについては :numref:`sec_linear-algebra` で詳しく説明します。

また、複数のテンソルを[***連結* する**]こともでき、端から端へつなげてより大きなテンソルを作れます。そのためには、テンソルのリストを与え、どの軸に沿って連結するかをシステムに指定するだけです。以下の例では、2つの行列を列（軸 1）ではなく行（軸 0）に沿って連結したときに何が起こるかを示しています。最初の出力の軸 0 の長さ（$6$）は、2つの入力テンソルの軸 0 の長さ（$3 + 3$）の和であり、2つ目の出力の軸 1 の長さ（$8$）は、2つの入力テンソルの軸 1 の長さ（$4 + 4$）の和であることが分かります。

```{.python .input}
%%tab mxnet
X = np.arange(12).reshape(3, 4)
Y = np.array([[2, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
np.concatenate([X, Y], axis=0), np.concatenate([X, Y], axis=1)
```

```{.python .input}
%%tab pytorch
X = torch.arange(12, dtype=torch.float32).reshape((3,4))
Y = torch.tensor([[2.0, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
torch.cat((X, Y), dim=0), torch.cat((X, Y), dim=1)
```

```{.python .input}
%%tab tensorflow
X = tf.reshape(tf.range(12, dtype=tf.float32), (3, 4))
Y = tf.constant([[2.0, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
tf.concat([X, Y], axis=0), tf.concat([X, Y], axis=1)
```

```{.python .input}
%%tab jax
X = jnp.arange(12, dtype=jnp.float32).reshape((3, 4))
Y = jnp.array([[2.0, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
jnp.concatenate((X, Y), axis=0), jnp.concatenate((X, Y), axis=1)
```

ときには、[**論理式を通じて二値テンソルを構築したい**]ことがあります。`X == Y` を例にしましょう。各位置 $i, j$ について、`X[i, j]` と `Y[i, j]` が等しければ、結果の対応する要素は 1 になり、そうでなければ 0 になります。

```{.python .input}
%%tab all
X == Y
```

[**テンソルのすべての要素を合計する**]と、要素が1つだけのテンソルが得られます。

```{.python .input}
%%tab mxnet, pytorch, jax
X.sum()
```

```{.python .input}
%%tab tensorflow
tf.reduce_sum(X)
```

## ブロードキャスティング
:label:`subsec_broadcasting`

ここまでで、同じ shape を持つ2つのテンソルに対して要素ごとの二項演算を行う方法を学びました。ある条件下では、shape が異なっていても、[**ブロードキャスティング機構を呼び出すことで要素ごとの二項演算を実行できます。**] ブロードキャスティングは次の2段階の手順に従います。(i) 長さ 1 の軸に沿って要素をコピーすることで、一方または両方の配列を拡張し、この変換後に2つのテンソルが同じ shape になるようにする。(ii) 得られた配列に対して要素ごとの演算を行う。

```{.python .input}
%%tab mxnet
a = np.arange(3).reshape(3, 1)
b = np.arange(2).reshape(1, 2)
a, b
```

```{.python .input}
%%tab pytorch
a = torch.arange(3).reshape((3, 1))
b = torch.arange(2).reshape((1, 2))
a, b
```

```{.python .input}
%%tab tensorflow
a = tf.reshape(tf.range(3), (3, 1))
b = tf.reshape(tf.range(2), (1, 2))
a, b
```

```{.python .input}
%%tab jax
a = jnp.arange(3).reshape((3, 1))
b = jnp.arange(2).reshape((1, 2))
a, b
```

`a` と `b` はそれぞれ $3\times1$ と $1\times2$ の行列なので、shape は一致しません。ブロードキャスティングでは、行列 `a` を列方向に、行列 `b` を行方向に複製してから要素ごとに加算することで、より大きな $3\times2$ 行列を生成します。

```{.python .input}
%%tab all
a + b
```

## メモリの節約

[**演算を実行すると、結果を保持するために新しいメモリが割り当てられることがあります。**] たとえば `Y = X + Y` と書くと、`Y` が以前指していたテンソルへの参照を外し、代わりに `Y` を新しく割り当てられたメモリに向けます。この問題は Python の `id()` 関数で確認できます。これは参照先オブジェクトのメモリアドレスを正確に返します。`Y = Y + X` を実行した後、`id(Y)` が別の場所を指していることに注意してください。これは、Python がまず `Y + X` を評価して結果用の新しいメモリを割り当て、その後 `Y` をこの新しいメモリ位置に向けるからです。

```{.python .input}
%%tab all
before = id(Y)
Y = Y + X
id(Y) == before
```

これは2つの理由で望ましくない場合があります。第一に、不要なメモリ割り当てを何度も行いたくないからです。機械学習では、しばしば数百メガバイト規模のパラメータを持ち、それらすべてを毎秒何度も更新します。可能な限り、これらの更新は *インプレース* で行いたいものです。第二に、同じパラメータを複数の変数から参照していることがあります。インプレース更新をしない場合、メモリリークを起こしたり、古いパラメータを誤って参照したりしないよう、これらすべての参照を注意深く更新しなければなりません。

:begin_tab:`mxnet, pytorch`
幸いなことに、[**インプレース演算の実行**]は簡単です。スライス記法 `Y[:] = <expression>` を使うことで、演算結果をすでに割り当て済みの配列 `Y` に代入できます。この概念を示すために、`zeros_like` を使って `Y` と同じ shape を持つように初期化したテンソル `Z` の値を上書きします。
:end_tab:

:begin_tab:`tensorflow`
TensorFlow の `Variables` は状態を保持する可変コンテナです。モデルパラメータを保存する方法を提供します。演算結果は `assign` を使って `Variable` に代入できます。この概念を示すために、`zeros_like` を使って `Y` と同じ shape を持つように初期化した `Variable` `Z` の値を上書きします。
:end_tab:

```{.python .input}
%%tab mxnet
Z = np.zeros_like(Y)
print('id(Z):', id(Z))
Z[:] = X + Y
print('id(Z):', id(Z))
```

```{.python .input}
%%tab pytorch
Z = torch.zeros_like(Y)
print('id(Z):', id(Z))
Z[:] = X + Y
print('id(Z):', id(Z))
```

```{.python .input}
%%tab tensorflow
Z = tf.Variable(tf.zeros_like(Y))
print('id(Z):', id(Z))
Z.assign(X + Y)
print('id(Z):', id(Z))
```

```{.python .input}
%%tab jax
# JAX arrays do not allow in-place operations
```

:begin_tab:`mxnet, pytorch`
[**`X` の値が後続の計算で再利用されないなら、`X[:] = X + Y` や `X += Y` を使って演算のメモリオーバーヘッドを減らすこともできます。**]
:end_tab:

:begin_tab:`tensorflow`
`Variable` に状態を永続的に保存した後でも、モデルパラメータではないテンソルに対する余分な割り当てを避けることで、さらにメモリ使用量を減らしたい場合があります。TensorFlow の `Tensors` は不変であり、勾配は `Variable` への代入を通じて流れないため、TensorFlow には個々の演算をインプレースで実行する明示的な方法はありません。

しかし、TensorFlow には `tf.function` デコレータがあり、計算を TensorFlow グラフでラップして、実行前にコンパイルおよび最適化できます。これにより、TensorFlow は不要な値を刈り込み、もはや必要ない以前の割り当てを再利用できます。これによって TensorFlow 計算のメモリオーバーヘッドが最小化されます。
:end_tab:

```{.python .input}
%%tab mxnet, pytorch
before = id(X)
X += Y
id(X) == before
```

```{.python .input}
%%tab tensorflow
@tf.function
def computation(X, Y):
    Z = tf.zeros_like(Y)  # This unused value will be pruned out
    A = X + Y  # Allocations will be reused when no longer needed
    B = A + Y
    C = B + Y
    return C + Y

computation(X, Y)
```

## 他の Python オブジェクトへの変換

:begin_tab:`mxnet, tensorflow`
[**NumPy テンソル（`ndarray`）への変換**]、またはその逆は簡単です。変換結果はメモリを共有しません。この小さな不便さは実は非常に重要です。CPU や GPU 上で演算を行うとき、Python の NumPy パッケージが同じメモリ領域を使って別のことをしたいかどうかを待つために、計算を止めたくはないからです。
:end_tab:

:begin_tab:`pytorch`
[**NumPy テンソル（`ndarray`）への変換**]、またはその逆は簡単です。torch テンソルと NumPy 配列は基盤となるメモリを共有し、一方をインプレース演算で変更すると、もう一方も変更されます。
:end_tab:

```{.python .input}
%%tab mxnet
A = X.asnumpy()
B = np.array(A)
type(A), type(B)
```

```{.python .input}
%%tab pytorch
A = X.numpy()
B = torch.from_numpy(A)
type(A), type(B)
```

```{.python .input}
%%tab tensorflow
A = X.numpy()
B = tf.constant(A)
type(A), type(B)
```

```{.python .input}
%%tab jax
A = jax.device_get(X)
B = jax.device_put(A)
type(A), type(B)
```

[**サイズ 1 のテンソルを Python のスカラーに変換する**]には、`item` 関数や Python の組み込み関数を呼び出せます。

```{.python .input}
%%tab mxnet
a = np.array([3.5])
a, a.item(), float(a), int(a)
```

```{.python .input}
%%tab pytorch
a = torch.tensor([3.5])
a, a.item(), float(a), int(a)
```

```{.python .input}
%%tab tensorflow
a = tf.constant([3.5]).numpy()
a, a.item(), float(a), int(a)
```

```{.python .input}
%%tab jax
a = jnp.array([3.5])
a, a.item(), float(a), int(a)
```

## 要約

テンソルクラスは、深層学習ライブラリにおいてデータを保存し操作するための主要なインターフェースです。テンソルは、構築ルーチン、インデックス付けとスライシング、基本的な数学演算、ブロードキャスティング、メモリ効率のよい代入、そして他の Python オブジェクトとの相互変換など、さまざまな機能を提供します。


## 演習

1. この節のコードを実行してください。条件式 `X == Y` を `X < Y` または `X > Y` に変えて、どのようなテンソルが得られるか確認してください。
1. ブロードキャスティング機構で要素ごとに作用する2つのテンソルを、3次元テンソルなど別の shape に置き換えてください。結果は期待どおりになりますか？

:begin_tab:`mxnet`
[議論](https://discuss.d2l.ai/t/26)
:end_tab:

:begin_tab:`pytorch`
[議論](https://discuss.d2l.ai/t/27)
:end_tab:

:begin_tab:`tensorflow`
[議論](https://discuss.d2l.ai/t/187)
:end_tab:

:begin_tab:`jax`
[議論](https://discuss.d2l.ai/t/17966)
:end_tab:\n