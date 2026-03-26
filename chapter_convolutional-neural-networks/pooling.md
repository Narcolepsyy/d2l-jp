{.python .input}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow', 'jax'])
```

# プーリング
:label:`sec_pooling`

多くの場合、最終的なタスクは画像に対する大域的な問いを求めます。
たとえば、*猫が含まれているか？* といった具合です。
したがって、最終層のユニットは入力全体に敏感であるべきです。
情報を徐々に集約し、より粗いマップを得ていくことで、
中間層では畳み込み層の利点を保ちながら、
最終的には大域的な表現を学習するというこの目標を達成できます。
ネットワークを深く進むほど、
各隠れノードが敏感になる受容野は
（入力に対して相対的に）大きくなります。
空間解像度を下げると、
畳み込みカーネルがより大きな有効領域を覆うため、
この過程が加速されます。

さらに、エッジのような低レベル特徴を検出する際には
（:numref:`sec_conv_layer` で議論したように）、
表現がある程度の平行移動不変性を持つことがしばしば望まれます。
たとえば、黒と白の境界がはっきりした画像 `X` を取り、
画像全体を右に1ピクセルずらす、
すなわち `Z[i, j] = X[i, j + 1]` とすると、
新しい画像 `Z` の出力は大きく異なるかもしれません。
エッジは1ピクセル分移動しているからです。
現実には、物体がまったく同じ場所に現れることはほとんどありません。
実際、三脚を使い静止した物体を撮影していても、
シャッターの動きによるカメラの振動で
全体が1ピクセル程度ずれることがあります
（高級カメラにはこの問題に対処するための特別な機能が搭載されています）。

この節では、*プーリング層*を導入します。
これは、畳み込み層の位置に対する感度を緩和することと、
表現を空間的にダウンサンプリングすることという
二つの目的を果たします。

```{.python .input}
%%tab mxnet
from d2l import mxnet as d2l
from mxnet import np, npx
from mxnet.gluon import nn
npx.set_np()
```

```{.python .input}
%%tab pytorch
from d2l import torch as d2l
import torch
from torch import nn
```

```{.python .input}
%%tab jax
from d2l import jax as d2l
from flax import linen as nn
import jax
from jax import numpy as jnp
```

## 最大プーリングと平均プーリング

畳み込み層と同様に、*プーリング*演算子は
固定形状のウィンドウからなり、
そのストライドに従って入力のすべての領域上をスライドし、
固定形状ウィンドウ（*プーリングウィンドウ*とも呼ばれます）が通過する各位置について
1つの出力を計算します。
しかし、畳み込み層における入力とカーネルの相互相関計算とは異なり、
プーリング層にはパラメータがありません（*カーネル*は存在しません）。
その代わり、プーリング演算子は決定論的であり、
通常はプーリングウィンドウ内の要素の最大値または平均値を計算します。
これらの操作はそれぞれ、*最大プーリング*（略して *max-pooling*）と
*平均プーリング*と呼ばれます。

*平均プーリング*は、CNN と同じくらい古くからある手法です。
その考え方は画像のダウンサンプリングに似ています。
低解像度画像を得るために単に2つおき（あるいは3つおき）の画素の値を取るのではなく、
隣接する画素を平均することで、
複数の隣接画素からの情報を統合し、
信号対雑音比の高い画像を得ることができます。
*max-pooling* は、認知神経科学の文脈で
物体認識のために情報集約がどのように階層的に行われうるかを記述する目的で
:citet:`Riesenhuber.Poggio.1999` において導入されました。
それ以前にも音声認識における先行版がありました :cite:`Yamaguchi.Sakamoto.Akabane.ea.1990`。
ほとんどすべての場合において、max-pooling とも呼ばれるこの手法は、
平均プーリングよりも望ましいです。

どちらの場合も、相互相関演算子と同様に、
プーリングウィンドウは入力テンソルの左上から始まり、
左から右へ、上から下へとスライドしていくと考えられます。
プーリングウィンドウが各位置に到達するたびに、
max か average かに応じて、
ウィンドウ内の入力部分テンソルの最大値または平均値を計算します。


![プーリングウィンドウの形状が $2\times 2$ の max-pooling。影付き部分は最初の出力要素であり、出力計算に使われる入力テンソル要素でもあります: $\max(0, 1, 3, 4)=4$.](../img/pooling.svg)
:label:`fig_pooling`

:numref:`fig_pooling` の出力テンソルの高さは 2、幅は 2 です。
4つの要素は、それぞれのプーリングウィンドウ内の最大値から得られます。

$$
\max(0, 1, 3, 4)=4,\\
\max(1, 2, 4, 5)=5,\\
\max(3, 4, 6, 7)=7,\\
\max(4, 5, 7, 8)=8.\\
$$

より一般には、$p \times q$ のプーリング層を、そのサイズの領域上で集約することで定義できます。
エッジ検出の問題に戻ると、
畳み込み層の出力を $2\times 2$ の max-pooling の入力として用います。
`X` を畳み込み層の入力、`Y` をプーリング層の出力とします。
`X[i, j]`、`X[i, j + 1]`、
`X[i+1, j]`、`X[i+1, j + 1]` の値が異なるかどうかにかかわらず、
プーリング層は常に `Y[i, j] = 1` を出力します。
つまり、$2\times 2$ の max-pooling 層を使えば、
畳み込み層が認識したパターンが高さまたは幅方向に1要素以内しか移動しない限り、
それを依然として検出できます。

以下のコードでは、`pool2d` 関数で
[**プーリング層の順伝播を実装**] します。
この関数は :numref:`sec_conv_layer` の `corr2d` 関数に似ています。
ただし、カーネルは不要で、入力の各領域の最大値または平均値として
出力を計算します。

```{.python .input}
%%tab mxnet, pytorch
def pool2d(X, pool_size, mode='max'):
    p_h, p_w = pool_size
    Y = d2l.zeros((X.shape[0] - p_h + 1, X.shape[1] - p_w + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            if mode == 'max':
                Y[i, j] = X[i: i + p_h, j: j + p_w].max()
            elif mode == 'avg':
                Y[i, j] = X[i: i + p_h, j: j + p_w].mean()
    return Y
```

```{.python .input}
%%tab jax
def pool2d(X, pool_size, mode='max'):
    p_h, p_w = pool_size
    Y = jnp.zeros((X.shape[0] - p_h + 1, X.shape[1] - p_w + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            if mode == 'max':
                Y = Y.at[i, j].set(X[i: i + p_h, j: j + p_w].max())
            elif mode == 'avg':
                Y = Y.at[i, j].set(X[i: i + p_h, j: j + p_w].mean())
    return Y
```

```{.python .input}
%%tab tensorflow
import tensorflow as tf

def pool2d(X, pool_size, mode='max'):
    p_h, p_w = pool_size
    Y = tf.Variable(tf.zeros((X.shape[0] - p_h + 1, X.shape[1] - p_w +1)))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            if mode == 'max':
                Y[i, j].assign(tf.reduce_max(X[i: i + p_h, j: j + p_w]))
            elif mode =='avg':
                Y[i, j].assign(tf.reduce_mean(X[i: i + p_h, j: j + p_w]))
    return Y
```

:numref:`fig_pooling` における入力テンソル `X` を構成して、
[**2次元 max-pooling 層の出力を検証**] できます。

```{.python .input}
%%tab all
X = d2l.tensor([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]])
pool2d(X, (2, 2))
```

また、[**平均プーリング層**] でも試すことができます。

```{.python .input}
%%tab all
pool2d(X, (2, 2), 'avg')
```

## [**パディングとストライド**]

畳み込み層と同様に、プーリング層も
出力形状を変化させます。
そしてこれまでと同様に、入力をパディングしストライドを調整することで、
望ましい出力形状を得るように演算を調整できます。
プーリング層におけるパディングとストライドの使用は、
深層学習フレームワークの組み込み2次元 max-pooling 層を通して示せます。
まず、形状が4次元の入力テンソル `X` を構成します。
ここで、サンプル数（バッチサイズ）とチャネル数はいずれも 1 です。

:begin_tab:`tensorflow`
TensorFlow は他のフレームワークと異なり、*channels-last* の入力を好み、またそれに最適化されています。
:end_tab:

```{.python .input}
%%tab mxnet, pytorch
X = d2l.reshape(d2l.arange(16, dtype=d2l.float32), (1, 1, 4, 4))
X
```

```{.python .input}
%%tab tensorflow, jax
X = d2l.reshape(d2l.arange(16, dtype=d2l.float32), (1, 4, 4, 1))
X
```

プーリングは領域から情報を集約するため、[**深層学習フレームワークではプーリングウィンドウサイズとストライドが一致するのが既定**] です。
たとえば、形状 `(3, 3)` のプーリングウィンドウを使うと、
既定では `(3, 3)` のストライドになります。

```{.python .input}
%%tab mxnet
pool2d = nn.MaxPool2D(3)
# Pooling has no model parameters, hence it needs no initialization
pool2d(X)
```

```{.python .input}
%%tab pytorch
pool2d = nn.MaxPool2d(3)
# Pooling has no model parameters, hence it needs no initialization
pool2d(X)
```

```{.python .input}
%%tab tensorflow
pool2d = tf.keras.layers.MaxPool2D(pool_size=[3, 3])
# Pooling has no model parameters, hence it needs no initialization
pool2d(X)
```

```{.python .input}
%%tab jax
# Pooling has no model parameters, hence it needs no initialization
nn.max_pool(X, window_shape=(3, 3), strides=(3, 3))
```

言うまでもなく、必要であれば [**ストライドとパディングを手動で指定**] して、フレームワークの既定値を上書きできます。

```{.python .input}
%%tab mxnet
pool2d = nn.MaxPool2D(3, padding=1, strides=2)
pool2d(X)
```

```{.python .input}
%%tab pytorch
pool2d = nn.MaxPool2d(3, padding=1, stride=2)
pool2d(X)
```

```{.python .input}
%%tab tensorflow
paddings = tf.constant([[0, 0], [1,0], [1,0], [0,0]])
X_padded = tf.pad(X, paddings, "CONSTANT")
pool2d = tf.keras.layers.MaxPool2D(pool_size=[3, 3], padding='valid',
                                   strides=2)
pool2d(X_padded)
```

```{.python .input}
%%tab jax
X_padded = jnp.pad(X, ((0, 0), (1, 0), (1, 0), (0, 0)), mode='constant')
nn.max_pool(X_padded, window_shape=(3, 3), padding='VALID', strides=(2, 2))
```

もちろん、以下の例が示すように、任意の高さと幅を持つ任意の長方形のプーリングウィンドウを指定できます。

```{.python .input}
%%tab mxnet
pool2d = nn.MaxPool2D((2, 3), padding=(0, 1), strides=(2, 3))
pool2d(X)
```

```{.python .input}
%%tab pytorch
pool2d = nn.MaxPool2d((2, 3), stride=(2, 3), padding=(0, 1))
pool2d(X)
```

```{.python .input}
%%tab tensorflow
paddings = tf.constant([[0, 0], [0, 0], [1, 1], [0, 0]])
X_padded = tf.pad(X, paddings, "CONSTANT")

pool2d = tf.keras.layers.MaxPool2D(pool_size=[2, 3], padding='valid',
                                   strides=(2, 3))
pool2d(X_padded)
```

```{.python .input}
%%tab jax

X_padded = jnp.pad(X, ((0, 0), (0, 0), (1, 1), (0, 0)), mode='constant')
nn.max_pool(X_padded, window_shape=(2, 3), strides=(2, 3), padding='VALID')
```

## 複数チャネル

多チャネル入力データを処理する場合、
[**プーリング層は入力チャネルごとに別々にプーリング**] し、
畳み込み層のようにチャネル方向に入力を合計することはしません。
これは、プーリング層の出力チャネル数が
入力チャネル数と同じであることを意味します。
以下では、テンソル `X` と `X + 1` をチャネル次元で連結し、
2チャネルの入力を構成します。

:begin_tab:`tensorflow`
TensorFlow では channels-last の構文のため、
最後の次元に沿った連結が必要になることに注意してください。
:end_tab:

```{.python .input}
%%tab mxnet, pytorch
X = d2l.concat((X, X + 1), 1)
X
```

```{.python .input}
%%tab tensorflow, jax
# Concatenate along `dim=3` due to channels-last syntax
X = d2l.concat([X, X + 1], 3)
X
```

見てわかるように、プーリング後も出力チャネル数は依然として 2 です。

```{.python .input}
%%tab mxnet
pool2d = nn.MaxPool2D(3, padding=1, strides=2)
pool2d(X)
```

```{.python .input}
%%tab pytorch
pool2d = nn.MaxPool2d(3, padding=1, stride=2)
pool2d(X)
```

```{.python .input}
%%tab tensorflow
paddings = tf.constant([[0, 0], [1,0], [1,0], [0,0]])
X_padded = tf.pad(X, paddings, "CONSTANT")
pool2d = tf.keras.layers.MaxPool2D(pool_size=[3, 3], padding='valid',
                                   strides=2)
pool2d(X_padded)

```

```{.python .input}
%%tab jax
X_padded = jnp.pad(X, ((0, 0), (1, 0), (1, 0), (0, 0)), mode='constant')
nn.max_pool(X_padded, window_shape=(3, 3), padding='VALID', strides=(2, 2))
```

:begin_tab:`tensorflow`
TensorFlow のプーリング出力は一見すると異なって見えますが、
数値的には MXNet や PyTorch と同じ結果です。
違いは次元の並びにあり、出力を縦方向に読むと他の実装と同じ出力になります。
:end_tab:

## まとめ

プーリングは非常に単純な操作です。その名の通り、値のウィンドウ上で結果を集約します。ストライドやパディングなど、畳み込みに関する意味論はこれまでと同様に適用されます。プーリングはチャネルに依存しない、すなわちチャネル数を変えず、各チャネルに別々に適用されることに注意してください。最後に、2つの代表的なプーリング手法のうち、max-pooling は平均プーリングよりも望ましく、出力にある程度の不変性を与えます。よく使われる選択として、出力の空間解像度を4分の1にするために $2 \times 2$ のプーリングウィンドウを選ぶ方法があります。 

プーリング以外にも解像度を下げる方法は数多くあります。たとえば、stochastic pooling :cite:`Zeiler.Fergus.2013` や fractional max-pooling :cite:`Graham.2014` では、集約にランダム化が組み合わされています。これにより、場合によっては精度がわずかに向上します。最後に、後で注意機構で見るように、出力を集約するより洗練された方法もあります。たとえば、クエリと表現ベクトルの整列を用いる方法です。 


## 演習

1. 畳み込みを通して平均プーリングを実装せよ。 
1. max-pooling は畳み込みだけでは実装できないことを証明せよ。 
1. max-pooling は ReLU 演算、すなわち $\textrm{ReLU}(x) = \max(0, x)$ を用いて実現できる。
    1. ReLU 演算だけを用いて $\max (a, b)$ を表せ。
    1. これを用いて、畳み込みと ReLU 層によって max-pooling を実装せよ。 
    1. $2 \times 2$ の畳み込みには何チャネル、何層必要か？ $3 \times 3$ の畳み込みではどうか？
1. プーリング層の計算コストはいくらか？ プーリング層への入力サイズが $c\times h\times w$、プーリングウィンドウの形状が $p_\textrm{h}\times p_\textrm{w}$、パディングが $(p_\textrm{h}, p_\textrm{w})$、ストライドが $(s_\textrm{h}, s_\textrm{w})$ であると仮定せよ。
1. max-pooling と average pooling が異なる働きをすると予想するのはなぜか？
1. 別個の最小プーリング層は必要か？ 別の演算で置き換えられるか？
1. プーリングに softmax 演算を使うこともできる。なぜそれほど一般的でないのだろうか？\n
