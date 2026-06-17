{.python .input}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow', 'jax'])
```

# プーリング層
:label:`sec_pooling`

**プーリング**とは、画像認識などの畳み込みニューラルネットワーク（CNN）において、空間解像度を下げることで計算量を削減し、位置のわずかなずれに対する不変性を与えるためのダウンサンプリング処理である。代表的な手法として最大プーリングと平均プーリングがある。

多くの場合、最終的な課題は画像全体に関する大域的な問いに答えることである。
たとえば、*猫が写っているか* といった問いである。
したがって、最終層のユニットは入力全体に反応すべきである。
情報を段階的に集約してより粗い特徴マップを得ることで、
中間層では畳み込み層の利点を保ちつつ、
最終的には大域的な表現を学習できる。
ネットワークが深くなるにつれて、
各隠れノードが反応する受容野は
（入力に対する相対的な意味で）大きくなる。
空間解像度を下げると、
畳み込みカーネルがより広い有効領域を覆うようになるため、
この過程はさらに加速する。

さらに、エッジのような低水準の特徴を検出する際には
（:numref:`sec_conv_layer` で述べたように）、
表現がある程度の平行移動不変性を持つことがしばしば望ましい。
たとえば、白黒の境界が明瞭な画像 `X` を考え、
画像全体を右に1ピクセルずらして
`Z[i, j] = X[i, j + 1]` とする。
このとき、新しい画像 `Z` に対する出力は大きく変わりうる。
エッジ自体が1ピクセル移動しているからである。
しかし現実には、物体が常にまったく同じ位置に現れることはほとんどない。
実際、三脚を用いて静止物体を撮影していても、
シャッター動作によるカメラの微小な振動で
画像全体が1ピクセル程度ずれることがある
（高級カメラにはこの問題に対処するための機構が備わっている）。

この節では、*プーリング層*を導入する。
これは、畳み込み層の位置に対する感度を弱めることと、
表現を空間的にダウンサンプリングすることという
二つの目的を担う。

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
固定形状のウィンドウを持ち、
そのストライドに従って入力の各領域上をスライドし、
各位置で1つの出力を計算する。
この固定形状のウィンドウは
*プーリングウィンドウ*とも呼ばれる。
ただし、畳み込み層における入力とカーネルの相互相関計算とは異なり、
プーリング層には学習すべきパラメータがない。
その代わり、プーリング演算子は決定論的であり、
通常はプーリングウィンドウ内の要素の最大値または平均値を計算する。
これらはそれぞれ、*最大プーリング*（*max-pooling*）と
*平均プーリング*と呼ばれる。

*平均プーリング*は、CNN と同じくらい古くから使われてきた手法である。
その発想は画像のダウンサンプリングに近い。
低解像度画像を得るために単に2つおき（あるいは3つおき）の画素値を取るのではなく、
隣接画素を平均すれば、
複数の近傍画素の情報を統合でき、
信号対雑音比の高い画像を得られる。
一方、*max-pooling* は、認知神経科学の文脈において、
物体認識のための情報集約がどのように階層的に行われうるかを記述する目的で
:citet:`Riesenhuber.Poggio.1999` により導入された。
それ以前にも、音声認識における先行研究が存在する :cite:`Yamaguchi.Sakamoto.Akabane.ea.1990`。
実際には、ほとんどの場合において、
max-pooling のほうが平均プーリングより好まれる。

いずれの場合も、相互相関演算子と同様に、
プーリングウィンドウは入力テンソルの左上から始まり、
左から右へ、上から下へとスライドする。
ウィンドウが各位置に到達するたびに、
max か average かに応じて、
その内部に含まれる入力部分テンソルの最大値または平均値を計算する。


![プーリングウィンドウの形状が $2\times 2$ の max-pooling。影付き部分は最初の出力要素の計算に用いられる入力テンソル要素であり、$\max(0, 1, 3, 4)=4$ である。](../img/pooling.svg)
:label:`fig_pooling`

:numref:`fig_pooling` の出力テンソルの高さは2、幅は2である。
4つの要素は、それぞれ対応するプーリングウィンドウ内の最大値から得られる。

$$
\max(0, 1, 3, 4)=4,\\
\max(1, 2, 4, 5)=5,\\
\max(3, 4, 6, 7)=7,\\
\max(4, 5, 7, 8)=8.\\
$$

より一般には、$p \times q$ のプーリング層は、
その大きさの領域上で集約を行うことで定義される。
エッジ検出の問題に戻ろう。
畳み込み層の出力を $2\times 2$ の max-pooling の入力として用いる。
`X` を畳み込み層の入力、`Y` をプーリング層の出力とする。
`X[i, j]`、`X[i, j + 1]`、
`X[i+1, j]`、`X[i+1, j + 1]` の値がどうであれ、
プーリング層は `Y[i, j] = 1` を出力する。
すなわち、$2\times 2$ の max-pooling 層を用いれば、
畳み込み層が検出したパターンが高さ方向または幅方向に1要素以内しか移動しない限り、
そのパターンを引き続き検出できる。

以下のコードでは、`pool2d` 関数によって
[**プーリング層の順伝播を実装**] する。
この関数は :numref:`sec_conv_layer` の `corr2d` 関数に似ている。
ただし、カーネルは不要であり、入力の各領域に対して最大値または平均値を計算して
出力を得る。

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

:numref:`fig_pooling` の入力テンソル `X` を作れば、
[**2次元 max-pooling 層の出力を確認**] できる。

```{.python .input}
%%tab all
X = d2l.tensor([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]])
pool2d(X, (2, 2))
```

同様に、[**平均プーリング層**] も試せる。

```{.python .input}
%%tab all
pool2d(X, (2, 2), 'avg')
```

## [**パディングとストライド**]

畳み込み層と同様に、プーリング層も
出力形状を変化させる。
そして同じく、入力にパディングを施し、ストライドを調整することで、
望ましい出力形状になるよう演算を制御できる。
プーリング層におけるパディングとストライドの使い方は、
深層学習フレームワークに組み込まれた2次元 max-pooling 層を通じて確認できる。
まず、4次元の入力テンソル `X` を作る。
ここでは、サンプル数（バッチサイズ）とチャネル数はいずれも1である。

:begin_tab:`tensorflow`
TensorFlow は他のフレームワークと異なり、*channels-last* 形式の入力を前提とし、それに最適化されている。
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

プーリングは局所領域から情報を集約するため、[**深層学習フレームワークではプーリングウィンドウの大きさとストライドが一致するのが既定**] である。
たとえば、形状 `(3, 3)` のプーリングウィンドウを使うと、
既定ではストライドも `(3, 3)` になる。

```{.python .input}
%%tab mxnet
pool2d = nn.MaxPool2D(3)
# プーリングにはモデルパラメータがないため、初期化は不要である
pool2d(X)
```

```{.python .input}
%%tab pytorch
pool2d = nn.MaxPool2d(3)
# プーリングにはモデルパラメータがないため、初期化は不要である
pool2d(X)
```

```{.python .input}
%%tab tensorflow
pool2d = tf.keras.layers.MaxPool2D(pool_size=[3, 3])
# プーリングにはモデルパラメータがないため、初期化は不要である
pool2d(X)
```

```{.python .input}
%%tab jax
# プーリングにはモデルパラメータがないため、初期化は不要である
nn.max_pool(X, window_shape=(3, 3), strides=(3, 3))
```

必要であれば、[**ストライドとパディングを明示的に指定**] して、フレームワークの既定値を上書きできる。

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

もちろん、次の例が示すように、高さと幅が異なる任意の長方形プーリングウィンドウも指定できる。

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

多チャネルの入力データを扱う場合、
[**プーリング層は各入力チャネルを独立にプーリング**] し、
畳み込み層のようにチャネル方向で和を取ることはしない。
したがって、プーリング層の出力チャネル数は
入力チャネル数と等しい。
以下では、テンソル `X` と `X + 1` をチャネル次元で連結し、
2チャネル入力を構成する。

:begin_tab:`tensorflow`
TensorFlow では channels-last 形式を用いるため、
最後の次元に沿って連結する必要がある。
:end_tab:

```{.python .input}
%%tab mxnet, pytorch
X = d2l.concat((X, X + 1), 1)
X
```

```{.python .input}
%%tab tensorflow, jax
# channels-last 形式のため`dim=3`に沿って連結する
X = d2l.concat([X, X + 1], 3)
X
```

見てのとおり、プーリング後も出力チャネル数は 2 のままである。

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
TensorFlow のプーリング出力は見かけ上は異なって見えるが、
数値的には MXNet や PyTorch と同一である。
違いは次元の並び順だけであり、出力を縦方向に読めば他の実装と一致する。
:end_tab:

## まとめ

プーリングは非常に単純な演算である。名前のとおり、値のウィンドウ上で結果を集約する。ストライドやパディングなど、畳み込みに関する考え方はそのまま適用される。プーリングはチャネルに依存せず、すなわちチャネル数を変えずに各チャネルへ独立に適用される点に注意されたい。代表的な二つの手法のうち、max-pooling は平均プーリングより好まれることが多く、出力にある程度の不変性を与える。よく用いられる設定として、出力の空間解像度を4分の1にするために $2 \times 2$ のプーリングウィンドウを選ぶ方法がある。

プーリング以外にも解像度を下げる方法は数多く存在する。たとえば、stochastic pooling :cite:`Zeiler.Fergus.2013` や fractional max-pooling :cite:`Graham.2014` では、集約にランダム性を導入する。これにより、場合によっては精度がわずかに向上する。さらに、後に注意機構で見るように、出力を集約するためのより洗練された方法もある。たとえば、クエリと表現ベクトルの整列を利用する方法である。


## 演習

1. 畳み込みによって平均プーリングを実装せよ。
1. max-pooling は畳み込みだけでは実装できないことを示せ。
1. max-pooling は ReLU 演算、すなわち $\textrm{ReLU}(x) = \max(0, x)$ を用いて実現できる。
    1. ReLU 演算だけを用いて $\max (a, b)$ を表せ。
    1. これを用いて、畳み込み層と ReLU 層によって max-pooling を実装せよ。
    1. $2 \times 2$ の畳み込みには何チャネル、何層必要か。$3 \times 3$ の畳み込みではどうか。
1. プーリング層の計算コストはいくらか。プーリング層への入力サイズが $c\times h\times w$、プーリングウィンドウの形状が $p_\textrm{h}\times p_\textrm{w}$、パディングが $(p_\textrm{h}, p_\textrm{w})$、ストライドが $(s_\textrm{h}, s_\textrm{w})$ であると仮定せよ。
1. max-pooling と average pooling が異なる働きをすると考えられるのはなぜか。
1. 独立した最小プーリング層は必要だろうか。別の演算で置き換えられるだろうか。
1. プーリングに softmax 演算を用いることもできる。にもかかわらず、それほど一般的でないのはなぜだろうか。