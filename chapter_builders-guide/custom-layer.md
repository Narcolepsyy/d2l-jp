{.python .input}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow', 'jax'])
```

# カスタム層

ディープラーニングの成功を支える要因の一つは、
さまざまな層が利用できることである。
それらを創造的に組み合わせることで、
多種多様なタスクに適したアーキテクチャを設計できる。
たとえば研究者たちは、画像、テキスト、
系列データの反復処理、
そして
動的計画法の実行に特化した層を発明してきた。
遅かれ早かれ、ディープラーニングフレームワークにはまだ存在しない層が必要になるだろう。
そのような場合には、カスタム層を作成しなければならない。
この節では、その方法を示す。

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
from torch.nn import functional as F
```

```{.python .input}
%%tab tensorflow
from d2l import tensorflow as d2l
import tensorflow as tf
```

```{.python .input}
%%tab jax
from d2l import jax as d2l
from flax import linen as nn
import jax
from jax import numpy as jnp
```

## [**パラメータを持たない層**]

まず、独自のパラメータを一切持たないカスタム層を構築してみよう。
これは、 :numref:`sec_model_construction` でモジュールを導入したときの内容を覚えていれば、見覚えがあるはずである。
次の `CenteredLayer` クラスは、入力から単に平均を引くだけである。
これを作るには、基底層クラスを継承し、順伝播関数を実装するだけで十分である。

```{.python .input}
%%tab mxnet
class CenteredLayer(nn.Block):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, X):
        return X - X.mean()
```

```{.python .input}
%%tab pytorch
class CenteredLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, X):
        return X - X.mean()
```

```{.python .input}
%%tab tensorflow
class CenteredLayer(tf.keras.Model):
    def __init__(self):
        super().__init__()

    def call(self, X):
        return X - tf.reduce_mean(X)
```

```{.python .input}
%%tab jax
class CenteredLayer(nn.Module):
    def __call__(self, X):
        return X - X.mean()
```

いくつかのデータをこの層に通して、意図したとおりに動作することを確認しよう。

```{.python .input}
%%tab all
layer = CenteredLayer()
layer(d2l.tensor([1.0, 2, 3, 4, 5]))
```

これで、[**この層をより複雑なモデルを構築する際の構成要素として組み込める。**]

```{.python .input}
%%tab mxnet
net = nn.Sequential()
net.add(nn.Dense(128), CenteredLayer())
net.initialize()
```

```{.python .input}
%%tab pytorch
net = nn.Sequential(nn.LazyLinear(128), CenteredLayer())
```

```{.python .input}
%%tab tensorflow
net = tf.keras.Sequential([tf.keras.layers.Dense(128), CenteredLayer()])
```

```{.python .input}
%%tab jax
net = nn.Sequential([nn.Dense(128), CenteredLayer()])
```

追加の健全性チェックとして、ランダムなデータを
ネットワークに通し、平均が実際に 0 であることを確認できる。
浮動小数点数を扱っているため、
量子化の影響で
ごく小さな 0 でない値が見えることもある。

:begin_tab:`jax`
ここでは `init_with_output` メソッドを利用する。これは
ネットワークの出力とパラメータの両方を返す。この場合は出力のみに注目する。
:end_tab:

```{.python .input}
%%tab pytorch, mxnet
Y = net(d2l.rand(4, 8))
Y.mean()
```

```{.python .input}
%%tab tensorflow
Y = net(tf.random.uniform((4, 8)))
tf.reduce_mean(Y)
```

```{.python .input}
%%tab jax
Y, _ = net.init_with_output(d2l.get_key(), jax.random.uniform(d2l.get_key(),
                                                              (4, 8)))
Y.mean()
```

## [**パラメータを持つ層**]

単純な層の定義方法がわかったので、
次は、学習によって調整できるパラメータを持つ層の定義に進もう。
組み込み関数を使ってパラメータを作成でき、
それらは基本的な管理機能を提供する。
特に、アクセス、初期化、
共有、保存、モデルパラメータの読み込みを管理する。
このようにして、ほかにも利点はあるが、カスタム層ごとに
独自のシリアライズ処理を書く必要がなくなる。

では、全結合層の独自実装を行ってみよう。
この層には 2 つのパラメータが必要であることを思い出してほしい。
一つは重みを表し、もう一つはバイアスである。
この実装では、ReLU 活性化をデフォルトとして組み込んでいる。
この層は `in_units` と `units` の 2 つの入力引数を必要とし、
それぞれ入力数と出力数を表す。

```{.python .input}
%%tab mxnet
class MyDense(nn.Block):
    def __init__(self, units, in_units, **kwargs):
        super().__init__(**kwargs)
        self.weight = self.params.get('weight', shape=(in_units, units))
        self.bias = self.params.get('bias', shape=(units,))

    def forward(self, x):
        linear = np.dot(x, self.weight.data(ctx=x.ctx)) + self.bias.data(
            ctx=x.ctx)
        return npx.relu(linear)
```

```{.python .input}
%%tab pytorch
class MyLinear(nn.Module):
    def __init__(self, in_units, units):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(in_units, units))
        self.bias = nn.Parameter(torch.randn(units,))
        
    def forward(self, X):
        linear = torch.matmul(X, self.weight.data) + self.bias.data
        return F.relu(linear)
```

```{.python .input}
%%tab tensorflow
class MyDense(tf.keras.Model):
    def __init__(self, units):
        super().__init__()
        self.units = units

    def build(self, X_shape):
        self.weight = self.add_weight(name='weight',
            shape=[X_shape[-1], self.units],
            initializer=tf.random_normal_initializer())
        self.bias = self.add_weight(
            name='bias', shape=[self.units],
            initializer=tf.zeros_initializer())

    def call(self, X):
        linear = tf.matmul(X, self.weight) + self.bias
        return tf.nn.relu(linear)
```

```{.python .input}
%%tab jax
class MyDense(nn.Module):
    in_units: int
    units: int

    def setup(self):
        self.weight = self.param('weight', nn.initializers.normal(stddev=1),
                                 (self.in_units, self.units))
        self.bias = self.param('bias', nn.initializers.zeros, self.units)

    def __call__(self, X):
        linear = jnp.matmul(X, self.weight) + self.bias
        return nn.relu(linear)
```

:begin_tab:`mxnet, tensorflow, jax`
次に、`MyDense` クラスをインスタンス化し、
そのモデルパラメータにアクセスする。
:end_tab:

:begin_tab:`pytorch`
次に、`MyLinear` クラスをインスタンス化し、
そのモデルパラメータにアクセスする。
:end_tab:

```{.python .input}
%%tab mxnet
dense = MyDense(units=3, in_units=5)
dense.params
```

```{.python .input}
%%tab pytorch
linear = MyLinear(5, 3)
linear.weight
```

```{.python .input}
%%tab tensorflow
dense = MyDense(3)
dense(tf.random.uniform((2, 5)))
dense.get_weights()
```

```{.python .input}
%%tab jax
dense = MyDense(5, 3)
params = dense.init(d2l.get_key(), jnp.zeros((3, 5)))
params
```

[**カスタム層を使って、順伝播計算を直接実行することもできる。**]

```{.python .input}
%%tab mxnet
dense.initialize()
dense(np.random.uniform(size=(2, 5)))
```

```{.python .input}
%%tab pytorch
linear(torch.rand(2, 5))
```

```{.python .input}
%%tab tensorflow
dense(tf.random.uniform((2, 5)))
```

```{.python .input}
%%tab jax
dense.apply(params, jax.random.uniform(d2l.get_key(),
                                       (2, 5)))
```

また、[**カスタム層を使ってモデルを構築することもできる。**]
一度それができれば、組み込みの全結合層と同じように使える。

```{.python .input}
%%tab mxnet
net = nn.Sequential()
net.add(MyDense(8, in_units=64),
        MyDense(1, in_units=8))
net.initialize()
net(np.random.uniform(size=(2, 64)))
```

```{.python .input}
%%tab pytorch
net = nn.Sequential(MyLinear(64, 8), MyLinear(8, 1))
net(torch.rand(2, 64))
```

```{.python .input}
%%tab tensorflow
net = tf.keras.models.Sequential([MyDense(8), MyDense(1)])
net(tf.random.uniform((2, 64)))
```

```{.python .input}
%%tab jax
net = nn.Sequential([MyDense(64, 8), MyDense(8, 1)])
Y, _ = net.init_with_output(d2l.get_key(), jax.random.uniform(d2l.get_key(),
                                                              (2, 64)))
Y
```

## まとめ

基本的な層クラスを使ってカスタム層を設計できる。これにより、ライブラリに既存のどの層とも異なる、柔軟な新しい層を定義できる。
一度定義すれば、カスタム層は任意の文脈やアーキテクチャで呼び出せる。
層はローカルなパラメータを持つことができ、それらは組み込み関数を通じて作成できる。


## 演習

1. 入力を受け取りテンソル縮約を計算する層を設計せよ。つまり、$y_k = \sum_{i, j} W_{ijk} x_i x_j$ を返す層である。
1. データのフーリエ係数の前半を返す層を設計せよ。
