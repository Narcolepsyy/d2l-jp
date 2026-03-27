{.python .input}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow', 'jax'])
```

# パラメータの初期化

パラメータへのアクセス方法がわかったので、
次はそれらを適切に初期化する方法を見ていこう。
適切な初期化の必要性については :numref:`sec_numerical_stability` で議論した。
深層学習フレームワークは、各層に対してデフォルトのランダム初期化を提供している。
しかし、私たちはしばしば、さまざまな別の手順に従って重みを初期化したいことがある。フレームワークは、最も一般的に
使われる手順の多くを提供しており、さらにカスタム初期化子を作成することもできる。

```{.python .input}
%%tab mxnet
from mxnet import init, np, npx
from mxnet.gluon import nn
npx.set_np()
```

```{.python .input}
%%tab pytorch
import torch
from torch import nn
```

```{.python .input}
%%tab tensorflow
import tensorflow as tf
```

```{.python .input}
%%tab jax
from d2l import jax as d2l
from flax import linen as nn
import jax
from jax import numpy as jnp
```

:begin_tab:`mxnet`
デフォルトでは、MXNet は重みパラメータを一様分布 $U(-0.07, 0.07)$ からランダムにサンプリングして初期化し、
バイアスパラメータはゼロにする。
MXNet の `init` モジュールは、さまざまな
既定の初期化方法を提供している。
:end_tab:

:begin_tab:`pytorch`
デフォルトでは、PyTorch は重み行列とバイアス行列を
入力次元と出力次元に基づいて計算される範囲から一様にサンプリングして初期化する。
PyTorch の `nn.init` モジュールは、さまざまな
既定の初期化方法を提供している。
:end_tab:

:begin_tab:`tensorflow`
デフォルトでは、Keras は重み行列を入力次元と出力次元に基づいて計算される範囲から一様にサンプリングして初期化し、バイアスパラメータはすべてゼロに設定される。
TensorFlow は、ルートモジュールと `keras.initializers` モジュールの両方で、さまざまな初期化方法を提供している。
:end_tab:

:begin_tab:`jax`
デフォルトでは、Flax は `jax.nn.initializers.lecun_normal` を使って重みを初期化する。つまり、
平均 0、標準偏差を $1 / \textrm{fan}_{\textrm{in}}$ の平方根に設定した切断正規分布からサンプルを生成する。
ここで `fan_in` は重みテンソルにおける入力ユニット数である。バイアス
パラメータはすべてゼロに設定される。
Jax の `nn.initializers` モジュールは、さまざまな
既定の初期化方法を提供している。
:end_tab:

```{.python .input}
%%tab mxnet
net = nn.Sequential()
net.add(nn.Dense(8, activation='relu'))
net.add(nn.Dense(1))
net.initialize()  # Use the default initialization method

X = np.random.uniform(size=(2, 4))
net(X).shape
```

```{.python .input}
%%tab pytorch
net = nn.Sequential(nn.LazyLinear(8), nn.ReLU(), nn.LazyLinear(1))
X = torch.rand(size=(2, 4))
net(X).shape
```

```{.python .input}
%%tab tensorflow
net = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(4, activation=tf.nn.relu),
    tf.keras.layers.Dense(1),
])

X = tf.random.uniform((2, 4))
net(X).shape
```

```{.python .input}
%%tab jax
net = nn.Sequential([nn.Dense(8), nn.relu, nn.Dense(1)])
X = jax.random.uniform(d2l.get_key(), (2, 4))
params = net.init(d2l.get_key(), X)
net.apply(params, X).shape
```

## [**組み込み初期化**]

まずは組み込みの初期化子を呼び出してみよう。
以下のコードでは、すべての重みパラメータを
標準偏差 0.01 のガウス乱数で初期化し、バイアスパラメータはゼロにする。

```{.python .input}
%%tab mxnet
# Here force_reinit ensures that parameters are freshly initialized even if
# they were already initialized previously
net.initialize(init=init.Normal(sigma=0.01), force_reinit=True)
net[0].weight.data()[0]
```

```{.python .input}
%%tab pytorch
def init_normal(module):
    if type(module) == nn.Linear:
        nn.init.normal_(module.weight, mean=0, std=0.01)
        nn.init.zeros_(module.bias)

net.apply(init_normal)
net[0].weight.data[0], net[0].bias.data[0]
```

```{.python .input}
%%tab tensorflow
net = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(
        4, activation=tf.nn.relu,
        kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.01),
        bias_initializer=tf.zeros_initializer()),
    tf.keras.layers.Dense(1)])

net(X)
net.weights[0], net.weights[1]
```

```{.python .input}
%%tab jax
weight_init = nn.initializers.normal(0.01)
bias_init = nn.initializers.zeros

net = nn.Sequential([nn.Dense(8, kernel_init=weight_init, bias_init=bias_init),
                     nn.relu,
                     nn.Dense(1, kernel_init=weight_init, bias_init=bias_init)])

params = net.init(jax.random.PRNGKey(d2l.get_seed()), X)
layer_0 = params['params']['layers_0']
layer_0['kernel'][:, 0], layer_0['bias'][0]
```

すべてのパラメータを
ある定数値（たとえば 1）に初期化することもできる。

```{.python .input}
%%tab mxnet
net.initialize(init=init.Constant(1), force_reinit=True)
net[0].weight.data()[0]
```

```{.python .input}
%%tab pytorch
def init_constant(module):
    if type(module) == nn.Linear:
        nn.init.constant_(module.weight, 1)
        nn.init.zeros_(module.bias)

net.apply(init_constant)
net[0].weight.data[0], net[0].bias.data[0]
```

```{.python .input}
%%tab tensorflow
net = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(
        4, activation=tf.nn.relu,
        kernel_initializer=tf.keras.initializers.Constant(1),
        bias_initializer=tf.zeros_initializer()),
    tf.keras.layers.Dense(1),
])

net(X)
net.weights[0], net.weights[1]
```

```{.python .input}
%%tab jax
weight_init = nn.initializers.constant(1)

net = nn.Sequential([nn.Dense(8, kernel_init=weight_init, bias_init=bias_init),
                     nn.relu,
                     nn.Dense(1, kernel_init=weight_init, bias_init=bias_init)])

params = net.init(jax.random.PRNGKey(d2l.get_seed()), X)
layer_0 = params['params']['layers_0']
layer_0['kernel'][:, 0], layer_0['bias'][0]
```

[**特定のブロックに対して異なる初期化子を適用することもできる。**]
たとえば、以下では第1層を
Xavier 初期化子で初期化し、第2層を
定数 42 で初期化する。

```{.python .input}
%%tab mxnet
net[0].weight.initialize(init=init.Xavier(), force_reinit=True)
net[1].initialize(init=init.Constant(42), force_reinit=True)
print(net[0].weight.data()[0])
print(net[1].weight.data())
```

```{.python .input}
%%tab pytorch
def init_xavier(module):
    if type(module) == nn.Linear:
        nn.init.xavier_uniform_(module.weight)

def init_42(module):
    if type(module) == nn.Linear:
        nn.init.constant_(module.weight, 42)

net[0].apply(init_xavier)
net[2].apply(init_42)
print(net[0].weight.data[0])
print(net[2].weight.data)
```

```{.python .input}
%%tab tensorflow
net = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(
        4,
        activation=tf.nn.relu,
        kernel_initializer=tf.keras.initializers.GlorotUniform()),
    tf.keras.layers.Dense(
        1, kernel_initializer=tf.keras.initializers.Constant(42)),
])

net(X)
print(net.layers[1].weights[0])
print(net.layers[2].weights[0])
```

```{.python .input}
%%tab jax
net = nn.Sequential([nn.Dense(8, kernel_init=nn.initializers.xavier_uniform(),
                              bias_init=bias_init),
                     nn.relu,
                     nn.Dense(1, kernel_init=nn.initializers.constant(42),
                              bias_init=bias_init)])

params = net.init(jax.random.PRNGKey(d2l.get_seed()), X)
params['params']['layers_0']['kernel'][:, 0], params['params']['layers_2']['kernel']
```

### [**カスタム初期化**]

ときには、必要な初期化方法が
深層学習フレームワークに用意されていないことがある。
以下の例では、任意の重みパラメータ $w$ に対して、次の奇妙な分布を用いる初期化子を定義する。

$$
\begin{aligned}
    w \sim \begin{cases}
        U(5, 10) & \textrm{ with probability } \frac{1}{4} \\
            0    & \textrm{ with probability } \frac{1}{2} \\
        U(-10, -5) & \textrm{ with probability } \frac{1}{4}
    \end{cases}
\end{aligned}
$$

:begin_tab:`mxnet`
ここでは `Initializer` クラスのサブクラスを定義する。
通常は、`_init_weight` 関数を実装するだけで十分である。
この関数はテンソル引数（`data`）を受け取り、
そこに望みの初期化値を代入する。
:end_tab:

:begin_tab:`pytorch`
ここでも、`net` に適用する `my_init` 関数を実装する。
:end_tab:

:begin_tab:`tensorflow`
ここでは `Initializer` のサブクラスを定義し、形状とデータ型が与えられたときに望みのテンソルを返す `__call__`
関数を実装する。
:end_tab:

:begin_tab:`jax`
Jax の初期化関数は、引数として `PRNGKey`、`shape`、および
`dtype` を取る。ここでは、形状とデータ型が与えられたときに望みの
テンソルを返す関数 `my_init` を実装する。
:end_tab:

```{.python .input}
%%tab mxnet
class MyInit(init.Initializer):
    def _init_weight(self, name, data):
        print('Init', name, data.shape)
        data[:] = np.random.uniform(-10, 10, data.shape)
        data *= np.abs(data) >= 5

net.initialize(MyInit(), force_reinit=True)
net[0].weight.data()[:2]
```

```{.python .input}
%%tab pytorch
def my_init(module):
    if type(module) == nn.Linear:
        print("Init", *[(name, param.shape)
                        for name, param in module.named_parameters()][0])
        nn.init.uniform_(module.weight, -10, 10)
        module.weight.data *= module.weight.data.abs() >= 5

net.apply(my_init)
net[0].weight[:2]
```

```{.python .input}
%%tab tensorflow
class MyInit(tf.keras.initializers.Initializer):
    def __call__(self, shape, dtype=None):
        data=tf.random.uniform(shape, -10, 10, dtype=dtype)
        factor=(tf.abs(data) >= 5)
        factor=tf.cast(factor, tf.float32)
        return data * factor

net = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(
        4,
        activation=tf.nn.relu,
        kernel_initializer=MyInit()),
    tf.keras.layers.Dense(1),
])

net(X)
print(net.layers[1].weights[0])
```

```{.python .input}
%%tab jax
def my_init(key, shape, dtype=jnp.float_):
    data = jax.random.uniform(key, shape, minval=-10, maxval=10)
    return data * (jnp.abs(data) >= 5)

net = nn.Sequential([nn.Dense(8, kernel_init=my_init), nn.relu, nn.Dense(1)])
params = net.init(d2l.get_key(), X)
print(params['params']['layers_0']['kernel'][:, :2])
```

:begin_tab:`mxnet, pytorch, tensorflow`
パラメータを直接設定するという選択肢も常にあることに注意してほしい。
:end_tab:

:begin_tab:`jax`
JAX と Flax でパラメータを初期化すると、返されるパラメータの辞書は
`flax.core.frozen_dict.FrozenDict` 型になる。Jax のエコシステムでは配列の値を直接変更することは推奨されないため、データ型は一般に不変である。変更を加えるには `params.unfreeze()` を使うことができる。
:end_tab:

```{.python .input}
%%tab mxnet
net[0].weight.data()[:] += 1
net[0].weight.data()[0, 0] = 42
net[0].weight.data()[0]
```

```{.python .input}
%%tab pytorch
net[0].weight.data[:] += 1
net[0].weight.data[0, 0] = 42
net[0].weight.data[0]
```

```{.python .input}
%%tab tensorflow
net.layers[1].weights[0][:].assign(net.layers[1].weights[0] + 1)
net.layers[1].weights[0][0, 0].assign(42)
net.layers[1].weights[0]
```

## まとめ

組み込み初期化子とカスタム初期化子を使ってパラメータを初期化できる。

## 演習

さらに多くの組み込み初期化子についてオンラインドキュメントを調べてみよう。
