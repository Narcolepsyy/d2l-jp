{.python .input}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow', 'jax'])
```

# パラメータ管理

アーキテクチャを選び、
ハイパーパラメータを設定したら、
次は学習ループに進む。
ここでの目標は、損失関数を最小化する
パラメータ値を見つけることである。
学習後には、将来の予測を行うために
これらのパラメータが必要になる。
さらに、場合によってはパラメータを取り出して、
別の文脈で再利用したり、
モデルをディスクに保存して
他のソフトウェアで実行できるようにしたり、
あるいは科学的理解を得ることを期待して
調べたりしたいこともある。

ほとんどの場合、パラメータがどのように宣言され、
操作されるかという細かな詳細は気にせず、
深層学習フレームワークに
重い処理を任せることができる。
しかし、標準的な層を積み重ねた
アーキテクチャから離れると、
パラメータの宣言や操作の細部に
踏み込む必要が出てくることがある。
この節では、次の内容を扱う。

* デバッグ、診断、可視化のためのパラメータへのアクセス。
* 異なるモデル構成要素間でのパラメータ共有。

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

[**まずは隠れ層が1つのMLPに注目する。**]

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
net = nn.Sequential(nn.LazyLinear(8),
                    nn.ReLU(),
                    nn.LazyLinear(1))

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

## [**パラメータへのアクセス**]
:label:`subsec_param-access`

まずは、すでに知っているモデルから
パラメータにアクセスする方法を見ていこう。

:begin_tab:`mxnet, pytorch, tensorflow`
モデルが `Sequential` クラスを使って定義されている場合、
まずモデルをリストのようにインデックス指定して
任意の層にアクセスできる。
各層のパラメータは、その属性に
簡単に格納されている。
:end_tab:

:begin_tab:`jax`
Flax と JAX では、先ほど定義したモデルで見たように、
モデルとパラメータが分離されている。
モデルが `Sequential` クラスを使って定義されている場合、
まずネットワークを初期化して
パラメータ辞書を生成する必要がある。
任意の層のパラメータには、この辞書のキーを通してアクセスできる。
:end_tab:

次のようにして、2番目の全結合層のパラメータを調べられる。

```{.python .input}
%%tab mxnet
net[1].params
```

```{.python .input}
%%tab pytorch
net[2].state_dict()
```

```{.python .input}
%%tab tensorflow
net.layers[2].weights
```

```{.python .input}
%%tab jax
params['params']['layers_2']
```

この全結合層には2つのパラメータが含まれており、
それぞれその層の重みとバイアスに対応している。

### [**対象を絞ったパラメータ**]

各パラメータは
パラメータクラスのインスタンスとして表現されることに注意してほしい。
パラメータを使って何か有用なことをするには、
まずその内部の数値を取り出す必要がある。
その方法はいくつかある。
より簡単なものもあれば、より一般的なものもある。
次のコードは、2番目のニューラルネットワーク層から
バイアスを取り出す。これはパラメータクラスのインスタンスを返し、
さらにそのパラメータの値にアクセスする。

```{.python .input}
%%tab mxnet
type(net[1].bias), net[1].bias.data()
```

```{.python .input}
%%tab pytorch
type(net[2].bias), net[2].bias.data
```

```{.python .input}
%%tab tensorflow
type(net.layers[2].weights[1]), tf.convert_to_tensor(net.layers[2].weights[1])
```

```{.python .input}
%%tab jax
bias = params['params']['layers_2']['bias']
type(bias), bias
```

:begin_tab:`mxnet,pytorch`
パラメータは、値、勾配、その他の情報を含む
複雑なオブジェクトである。
そのため、値を明示的に要求する必要がある。

値に加えて、各パラメータからは勾配にもアクセスできる。このネットワークではまだ逆伝播を呼び出していないため、初期状態のままである。
:end_tab:

:begin_tab:`jax`
他のフレームワークとは異なり、JAX はニューラルネットワークのパラメータに対する勾配を追跡しない。その代わり、パラメータとネットワークは分離されている。
これにより、ユーザーは計算を Python 関数として表現し、同じ目的のために `grad` 変換を使うことができる。
:end_tab:

```{.python .input}
%%tab mxnet
net[1].weight.grad()
```

```{.python .input}
%%tab pytorch
net[2].weight.grad == None
```

### [**すべてのパラメータを一度に**]

すべてのパラメータに対して操作を行う必要があるとき、
1つずつアクセスするのは面倒になりがちである。
特に、より複雑な、たとえば入れ子になったモジュールを扱う場合には、
各サブモジュールのパラメータを取り出すために
ツリー全体を再帰的にたどる必要があるため、
状況はさらに扱いにくくなる。以下では、すべての層のパラメータにアクセスする方法を示す。

```{.python .input}
%%tab mxnet
net.collect_params()
```

```{.python .input}
%%tab pytorch
[(name, param.shape) for name, param in net.named_parameters()]
```

```{.python .input}
%%tab tensorflow
net.get_weights()
```

```{.python .input}
%%tab jax
jax.tree_util.tree_map(lambda x: x.shape, params)
```

## [**共有パラメータ**]

しばしば、複数の層にまたがってパラメータを共有したいことがある。
それをエレガントに行う方法を見てみよう。
以下では全結合層を1つ用意し、
そのパラメータを使って別の層のパラメータを設定する。
ここでは、パラメータにアクセスする前に
順伝播 `net(X)` を実行する必要がある。

```{.python .input}
%%tab mxnet
net = nn.Sequential()
# We need to give the shared layer a name so that we can refer to its
# parameters
shared = nn.Dense(8, activation='relu')
net.add(nn.Dense(8, activation='relu'),
        shared,
        nn.Dense(8, activation='relu', params=shared.params),
        nn.Dense(10))
net.initialize()

X = np.random.uniform(size=(2, 20))

net(X)
# Check whether the parameters are the same
print(net[1].weight.data()[0] == net[2].weight.data()[0])
net[1].weight.data()[0, 0] = 100
# Make sure that they are actually the same object rather than just having the
# same value
print(net[1].weight.data()[0] == net[2].weight.data()[0])
```

```{.python .input}
%%tab pytorch
# We need to give the shared layer a name so that we can refer to its
# parameters
shared = nn.LazyLinear(8)
net = nn.Sequential(nn.LazyLinear(8), nn.ReLU(),
                    shared, nn.ReLU(),
                    shared, nn.ReLU(),
                    nn.LazyLinear(1))

net(X)
# Check whether the parameters are the same
print(net[2].weight.data[0] == net[4].weight.data[0])
net[2].weight.data[0, 0] = 100
# Make sure that they are actually the same object rather than just having the
# same value
print(net[2].weight.data[0] == net[4].weight.data[0])
```

```{.python .input}
%%tab tensorflow
# tf.keras behaves a bit differently. It removes the duplicate layer
# automatically
shared = tf.keras.layers.Dense(4, activation=tf.nn.relu)
net = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),
    shared,
    shared,
    tf.keras.layers.Dense(1),
])

net(X)
# Check whether the parameters are different
print(len(net.layers) == 3)
```

```{.python .input}
%%tab jax
# We need to give the shared layer a name so that we can refer to its
# parameters
shared = nn.Dense(8)
net = nn.Sequential([nn.Dense(8), nn.relu,
                     shared, nn.relu,
                     shared, nn.relu,
                     nn.Dense(1)])

params = net.init(jax.random.PRNGKey(d2l.get_seed()), X)

# Check whether the parameters are different
print(len(params['params']) == 3)
```

この例は、第2層と第3層のパラメータが
結び付けられていることを示している。
それらは単に等しいだけではなく、
まったく同じテンソルとして表現されている。
したがって、どちらか一方のパラメータを変更すると、
もう一方も変化する。

:begin_tab:`mxnet, pytorch, tensorflow`
パラメータが共有されているとき、
勾配はどうなるのか疑問に思うかもしれない。
モデルのパラメータには勾配が含まれているため、
第2の隠れ層と第3の隠れ層の勾配は
逆伝播の際に加算される。
:end_tab:


## まとめ

モデルパラメータにアクセスし、共有するための
いくつかの方法がある。


## 演習

1. :numref:`sec_model_construction` で定義した `NestMLP` モデルを使い、各層のパラメータにアクセスせよ。
1. 共有パラメータ層を含むMLPを構成して学習せよ。学習の過程で、各層のモデルパラメータと勾配を観察せよ。
1. パラメータ共有はなぜ良い考えなのか。
