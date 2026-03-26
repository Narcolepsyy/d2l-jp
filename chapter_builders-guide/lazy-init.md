{.python .input}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow', 'jax'])
```

# 遅延初期化
:label:`sec_lazy_init`

ここまでのところ、ネットワークの構築においてかなり大雑把にやってもうまくいっているように見えたかもしれません。
具体的には、次のような直感に反することを行ってきました。これらは本来うまく動くようには思えないかもしれません。

* 入力次元を指定せずにネットワークアーキテクチャを定義した。
* 直前の層の出力次元を指定せずに層を追加した。
* さらに、モデルが何個のパラメータを持つべきかを決めるのに十分な情報を与える前に、これらのパラメータを「初期化」した。

コードが実際に動いていることに驚くかもしれません。
そもそも、深層学習フレームワークがネットワークの入力次元を知る方法はありません。
ここでの工夫は、フレームワークが*初期化を遅延*し、最初にデータをモデルに通すまで待って、その場で各層のサイズを推論することです。


後で畳み込みニューラルネットワークを扱うときには、この手法はさらに便利になります。
なぜなら、入力次元
（たとえば画像の解像度）
が、その後に続く各層の次元に影響するからです。
したがって、コードを書く時点では次元の値を知らなくてもパラメータを設定できる能力は、モデルの指定やその後の修正を大幅に簡単にしてくれます。
それでは、初期化の仕組みをさらに詳しく見ていきましょう。

```{.python .input}
%%tab mxnet
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

まず、MLP をインスタンス化してみましょう。

```{.python .input}
%%tab mxnet
net = nn.Sequential()
net.add(nn.Dense(256, activation='relu'))
net.add(nn.Dense(10))
```

```{.python .input}
%%tab pytorch
net = nn.Sequential(nn.LazyLinear(256), nn.ReLU(), nn.LazyLinear(10))
```

```{.python .input}
%%tab tensorflow
net = tf.keras.models.Sequential([
    tf.keras.layers.Dense(256, activation=tf.nn.relu),
    tf.keras.layers.Dense(10),
])
```

```{.python .input}
%%tab jax
net = nn.Sequential([nn.Dense(256), nn.relu, nn.Dense(10)])
```

この時点では、入力次元がまだ不明なので、ネットワークは入力層の重みの次元を知ることができません。

:begin_tab:`mxnet, pytorch, tensorflow`
したがって、フレームワークはまだどのパラメータも初期化していません。
以下でパラメータにアクセスしようとして確認してみましょう。
:end_tab:

:begin_tab:`jax`
:numref:`subsec_param-access` で述べたように、Jax と Flax ではパラメータとネットワーク定義は分離されており、ユーザーが両方を手動で扱います。Flax のモデルはステートレスなので、`parameters` 属性はありません。
:end_tab:

```{.python .input}
%%tab mxnet
print(net.collect_params)
print(net.collect_params())
```

```{.python .input}
%%tab pytorch
net[0].weight
```

```{.python .input}
%%tab tensorflow
[net.layers[i].get_weights() for i in range(len(net.layers))]
```

:begin_tab:`mxnet`
パラメータオブジェクト自体は存在しているものの、
各層への入力次元は -1 として表示されていることに注意してください。
MXNet では特別な値 -1 を使って、
パラメータの次元がまだ不明であることを示します。
この時点で `net[0].weight.data()` にアクセスしようとすると、
パラメータにアクセスする前にネットワークを初期化しなければならないという実行時エラーが発生します。
それでは、`initialize` メソッドを使ってパラメータを初期化しようとすると何が起こるか見てみましょう。
:end_tab:

:begin_tab:`tensorflow`
各層オブジェクトは存在していますが、重みは空です。
重みがまだ初期化されていないため、`net.get_weights()` を使うとエラーになります。
:end_tab:

```{.python .input}
%%tab mxnet
net.initialize()
net.collect_params()
```

:begin_tab:`mxnet`
ご覧のとおり、何も変わっていません。
入力次元が不明な場合、initialize の呼び出しは実際にはパラメータを初期化しません。
代わりにこの呼び出しは、パラメータを初期化したいこと
（必要に応じて、どの分布に従って初期化するかも含めて）
を MXNet に登録します。
:end_tab:

次に、ネットワークにデータを通して、
フレームワークにようやくパラメータを初期化させてみましょう。

```{.python .input}
%%tab mxnet
X = np.random.uniform(size=(2, 20))
net(X)

net.collect_params()
```

```{.python .input}
%%tab pytorch
X = torch.rand(2, 20)
net(X)

net[0].weight.shape
```

```{.python .input}
%%tab tensorflow
X = tf.random.uniform((2, 20))
net(X)
[w.shape for w in net.get_weights()]
```

```{.python .input}
%%tab jax
params = net.init(d2l.get_key(), jnp.zeros((2, 20)))
jax.tree_util.tree_map(lambda x: x.shape, params).tree_flatten_with_keys()
```

入力次元
20
が分かればすぐに、
フレームワークは 20 の値を代入することで最初の層の重み行列の形状を特定できます。
最初の層の形状が分かると、フレームワークは次の層へ進み、
計算グラフに沿って順に処理し、
すべての形状が分かるまで続けます。
この場合、遅延初期化が必要なのは最初の層だけですが、フレームワークは順次に初期化を行います。
すべてのパラメータ形状が分かると、フレームワークはようやくパラメータを初期化できます。

:begin_tab:`pytorch`
次のメソッドは、
ダミー入力をネットワークに通して
予備実行を行い、
すべてのパラメータ形状を推論したうえで
パラメータを初期化します。
これは、デフォルトのランダム初期化を望まない場合に後で使われます。
:end_tab:

:begin_tab:`jax`
Flax におけるパラメータ初期化は常に手動で行われ、ユーザーが管理します。次のメソッドはダミー入力とキーの辞書を引数に取ります。
このキー辞書には、モデルパラメータを初期化するための rngs と、dropout 層を持つモデルで dropout マスクを生成するための dropout rng が含まれます。dropout については後ほど :numref:`sec_dropout` で詳しく説明します。
最終的にこのメソッドはモデルを初期化し、パラメータを返します。
これまでの節でも、内部ではこれを使ってきました。
:end_tab:

```{.python .input}
%%tab pytorch
@d2l.add_to_class(d2l.Module)  #@save
def apply_init(self, inputs, init=None):
    self.forward(*inputs)
    if init is not None:
        self.net.apply(init)
```

```{.python .input}
%%tab jax
@d2l.add_to_class(d2l.Module)  #@save
def apply_init(self, dummy_input, key):
    params = self.init(key, *dummy_input)  # dummy_input tuple unpacked
    return params
```

## 要約

遅延初期化は便利です。フレームワークがパラメータ形状を自動的に推論できるため、アーキテクチャの修正が容易になり、よくあるエラーの原因を一つ取り除けます。
モデルにデータを通すことで、フレームワークにようやくパラメータを初期化させることができます。


## 演習

1. 最初の層には入力次元を指定するが、その後の層には指定しない場合、どうなりますか？ すぐに初期化されますか？
1. 次元が一致しないように指定した場合、どうなりますか？
1. 入力の次元が変化する場合、何をする必要がありますか？ ヒント: パラメータ共有を見てください。\n
