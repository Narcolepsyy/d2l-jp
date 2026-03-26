{.python .input}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow', 'jax'])
```

# ファイル入出力

これまで、データの処理方法や、深層学習モデルの構築、学習、テストの方法について議論してきました。
しかし、いずれは学習済みモデルに十分満足し、さまざまな文脈で後から利用するために結果を保存したくなるはずです
（あるいは、デプロイ時に予測を行うためかもしれません）。
さらに、長時間の学習処理を実行しているときには、途中結果を定期的に保存する（チェックポイントを取る）ことがベストプラクティスです。
そうしておけば、サーバーの電源コードにつまずいてしまっても、数日分の計算を失わずに済みます。
そこで、個々の重みベクトルとモデル全体の両方を、どのように読み書きするかを学ぶ時が来ました。
この節ではその両方を扱います。

```{.python .input}
%%tab mxnet
from mxnet import np, npx
from mxnet.gluon import nn
npx.set_np()
```

```{.python .input}
%%tab pytorch
import torch
from torch import nn
from torch.nn import functional as F
```

```{.python .input}
%%tab tensorflow
import tensorflow as tf
import numpy as np
```

```{.python .input}
%%tab jax
from d2l import jax as d2l
import flax
from flax import linen as nn
from flax.training import checkpoints
import jax
from jax import numpy as jnp
```

## [**テンソルの読み込みと保存**]

個々のテンソルについては、`load` と `save` 関数を直接呼び出して、それぞれ読み込みと書き込みを行えます。
どちらの関数も名前を指定する必要があり、`save` には保存する変数を入力として与える必要があります。

```{.python .input}
%%tab mxnet
x = np.arange(4)
npx.save('x-file', x)
```

```{.python .input}
%%tab pytorch
x = torch.arange(4)
torch.save(x, 'x-file')
```

```{.python .input}
%%tab tensorflow
x = tf.range(4)
np.save('x-file.npy', x)
```

```{.python .input}
%%tab jax
x = jnp.arange(4)
jnp.save('x-file.npy', x)
```

これで、保存されたファイルからデータをメモリに読み戻せます。

```{.python .input}
%%tab mxnet
x2 = npx.load('x-file')
x2
```

```{.python .input}
%%tab pytorch
x2 = torch.load('x-file')
x2
```

```{.python .input}
%%tab tensorflow
x2 = np.load('x-file.npy', allow_pickle=True)
x2
```

```{.python .input}
%%tab jax
x2 = jnp.load('x-file.npy', allow_pickle=True)
x2
```

[**テンソルのリストを保存して、メモリに読み戻すこともできます。**]

```{.python .input}
%%tab mxnet
y = np.zeros(4)
npx.save('x-files', [x, y])
x2, y2 = npx.load('x-files')
(x2, y2)
```

```{.python .input}
%%tab pytorch
y = torch.zeros(4)
torch.save([x, y],'x-files')
x2, y2 = torch.load('x-files')
(x2, y2)
```

```{.python .input}
%%tab tensorflow
y = tf.zeros(4)
np.save('xy-files.npy', [x, y])
x2, y2 = np.load('xy-files.npy', allow_pickle=True)
(x2, y2)
```

```{.python .input}
%%tab jax
y = jnp.zeros(4)
jnp.save('xy-files.npy', [x, y])
x2, y2 = jnp.load('xy-files.npy', allow_pickle=True)
(x2, y2)
```

さらに、文字列からテンソルへの対応を表す辞書を[**書き込んで読み込むこともできます。**]
これは、モデル内のすべての重みを読み書きしたいときに便利です。

```{.python .input}
%%tab mxnet
mydict = {'x': x, 'y': y}
npx.save('mydict', mydict)
mydict2 = npx.load('mydict')
mydict2
```

```{.python .input}
%%tab pytorch
mydict = {'x': x, 'y': y}
torch.save(mydict, 'mydict')
mydict2 = torch.load('mydict')
mydict2
```

```{.python .input}
%%tab tensorflow
mydict = {'x': x, 'y': y}
np.save('mydict.npy', mydict)
mydict2 = np.load('mydict.npy', allow_pickle=True)
mydict2
```

```{.python .input}
%%tab jax
mydict = {'x': x, 'y': y}
jnp.save('mydict.npy', mydict)
mydict2 = jnp.load('mydict.npy', allow_pickle=True)
mydict2
```

## [**モデルパラメータの読み込みと保存**]

個々の重みベクトル（あるいは他のテンソル）を保存するのは便利ですが、モデル全体を保存（および後で読み込み）したい場合には非常に面倒になります。
何しろ、あちこちに散らばった多数のパラメータ群があるかもしれないからです。
このため、深層学習フレームワークには、ネットワーク全体を読み込み・保存するための組み込み機能が用意されています。
ここで注意すべき重要な点は、これが保存するのはモデルの*パラメータ*であって、モデル全体ではないということです。
たとえば、3層のMLPがある場合、アーキテクチャは別途指定する必要があります。
その理由は、モデル自体が任意のコードを含みうるため、自然な形ではシリアライズできないからです。
したがって、モデルを復元するには、コードでアーキテクチャを生成し、その後でディスクからパラメータを読み込む必要があります。
[**まずはおなじみのMLPから始めましょう。**]

```{.python .input}
%%tab mxnet
class MLP(nn.Block):
    def __init__(self, **kwargs):
        super(MLP, self).__init__(**kwargs)
        self.hidden = nn.Dense(256, activation='relu')
        self.output = nn.Dense(10)

    def forward(self, x):
        return self.output(self.hidden(x))

net = MLP()
net.initialize()
X = np.random.uniform(size=(2, 20))
Y = net(X)
```

```{.python .input}
%%tab pytorch
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.LazyLinear(256)
        self.output = nn.LazyLinear(10)

    def forward(self, x):
        return self.output(F.relu(self.hidden(x)))

net = MLP()
X = torch.randn(size=(2, 20))
Y = net(X)
```

```{.python .input}
%%tab tensorflow
class MLP(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.flatten = tf.keras.layers.Flatten()
        self.hidden = tf.keras.layers.Dense(units=256, activation=tf.nn.relu)
        self.out = tf.keras.layers.Dense(units=10)

    def call(self, inputs):
        x = self.flatten(inputs)
        x = self.hidden(x)
        return self.out(x)

net = MLP()
X = tf.random.uniform((2, 20))
Y = net(X)
```

```{.python .input}
%%tab jax
class MLP(nn.Module):
    def setup(self):
        self.hidden = nn.Dense(256)
        self.output = nn.Dense(10)

    def __call__(self, x):
        return self.output(nn.relu(self.hidden(x)))

net = MLP()
X = jax.random.normal(jax.random.PRNGKey(d2l.get_seed()), (2, 20))
Y, params = net.init_with_output(jax.random.PRNGKey(d2l.get_seed()), X)
```

次に、モデルのパラメータを "mlp.params" という名前のファイルとして[**保存します**]。

```{.python .input}
%%tab mxnet
net.save_parameters('mlp.params')
```

```{.python .input}
%%tab pytorch
torch.save(net.state_dict(), 'mlp.params')
```

```{.python .input}
%%tab tensorflow
net.save_weights('mlp.params')
```

```{.python .input}
%%tab jax
checkpoints.save_checkpoint('ckpt_dir', params, step=1, overwrite=True)
```

モデルを復元するには、元のMLPモデルのクローンをインスタンス化します。
モデルパラメータをランダムに初期化する代わりに、[**ファイルに保存されたパラメータを直接読み込みます**]。

```{.python .input}
%%tab mxnet
clone = MLP()
clone.load_parameters('mlp.params')
```

```{.python .input}
%%tab pytorch
clone = MLP()
clone.load_state_dict(torch.load('mlp.params'))
clone.eval()
```

```{.python .input}
%%tab tensorflow
clone = MLP()
clone.load_weights('mlp.params')
```

```{.python .input}
%%tab jax
clone = MLP()
cloned_params = flax.core.freeze(checkpoints.restore_checkpoint('ckpt_dir',
                                                                target=None))
```

両方のインスタンスは同じモデルパラメータを持っているので、同じ入力 `X` に対する計算結果も同じになるはずです。
確認してみましょう。

```{.python .input}
%%tab pytorch, mxnet, tensorflow
Y_clone = clone(X)
Y_clone == Y
```

```{.python .input}
%%tab jax
Y_clone = clone.apply(cloned_params, X)
Y_clone == Y
```

## まとめ

`save` と `load` 関数は、テンソルオブジェクトに対するファイル入出力に使えます。
ネットワーク全体のパラメータ集合は、パラメータ辞書を介して保存・読み込みできます。
アーキテクチャの保存は、パラメータではなくコードで行う必要があります。

## 演習

1. 学習済みモデルを別のデバイスにデプロイする必要がない場合でも、モデルパラメータを保存する実用上の利点は何ですか？
1. 異なるアーキテクチャを持つネットワークに組み込むために、ネットワークの一部だけを再利用したいとします。たとえば、以前のネットワークの最初の2層を新しいネットワークで使うには、どのようにすればよいでしょうか？
1. ネットワークのアーキテクチャとパラメータを保存するには、どのようにすればよいでしょうか？ アーキテクチャにはどのような制約を課しますか？\n
