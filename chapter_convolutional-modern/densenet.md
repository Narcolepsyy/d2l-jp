{.python .input}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow', 'jax'])
```

# 密に接続されたネットワーク（DenseNet）
:label:`sec_densenet`

ResNet は、深いネットワークにおける関数をどのようにパラメータ化するかという見方を大きく変えました。*DenseNet*（dense convolutional network）は、ある意味でその論理的な拡張です :cite:`Huang.Liu.Van-Der-Maaten.ea.2017`。
DenseNet の特徴は、各層がそれ以前のすべての層と接続される接続パターンと、ResNet の加算演算子ではなく連結演算を用いて、以前の層からの特徴を保持し再利用する点にあります。
これをどのように導くかを理解するために、少し数学に寄り道しましょう。

```{.python .input}
%%tab mxnet
from d2l import mxnet as d2l
from mxnet import init, np, npx
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
from d2l import tensorflow as d2l
import tensorflow as tf
```

```{.python .input}
%%tab jax
from d2l import jax as d2l
from flax import linen as nn
from jax import numpy as jnp
import jax
```

## ResNet から DenseNet へ

関数のテイラー展開を思い出してください。点 $x = 0$ においては、次のように書けます。

$$f(x) = f(0) + x \cdot \left[f'(0) + x \cdot \left[\frac{f''(0)}{2!}  + x \cdot \left[\frac{f'''(0)}{3!}  + \cdots \right]\right]\right].$$


重要なのは、関数を次数の高い項へと分解している点です。同様に、ResNet は関数を次のように分解します。

$$f(\mathbf{x}) = \mathbf{x} + g(\mathbf{x}).$$

つまり、ResNet は $f$ を単純な線形項と、より複雑な非線形項に分解します。
2項を超える情報を（必ずしも加算せずに）取り込みたいとしたらどうでしょうか？
その一つの解が DenseNet です :cite:`Huang.Liu.Van-Der-Maaten.ea.2017`。

![ResNet（左）と DenseNet（右）の層間接続における主な違い：加算の使用と連結の使用。 ](../img/densenet-block.svg)
:label:`fig_densenet_block`

:numref:`fig_densenet_block` に示すように、ResNet と DenseNet の主な違いは、後者では出力を加算するのではなく、*連結*（$[,]$ で表す）する点です。
その結果、ますます複雑な関数列を適用した後の値へと、$\mathbf{x}$ を写像します。

$$\mathbf{x} \to \left[
\mathbf{x},
f_1(\mathbf{x}),
f_2\left(\left[\mathbf{x}, f_1\left(\mathbf{x}\right)\right]\right), f_3\left(\left[\mathbf{x}, f_1\left(\mathbf{x}\right), f_2\left(\left[\mathbf{x}, f_1\left(\mathbf{x}\right)\right]\right)\right]\right), \ldots\right].$$

最終的には、これらすべての関数を MLP でまとめて、特徴数を再び減らします。実装の観点では、これは非常に単純です。
項を加算する代わりに、それらを連結するだけです。DenseNet という名前は、変数間の依存グラフが非常に密になることに由来します。このような連鎖の最終層は、すべての前の層と密に接続されています。密な接続は :numref:`fig_densenet` に示されています。

![DenseNet における密な接続。深くなるにつれて次元が増加することに注意。](../img/densenet.svg)
:label:`fig_densenet`

DenseNet を構成する主な要素は、*dense block* と *transition layer* です。前者は入力と出力をどのように連結するかを定義し、後者はチャネル数を制御して大きくなりすぎないようにします。というのも、$\mathbf{x} \to \left[\mathbf{x}, f_1(\mathbf{x}),
f_2\left(\left[\mathbf{x}, f_1\left(\mathbf{x}\right)\right]\right), \ldots \right]$ という拡張は、かなり高次元になりうるからです。


## [**Dense Block**]

DenseNet は、ResNet の「バッチ正規化、活性化、畳み込み」を組み合わせた修正版の構造を使います（:numref:`sec_resnet` の演習を参照）。
まず、この畳み込みブロック構造を実装します。

```{.python .input}
%%tab mxnet
def conv_block(num_channels):
    blk = nn.Sequential()
    blk.add(nn.BatchNorm(),
            nn.Activation('relu'),
            nn.Conv2D(num_channels, kernel_size=3, padding=1))
    return blk
```

```{.python .input}
%%tab pytorch
def conv_block(num_channels):
    return nn.Sequential(
        nn.LazyBatchNorm2d(), nn.ReLU(),
        nn.LazyConv2d(num_channels, kernel_size=3, padding=1))
```

```{.python .input}
%%tab tensorflow
class ConvBlock(tf.keras.layers.Layer):
    def __init__(self, num_channels):
        super(ConvBlock, self).__init__()
        self.bn = tf.keras.layers.BatchNormalization()
        self.relu = tf.keras.layers.ReLU()
        self.conv = tf.keras.layers.Conv2D(
            filters=num_channels, kernel_size=(3, 3), padding='same')

        self.listLayers = [self.bn, self.relu, self.conv]

    def call(self, x):
        y = x
        for layer in self.listLayers.layers:
            y = layer(y)
        y = tf.keras.layers.concatenate([x,y], axis=-1)
        return y
```

```{.python .input}
%%tab jax
class ConvBlock(nn.Module):
    num_channels: int
    training: bool = True

    @nn.compact
    def __call__(self, X):
        Y = nn.relu(nn.BatchNorm(not self.training)(X))
        Y = nn.Conv(self.num_channels, kernel_size=(3, 3), padding=(1, 1))(Y)
        Y = jnp.concatenate((X, Y), axis=-1)
        return Y
```

*dense block* は複数の畳み込みブロックからなり、それぞれが同じ数の出力チャネルを使います。ただし順伝播では、各畳み込みブロックの入力と出力をチャネル次元で連結します。遅延評価により、次元を自動的に調整できます。

```{.python .input}
%%tab mxnet
class DenseBlock(nn.Block):
    def __init__(self, num_convs, num_channels):
        super().__init__()
        self.net = nn.Sequential()
        for _ in range(num_convs):
            self.net.add(conv_block(num_channels))

    def forward(self, X):
        for blk in self.net:
            Y = blk(X)
            # Concatenate input and output of each block along the channels
            X = np.concatenate((X, Y), axis=1)
        return X
```

```{.python .input}
%%tab pytorch
class DenseBlock(nn.Module):
    def __init__(self, num_convs, num_channels):
        super(DenseBlock, self).__init__()
        layer = []
        for i in range(num_convs):
            layer.append(conv_block(num_channels))
        self.net = nn.Sequential(*layer)

    def forward(self, X):
        for blk in self.net:
            Y = blk(X)
            # Concatenate input and output of each block along the channels
            X = torch.cat((X, Y), dim=1)
        return X
```

```{.python .input}
%%tab tensorflow
class DenseBlock(tf.keras.layers.Layer):
    def __init__(self, num_convs, num_channels):
        super(DenseBlock, self).__init__()
        self.listLayers = []
        for _ in range(num_convs):
            self.listLayers.append(ConvBlock(num_channels))

    def call(self, x):
        for layer in self.listLayers.layers:
            x = layer(x)
        return x
```

```{.python .input}
%%tab jax
class DenseBlock(nn.Module):
    num_convs: int
    num_channels: int
    training: bool = True

    def setup(self):
        layer = []
        for i in range(self.num_convs):
            layer.append(ConvBlock(self.num_channels, self.training))
        self.net = nn.Sequential(layer)

    def __call__(self, X):
        return self.net(X)
```

次の例では、出力チャネル数が 10 の畳み込みブロックを 2 つ持つ `DenseBlock` インスタンスを[**定義**]します。
3 チャネルの入力を使うと、出力は $3 + 10 + 10=23$ チャネルになります。畳み込みブロックのチャネル数は、入力チャネル数に対する出力チャネル数の増加量を制御します。これは *growth rate* とも呼ばれます。

```{.python .input}
%%tab pytorch, mxnet, tensorflow
blk = DenseBlock(2, 10)
if tab.selected('mxnet'):
    X = np.random.uniform(size=(4, 3, 8, 8))
    blk.initialize()
if tab.selected('pytorch'):
    X = torch.randn(4, 3, 8, 8)
if tab.selected('tensorflow'):
    X = tf.random.uniform((4, 8, 8, 3))
Y = blk(X)
Y.shape
```

```{.python .input}
%%tab jax
blk = DenseBlock(2, 10)
X = jnp.zeros((4, 8, 8, 3))
Y = blk.init_with_output(d2l.get_key(), X)[0]
Y.shape
```

## [**Transition Layer**]

各 dense block はチャネル数を増やすため、それをあまり多く重ねると、モデルが過度に複雑になります。*transition layer* はモデルの複雑さを制御するために使われます。これは $1\times 1$ 畳み込みを用いてチャネル数を減らします。さらに、ストライド 2 の平均プーリングによって高さと幅を半分にします。

```{.python .input}
%%tab mxnet
def transition_block(num_channels):
    blk = nn.Sequential()
    blk.add(nn.BatchNorm(), nn.Activation('relu'),
            nn.Conv2D(num_channels, kernel_size=1),
            nn.AvgPool2D(pool_size=2, strides=2))
    return blk
```

```{.python .input}
%%tab pytorch
def transition_block(num_channels):
    return nn.Sequential(
        nn.LazyBatchNorm2d(), nn.ReLU(),
        nn.LazyConv2d(num_channels, kernel_size=1),
        nn.AvgPool2d(kernel_size=2, stride=2))
```

```{.python .input}
%%tab tensorflow
class TransitionBlock(tf.keras.layers.Layer):
    def __init__(self, num_channels, **kwargs):
        super(TransitionBlock, self).__init__(**kwargs)
        self.batch_norm = tf.keras.layers.BatchNormalization()
        self.relu = tf.keras.layers.ReLU()
        self.conv = tf.keras.layers.Conv2D(num_channels, kernel_size=1)
        self.avg_pool = tf.keras.layers.AvgPool2D(pool_size=2, strides=2)

    def call(self, x):
        x = self.batch_norm(x)
        x = self.relu(x)
        x = self.conv(x)
        return self.avg_pool(x)
```

```{.python .input}
%%tab jax
class TransitionBlock(nn.Module):
    num_channels: int
    training: bool = True

    @nn.compact
    def __call__(self, X):
        X = nn.BatchNorm(not self.training)(X)
        X = nn.relu(X)
        X = nn.Conv(self.num_channels, kernel_size=(1, 1))(X)
        X = nn.avg_pool(X, window_shape=(2, 2), strides=(2, 2))
        return X
```

前の例の dense block の出力に対して、10 チャネルの transition layer を[**適用**]します。これにより出力チャネル数は 10 に減り、高さと幅は半分になります。

```{.python .input}
%%tab mxnet
blk = transition_block(10)
blk.initialize()
blk(Y).shape
```

```{.python .input}
%%tab pytorch
blk = transition_block(10)
blk(Y).shape
```

```{.python .input}
%%tab tensorflow
blk = TransitionBlock(10)
blk(Y).shape
```

```{.python .input}
%%tab jax
blk = TransitionBlock(10)
blk.init_with_output(d2l.get_key(), Y)[0].shape
```

## [**DenseNet モデル**]

次に、DenseNet モデルを構築します。DenseNet はまず、ResNet と同じ単一の畳み込み層と max-pooling 層を使います。

```{.python .input}
%%tab pytorch, mxnet, tensorflow
class DenseNet(d2l.Classifier):
    def b1(self):
        if tab.selected('mxnet'):
            net = nn.Sequential()
            net.add(nn.Conv2D(64, kernel_size=7, strides=2, padding=3),
                nn.BatchNorm(), nn.Activation('relu'),
                nn.MaxPool2D(pool_size=3, strides=2, padding=1))
            return net
        if tab.selected('pytorch'):
            return nn.Sequential(
                nn.LazyConv2d(64, kernel_size=7, stride=2, padding=3),
                nn.LazyBatchNorm2d(), nn.ReLU(),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        if tab.selected('tensorflow'):
            return tf.keras.models.Sequential([
                tf.keras.layers.Conv2D(
                    64, kernel_size=7, strides=2, padding='same'),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.ReLU(),
                tf.keras.layers.MaxPool2D(
                    pool_size=3, strides=2, padding='same')])
```

```{.python .input}
%%tab jax
class DenseNet(d2l.Classifier):
    num_channels: int = 64
    growth_rate: int = 32
    arch: tuple = (4, 4, 4, 4)
    lr: float = 0.1
    num_classes: int = 10
    training: bool = True

    def setup(self):
        self.net = self.create_net()

    def b1(self):
        return nn.Sequential([
            nn.Conv(64, kernel_size=(7, 7), strides=(2, 2), padding='same'),
            nn.BatchNorm(not self.training),
            nn.relu,
            lambda x: nn.max_pool(x, window_shape=(3, 3),
                                  strides=(2, 2), padding='same')
        ])
```

その後、ResNet が残差ブロックからなる 4 つのモジュールを使うのと同様に、DenseNet は 4 つの dense block を使います。
ResNet と同様に、各 dense block で使う畳み込み層の数を設定できます。ここでは、 :numref:`sec_resnet` の ResNet-18 モデルと一致するように 4 に設定します。さらに、dense block 内の畳み込み層のチャネル数（すなわち growth rate）を 32 に設定するので、各 dense block には 128 チャネルが追加されます。

ResNet では、各モジュールの間でストライド 2 の残差ブロックによって高さと幅が減少します。ここでは、transition layer を使って高さと幅を半分にし、チャネル数も半分にします。ResNet と同様に、最後に global pooling 層と全結合層を接続して出力を生成します。

```{.python .input}
%%tab pytorch, mxnet, tensorflow
@d2l.add_to_class(DenseNet)
def __init__(self, num_channels=64, growth_rate=32, arch=(4, 4, 4, 4),
             lr=0.1, num_classes=10):
    super(DenseNet, self).__init__()
    self.save_hyperparameters()
    if tab.selected('mxnet'):
        self.net = nn.Sequential()
        self.net.add(self.b1())
        for i, num_convs in enumerate(arch):
            self.net.add(DenseBlock(num_convs, growth_rate))
            # The number of output channels in the previous dense block
            num_channels += num_convs * growth_rate
            # A transition layer that halves the number of channels is added
            # between the dense blocks
            if i != len(arch) - 1:
                num_channels //= 2
                self.net.add(transition_block(num_channels))
        self.net.add(nn.BatchNorm(), nn.Activation('relu'),
                     nn.GlobalAvgPool2D(), nn.Dense(num_classes))
        self.net.initialize(init.Xavier())
    if tab.selected('pytorch'):
        self.net = nn.Sequential(self.b1())
        for i, num_convs in enumerate(arch):
            self.net.add_module(f'dense_blk{i+1}', DenseBlock(num_convs,
                                                              growth_rate))
            # The number of output channels in the previous dense block
            num_channels += num_convs * growth_rate
            # A transition layer that halves the number of channels is added
            # between the dense blocks
            if i != len(arch) - 1:
                num_channels //= 2
                self.net.add_module(f'tran_blk{i+1}', transition_block(
                    num_channels))
        self.net.add_module('last', nn.Sequential(
            nn.LazyBatchNorm2d(), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(),
            nn.LazyLinear(num_classes)))
        self.net.apply(d2l.init_cnn)
    if tab.selected('tensorflow'):
        self.net = tf.keras.models.Sequential(self.b1())
        for i, num_convs in enumerate(arch):
            self.net.add(DenseBlock(num_convs, growth_rate))
            # The number of output channels in the previous dense block
            num_channels += num_convs * growth_rate
            # A transition layer that halves the number of channels is added
            # between the dense blocks
            if i != len(arch) - 1:
                num_channels //= 2
                self.net.add(TransitionBlock(num_channels))
        self.net.add(tf.keras.models.Sequential([
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.GlobalAvgPool2D(),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(num_classes)]))
```

```{.python .input}
%%tab jax
@d2l.add_to_class(DenseNet)
def create_net(self):
    net = self.b1()
    for i, num_convs in enumerate(self.arch):
        net.layers.extend([DenseBlock(num_convs, self.growth_rate,
                                      training=self.training)])
        # The number of output channels in the previous dense block
        num_channels = self.num_channels + (num_convs * self.growth_rate)
        # A transition layer that halves the number of channels is added
        # between the dense blocks
        if i != len(self.arch) - 1:
            num_channels //= 2
            net.layers.extend([TransitionBlock(num_channels,
                                               training=self.training)])
    net.layers.extend([
        nn.BatchNorm(not self.training),
        nn.relu,
        lambda x: nn.avg_pool(x, window_shape=x.shape[1:3],
                              strides=x.shape[1:3], padding='valid'),
        lambda x: x.reshape((x.shape[0], -1)),
        nn.Dense(self.num_classes)
    ])
    return net
```

## [**学習**]

ここではより深いネットワークを使うため、計算を簡単にする目的で、入力の高さと幅を 224 から 96 に縮小します。

```{.python .input}
%%tab mxnet, pytorch, jax
model = DenseNet(lr=0.01)
trainer = d2l.Trainer(max_epochs=10, num_gpus=1)
data = d2l.FashionMNIST(batch_size=128, resize=(96, 96))
trainer.fit(model, data)
```

```{.python .input}
%%tab tensorflow
trainer = d2l.Trainer(max_epochs=10)
data = d2l.FashionMNIST(batch_size=128, resize=(96, 96))
with d2l.try_gpu():
    model = DenseNet(lr=0.01)
    trainer.fit(model, data)
```

## 要約と考察

DenseNet を構成する主な要素は dense block と transition layer です。後者については、ネットワークを組み立てる際にチャネル数が再び縮小する transition layer を追加して、次元を制御する必要があります。
層間接続の観点では、入力と出力を加算する ResNet とは対照的に、DenseNet は入力と出力をチャネル次元で連結します。
これらの連結操作は、特徴を再利用して計算効率を高めますが、残念ながら GPU メモリの消費が大きくなります。
その結果、DenseNet を適用するには、学習時間が増える可能性のある、よりメモリ効率の高い実装が必要になることがあります :cite:`pleiss2017memory`。


## 演習

1. なぜ transition layer では max-pooling ではなく average pooling を使うのでしょうか？
1. DenseNet の論文で挙げられている利点の一つは、モデルパラメータが ResNet より小さいことです。なぜそうなるのでしょうか？
1. DenseNet が批判される問題の一つに、高いメモリ消費があります。
    1. これは本当にそうでしょうか？入力形状を $224\times 224$ に変更して、実際の GPU メモリ消費を経験的に比較してみてください。
    1. メモリ消費を減らす別の方法を考えられますか？その場合、フレームワークをどのように変更する必要があるでしょうか？
1. DenseNet 論文 :cite:`Huang.Liu.Van-Der-Maaten.ea.2017` の Table 1 に示されているさまざまな DenseNet 版を実装してください。
1. DenseNet の考え方を適用して、MLP ベースのモデルを設計してください。それを :numref:`sec_kaggle_house` の住宅価格予測タスクに適用してください。\n
