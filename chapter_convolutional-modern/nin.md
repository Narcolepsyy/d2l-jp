{.python .input}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow', 'jax'])
```

# Network in Network（NiN）
:label:`sec_nin`

LeNet、AlexNet、VGG はいずれも共通の設計パターンを共有している。
すなわち、畳み込み層とプーリング層の系列を通じて *空間的* 構造を利用して特徴を抽出し、
その後、全結合層によって表現を後処理する。
AlexNet と VGG による LeNet からの改良は主として、
これら後続のネットワークがこの 2 つのモジュールをどのように広げ、深くしたかにある。

この設計には 2 つの大きな課題がある。
第 1 に、アーキテクチャの最後にある全結合層は、膨大な数のパラメータを消費する。
たとえば、VGG-11 のような単純なモデルでさえ、単精度（FP32）では約 400MB の RAM を占有する巨大な行列を必要とする。
これは、特にモバイル機器や組み込み機器では、計算に対する大きな障害となる。
そもそも、高性能なスマートフォンであっても RAM は 8GB を超えない。
VGG が考案された当時はそれより 1 桁少なく（iPhone 4S は 512MB だった）、画像分類器にメモリの大半を費やすことを正当化するのは難しかっただろう。

第 2 に、非線形性の程度を高めるために、ネットワークのより前段に全結合層を追加することも同様に不可能である。
そうすると空間構造が失われ、さらに多くのメモリを要する可能性があるからだ。

*network in network*（*NiN*）ブロック :cite:`Lin.Chen.Yan.2013` は、これら 2 つの問題を 1 つの単純な戦略で解決できる代替案を提供する。
これは非常に単純な洞察に基づいて提案された。(i) $1 \times 1$ 畳み込みを用いてチャネル活性化全体に局所的な非線形性を追加し、(ii) 最後の表現層のすべての位置を統合するためにグローバル平均プーリングを用いる。
なお、追加された非線形性がなければ、グローバル平均プーリングは有効ではない。
以下でこれを詳しく見ていこう。

```{.python .input}
%%tab mxnet
from d2l import mxnet as d2l
from mxnet import np, npx, init
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
from d2l import tensorflow as d2l
```

```{.python .input}
%%tab jax
from d2l import jax as d2l
from flax import linen as nn
import jax
from jax import numpy as jnp
```

## [**NiN ブロック**]

:numref:`subsec_1x1` を思い出そう。そこで、畳み込み層の入力と出力は、例、チャネル、高さ、幅に対応する軸を持つ 4 次元テンソルからなると述べた。
また、全結合層の入力と出力は、通常、例と特徴に対応する 2 次元テンソルであることも思い出してほしい。
NiN の考え方は、各ピクセル位置（高さと幅の各位置）に対して全結合層を適用することである。
その結果として得られる $1 \times 1$ 畳み込みは、各ピクセル位置で独立に作用する全結合層とみなせる。

:numref:`fig_nin` は、VGG と NiN の主な構造上の違いと、それぞれのブロックを示している。
NiN ブロックの違い（最初の畳み込みの後に $1 \times 1$ 畳み込みが続くのに対し、VGG では $3 \times 3$ 畳み込みを維持する）と、最後に巨大な全結合層が不要になっている点の両方に注目してほしい。

![VGG と NiN のアーキテクチャ、およびそれぞれのブロックの比較。](../img/nin.svg)
:width:`600px`
:label:`fig_nin`

```{.python .input}
%%tab mxnet
def nin_block(num_channels, kernel_size, strides, padding):
    blk = nn.Sequential()
    blk.add(nn.Conv2D(num_channels, kernel_size, strides, padding,
                      activation='relu'),
            nn.Conv2D(num_channels, kernel_size=1, activation='relu'),
            nn.Conv2D(num_channels, kernel_size=1, activation='relu'))
    return blk
```

```{.python .input}
%%tab pytorch
def nin_block(out_channels, kernel_size, strides, padding):
    return nn.Sequential(
        nn.LazyConv2d(out_channels, kernel_size, strides, padding), nn.ReLU(),
        nn.LazyConv2d(out_channels, kernel_size=1), nn.ReLU(),
        nn.LazyConv2d(out_channels, kernel_size=1), nn.ReLU())
```

```{.python .input}
%%tab tensorflow
def nin_block(out_channels, kernel_size, strides, padding):
    return tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(out_channels, kernel_size, strides=strides,
                           padding=padding),
    tf.keras.layers.Activation('relu'),
    tf.keras.layers.Conv2D(out_channels, 1),
    tf.keras.layers.Activation('relu'),
    tf.keras.layers.Conv2D(out_channels, 1),
    tf.keras.layers.Activation('relu')])
```

```{.python .input}
%%tab jax
def nin_block(out_channels, kernel_size, strides, padding):
    return nn.Sequential([
        nn.Conv(out_channels, kernel_size, strides, padding),
        nn.relu,
        nn.Conv(out_channels, kernel_size=(1, 1)), nn.relu,
        nn.Conv(out_channels, kernel_size=(1, 1)), nn.relu])
```

## [**NiN モデル**]

NiN は AlexNet と同じ初期の畳み込みサイズを用いる（NiN はその直後に提案された）。
カーネルサイズはそれぞれ $11\times 11$、$5\times 5$、$3\times 3$ であり、出力チャネル数は AlexNet と一致する。各 NiN ブロックの後には、ストライド 2、ウィンドウ形状 $3\times 3$ の最大プーリング層が続く。

NiN と AlexNet および VGG の第 2 の重要な違いは、NiN が全結合層を完全に避けることである。
その代わりに、NiN はラベルクラス数と等しい出力チャネル数を持つ NiN ブロックの後に、*グローバル* 平均プーリング層を置き、ロジットのベクトルを得る。
この設計は、学習時間が増える可能性という代償はあるものの、必要なモデルパラメータ数を大幅に削減する。

```{.python .input}
%%tab pytorch, mxnet, tensorflow
class NiN(d2l.Classifier):
    def __init__(self, lr=0.1, num_classes=10):
        super().__init__()
        self.save_hyperparameters()
        if tab.selected('mxnet'):
            self.net = nn.Sequential()
            self.net.add(
                nin_block(96, kernel_size=11, strides=4, padding=0),
                nn.MaxPool2D(pool_size=3, strides=2),
                nin_block(256, kernel_size=5, strides=1, padding=2),
                nn.MaxPool2D(pool_size=3, strides=2),
                nin_block(384, kernel_size=3, strides=1, padding=1),
                nn.MaxPool2D(pool_size=3, strides=2),
                nn.Dropout(0.5),
                nin_block(num_classes, kernel_size=3, strides=1, padding=1),
                nn.GlobalAvgPool2D(),
                nn.Flatten())
            self.net.initialize(init.Xavier())
        if tab.selected('pytorch'):
            self.net = nn.Sequential(
                nin_block(96, kernel_size=11, strides=4, padding=0),
                nn.MaxPool2d(3, stride=2),
                nin_block(256, kernel_size=5, strides=1, padding=2),
                nn.MaxPool2d(3, stride=2),
                nin_block(384, kernel_size=3, strides=1, padding=1),
                nn.MaxPool2d(3, stride=2),
                nn.Dropout(0.5),
                nin_block(num_classes, kernel_size=3, strides=1, padding=1),
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten())
            self.net.apply(d2l.init_cnn)
        if tab.selected('tensorflow'):
            self.net = tf.keras.models.Sequential([
                nin_block(96, kernel_size=11, strides=4, padding='valid'),
                tf.keras.layers.MaxPool2D(pool_size=3, strides=2),
                nin_block(256, kernel_size=5, strides=1, padding='same'),
                tf.keras.layers.MaxPool2D(pool_size=3, strides=2),
                nin_block(384, kernel_size=3, strides=1, padding='same'),
                tf.keras.layers.MaxPool2D(pool_size=3, strides=2),
                tf.keras.layers.Dropout(0.5),
                nin_block(num_classes, kernel_size=3, strides=1, padding='same'),
                tf.keras.layers.GlobalAvgPool2D(),
                tf.keras.layers.Flatten()])
```

```{.python .input}
%%tab jax
class NiN(d2l.Classifier):
    lr: float = 0.1
    num_classes = 10
    training: bool = True

    def setup(self):
        self.net = nn.Sequential([
            nin_block(96, kernel_size=(11, 11), strides=(4, 4), padding=(0, 0)),
            lambda x: nn.max_pool(x, (3, 3), strides=(2, 2)),
            nin_block(256, kernel_size=(5, 5), strides=(1, 1), padding=(2, 2)),
            lambda x: nn.max_pool(x, (3, 3), strides=(2, 2)),
            nin_block(384, kernel_size=(3, 3), strides=(1, 1), padding=(1, 1)),
            lambda x: nn.max_pool(x, (3, 3), strides=(2, 2)),
            nn.Dropout(0.5, deterministic=not self.training),
            nin_block(self.num_classes, kernel_size=(3, 3), strides=1, padding=(1, 1)),
            lambda x: nn.avg_pool(x, (5, 5)),  # global avg pooling
            lambda x: x.reshape((x.shape[0], -1))  # flatten
        ])
```

各ブロックの [**出力形状**] を確認するためにデータ例を作成する。

```{.python .input}
%%tab mxnet, pytorch
NiN().layer_summary((1, 1, 224, 224))
```

```{.python .input}
%%tab tensorflow
NiN().layer_summary((1, 224, 224, 1))
```

```{.python .input}
%%tab jax
NiN(training=False).layer_summary((1, 224, 224, 1))
```

## [**学習**]

これまでと同様に、AlexNet と VGG で用いたのと同じオプティマイザを使って Fashion-MNIST でモデルを学習する。

```{.python .input}
%%tab mxnet, pytorch, jax
model = NiN(lr=0.05)
trainer = d2l.Trainer(max_epochs=10, num_gpus=1)
data = d2l.FashionMNIST(batch_size=128, resize=(224, 224))
if tab.selected('pytorch'):
    model.apply_init([next(iter(data.get_dataloader(True)))[0]], d2l.init_cnn)
trainer.fit(model, data)
```

```{.python .input}
%%tab tensorflow
trainer = d2l.Trainer(max_epochs=10)
data = d2l.FashionMNIST(batch_size=128, resize=(224, 224))
with d2l.try_gpu():
    model = NiN(lr=0.05)
    trainer.fit(model, data)
```

## まとめ

NiN は AlexNet や VGG よりも劇的に少ないパラメータしか持たない。これは主として、巨大な全結合層を必要としないことに由来する。
その代わりに、ネットワーク本体の最後の段の後で、すべての画像位置を集約するためにグローバル平均プーリングを用いる。
これにより、高価な（学習された）縮約演算が不要になり、単純な平均に置き換えられる。
当時研究者を驚かせたのは、この平均化操作が精度を損なわなかったことである。
低解像度の表現（多くのチャネルを持つ）に対して平均を取ることは、ネットワークが扱える平行移動不変性の量を増やすことにも注意してほしい。

広いカーネルを持つ畳み込みを減らし、それらを $1 \times 1$ 畳み込みに置き換えることは、さらにパラメータ削減に役立つ。
これは、任意の位置におけるチャネル間でかなりの非線形性を持たせることができる。
$1 \times 1$ 畳み込みとグローバル平均プーリングの両方が、その後の CNN 設計に大きな影響を与えた。

## 演習

1. NiN ブロックに $1\times 1$ 畳み込み層が 2 つあるのはなぜか。3 つに増やすとどうなるか。1 つに減らすとどうなるか。何が変わるか。
1. $1 \times 1$ 畳み込みを $3 \times 3$ 畳み込みに置き換えると何が変わるか。
1. グローバル平均プーリングを全結合層に置き換えるとどうなるか（速度、精度、パラメータ数）。
1. NiN のリソース使用量を計算せよ。
    1. パラメータ数はいくつか。
    1. 計算量はどれくらいか。
    1. 学習中に必要なメモリ量はどれくらいか。
    1. 予測時に必要なメモリ量はどれくらいか。
1. $384 \times 5 \times 5$ の表現を一度に $10 \times 5 \times 5$ の表現へ縮約することには、どのような問題がありうるか。
1. VGG-11、VGG-16、VGG-19 を生み出した VGG の構造設計上の決定を用いて、NiN 風ネットワークの系列を設計せよ。

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/79)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/80)
:end_tab:

:begin_tab:`jax`
[Discussions](https://discuss.d2l.ai/t/18003)
:end_tab:\n