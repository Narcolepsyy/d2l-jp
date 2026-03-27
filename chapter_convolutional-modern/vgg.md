{.python .input  n=1}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow', 'jax'])
```

# ブロックを用いたネットワーク（VGG）
:label:`sec_vgg`

AlexNet は、深い CNN が良い結果を達成できるという実証的証拠を示したが、その後の研究者が新しいネットワークを設計する際の一般的なひな形は提示しませんでした。以下の節では、深層ネットワークの設計によく用いられるいくつかのヒューリスティックな概念を紹介する。

この分野の進歩は、チップ設計における VLSI（very large scale integration）の進歩を思わせる。そこでは、エンジニアはトランジスタの配置から論理要素へ、さらに論理ブロックへと移っていきました :cite:`Mead.1980`。同様に、ニューラルネットワークアーキテクチャの設計も徐々に抽象化が進み、研究者は個々のニューロンから層全体へ、そして現在では層の反復パターンであるブロックへと考えるようになりました。さらに10年後の現在では、研究者が学習済みのモデル全体を別の、とはいえ関連するタスクに転用するところまで進んでいる。このような大規模な事前学習済みモデルは、通常 *foundation models* と呼ばれます :cite:`bommasani2021opportunities`。 

話をネットワーク設計に戻しよう。ブロックを用いるという考え方は、オックスフォード大学の Visual Geometry Group（VGG）による、同名の *VGG* ネットワークで最初に現れました :cite:`Simonyan.Zisserman.2014`。ループとサブルーチンを使えば、現代のどの深層学習フレームワークでも、こうした反復構造をコードで簡単に実装できる。

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
```

## [**VGG ブロック**]
:label:`subsec_vgg-blocks`

CNN の基本的な構成要素は、次の順序からなる系列である。  
(i) 解像度を維持するためにパディングを施した畳み込み層、  
(ii) ReLU などの非線形性、  
(iii) 解像度を下げるための max-pooling などのプーリング層。  
この方法の問題の一つは、空間解像度がかなり急速に低下することである。特に、すべての次元（$d$）を使い切る前に、ネットワークに許される畳み込み層の数は $\log_2 d$ に厳しく制限される。たとえば ImageNet の場合、この方法では 8 層を超える畳み込み層を持つことは不可能である。 

:citet:`Simonyan.Zisserman.2014` の重要なアイデアは、max-pooling によるダウンサンプリングの間に、*複数* の畳み込みをブロックとして挟むことでした。彼らは主として、深いネットワークと幅広いネットワークのどちらがより良い性能を示すかに関心を持っていました。たとえば、2 つの $3 \times 3$ 畳み込みを連続して適用すると、1 つの $5 \times 5$ 畳み込みと同じ画素に触れます。同時に、後者はおよそ同じ数のパラメータ（$25 \cdot c^2$）を使いますが、これは 3 つの $3 \times 3$ 畳み込み（$3 \cdot 9 \cdot c^2$）とほぼ同じです。かなり詳細な解析により、深くて狭いネットワークが浅いネットワークよりも大幅に優れていることを示しました。これにより深層学習は、典型的な応用で 100 層を超える、ますます深いネットワークを目指す方向へ進みました。  
$3 \times 3$ 畳み込みを積み重ねる設計は、その後の深層ネットワークにおける黄金標準となりました（この設計判断が最近になって再検討されたのは :citet:`liu2022convnet` です）。その結果、小さな畳み込みの高速実装は GPU の定番となりました :cite:`lavin2016fast`。 

VGG に戻ると、VGG ブロックは、パディング 1 の $3\times3$ カーネルを持つ畳み込みの *系列* に続いて、ストライド 2 の $2 \times 2$ max-pooling 層（各ブロック後に高さと幅を半分にする）から構成される。以下のコードでは、1 つの VGG ブロックを実装する `vgg_block` という関数を定義する。

以下の関数は 2 つの引数を取り、畳み込み層の数 `num_convs` と出力チャネル数 `num_channels` に対応する。

```{.python .input  n=2}
%%tab mxnet
def vgg_block(num_convs, num_channels):
    blk = nn.Sequential()
    for _ in range(num_convs):
        blk.add(nn.Conv2D(num_channels, kernel_size=3,
                          padding=1, activation='relu'))
    blk.add(nn.MaxPool2D(pool_size=2, strides=2))
    return blk
```

```{.python .input  n=3}
%%tab pytorch
def vgg_block(num_convs, out_channels):
    layers = []
    for _ in range(num_convs):
        layers.append(nn.LazyConv2d(out_channels, kernel_size=3, padding=1))
        layers.append(nn.ReLU())
    layers.append(nn.MaxPool2d(kernel_size=2,stride=2))
    return nn.Sequential(*layers)
```

```{.python .input  n=4}
%%tab tensorflow
def vgg_block(num_convs, num_channels):
    blk = tf.keras.models.Sequential()
    for _ in range(num_convs):
        blk.add(
            tf.keras.layers.Conv2D(num_channels, kernel_size=3,
                                   padding='same', activation='relu'))
    blk.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))
    return blk
```

```{.python .input}
%%tab jax
def vgg_block(num_convs, out_channels):
    layers = []
    for _ in range(num_convs):
        layers.append(nn.Conv(out_channels, kernel_size=(3, 3), padding=(1, 1)))
        layers.append(nn.relu)
    layers.append(lambda x: nn.max_pool(x, window_shape=(2, 2), strides=(2, 2)))
    return nn.Sequential(layers)
```

## [**VGG ネットワーク**]
:label:`subsec_vgg-network`

AlexNet や LeNet と同様に、VGG ネットワークは 2 つの部分に分けられる。  
前半は主として畳み込み層とプーリング層からなり、後半は AlexNet と同じ全結合層からなる。  
重要な違いは、畳み込み層が、次元を変えない非線形変換のまとまりとしてグループ化され、その後に解像度を下げるステップが続く点である。これは :numref:`fig_vgg` に示されている。 

![AlexNet から VGG へ。重要な違いは、VGG が層のブロックから構成されるのに対し、AlexNet の層はすべて個別に設計されている点である。](../img/vgg.svg)
:width:`400px`
:label:`fig_vgg`

ネットワークの畳み込み部分は、 :numref:`fig_vgg` にあるいくつかの VGG ブロック（`vgg_block` 関数でも定義済み）を順番に接続したものである。この畳み込みのグループ化は、具体的な演算の選択は大きく変化してきたものの、過去10年ほとんど変わらずに残ってきたパターンである。  
変数 `arch` はタプルのリスト（ブロックごとに 1 つ）で構成され、各タプルは 2 つの値、すなわち畳み込み層の数と出力チャネル数を含みる。これはちょうど `vgg_block` 関数を呼び出すために必要な引数である。したがって、VGG は単一の具体的なモデルというより、ネットワークの *ファミリー* を定義している。特定のネットワークを構築するには、`arch` を順にたどってブロックを組み立てればよいだけである。

```{.python .input  n=5}
%%tab pytorch, mxnet, tensorflow
class VGG(d2l.Classifier):
    def __init__(self, arch, lr=0.1, num_classes=10):
        super().__init__()
        self.save_hyperparameters()
        if tab.selected('mxnet'):
            self.net = nn.Sequential()
            for (num_convs, num_channels) in arch:
                self.net.add(vgg_block(num_convs, num_channels))
            self.net.add(nn.Dense(4096, activation='relu'), nn.Dropout(0.5),
                         nn.Dense(4096, activation='relu'), nn.Dropout(0.5),
                         nn.Dense(num_classes))
            self.net.initialize(init.Xavier())
        if tab.selected('pytorch'):
            conv_blks = []
            for (num_convs, out_channels) in arch:
                conv_blks.append(vgg_block(num_convs, out_channels))
            self.net = nn.Sequential(
                *conv_blks, nn.Flatten(),
                nn.LazyLinear(4096), nn.ReLU(), nn.Dropout(0.5),
                nn.LazyLinear(4096), nn.ReLU(), nn.Dropout(0.5),
                nn.LazyLinear(num_classes))
            self.net.apply(d2l.init_cnn)
        if tab.selected('tensorflow'):
            self.net = tf.keras.models.Sequential()
            for (num_convs, num_channels) in arch:
                self.net.add(vgg_block(num_convs, num_channels))
            self.net.add(
                tf.keras.models.Sequential([
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(4096, activation='relu'),
                tf.keras.layers.Dropout(0.5),
                tf.keras.layers.Dense(4096, activation='relu'),
                tf.keras.layers.Dropout(0.5),
                tf.keras.layers.Dense(num_classes)]))
```

```{.python .input  n=5}
%%tab jax
class VGG(d2l.Classifier):
    arch: list
    lr: float = 0.1
    num_classes: int = 10
    training: bool = True

    def setup(self):
        conv_blks = []
        for (num_convs, out_channels) in self.arch:
            conv_blks.append(vgg_block(num_convs, out_channels))

        self.net = nn.Sequential([
            *conv_blks,
            lambda x: x.reshape((x.shape[0], -1)),  # flatten
            nn.Dense(4096), nn.relu,
            nn.Dropout(0.5, deterministic=not self.training),
            nn.Dense(4096), nn.relu,
            nn.Dropout(0.5, deterministic=not self.training),
            nn.Dense(self.num_classes)])
```

元の VGG ネットワークは 5 つの畳み込みブロックからなり、そのうち最初の 2 つはそれぞれ 1 層の畳み込み層を持ち、残りの 3 つはそれぞれ 2 層の畳み込み層を含みる。最初のブロックの出力チャネル数は 64 で、その後の各ブロックでは出力チャネル数が 2 倍ずつ増え、512 に達するまで続く。このネットワークは 8 層の畳み込み層と 3 層の全結合層を使うため、しばしば VGG-11 と呼ばれる。

```{.python .input  n=6}
%%tab pytorch, mxnet
VGG(arch=((1, 64), (1, 128), (2, 256), (2, 512), (2, 512))).layer_summary(
    (1, 1, 224, 224))
```

```{.python .input  n=7}
%%tab tensorflow
VGG(arch=((1, 64), (1, 128), (2, 256), (2, 512), (2, 512))).layer_summary(
    (1, 224, 224, 1))
```

```{.python .input}
%%tab jax
VGG(arch=((1, 64), (1, 128), (2, 256), (2, 512), (2, 512)),
    training=False).layer_summary((1, 224, 224, 1))
```

見てわかるように、各ブロックで高さと幅を半分にし、最終的に高さと幅が 7 になってから表現を平坦化し、ネットワークの全結合部分で処理する。  
:citet:`Simonyan.Zisserman.2014` では、VGG の他のいくつかの変種も説明されています。実際、新しいアーキテクチャを導入する際には、速度と精度のトレードオフが異なる *ファミリー* のネットワークを提案するのが今では標準になっています。 

## 学習

[**VGG-11 は AlexNet より計算負荷が高いため、より少ないチャネル数のネットワークを構成する。**]
これは Fashion-MNIST の学習には十分すぎるほどである。[**モデル学習**] の過程は :numref:`sec_alexnet` における AlexNet と同様である。  
ここでも、検証損失と訓練損失がよく一致しており、過学習はわずかであることが示唆される。

```{.python .input  n=8}
%%tab mxnet, pytorch, jax
model = VGG(arch=((1, 16), (1, 32), (2, 64), (2, 128), (2, 128)), lr=0.01)
trainer = d2l.Trainer(max_epochs=10, num_gpus=1)
data = d2l.FashionMNIST(batch_size=128, resize=(224, 224))
if tab.selected('pytorch'):
    model.apply_init([next(iter(data.get_dataloader(True)))[0]], d2l.init_cnn)
trainer.fit(model, data)
```

```{.python .input  n=9}
%%tab tensorflow
trainer = d2l.Trainer(max_epochs=10)
data = d2l.FashionMNIST(batch_size=128, resize=(224, 224))
with d2l.try_gpu():
    model = VGG(arch=((1, 16), (1, 32), (2, 64), (2, 128), (2, 128)), lr=0.01)
    trainer.fit(model, data)
```

## 要約

VGG は、真に現代的な畳み込みニューラルネットワークの最初のものだと主張できるかもしれません。AlexNet は、深層学習を大規模で有効にする多くの構成要素を導入したが、複数の畳み込みからなるブロックや、深くて狭いネットワークを好む設計など、重要な性質を導入したのは VGG だと言えるでしょう。また、実際に同じパラメータ化を持つモデルのファミリー全体である最初のネットワークでもあり、実践者に対して複雑さと速度の間で十分なトレードオフを与える。ここは、現代の深層学習フレームワークが真価を発揮する場でもある。もはやネットワークを指定するために XML の設定ファイルを生成する必要はなく、単純な Python コードでそのようなネットワークを組み立てればよいのである。 

より最近では、ParNet :cite:`Goyal.Bochkovskiy.Deng.ea.2021` が、多数の並列計算を用いることで、はるかに浅いアーキテクチャでも競争力のある性能を達成できることを示した。これは刺激的な進展であり、将来のアーキテクチャ設計に影響を与えることが期待される。とはいえ、この章の残りでは、過去10年の科学的進歩の道筋に従うことにする。 

## 演習


1. AlexNet と比べると、VGG は計算の面でかなり遅く、GPU メモリもより多く必要とする。 
    1. AlexNet と VGG に必要なパラメータ数を比較せよ。
    1. 畳み込み層と全結合層で使われる浮動小数点演算回数を比較せよ。 
    1. 全結合層によって生じる計算コストをどのように削減できるか。
1. ネットワークの各層に対応する次元を表示すると、ネットワークは 11 層あるにもかかわらず、8 つのブロック（に加えていくつかの補助変換）に関する情報しか見えません。残りの 3 層はどこへ行ったのでしょうか。
1. VGG 論文 :cite:`Simonyan.Zisserman.2014` の表 1 を用いて、VGG-16 や VGG-19 などの他の一般的なモデルを構成せよ。
1. Fashion-MNIST の解像度を $28 \times 28$ から $224 \times 224$ へ 8 倍にアップサンプリングするのは非常に無駄である。ネットワークアーキテクチャと解像度変換を変更して、たとえば入力を 56 または 84 次元にしてみよ。ネットワークの精度を下げずにそれを実現できるか。ダウンサンプリングの前により多くの非線形性を加える方法については、VGG 論文 :cite:`Simonyan.Zisserman.2014` を参照せよ。
