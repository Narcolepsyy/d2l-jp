{.python .input}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow', 'jax'])
```

# 畳み込みニューラルネットワーク（LeNet）
:label:`sec_lenet`

これで、完全に機能する CNN を組み立てるために必要な材料はすべてそろいました。
以前、画像データに初めて触れたときには、Fashion-MNIST データセットの衣服画像に対して、ソフトマックス回帰による線形モデル (:numref:`sec_softmax_scratch`) と MLP (:numref:`sec_mlp-implementation`) を適用しました。
このようなデータを扱いやすくするために、まず各画像を $28\times28$ の行列から固定長の $784$ 次元ベクトルへと平坦化し、その後、全結合層で処理しました。
いまや畳み込み層を扱えるようになったので、画像の空間構造を保持したまま処理できます。
さらに、全結合層を畳み込み層に置き換えることで、必要なパラメータ数がはるかに少ない、より簡潔なモデルを得られます。

この節では、*LeNet* を紹介します。
LeNet は、コンピュータビジョンタスクでの性能によって広く注目を集めた、最初期の公開 CNN の一つです。
このモデルは、当時 AT&T Bell Labs の研究者だった Yann LeCun によって、画像中の手書き数字を認識する目的で提案され、その名も彼にちなんでいます :cite:`LeCun.Bottou.Bengio.ea.1998`。
この研究は、技術開発における10年にわたる研究の集大成でした。
LeCun のチームは、バックプロパゲーションによって CNN を訓練することに初めて成功した研究を発表しました :cite:`LeCun.Boser.Denker.ea.1989`。

当時、LeNet はサポートベクターマシンに匹敵する優れた結果を達成し、教師あり学習における支配的な手法であったそれに対して、1桁あたり1%未満の誤り率を実現しました。
LeNet はその後、ATM 機での入金処理のために数字認識へと応用されました。
今日に至るまで、1990年代に Yann LeCun と同僚の Leon Bottou が書いたコードをそのまま動かしている ATM もあります！

```{.python .input}
%%tab mxnet
from d2l import mxnet as d2l
from mxnet import autograd, gluon, init, np, npx
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
from types import FunctionType
```

## LeNet

大まかに言うと、(**LeNet（LeNet-5）は2つの部分から構成されます。
(i) 2つの畳み込み層からなる畳み込みエンコーダと、
(ii) 3つの全結合層からなる密なブロックです**）。
アーキテクチャは :numref:`img_lenet` に要約されています。

![LeNet におけるデータの流れ。入力は手書き数字で、出力は10個の候補に対する確率です。](../img/lenet.svg)
:label:`img_lenet`

各畳み込みブロックの基本単位は、
畳み込み層、シグモイド活性化関数、
そしてその後に続く平均プーリング操作です。
ReLU と最大プーリングのほうが性能は良いのですが、
当時はまだ発見されていませんでした。
各畳み込み層は $5\times 5$ のカーネルと
シグモイド活性化関数を使います。
これらの層は、空間的に配置された入力を
複数の2次元特徴マップへと写像し、通常は
チャネル数を増やします。
最初の畳み込み層の出力チャネル数は 6、
2番目は 16 です。
各 $2\times2$ のプーリング操作（ストライド 2）は、
空間的なダウンサンプリングによって次元を 4 分の 1 に削減します。
畳み込みブロックの出力形状は
（バッチサイズ、チャネル数、高さ、幅）で与えられます。

畳み込みブロックの出力を
密なブロックへ渡すためには、
ミニバッチ内の各サンプルを平坦化する必要があります。
言い換えると、この4次元入力を、全結合層が期待する2次元入力へ変換します。
念のため言うと、ここで欲しい2次元表現では、第1次元がミニバッチ内のサンプルを表し、
第2次元が各サンプルの平坦なベクトル表現を表します。
LeNet の密なブロックは3つの全結合層からなり、
それぞれの出力数は 120、84、10 です。
なお、これは分類を行っているので、
10次元の出力層は可能な出力クラス数に対応します。

LeNet の内部で何が起きているのかを本当に理解するところまで来るには少し手間がかかるかもしれませんが、
次のコード片を見れば、
このようなモデルを現代の深層学習フレームワークで実装するのが驚くほど簡単だと納得していただけるはずです。
必要なのは `Sequential` ブロックをインスタンス化し、
:numref:`subsec_xavier` で紹介した Xavier 初期化を用いて、
適切な層を順につなげるだけです。

```{.python .input}
%%tab pytorch
def init_cnn(module):  #@save
    """Initialize weights for CNNs."""
    if type(module) == nn.Linear or type(module) == nn.Conv2d:
        nn.init.xavier_uniform_(module.weight)
```

```{.python .input}
%%tab pytorch, mxnet, tensorflow
class LeNet(d2l.Classifier):  #@save
    """The LeNet-5 model."""
    def __init__(self, lr=0.1, num_classes=10):
        super().__init__()
        self.save_hyperparameters()
        if tab.selected('mxnet'):
            self.net = nn.Sequential()
            self.net.add(
                nn.Conv2D(channels=6, kernel_size=5, padding=2,
                          activation='sigmoid'),
                nn.AvgPool2D(pool_size=2, strides=2),
                nn.Conv2D(channels=16, kernel_size=5, activation='sigmoid'),
                nn.AvgPool2D(pool_size=2, strides=2),
                nn.Dense(120, activation='sigmoid'),
                nn.Dense(84, activation='sigmoid'),
                nn.Dense(num_classes))
            self.net.initialize(init.Xavier())
        if tab.selected('pytorch'):
            self.net = nn.Sequential(
                nn.LazyConv2d(6, kernel_size=5, padding=2), nn.Sigmoid(),
                nn.AvgPool2d(kernel_size=2, stride=2),
                nn.LazyConv2d(16, kernel_size=5), nn.Sigmoid(),
                nn.AvgPool2d(kernel_size=2, stride=2),
                nn.Flatten(),
                nn.LazyLinear(120), nn.Sigmoid(),
                nn.LazyLinear(84), nn.Sigmoid(),
                nn.LazyLinear(num_classes))
        if tab.selected('tensorflow'):
            self.net = tf.keras.models.Sequential([
                tf.keras.layers.Conv2D(filters=6, kernel_size=5,
                                       activation='sigmoid', padding='same'),
                tf.keras.layers.AvgPool2D(pool_size=2, strides=2),
                tf.keras.layers.Conv2D(filters=16, kernel_size=5,
                                       activation='sigmoid'),
                tf.keras.layers.AvgPool2D(pool_size=2, strides=2),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(120, activation='sigmoid'),
                tf.keras.layers.Dense(84, activation='sigmoid'),
                tf.keras.layers.Dense(num_classes)])
```

```{.python .input}
%%tab jax
class LeNet(d2l.Classifier):  #@save
    """The LeNet-5 model."""
    lr: float = 0.1
    num_classes: int = 10
    kernel_init: FunctionType = nn.initializers.xavier_uniform

    def setup(self):
        self.net = nn.Sequential([
            nn.Conv(features=6, kernel_size=(5, 5), padding='SAME',
                    kernel_init=self.kernel_init()),
            nn.sigmoid,
            lambda x: nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2)),
            nn.Conv(features=16, kernel_size=(5, 5), padding='VALID',
                    kernel_init=self.kernel_init()),
            nn.sigmoid,
            lambda x: nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2)),
            lambda x: x.reshape((x.shape[0], -1)),  # flatten
            nn.Dense(features=120, kernel_init=self.kernel_init()),
            nn.sigmoid,
            nn.Dense(features=84, kernel_init=self.kernel_init()),
            nn.sigmoid,
            nn.Dense(features=self.num_classes, kernel_init=self.kernel_init())
        ])
```

ここでは LeNet の再現にあたり少し手を加え、ガウス活性化層をソフトマックス層に置き換えています。
ガウスデコーダは現在ではほとんど使われないため、これにより実装が大幅に सरल化されます。
それ以外は、このネットワークは元の LeNet-5 アーキテクチャと一致しています。

:begin_tab:`pytorch, mxnet, tensorflow`
ネットワークの内部で何が起きているか見てみましょう。単一チャネル（白黒）の
$28 \times 28$ 画像をネットワークに通し、
各層での出力形状を表示することで、:numref:`img_lenet_vert` で期待される動作と
整合していることを確認するために、[**モデルを調べる**]ことができます。
:end_tab:

:begin_tab:`jax`
ネットワークの内部で何が起きているか見てみましょう。単一チャネル（白黒）の
$28 \times 28$ 画像をネットワークに通し、
各層での出力形状を表示することで、:numref:`img_lenet_vert` で期待される動作と
整合していることを確認するために、[**モデルを調べる**]ことができます。
Flax には `nn.tabulate` という、ネットワーク内の層と
パラメータを要約する便利なメソッドがあります。ここでは `bind` メソッドを使って束縛済みモデルを作成します。
変数は `d2l.Module` クラスに束縛され、つまりこの束縛済みモデルは状態を持つオブジェクトとなり、
その後 `Sequential` オブジェクト属性 `net` とその中の `layers` にアクセスできるようになります。
なお、`bind` メソッドは対話的な実験にのみ使うべきであり、`apply` メソッドの直接の代替ではありません。
:end_tab:

![LeNet-5 の圧縮表記。](../img/lenet-vert.svg)
:label:`img_lenet_vert`

```{.python .input}
%%tab mxnet, pytorch
@d2l.add_to_class(d2l.Classifier)  #@save
def layer_summary(self, X_shape):
    X = d2l.randn(*X_shape)
    for layer in self.net:
        X = layer(X)
        print(layer.__class__.__name__, 'output shape:\t', X.shape)
        
model = LeNet()
model.layer_summary((1, 1, 28, 28))
```

```{.python .input}
%%tab tensorflow
@d2l.add_to_class(d2l.Classifier)  #@save
def layer_summary(self, X_shape):
    X = d2l.normal(X_shape)
    for layer in self.net.layers:
        X = layer(X)
        print(layer.__class__.__name__, 'output shape:\t', X.shape)

model = LeNet()
model.layer_summary((1, 28, 28, 1))
```

```{.python .input}
%%tab jax
@d2l.add_to_class(d2l.Classifier)  #@save
def layer_summary(self, X_shape, key=d2l.get_key()):
    X = jnp.zeros(X_shape)
    params = self.init(key, X)
    bound_model = self.clone().bind(params, mutable=['batch_stats'])
    _ = bound_model(X)
    for layer in bound_model.net.layers:
        X = layer(X)
        print(layer.__class__.__name__, 'output shape:\t', X.shape)

model = LeNet()
model.layer_summary((1, 28, 28, 1))
```

畳み込みブロック全体を通して、各層における表現の高さと幅は
（前の層と比べて）減少していることに注意してください。
最初の畳み込み層では、$5 \times 5$ カーネルを使ったときに生じる高さと幅の減少を補うために、
2ピクセルのパディングを使っています。
余談ですが、元の MNIST OCR データセットにおける $28 \times 28$ ピクセルという画像サイズは、
元の $32 \times 32$ ピクセルのスキャン画像から2ピクセル分の行（および列）を*切り取った*結果です。
これは主として、当時はメガバイト単位の節約が重要だったため、容量を節約する（30% の削減）ために行われました。

対照的に、2番目の畳み込み層ではパディングを行わないため、
高さと幅はどちらも4ピクセルずつ減少します。
層を上へ進むにつれて、
チャネル数は入力の 1 から、最初の畳み込み層の後に 6、
2番目の畳み込み層の後に 16 へと増加します。
しかし、各プーリング層は高さと幅を半分にします。
最後に、各全結合層が次元を削減し、
最終的にクラス数と一致する次元の出力を生成します。


## 学習

モデルを実装したので、
[**LeNet-5 モデルが Fashion-MNIST でどの程度うまくいくか実験してみましょう**]。

CNN はパラメータ数が少ない一方で、
各パラメータがより多くの乗算に関与するため、
同程度の深さの MLP よりも計算コストが高くなることがあります。
GPU にアクセスできるなら、学習を高速化するために
ここで活用するよい機会かもしれません。
なお、
`d2l.Trainer` クラスが細部をすべて処理してくれます。
デフォルトでは、利用可能なデバイス上でモデルパラメータを初期化します。
MLP のときと同様に、損失関数はクロスエントロピーであり、
ミニバッチ確率的勾配降下法によって最小化します。

```{.python .input}
%%tab pytorch, mxnet, jax
trainer = d2l.Trainer(max_epochs=10, num_gpus=1)
data = d2l.FashionMNIST(batch_size=128)
model = LeNet(lr=0.1)
if tab.selected('pytorch'):
    model.apply_init([next(iter(data.get_dataloader(True)))[0]], init_cnn)
trainer.fit(model, data)
```

```{.python .input}
%%tab tensorflow
trainer = d2l.Trainer(max_epochs=10)
data = d2l.FashionMNIST(batch_size=128)
with d2l.try_gpu():
    model = LeNet(lr=0.1)
    trainer.fit(model, data)
```

## 要約

この章では大きな進歩を遂げました。1980年代の MLP から、1990年代から2000年代初頭の CNN へと移りました。たとえば LeNet-5 の形で提案されたアーキテクチャは、今日においてもなお意味を持っています。Fashion-MNIST における誤り率について、LeNet-5 で達成できる性能を、MLP (:numref:`sec_mlp-implementation`) で達成可能な最良のものや、ResNet (:numref:`sec_resnet`) のようなはるかに高度なアーキテクチャと比較してみる価値があります。LeNet は前者よりも後者にずっと近いものです。後ほど見るように、主な違いの一つは、より多くの計算資源が、はるかに複雑なアーキテクチャを可能にしたことです。

2つ目の違いは、LeNet を実装することの相対的な容易さです。
かつては、C++ とアセンブリコードで何か月もかかる工学的課題であり、SN を改善するためのエンジニアリング、初期の Lisp ベースの深層学習ツール :cite:`Bottou.Le-Cun.1988`、そして最終的にはモデルを用いた試行錯誤が必要でしたが、今では数分で実現できます。
この驚くべき生産性向上こそが、深層学習モデル開発を大きく民主化したのです。
次の章では、このウサギの穴をさらに下っていき、どこへたどり着くのかを見ていきます。

## 演習

1. LeNet を現代風にしましょう。以下の変更を実装してテストしてください。
    1. 平均プーリングを最大プーリングに置き換える。
    1. ソフトマックス層を ReLU に置き換える。
1. 最大プーリングと ReLU に加えて、LeNet 風ネットワークのサイズを変更して精度を改善してみましょう。
    1. 畳み込みウィンドウのサイズを調整する。
    1. 出力チャネル数を調整する。
    1. 畳み込み層の数を調整する。
    1. 全結合層の数を調整する。
    1. 学習率やその他の学習設定（たとえば初期化やエポック数）を調整する。
1. 改良したネットワークを元の MNIST データセットで試してみましょう。
1. 異なる入力（たとえばセーターやコート）に対する LeNet の第1層と第2層の活性化を表示しましょう。
1. ネットワークに大きく異なる画像（たとえば猫、車、あるいはランダムノイズ）を入力すると、活性化はどうなりますか？

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/73)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/74)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/275)
:end_tab:

:begin_tab:`jax`
[Discussions](https://discuss.d2l.ai/t/18000)
:end_tab:\n