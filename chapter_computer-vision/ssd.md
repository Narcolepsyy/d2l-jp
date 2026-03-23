# シングルショット・マルチボックス検出
:label:`sec_ssd`

:numref:`sec_bbox`--:numref:`sec_object-detection-dataset` では、
バウンディングボックス、アンカーボックス、
マルチスケール物体検出、そして物体検出用データセットを紹介しました。
ここでは、こうした背景知識を用いて
物体検出モデルである
シングルショット・マルチボックス検出
（SSD） :cite:`Liu.Anguelov.Erhan.ea.2016`
を設計する準備が整いました。
このモデルはシンプルで高速であり、広く使われています。
これは数多くある物体検出モデルのうちの一つにすぎませんが、
この節で扱う設計原理や実装の詳細の一部は、
他のモデルにも適用できます。


## モデル

:numref:`fig_ssd` は
シングルショット・マルチボックス検出の設計概要を示しています。
このモデルは主に
ベースネットワークと、
それに続く
いくつかのマルチスケール特徴マップブロックから構成されます。
ベースネットワークは
入力画像から特徴を抽出するためのもので、
深いCNNを使うことができます。
たとえば、
元のシングルショット・マルチボックス検出の論文では、
分類層の手前で切り詰めたVGGネットワークを採用しており :cite:`Liu.Anguelov.Erhan.ea.2016`、
ResNetも一般的に使われています。
この設計により、
ベースネットワークの出力する特徴マップを
より大きくして、
より多くのアンカーボックスを生成し、
より小さな物体を検出できるようにします。
その後、
各マルチスケール特徴マップブロックは
前のブロックからの特徴マップの高さと幅を
（たとえば半分に）縮小し、
特徴マップの各ユニットが
入力画像上で持つ受容野を
拡大できるようにします。


:numref:`sec_multiscale-object-detection` で
深層ニューラルネットワークによる画像の層ごとの表現を通じた
マルチスケール物体検出の設計を思い出してください。
:numref:`fig_ssd` の上部に近い
マルチスケール特徴マップほど小さいですが、受容野は大きいため、
数は少ないがより大きな物体の検出に適しています。

要するに、
ベースネットワークといくつかのマルチスケール特徴マップブロックを通じて、
シングルショット・マルチボックス検出は
異なるサイズを持つさまざまな数のアンカーボックスを生成し、
これらのアンカーボックス（したがってバウンディングボックス）の
クラスとオフセットを予測することで
さまざまなサイズの物体を検出します。
したがって、これはマルチスケール物体検出モデルです。


![マルチスケール物体検出モデルとして、シングルショット・マルチボックス検出は主にベースネットワークと、それに続くいくつかのマルチスケール特徴マップブロックから構成される。](../img/ssd.svg)
:label:`fig_ssd`


以下では、
:numref:`fig_ssd` における異なるブロックの実装詳細を説明します。まず、
クラス予測とバウンディングボックス予測をどのように実装するかを議論します。



### [**クラス予測層**]

物体クラス数を $q$ とします。
するとアンカーボックスは $q+1$ 個のクラスを持ち、
クラス0は背景です。
あるスケールで、
特徴マップの高さと幅がそれぞれ $h$ と $w$ であるとします。
これらの特徴マップの各空間位置を中心として
$a$ 個のアンカーボックスが生成されるとき、
合計 $hwa$ 個のアンカーボックスを分類する必要があります。
これは、パラメータ数が非常に多くなりがちなため、
全結合層による分類を非現実的にします。
:numref:`sec_nin` で
畳み込み層のチャネルを使ってクラスを予測した方法を思い出してください。
シングルショット・マルチボックス検出は
モデルの複雑さを下げるために
同じ手法を使います。

具体的には、
クラス予測層は特徴マップの幅と高さを変えずに
畳み込み層を使います。
このようにして、
特徴マップの同じ空間次元（幅と高さ）において
出力と入力の間に
1対1の対応を持たせることができます。
より具体的には、
任意の空間位置 ($x$, $y$) における
出力特徴マップのチャネルは、
入力特徴マップの ($x$, $y$) を中心とする
すべてのアンカーボックスに対する
クラス予測を表します。
有効な予測を得るには、
$a(q+1)$ 個の出力チャネルが必要です。
ここで、同じ空間位置に対して
インデックス $i(q+1) + j$ の出力チャネルは、
アンカーボックス $i$ ($0 \leq i < a$) に対する
クラス $j$ ($0 \leq j \leq q$) の予測を表します。

以下では、このようなクラス予測層を定義します。
引数 `num_anchors` と `num_classes` により、それぞれ $a$ と $q$ を指定します。
この層は、パディング1の $3\times3$ 畳み込み層を使います。
この畳み込み層の入力と出力の幅と高さは変わりません。

```{.python .input}
#@tab mxnet
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import autograd, gluon, image, init, np, npx
from mxnet.gluon import nn

npx.set_np()

def cls_predictor(num_anchors, num_classes):
    return nn.Conv2D(num_anchors * (num_classes + 1), kernel_size=3,
                     padding=1)
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
import torch
import torchvision
from torch import nn
from torch.nn import functional as F

def cls_predictor(num_inputs, num_anchors, num_classes):
    return nn.Conv2d(num_inputs, num_anchors * (num_classes + 1),
                     kernel_size=3, padding=1)
```

### (**バウンディングボックス予測層**)

バウンディングボックス予測層の設計は、クラス予測層と似ています。
唯一の違いは、各アンカーボックスに対する出力数です。
ここでは $q+1$ 個のクラスではなく、4つのオフセットを予測する必要があります。

```{.python .input}
#@tab mxnet
def bbox_predictor(num_anchors):
    return nn.Conv2D(num_anchors * 4, kernel_size=3, padding=1)
```

```{.python .input}
#@tab pytorch
def bbox_predictor(num_inputs, num_anchors):
    return nn.Conv2d(num_inputs, num_anchors * 4, kernel_size=3, padding=1)
```

### [**複数スケールの予測を連結する**]

前述したように、シングルショット・マルチボックス検出は
マルチスケール特徴マップを使ってアンカーボックスを生成し、そのクラスとオフセットを予測します。
異なるスケールでは、
特徴マップの形状や
同じユニットを中心とするアンカーボックスの数が
異なる場合があります。
したがって、
異なるスケールでの予測出力の形状も
異なりえます。

以下の例では、
同じミニバッチに対して
2つの異なるスケールの特徴マップ `Y1` と `Y2` を構成します。
ここで `Y2` の高さと幅は `Y1` の半分です。
クラス予測を例に取りましょう。
`Y1` と `Y2` の各ユニットに対して
それぞれ5個と3個のアンカーボックスが生成されるとします。
さらに、
物体クラス数が10であるとします。
特徴マップ `Y1` と `Y2` に対する
クラス予測出力のチャネル数はそれぞれ
$5\times(10+1)=55$ と $3\times(10+1)=33$ であり、
いずれの出力形状も
（バッチサイズ，チャネル数，高さ，幅）です。

```{.python .input}
#@tab mxnet
def forward(x, block):
    block.initialize()
    return block(x)

Y1 = forward(np.zeros((2, 8, 20, 20)), cls_predictor(5, 10))
Y2 = forward(np.zeros((2, 16, 10, 10)), cls_predictor(3, 10))
Y1.shape, Y2.shape
```

```{.python .input}
#@tab pytorch
def forward(x, block):
    return block(x)

Y1 = forward(torch.zeros((2, 8, 20, 20)), cls_predictor(8, 5, 10))
Y2 = forward(torch.zeros((2, 16, 10, 10)), cls_predictor(16, 3, 10))
Y1.shape, Y2.shape
```

見てのとおり、
バッチサイズの次元を除けば、
他の3つの次元はすべて異なるサイズを持っています。
これら2つの予測出力をより効率的に計算するために連結するには、
これらのテンソルをより一貫した形式に変換します。

チャネル次元には、
同じ中心を持つアンカーボックスに対する予測が格納されていることに注意してください。
まずこの次元を最内側に移します。
バッチサイズは異なるスケールでも同じなので、
予測出力を
（バッチサイズ，高さ $\times$ 幅 $\times$ チャネル数）
という形状の2次元テンソルに変換できます。
その後、
異なるスケールのこのような出力を
次元1に沿って連結できます。

```{.python .input}
#@tab mxnet
def flatten_pred(pred):
    return npx.batch_flatten(pred.transpose(0, 2, 3, 1))

def concat_preds(preds):
    return np.concatenate([flatten_pred(p) for p in preds], axis=1)
```

```{.python .input}
#@tab pytorch
def flatten_pred(pred):
    return torch.flatten(pred.permute(0, 2, 3, 1), start_dim=1)

def concat_preds(preds):
    return torch.cat([flatten_pred(p) for p in preds], dim=1)
```

このようにして、
`Y1` と `Y2` はチャネル数、高さ、幅が異なっていても、
同じミニバッチに対する2つの異なるスケールの予測出力を
連結できます。

```{.python .input}
#@tab all
concat_preds([Y1, Y2]).shape
```

### [**ダウンサンプリングブロック**]

複数スケールの物体を検出するために、
入力特徴マップの高さと幅を半分にする
以下のダウンサンプリングブロック `down_sample_blk` を定義します。
実際には、
このブロックは :numref:`subsec_vgg-blocks` の
VGGブロックの設計を適用しています。
より具体的には、
各ダウンサンプリングブロックは
パディング1の $3\times3$ 畳み込み層2つと、
ストライド2の $2\times2$ 最大プーリング層1つから構成されます。
ご存じのように、パディング1の $3\times3$ 畳み込み層は特徴マップの形状を変えません。
しかし、その後に続く $2\times2$ 最大プーリングは
入力特徴マップの高さと幅を半分にします。
このダウンサンプリングブロックの入力特徴マップと出力特徴マップの両方について、
$1\times 2+(3-1)+(3-1)=6$ であるため、
出力の各ユニットは入力上で $6\times6$ の受容野を持ちます。
したがって、ダウンサンプリングブロックは
出力特徴マップの各ユニットの受容野を拡大します。

```{.python .input}
#@tab mxnet
def down_sample_blk(num_channels):
    blk = nn.Sequential()
    for _ in range(2):
        blk.add(nn.Conv2D(num_channels, kernel_size=3, padding=1),
                nn.BatchNorm(in_channels=num_channels),
                nn.Activation('relu'))
    blk.add(nn.MaxPool2D(2))
    return blk
```

```{.python .input}
#@tab pytorch
def down_sample_blk(in_channels, out_channels):
    blk = []
    for _ in range(2):
        blk.append(nn.Conv2d(in_channels, out_channels,
                             kernel_size=3, padding=1))
        blk.append(nn.BatchNorm2d(out_channels))
        blk.append(nn.ReLU())
        in_channels = out_channels
    blk.append(nn.MaxPool2d(2))
    return nn.Sequential(*blk)
```

以下の例では、構成したダウンサンプリングブロックは入力チャネル数を変え、
入力特徴マップの高さと幅を半分にします。

```{.python .input}
#@tab mxnet
forward(np.zeros((2, 3, 20, 20)), down_sample_blk(10)).shape
```

```{.python .input}
#@tab pytorch
forward(torch.zeros((2, 3, 20, 20)), down_sample_blk(3, 10)).shape
```

### [**ベースネットワークブロック**]

ベースネットワークブロックは入力画像から特徴を抽出するために使われます。
簡単のために、
3つのダウンサンプリングブロックからなる
小さなベースネットワークを構成し、
各ブロックでチャネル数を2倍にします。
$256\times256$ の入力画像に対して、
このベースネットワークブロックは
$32 \times 32$ の特徴マップを出力します（$256/2^3=32$）。

```{.python .input}
#@tab mxnet
def base_net():
    blk = nn.Sequential()
    for num_filters in [16, 32, 64]:
        blk.add(down_sample_blk(num_filters))
    return blk

forward(np.zeros((2, 3, 256, 256)), base_net()).shape
```

```{.python .input}
#@tab pytorch
def base_net():
    blk = []
    num_filters = [3, 16, 32, 64]
    for i in range(len(num_filters) - 1):
        blk.append(down_sample_blk(num_filters[i], num_filters[i+1]))
    return nn.Sequential(*blk)

forward(torch.zeros((2, 3, 256, 256)), base_net()).shape
```

### 完全なモデル


[**完全な
シングルショット・マルチボックス検出モデルは
5つのブロックから構成されます。**]
各ブロックで生成された特徴マップは、
(i) アンカーボックスの生成と、
(ii) それらのアンカーボックスのクラスとオフセットの予測
の両方に使われます。
この5つのブロックのうち、
最初のものがベースネットワークブロック、
2番目から4番目が
ダウンサンプリングブロック、
最後のブロックは
グローバル最大プーリングを使って
高さと幅の両方を1に縮小します。
技術的には、
2番目から5番目のブロックはすべて
:numref:`fig_ssd` における
マルチスケール特徴マップブロックです。

```{.python .input}
#@tab mxnet
def get_blk(i):
    if i == 0:
        blk = base_net()
    elif i == 4:
        blk = nn.GlobalMaxPool2D()
    else:
        blk = down_sample_blk(128)
    return blk
```

```{.python .input}
#@tab pytorch
def get_blk(i):
    if i == 0:
        blk = base_net()
    elif i == 1:
        blk = down_sample_blk(64, 128)
    elif i == 4:
        blk = nn.AdaptiveMaxPool2d((1,1))
    else:
        blk = down_sample_blk(128, 128)
    return blk
```

ここで、各ブロックの[**順伝播を定義**]します。
画像分類タスクとは異なり、
ここでの出力には
(i) CNN特徴マップ `Y`、
(ii) 現在のスケールで `Y` を用いて生成されたアンカーボックス、
(iii) これらのアンカーボックスに対して `Y` に基づいて予測されたクラスとオフセット
が含まれます。

```{.python .input}
#@tab mxnet
def blk_forward(X, blk, size, ratio, cls_predictor, bbox_predictor):
    Y = blk(X)
    anchors = d2l.multibox_prior(Y, sizes=size, ratios=ratio)
    cls_preds = cls_predictor(Y)
    bbox_preds = bbox_predictor(Y)
    return (Y, anchors, cls_preds, bbox_preds)
```

```{.python .input}
#@tab pytorch
def blk_forward(X, blk, size, ratio, cls_predictor, bbox_predictor):
    Y = blk(X)
    anchors = d2l.multibox_prior(Y, sizes=size, ratios=ratio)
    cls_preds = cls_predictor(Y)
    bbox_preds = bbox_predictor(Y)
    return (Y, anchors, cls_preds, bbox_preds)
```

:numref:`fig_ssd` で見たように、
上部に近いマルチスケール特徴マップブロックほど
より大きな物体の検出に使われるため、
より大きなアンカーボックスを生成する必要があります。
上の順伝播では、
各マルチスケール特徴マップブロックにおいて、
呼び出される `multibox_prior` 関数（:numref:`sec_anchor` で説明） の
`size` 引数を通じて
2つのスケール値のリストを渡しています。
以下では、
0.2 と 1.05 の間の区間を
5つの区間に等分し、
5つのブロックにおける小さいスケール値を
0.2, 0.37, 0.54, 0.71, 0.88 と決めます。
その後、それらの大きいスケール値は
$\sqrt{0.2 \times 0.37} = 0.272$, $\sqrt{0.37 \times 0.54} = 0.447$ などで与えられます。

[~~各ブロックのハイパーパラメータ~~]

```{.python .input}
#@tab all
sizes = [[0.2, 0.272], [0.37, 0.447], [0.54, 0.619], [0.71, 0.79],
         [0.88, 0.961]]
ratios = [[1, 2, 0.5]] * 5
num_anchors = len(sizes[0]) + len(ratios[0]) - 1
```

これで、以下のように完全なモデル `TinySSD` を[**定義**]できます。

```{.python .input}
#@tab mxnet
class TinySSD(nn.Block):
    def __init__(self, num_classes, **kwargs):
        super(TinySSD, self).__init__(**kwargs)
        self.num_classes = num_classes
        for i in range(5):
            # Equivalent to the assignment statement `self.blk_i = get_blk(i)`
            setattr(self, f'blk_{i}', get_blk(i))
            setattr(self, f'cls_{i}', cls_predictor(num_anchors, num_classes))
            setattr(self, f'bbox_{i}', bbox_predictor(num_anchors))

    def forward(self, X):
        anchors, cls_preds, bbox_preds = [None] * 5, [None] * 5, [None] * 5
        for i in range(5):
            # Here `getattr(self, 'blk_%d' % i)` accesses `self.blk_i`
            X, anchors[i], cls_preds[i], bbox_preds[i] = blk_forward(
                X, getattr(self, f'blk_{i}'), sizes[i], ratios[i],
                getattr(self, f'cls_{i}'), getattr(self, f'bbox_{i}'))
        anchors = np.concatenate(anchors, axis=1)
        cls_preds = concat_preds(cls_preds)
        cls_preds = cls_preds.reshape(
            cls_preds.shape[0], -1, self.num_classes + 1)
        bbox_preds = concat_preds(bbox_preds)
        return anchors, cls_preds, bbox_preds
```

```{.python .input}
#@tab pytorch
class TinySSD(nn.Module):
    def __init__(self, num_classes, **kwargs):
        super(TinySSD, self).__init__(**kwargs)
        self.num_classes = num_classes
        idx_to_in_channels = [64, 128, 128, 128, 128]
        for i in range(5):
            # Equivalent to the assignment statement `self.blk_i = get_blk(i)`
            setattr(self, f'blk_{i}', get_blk(i))
            setattr(self, f'cls_{i}', cls_predictor(idx_to_in_channels[i],
                                                    num_anchors, num_classes))
            setattr(self, f'bbox_{i}', bbox_predictor(idx_to_in_channels[i],
                                                      num_anchors))

    def forward(self, X):
        anchors, cls_preds, bbox_preds = [None] * 5, [None] * 5, [None] * 5
        for i in range(5):
            # Here `getattr(self, 'blk_%d' % i)` accesses `self.blk_i`
            X, anchors[i], cls_preds[i], bbox_preds[i] = blk_forward(
                X, getattr(self, f'blk_{i}'), sizes[i], ratios[i],
                getattr(self, f'cls_{i}'), getattr(self, f'bbox_{i}'))
        anchors = torch.cat(anchors, dim=1)
        cls_preds = concat_preds(cls_preds)
        cls_preds = cls_preds.reshape(
            cls_preds.shape[0], -1, self.num_classes + 1)
        bbox_preds = concat_preds(bbox_preds)
        return anchors, cls_preds, bbox_preds
```

ここで、$256 \times 256$ 画像のミニバッチ `X` に対して
順伝播を行うためにモデルインスタンスを[**作成**]します。

この節の前半で示したように、
最初のブロックは $32 \times 32$ の特徴マップを出力します。
2番目から4番目のダウンサンプリングブロックは
高さと幅を半分にし、
5番目のブロックはグローバルプーリングを使うことを思い出してください。
特徴マップの空間次元の各ユニットに対して
4個のアンカーボックスが生成されるので、
5つのスケール全体では
各画像に対して合計 $(32^2 + 16^2 + 8^2 + 4^2 + 1)\times 4 = 5444$ 個のアンカーボックスが生成されます。

```{.python .input}
#@tab mxnet
net = TinySSD(num_classes=1)
net.initialize()
X = np.zeros((32, 3, 256, 256))
anchors, cls_preds, bbox_preds = net(X)

print('output anchors:', anchors.shape)
print('output class preds:', cls_preds.shape)
print('output bbox preds:', bbox_preds.shape)
```

```{.python .input}
#@tab pytorch
net = TinySSD(num_classes=1)
X = torch.zeros((32, 3, 256, 256))
anchors, cls_preds, bbox_preds = net(X)

print('output anchors:', anchors.shape)
print('output class preds:', cls_preds.shape)
print('output bbox preds:', bbox_preds.shape)
```

## 学習

ここでは、
物体検出のためにシングルショット・マルチボックス検出モデルを
どのように学習するかを説明します。


### データセットの読み込みとモデルの初期化

まず、
:numref:`sec_object-detection-dataset` で説明した
バナナ検出データセットを[**読み込みます**]。

```{.python .input}
#@tab all
batch_size = 32
train_iter, _ = d2l.load_data_bananas(batch_size)
```

バナナ検出データセットには1つのクラスしかありません。モデルを定義した後は、
そのパラメータを[**初期化し、最適化アルゴリズムを定義する**]必要があります。

```{.python .input}
#@tab mxnet
device, net = d2l.try_gpu(), TinySSD(num_classes=1)
net.initialize(init=init.Xavier(), ctx=device)
trainer = gluon.Trainer(net.collect_params(), 'sgd',
                        {'learning_rate': 0.2, 'wd': 5e-4})
```

```{.python .input}
#@tab pytorch
device, net = d2l.try_gpu(), TinySSD(num_classes=1)
trainer = torch.optim.SGD(net.parameters(), lr=0.2, weight_decay=5e-4)
```

### [**損失関数と評価関数の定義**]

物体検出には2種類の損失があります。
1つ目の損失はアンカーボックスのクラスに関するものです。
その計算には、
画像分類で使った
交差エントロピー損失関数を
そのまま再利用できます。
2つ目の損失は
正例（背景でない）アンカーボックスのオフセットに関するものです。
これは回帰問題です。
この回帰問題では、
しかし、
ここでは :numref:`subsec_normal_distribution_and_squared_loss` で説明した二乗損失は使いません。
代わりに、
$\ell_1$ ノルム損失、
すなわち予測値と正解値の差の絶対値を使います。
マスク変数 `bbox_masks` は、
損失計算において
負例アンカーボックスと不正な（パディングされた）アンカーボックスを除外します。
最後に、
アンカーボックスのクラス損失と
アンカーボックスのオフセット損失を足し合わせて、
モデルの損失関数を得ます。

```{.python .input}
#@tab mxnet
cls_loss = gluon.loss.SoftmaxCrossEntropyLoss()
bbox_loss = gluon.loss.L1Loss()

def calc_loss(cls_preds, cls_labels, bbox_preds, bbox_labels, bbox_masks):
    cls = cls_loss(cls_preds, cls_labels)
    bbox = bbox_loss(bbox_preds * bbox_masks, bbox_labels * bbox_masks)
    return cls + bbox
```

```{.python .input}
#@tab pytorch
cls_loss = nn.CrossEntropyLoss(reduction='none')
bbox_loss = nn.L1Loss(reduction='none')

def calc_loss(cls_preds, cls_labels, bbox_preds, bbox_labels, bbox_masks):
    batch_size, num_classes = cls_preds.shape[0], cls_preds.shape[2]
    cls = cls_loss(cls_preds.reshape(-1, num_classes),
                   cls_labels.reshape(-1)).reshape(batch_size, -1).mean(dim=1)
    bbox = bbox_loss(bbox_preds * bbox_masks,
                     bbox_labels * bbox_masks).mean(dim=1)
    return cls + bbox
```

分類結果の評価には精度を使えます。
オフセットに対して $\ell_1$ ノルム損失を使っているため、
予測されたバウンディングボックスの評価には
*平均絶対誤差* を使います。
これらの予測結果は、
生成されたアンカーボックスと
それらに対する予測オフセットから得られます。

```{.python .input}
#@tab mxnet
def cls_eval(cls_preds, cls_labels):
    # Because the class prediction results are on the final dimension,
    # `argmax` needs to specify this dimension
    return float((cls_preds.argmax(axis=-1).astype(
        cls_labels.dtype) == cls_labels).sum())

def bbox_eval(bbox_preds, bbox_labels, bbox_masks):
    return float((np.abs((bbox_labels - bbox_preds) * bbox_masks)).sum())
```

```{.python .input}
#@tab pytorch
def cls_eval(cls_preds, cls_labels):
    # Because the class prediction results are on the final dimension,
    # `argmax` needs to specify this dimension
    return float((cls_preds.argmax(dim=-1).type(
        cls_labels.dtype) == cls_labels).sum())

def bbox_eval(bbox_preds, bbox_labels, bbox_masks):
    return float((torch.abs((bbox_labels - bbox_preds) * bbox_masks)).sum())
```

### [**モデルの学習**]

モデルを学習するときには、
順伝播でマルチスケールのアンカーボックス (`anchors`) を生成し、
それらのクラス (`cls_preds`) とオフセット (`bbox_preds`) を予測する必要があります。
次に、ラベル情報 `Y` に基づいて、
このように生成されたアンカーボックスのクラス (`cls_labels`) とオフセット (`bbox_labels`) にラベル付けします。
最後に、クラスとオフセットの予測値とラベル値を使って
損失関数を計算します。
簡潔な実装のため、
ここではテストデータセットの評価は省略します。

```{.python .input}
#@tab mxnet
num_epochs, timer = 20, d2l.Timer()
animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs],
                        legend=['class error', 'bbox mae'])
for epoch in range(num_epochs):
    # Sum of training accuracy, no. of examples in sum of training accuracy,
    # Sum of absolute error, no. of examples in sum of absolute error
    metric = d2l.Accumulator(4)
    for features, target in train_iter:
        timer.start()
        X = features.as_in_ctx(device)
        Y = target.as_in_ctx(device)
        with autograd.record():
            # Generate multiscale anchor boxes and predict their classes and
            # offsets
            anchors, cls_preds, bbox_preds = net(X)
            # Label the classes and offsets of these anchor boxes
            bbox_labels, bbox_masks, cls_labels = d2l.multibox_target(anchors,
                                                                      Y)
            # Calculate the loss function using the predicted and labeled
            # values of the classes and offsets
            l = calc_loss(cls_preds, cls_labels, bbox_preds, bbox_labels,
                          bbox_masks)
        l.backward()
        trainer.step(batch_size)
        metric.add(cls_eval(cls_preds, cls_labels), cls_labels.size,
                   bbox_eval(bbox_preds, bbox_labels, bbox_masks),
                   bbox_labels.size)
    cls_err, bbox_mae = 1 - metric[0] / metric[1], metric[2] / metric[3]
    animator.add(epoch + 1, (cls_err, bbox_mae))
print(f'class err {cls_err:.2e}, bbox mae {bbox_mae:.2e}')
print(f'{len(train_iter._dataset) / timer.stop():.1f} examples/sec on '
      f'{str(device)}')
```

```{.python .input}
#@tab pytorch
num_epochs, timer = 20, d2l.Timer()
animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs],
                        legend=['class error', 'bbox mae'])
net = net.to(device)
for epoch in range(num_epochs):
    # Sum of training accuracy, no. of examples in sum of training accuracy,
    # Sum of absolute error, no. of examples in sum of absolute error
    metric = d2l.Accumulator(4)
    net.train()
    for features, target in train_iter:
        timer.start()
        trainer.zero_grad()
        X, Y = features.to(device), target.to(device)
        # Generate multiscale anchor boxes and predict their classes and
        # offsets
        anchors, cls_preds, bbox_preds = net(X)
        # Label the classes and offsets of these anchor boxes
        bbox_labels, bbox_masks, cls_labels = d2l.multibox_target(anchors, Y)
        # Calculate the loss function using the predicted and labeled values
        # of the classes and offsets
        l = calc_loss(cls_preds, cls_labels, bbox_preds, bbox_labels,
                      bbox_masks)
        l.mean().backward()
        trainer.step()
        metric.add(cls_eval(cls_preds, cls_labels), cls_labels.numel(),
                   bbox_eval(bbox_preds, bbox_labels, bbox_masks),
                   bbox_labels.numel())
    cls_err, bbox_mae = 1 - metric[0] / metric[1], metric[2] / metric[3]
    animator.add(epoch + 1, (cls_err, bbox_mae))
print(f'class err {cls_err:.2e}, bbox mae {bbox_mae:.2e}')
print(f'{len(train_iter.dataset) / timer.stop():.1f} examples/sec on '
      f'{str(device)}')
```

## [**予測**]

予測時の目標は、
画像上の関心対象のすべての物体を検出することです。
以下では、
テスト画像を読み込み、リサイズして、
畳み込み層に必要な4次元テンソルに変換します。

```{.python .input}
#@tab mxnet
img = image.imread('../img/banana.jpg')
feature = image.imresize(img, 256, 256).astype('float32')
X = np.expand_dims(feature.transpose(2, 0, 1), axis=0)
```

```{.python .input}
#@tab pytorch
X = torchvision.io.read_image('../img/banana.jpg').unsqueeze(0).float()
img = X.squeeze(0).permute(1, 2, 0).long()
```

以下の `multibox_detection` 関数を使うと、
予測されたバウンディングボックスは
アンカーボックスとそれらの予測オフセットから得られます。
その後、非最大抑制を使って
類似した予測バウンディングボックスを除去します。

```{.python .input}
#@tab mxnet
def predict(X):
    anchors, cls_preds, bbox_preds = net(X.as_in_ctx(device))
    cls_probs = npx.softmax(cls_preds).transpose(0, 2, 1)
    output = d2l.multibox_detection(cls_probs, bbox_preds, anchors)
    idx = [i for i, row in enumerate(output[0]) if row[0] != -1]
    return output[0, idx]

output = predict(X)
```

```{.python .input}
#@tab pytorch
def predict(X):
    net.eval()
    anchors, cls_preds, bbox_preds = net(X.to(device))
    cls_probs = F.softmax(cls_preds, dim=2).permute(0, 2, 1)
    output = d2l.multibox_detection(cls_probs, bbox_preds, anchors)
    idx = [i for i, row in enumerate(output[0]) if row[0] != -1]
    return output[0, idx]

output = predict(X)
```

最後に、[**信頼度が0.9以上のすべての予測バウンディングボックスを表示**]します。

```{.python .input}
#@tab mxnet
def display(img, output, threshold):
    d2l.set_figsize((5, 5))
    fig = d2l.plt.imshow(img.asnumpy())
    for row in output:
        score = float(row[1])
        if score < threshold:
            continue
        h, w = img.shape[:2]
        bbox = [row[2:6] * np.array((w, h, w, h), ctx=row.ctx)]
        d2l.show_bboxes(fig.axes, bbox, '%.2f' % score, 'w')

display(img, output, threshold=0.9)
```

```{.python .input}
#@tab pytorch
def display(img, output, threshold):
    d2l.set_figsize((5, 5))
    fig = d2l.plt.imshow(img)
    for row in output:
        score = float(row[1])
        if score < threshold:
            continue
        h, w = img.shape[:2]
        bbox = [row[2:6] * torch.tensor((w, h, w, h), device=row.device)]
        d2l.show_bboxes(fig.axes, bbox, '%.2f' % score, 'w')

display(img, output.cpu(), threshold=0.9)
```

## まとめ

* シングルショット・マルチボックス検出はマルチスケール物体検出モデルです。ベースネットワークといくつかのマルチスケール特徴マップブロックを通じて、シングルショット・マルチボックス検出は異なるサイズを持つさまざまな数のアンカーボックスを生成し、これらのアンカーボックス（したがってバウンディングボックス）のクラスとオフセットを予測することでさまざまなサイズの物体を検出します。
* シングルショット・マルチボックス検出モデルを学習するとき、損失関数はアンカーボックスのクラスとオフセットの予測値とラベル値に基づいて計算されます。



## 演習

1. 損失関数を改善することで、シングルショット・マルチボックス検出を改良できますか？ たとえば、予測オフセットに対する $\ell_1$ ノルム損失を smooth $\ell_1$ ノルム損失に置き換えてみてください。この損失関数は、滑らかさのためにゼロ付近で二乗関数を使い、その形はハイパーパラメータ $\sigma$ によって制御されます。

$$
f(x) =
    \begin{cases}
    (\sigma x)^2/2,& \textrm{if }|x| < 1/\sigma^2\\
    |x|-0.5/\sigma^2,& \textrm{otherwise}
    \end{cases}
$$

$\sigma$ が非常に大きいとき、この損失は $\ell_1$ ノルム損失に似ています。値が小さいほど、損失関数はより滑らかになります。

```{.python .input}
#@tab mxnet
sigmas = [10, 1, 0.5]
lines = ['-', '--', '-.']
x = np.arange(-2, 2, 0.1)
d2l.set_figsize()

for l, s in zip(lines, sigmas):
    y = npx.smooth_l1(x, scalar=s)
    d2l.plt.plot(x.asnumpy(), y.asnumpy(), l, label='sigma=%.1f' % s)
d2l.plt.legend();
```

```{.python .input}
#@tab pytorch
def smooth_l1(data, scalar):
    out = []
    for i in data:
        if abs(i) < 1 / (scalar ** 2):
            out.append(((scalar * i) ** 2) / 2)
        else:
            out.append(abs(i) - 0.5 / (scalar ** 2))
    return torch.tensor(out)

sigmas = [10, 1, 0.5]
lines = ['-', '--', '-.']
x = torch.arange(-2, 2, 0.1)
d2l.set_figsize()

for l, s in zip(lines, sigmas):
    y = smooth_l1(x, scalar=s)
    d2l.plt.plot(x, y, l, label='sigma=%.1f' % s)
d2l.plt.legend();
```

さらに、この実験ではクラス予測に交差エントロピー損失を使いました。
正解クラス $j$ に対する予測確率を $p_j$ とすると、交差エントロピー損失は $-\log p_j$ です。focal loss
:cite:`Lin.Goyal.Girshick.ea.2017` も使えます。ハイパーパラメータ $\gamma > 0$
と $\alpha > 0$ を用いると、この損失は次のように定義されます。

$$ - \alpha (1-p_j)^{\gamma} \log p_j.$$

見てのとおり、$\gamma$ を増やすと、
うまく分類された例（たとえば $p_j > 0.5$）に対する相対的な損失を
効果的に減らせるので、
学習は誤分類された難しい例に
より重点を置けるようになります。

```{.python .input}
#@tab mxnet
def focal_loss(gamma, x):
    return -(1 - x) ** gamma * np.log(x)

x = np.arange(0.01, 1, 0.01)
for l, gamma in zip(lines, [0, 1, 5]):
    y = d2l.plt.plot(x.asnumpy(), focal_loss(gamma, x).asnumpy(), l,
                     label='gamma=%.1f' % gamma)
d2l.plt.legend();
```

```{.python .input}
#@tab pytorch
def focal_loss(gamma, x):
    return -(1 - x) ** gamma * torch.log(x)

x = torch.arange(0.01, 1, 0.01)
for l, gamma in zip(lines, [0, 1, 5]):
    y = d2l.plt.plot(x, focal_loss(gamma, x), l, label='gamma=%.1f' % gamma)
d2l.plt.legend();
```

2. 紙幅の制約のため、この節ではシングルショット・マルチボックス検出モデルのいくつかの実装詳細を省略しました。次の観点でさらにモデルを改善できますか。
    1. 物体が画像に比べて非常に小さい場合、モデルは入力画像をより大きくリサイズできます。
    1. 通常、負例アンカーボックスは非常に多数あります。クラス分布をより均衡にするために、負例アンカーボックスをダウンサンプリングできます。
    1. 損失関数では、クラス損失とオフセット損失に異なる重みのハイパーパラメータを割り当てます。
    1. シングルショット・マルチボックス検出の論文 :cite:`Liu.Anguelov.Erhan.ea.2016` にあるような、他の方法で物体検出モデルを評価します。



:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/373)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1604)
:end_tab:\n