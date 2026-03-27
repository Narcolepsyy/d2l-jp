# 完全畳み込みネットワーク
:label:`sec_fcn`

:numref:`sec_semantic_segmentation` で述べたように、
セマンティックセグメンテーションは
画像をピクセル単位で分類します。
完全畳み込みネットワーク（FCN; fully convolutional network）は、
畳み込みニューラルネットワークを用いて
画像の各ピクセルをピクセルクラスへ変換します :cite:`Long.Shelhamer.Darrell.2015`。
これまで画像分類や物体検出で見てきたCNNとは異なり、
完全畳み込みネットワークは
中間特徴マップの高さと幅を
入力画像のそれに戻します。
これは :numref:`sec_transposed_conv` で導入した
転置畳み込み層によって実現されます。
その結果、
分類出力と入力画像は
ピクセルレベルで
1対1に対応します。
すなわち、任意の出力ピクセルにおけるチャネル次元には、
同じ空間位置にある入力ピクセルに対する
分類結果が格納されます。

```{.python .input}
#@tab mxnet
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import gluon, image, init, np, npx
from mxnet.gluon import nn

npx.set_np()
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
import torch
import torchvision
from torch import nn
from torch.nn import functional as F
```

## モデル

ここでは、完全畳み込みネットワークモデルの基本設計を説明します。
:numref:`fig_fcn` に示すように、
このモデルはまずCNNを用いて画像特徴を抽出し、
次に $1\times 1$ 畳み込み層を介してチャネル数を
クラス数へ変換し、
最後に :numref:`sec_transposed_conv` で導入した転置畳み込みを用いて
特徴マップの高さと幅を
入力画像のそれへ変換します。
その結果、
モデル出力は入力画像と同じ高さと幅を持ち、
出力チャネルには同じ空間位置にある入力ピクセルの
予測クラスが含まれます。


![完全畳み込みネットワーク。](../img/fcn.svg)
:label:`fig_fcn`

以下では、[**ImageNetデータセットで事前学習されたResNet-18モデルを用いて画像特徴を抽出**]し、
そのモデルインスタンスを `pretrained_net` と表します。
このモデルの最後の数層には
グローバル平均プーリング層と
全結合層が含まれていますが、
これらは完全畳み込みネットワークでは不要です。

```{.python .input}
#@tab mxnet
pretrained_net = gluon.model_zoo.vision.resnet18_v2(pretrained=True)
pretrained_net.features[-3:], pretrained_net.output
```

```{.python .input}
#@tab pytorch
pretrained_net = torchvision.models.resnet18(pretrained=True)
list(pretrained_net.children())[-3:]
```

次に、[**完全畳み込みネットワークのインスタンス `net` を作成**]します。
これは、出力に最も近い最後のグローバル平均プーリング層と
全結合層を除く、
ResNet-18の事前学習済み層をすべてコピーします。

```{.python .input}
#@tab mxnet
net = nn.HybridSequential()
for layer in pretrained_net.features[:-2]:
    net.add(layer)
```

```{.python .input}
#@tab pytorch
net = nn.Sequential(*list(pretrained_net.children())[:-2])
```

高さと幅がそれぞれ320と480の入力を与えると、
`net` の順伝播は
入力の高さと幅を元の1/32、すなわち10と15にまで縮小します。

```{.python .input}
#@tab mxnet
X = np.random.uniform(size=(1, 3, 320, 480))
net(X).shape
```

```{.python .input}
#@tab pytorch
X = torch.rand(size=(1, 3, 320, 480))
net(X).shape
```

次に、[**$1\times 1$ 畳み込み層を用いて出力チャネル数をPascal VOC2012データセットのクラス数（21）へ変換**]します。
最後に、特徴マップの高さと幅を
**32倍に増やして**入力画像の高さと幅に戻す必要があります。
:numref:`sec_padding` で畳み込み層の出力形状の計算方法を思い出してください。
$(320-64+16\times2+32)/32=10$ かつ $(480-64+16\times2+32)/32=15$ なので、ストライドを $32$ に設定した転置畳み込み層を構成し、
カーネルの高さと幅を $64$、パディングを $16$ に設定します。
一般に、
ストライドを $s$、
パディングを $s/2$（$s/2$ が整数であると仮定）、
カーネルの高さと幅を $2s$ とすると、
転置畳み込みは入力の高さと幅を
$s$ 倍に増やします。

```{.python .input}
#@tab mxnet
num_classes = 21
net.add(nn.Conv2D(num_classes, kernel_size=1),
        nn.Conv2DTranspose(
            num_classes, kernel_size=64, padding=16, strides=32))
```

```{.python .input}
#@tab pytorch
num_classes = 21
net.add_module('final_conv', nn.Conv2d(512, num_classes, kernel_size=1))
net.add_module('transpose_conv', nn.ConvTranspose2d(num_classes, num_classes,
                                    kernel_size=64, padding=16, stride=32))
```

## [**転置畳み込み層の初期化**]


すでに見たように、
転置畳み込み層は
特徴マップの高さと幅を増やすことができます。
画像処理では、画像を拡大する、すなわち *アップサンプリング* が必要になることがあります。
*バイリニア補間* は、よく使われるアップサンプリング手法の1つです。
これは転置畳み込み層の初期化にもよく用いられます。

バイリニア補間を説明するために、
入力画像が与えられたときに
アップサンプリング後の出力画像の各ピクセルを
計算したいとします。
出力画像の座標 $(x, y)$ にあるピクセルを計算するには、
まず $(x, y)$ を入力画像上の座標 $(x', y')$ に写像します。たとえば、入力サイズと出力サイズの比に従います。
写像された $x'$ と $y'$ は実数であることに注意してください。
次に、入力画像上で座標 $(x', y')$ に最も近い4つのピクセルを見つけます。
最後に、座標 $(x, y)$ にある出力画像のピクセルは、
入力画像上のこの4つの最近傍ピクセルと、
$(x', y')$ からの相対距離に基づいて計算されます。

バイリニア補間によるアップサンプリングは、
以下の `bilinear_kernel` 関数で構成したカーネルを持つ転置畳み込み層によって実装できます。
紙幅の都合上、
ここでは `bilinear_kernel` 関数の実装のみを示し、
そのアルゴリズム設計についての議論は省略します。

```{.python .input}
#@tab mxnet
def bilinear_kernel(in_channels, out_channels, kernel_size):
    factor = (kernel_size + 1) // 2
    if kernel_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = (np.arange(kernel_size).reshape(-1, 1),
          np.arange(kernel_size).reshape(1, -1))
    filt = (1 - np.abs(og[0] - center) / factor) * \
           (1 - np.abs(og[1] - center) / factor)
    weight = np.zeros((in_channels, out_channels, kernel_size, kernel_size))
    weight[range(in_channels), range(out_channels), :, :] = filt
    return np.array(weight)
```

```{.python .input}
#@tab pytorch
def bilinear_kernel(in_channels, out_channels, kernel_size):
    factor = (kernel_size + 1) // 2
    if kernel_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = (torch.arange(kernel_size).reshape(-1, 1),
          torch.arange(kernel_size).reshape(1, -1))
    filt = (1 - torch.abs(og[0] - center) / factor) * \
           (1 - torch.abs(og[1] - center) / factor)
    weight = torch.zeros((in_channels, out_channels,
                          kernel_size, kernel_size))
    weight[range(in_channels), range(out_channels), :, :] = filt
    return weight
```

転置畳み込み層で実装された[**バイリニア補間によるアップサンプリングを試してみましょう**]。
高さと幅を2倍にする転置畳み込み層を構成し、
そのカーネルを `bilinear_kernel` 関数で初期化します。

```{.python .input}
#@tab mxnet
conv_trans = nn.Conv2DTranspose(3, kernel_size=4, padding=1, strides=2)
conv_trans.initialize(init.Constant(bilinear_kernel(3, 3, 4)))
```

```{.python .input}
#@tab pytorch
conv_trans = nn.ConvTranspose2d(3, 3, kernel_size=4, padding=1, stride=2,
                                bias=False)
conv_trans.weight.data.copy_(bilinear_kernel(3, 3, 4));
```

画像 `X` を読み込み、アップサンプリングの出力を `Y` に代入します。画像を表示するために、チャネル次元の位置を調整する必要があります。

```{.python .input}
#@tab mxnet
img = image.imread('../img/catdog.jpg')
X = np.expand_dims(img.astype('float32').transpose(2, 0, 1), axis=0) / 255
Y = conv_trans(X)
out_img = Y[0].transpose(1, 2, 0)
```

```{.python .input}
#@tab pytorch
img = torchvision.transforms.ToTensor()(d2l.Image.open('../img/catdog.jpg'))
X = img.unsqueeze(0)
Y = conv_trans(X)
out_img = Y[0].permute(1, 2, 0).detach()
```

ご覧のとおり、転置畳み込み層は画像の高さと幅の両方を2倍にします。
座標のスケールが異なることを除けば、
バイリニア補間で拡大した画像と :numref:`sec_bbox` で表示した元の画像は同じように見えます。

```{.python .input}
#@tab mxnet
d2l.set_figsize()
print('input image shape:', img.shape)
d2l.plt.imshow(img.asnumpy());
print('output image shape:', out_img.shape)
d2l.plt.imshow(out_img.asnumpy());
```

```{.python .input}
#@tab pytorch
d2l.set_figsize()
print('input image shape:', img.permute(1, 2, 0).shape)
d2l.plt.imshow(img.permute(1, 2, 0));
print('output image shape:', out_img.shape)
d2l.plt.imshow(out_img);
```

完全畳み込みネットワークでは、[**転置畳み込み層をバイリニア補間によるアップサンプリングで初期化します。$1\times 1$ 畳み込み層にはXavier初期化を用います。**]

```{.python .input}
#@tab mxnet
W = bilinear_kernel(num_classes, num_classes, 64)
net[-1].initialize(init.Constant(W))
net[-2].initialize(init=init.Xavier())
```

```{.python .input}
#@tab pytorch
W = bilinear_kernel(num_classes, num_classes, 64)
net.transpose_conv.weight.data.copy_(W);
```

## [**データセットの読み込み**]

:numref:`sec_semantic_segmentation` で導入した
セマンティックセグメンテーションデータセットを読み込みます。
ランダムクロップの出力画像形状は
$320\times 480$ に指定します。高さと幅の両方が $32$ で割り切れるようにします。

```{.python .input}
#@tab all
batch_size, crop_size = 32, (320, 480)
train_iter, test_iter = d2l.load_data_voc(batch_size, crop_size)
```

## [**学習**]


これで、構築した
完全畳み込みネットワークを学習できます。
ここでの損失関数と精度計算は、以前の章の画像分類と本質的には変わりません。
転置畳み込み層の出力チャネルを用いて
各ピクセルのクラスを予測するため、
損失計算ではチャネル次元が指定されます。
さらに、精度は
すべてのピクセルについての予測クラスの正しさに基づいて計算されます。

```{.python .input}
#@tab mxnet
num_epochs, lr, wd, devices = 5, 0.1, 1e-3, d2l.try_all_gpus()
loss = gluon.loss.SoftmaxCrossEntropyLoss(axis=1)
net.collect_params().reset_ctx(devices)
trainer = gluon.Trainer(net.collect_params(), 'sgd',
                        {'learning_rate': lr, 'wd': wd})
d2l.train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs, devices)
```

```{.python .input}
#@tab pytorch
def loss(inputs, targets):
    return F.cross_entropy(inputs, targets, reduction='none').mean(1).mean(1)

num_epochs, lr, wd, devices = 5, 0.001, 1e-3, d2l.try_all_gpus()
trainer = torch.optim.SGD(net.parameters(), lr=lr, weight_decay=wd)
d2l.train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs, devices)
```

## [**予測**]


予測時には、入力画像の各チャネルを標準化し、
画像をCNNが必要とする4次元の入力形式に変換する必要があります。

```{.python .input}
#@tab mxnet
def predict(img):
    X = test_iter._dataset.normalize_image(img)
    X = np.expand_dims(X.transpose(2, 0, 1), axis=0)
    pred = net(X.as_in_ctx(devices[0])).argmax(axis=1)
    return pred.reshape(pred.shape[1], pred.shape[2])
```

```{.python .input}
#@tab pytorch
def predict(img):
    X = test_iter.dataset.normalize_image(img).unsqueeze(0)
    pred = net(X.to(devices[0])).argmax(dim=1)
    return pred.reshape(pred.shape[1], pred.shape[2])
```

各ピクセルの[**予測クラスを可視化する**]ために、
予測されたクラスをデータセット内のラベル色に戻します。

```{.python .input}
#@tab mxnet
def label2image(pred):
    colormap = np.array(d2l.VOC_COLORMAP, ctx=devices[0], dtype='uint8')
    X = pred.astype('int32')
    return colormap[X, :]
```

```{.python .input}
#@tab pytorch
def label2image(pred):
    colormap = torch.tensor(d2l.VOC_COLORMAP, device=devices[0])
    X = pred.long()
    return colormap[X, :]
```

テストデータセットの画像はサイズも形状もさまざまです。
モデルはストライド32の転置畳み込み層を使うため、
入力画像の高さまたは幅が32で割り切れない場合、
転置畳み込み層の出力の高さまたは幅は入力画像の形状とずれます。
この問題に対処するために、
画像内で高さと幅が32の整数倍である複数の矩形領域を切り出し、
それぞれの領域の画素に対して個別に順伝播を行うことができます。
なお、
これらの矩形領域の和集合が入力画像全体を完全に覆う必要があります。
あるピクセルが複数の矩形領域に含まれる場合、
同じピクセルに対する別々の領域での転置畳み込み出力の平均を
softmax演算への入力として
クラスを予測できます。


簡単のため、ここでは大きめのテスト画像をいくつか読み込み、
画像の左上隅から始まる $320\times480$ の領域を予測に用います。
これらのテスト画像について、
切り出した領域、
予測結果、
および正解を行ごとに表示します。

```{.python .input}
#@tab mxnet
voc_dir = d2l.download_extract('voc2012', 'VOCdevkit/VOC2012')
test_images, test_labels = d2l.read_voc_images(voc_dir, False)
n, imgs = 4, []
for i in range(n):
    crop_rect = (0, 0, 480, 320)
    X = image.fixed_crop(test_images[i], *crop_rect)
    pred = label2image(predict(X))
    imgs += [X, pred, image.fixed_crop(test_labels[i], *crop_rect)]
d2l.show_images(imgs[::3] + imgs[1::3] + imgs[2::3], 3, n, scale=2);
```

```{.python .input}
#@tab pytorch
voc_dir = d2l.download_extract('voc2012', 'VOCdevkit/VOC2012')
test_images, test_labels = d2l.read_voc_images(voc_dir, False)
n, imgs = 4, []
for i in range(n):
    crop_rect = (0, 0, 320, 480)
    X = torchvision.transforms.functional.crop(test_images[i], *crop_rect)
    pred = label2image(predict(X))
    imgs += [X.permute(1,2,0), pred.cpu(),
             torchvision.transforms.functional.crop(
                 test_labels[i], *crop_rect).permute(1,2,0)]
d2l.show_images(imgs[::3] + imgs[1::3] + imgs[2::3], 3, n, scale=2);
```

## まとめ

* 完全畳み込みネットワークはまずCNNを用いて画像特徴を抽出し、次に $1\times 1$ 畳み込み層を介してチャネル数をクラス数へ変換し、最後に転置畳み込みを用いて特徴マップの高さと幅を入力画像のそれへ変換します。
* 完全畳み込みネットワークでは、転置畳み込み層の初期化にバイリニア補間によるアップサンプリングを用いることができます。


## 演習

1. 実験で転置畳み込み層にXavier初期化を用いると、結果はどのように変わりますか？
1. ハイパーパラメータを調整することで、モデルの精度をさらに改善できますか？
1. テスト画像内のすべてのピクセルのクラスを予測してください。
1. 元の完全畳み込みネットワークの論文では、いくつかの中間CNN層の出力も使用しています :cite:`Long.Shelhamer.Darrell.2015`。このアイデアを実装してみてください。
