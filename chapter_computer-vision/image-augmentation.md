# 画像拡張
:label:`sec_image_augmentation`

:numref:`sec_alexnet` では、さまざまな応用において深層ニューラルネットワークが成功するための前提条件として、大規模データセットが必要であることを述べました。  
*画像拡張* は、訓練画像に一連のランダムな変換を施すことで、元の画像に似ているが異なる訓練例を生成し、訓練データセットの規模を拡大します。  
別の見方をすると、画像拡張は、訓練例に対するランダムな微調整によって、モデルが特定の属性に過度に依存しにくくなり、その結果として汎化能力が向上する、という事実に動機づけられています。  
たとえば、画像をさまざまな方法で切り抜くことで、注目対象が異なる位置に現れるようにでき、これにより対象の位置へのモデルの依存を減らせます。  
また、明るさや色などの要素を調整して、色に対するモデルの感度を下げることもできます。  
当時の AlexNet の成功には、画像拡張が不可欠だった可能性が高いでしょう。  
この節では、コンピュータビジョンで広く使われているこの手法について説明します。

```{.python .input}
#@tab mxnet
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import autograd, gluon, image, init, np, npx
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
```

## 一般的な画像拡張手法

一般的な画像拡張手法を調べるにあたり、以下の $400\times 500$ の画像を例として用います。

```{.python .input}
#@tab mxnet
d2l.set_figsize()
img = image.imread('../img/cat1.jpg')
d2l.plt.imshow(img.asnumpy());
```

```{.python .input}
#@tab pytorch
d2l.set_figsize()
img = d2l.Image.open('../img/cat1.jpg')
d2l.plt.imshow(img);
```

多くの画像拡張手法には、ある程度のランダム性があります。画像拡張の効果を観察しやすくするために、まず補助関数 `apply` を定義します。この関数は、画像拡張手法 `aug` を入力画像 `img` に対して複数回実行し、その結果をすべて表示します。

```{.python .input}
#@tab all
def apply(img, aug, num_rows=2, num_cols=4, scale=1.5):
    Y = [aug(img) for _ in range(num_rows * num_cols)]
    d2l.show_images(Y, num_rows, num_cols, scale=scale)
```

### 反転と切り抜き

:begin_tab:`mxnet`
[**画像を左右反転する**] ことは、通常、対象のカテゴリを変えません。  
これは、最も早くから使われ、かつ最も広く用いられている画像拡張手法の一つです。  
次に、`transforms` モジュールを使って `RandomFlipLeftRight` のインスタンスを作成します。これは、50% の確率で画像を左右反転します。
:end_tab:

:begin_tab:`pytorch`
[**画像を左右反転する**] ことは、通常、対象のカテゴリを変えません。  
これは、最も早くから使われ、かつ最も広く用いられている画像拡張手法の一つです。  
次に、`transforms` モジュールを使って `RandomHorizontalFlip` のインスタンスを作成します。これは、50% の確率で画像を左右反転します。
:end_tab:

```{.python .input}
#@tab mxnet
apply(img, gluon.data.vision.transforms.RandomFlipLeftRight())
```

```{.python .input}
#@tab pytorch
apply(img, torchvision.transforms.RandomHorizontalFlip())
```

:begin_tab:`mxnet`
[**上下反転する**] ことは、左右反転ほど一般的ではありません。とはいえ、この例の画像では、上下反転しても認識の妨げにはなりません。  
次に、`RandomFlipTopBottom` のインスタンスを作成し、50% の確率で画像を上下反転します。
:end_tab:

:begin_tab:`pytorch`
[**上下反転する**] ことは、左右反転ほど一般的ではありません。とはいえ、この例の画像では、上下反転しても認識の妨げにはなりません。  
次に、`RandomVerticalFlip` のインスタンスを作成し、50% の確率で画像を上下反転します。
:end_tab:

```{.python .input}
#@tab mxnet
apply(img, gluon.data.vision.transforms.RandomFlipTopBottom())
```

```{.python .input}
#@tab pytorch
apply(img, torchvision.transforms.RandomVerticalFlip())
```

ここで用いた例の画像では猫が画像の中央にいますが、一般にはそうとは限りません。  
:numref:`sec_pooling` では、プーリング層によって畳み込み層の対象位置に対する感度を下げられることを説明しました。  
さらに、画像をランダムに切り抜くことで、対象が画像内の異なる位置や異なるスケールで現れるようにでき、これもモデルの対象位置への感度を下げるのに役立ちます。

以下のコードでは、毎回元の面積の $10\% \sim 100\%$ の領域を [**ランダムに切り抜き**]、その領域の幅と高さの比を $0.5 \sim 2$ の範囲からランダムに選びます。次に、その領域の幅と高さをどちらも 200 ピクセルに拡大縮小します。  
特に断りがない限り、この節での $a$ と $b$ の間の乱数とは、区間 $[a, b]$ から一様分布に従ってランダムサンプリングされた連続値を指します。

```{.python .input}
#@tab mxnet
shape_aug = gluon.data.vision.transforms.RandomResizedCrop(
    (200, 200), scale=(0.1, 1), ratio=(0.5, 2))
apply(img, shape_aug)
```

```{.python .input}
#@tab pytorch
shape_aug = torchvision.transforms.RandomResizedCrop(
    (200, 200), scale=(0.1, 1), ratio=(0.5, 2))
apply(img, shape_aug)
```

### 色の変更

別の拡張手法として、色を変更する方法があります。画像の色に関して、明るさ、コントラスト、彩度、色相の 4 つの側面を変更できます。以下の例では、画像の [**明るさをランダムに変更**] し、元の画像の 50% ($1-0.5$) から 150% ($1+0.5$) の範囲の値にします。

```{.python .input}
#@tab mxnet
apply(img, gluon.data.vision.transforms.RandomBrightness(0.5))
```

```{.python .input}
#@tab pytorch
apply(img, torchvision.transforms.ColorJitter(
    brightness=0.5, contrast=0, saturation=0, hue=0))
```

同様に、画像の [**色相をランダムに変更**] することもできます。

```{.python .input}
#@tab mxnet
apply(img, gluon.data.vision.transforms.RandomHue(0.5))
```

```{.python .input}
#@tab pytorch
apply(img, torchvision.transforms.ColorJitter(
    brightness=0, contrast=0, saturation=0, hue=0.5))
```

また、`RandomColorJitter` のインスタンスを作成し、画像の `brightness`、`contrast`、`saturation`、`hue` を [**同時にランダム変更する方法**] を設定することもできます。

```{.python .input}
#@tab mxnet
color_aug = gluon.data.vision.transforms.RandomColorJitter(
    brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5)
apply(img, color_aug)
```

```{.python .input}
#@tab pytorch
color_aug = torchvision.transforms.ColorJitter(
    brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5)
apply(img, color_aug)
```

### 複数の画像拡張手法の組み合わせ

実際には、[**複数の画像拡張手法を組み合わせる**] ことがよくあります。  
たとえば、上で定義したさまざまな画像拡張手法を組み合わせ、`Compose` インスタンスを通して各画像に適用できます。

```{.python .input}
#@tab mxnet
augs = gluon.data.vision.transforms.Compose([
    gluon.data.vision.transforms.RandomFlipLeftRight(), color_aug, shape_aug])
apply(img, augs)
```

```{.python .input}
#@tab pytorch
augs = torchvision.transforms.Compose([
    torchvision.transforms.RandomHorizontalFlip(), color_aug, shape_aug])
apply(img, augs)
```

## [**画像拡張を用いた学習**]

画像拡張を用いてモデルを学習してみましょう。  
ここでは、以前使った Fashion-MNIST データセットの代わりに CIFAR-10 データセットを使います。  
これは、Fashion-MNIST データセットでは対象の位置とサイズが正規化されている一方、CIFAR-10 データセットでは対象の色とサイズにより大きな違いがあるためです。  
CIFAR-10 データセットの最初の 32 個の訓練画像を以下に示します。

```{.python .input}
#@tab mxnet
d2l.show_images(gluon.data.vision.CIFAR10(
    train=True)[:32][0], 4, 8, scale=0.8);
```

```{.python .input}
#@tab pytorch
all_images = torchvision.datasets.CIFAR10(train=True, root="../data",
                                          download=True)
d2l.show_images([all_images[i][0] for i in range(32)], 4, 8, scale=0.8);
```

予測時に確定的な結果を得るため、通常は画像拡張を訓練例にのみ適用し、予測時にはランダム操作を伴う画像拡張を使いません。  
[**ここでは最も単純な左右反転のランダム拡張のみを使います**]。さらに、`ToTensor` インスタンスを使って画像のミニバッチを深層学習フレームワークが要求する形式、すなわち  
形状が (バッチサイズ, チャネル数, 高さ, 幅) で、0 から 1 の間の 32 ビット浮動小数点数に変換します。

```{.python .input}
#@tab mxnet
train_augs = gluon.data.vision.transforms.Compose([
    gluon.data.vision.transforms.RandomFlipLeftRight(),
    gluon.data.vision.transforms.ToTensor()])

test_augs = gluon.data.vision.transforms.Compose([
    gluon.data.vision.transforms.ToTensor()])
```

```{.python .input}
#@tab pytorch
train_augs = torchvision.transforms.Compose([
     torchvision.transforms.RandomHorizontalFlip(),
     torchvision.transforms.ToTensor()])

test_augs = torchvision.transforms.Compose([
     torchvision.transforms.ToTensor()])
```

:begin_tab:`mxnet`
次に、画像の読み込みと画像拡張の適用を容易にする補助関数を定義します。  
Gluon のデータセットが提供する `transform_first` 関数は、各訓練例（画像とラベル）の最初の要素、すなわち画像に画像拡張を適用します。  
`DataLoader` の詳細な説明については、:numref:`sec_fashion_mnist` を参照してください。
:end_tab:

:begin_tab:`pytorch`
次に、画像の読み込みと画像拡張の適用を容易にする [**補助関数を定義します**]。  
PyTorch のデータセットが提供する `transform` 引数は、画像を変換する際に拡張を適用します。  
`DataLoader` の詳細な説明については、:numref:`sec_fashion_mnist` を参照してください。
:end_tab:

```{.python .input}
#@tab mxnet
def load_cifar10(is_train, augs, batch_size):
    return gluon.data.DataLoader(
        gluon.data.vision.CIFAR10(train=is_train).transform_first(augs),
        batch_size=batch_size, shuffle=is_train,
        num_workers=d2l.get_dataloader_workers())
```

```{.python .input}
#@tab pytorch
def load_cifar10(is_train, augs, batch_size):
    dataset = torchvision.datasets.CIFAR10(root="../data", train=is_train,
                                           transform=augs, download=True)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                    shuffle=is_train, num_workers=d2l.get_dataloader_workers())
    return dataloader
```

### マルチ GPU 学習

:numref:`sec_resnet` の ResNet-18 モデルを CIFAR-10 データセットで学習します。  
:numref:`sec_multi_gpu_concise` におけるマルチ GPU 学習の説明を思い出してください。  
以下では、[**複数の GPU を使ってモデルを学習・評価する関数を定義します**]。

```{.python .input}
#@tab mxnet
#@save
def train_batch_ch13(net, features, labels, loss, trainer, devices,
                     split_f=d2l.split_batch):
    """Train for a minibatch with multiple GPUs (defined in Chapter 13)."""
    X_shards, y_shards = split_f(features, labels, devices)
    with autograd.record():
        pred_shards = [net(X_shard) for X_shard in X_shards]
        ls = [loss(pred_shard, y_shard) for pred_shard, y_shard
              in zip(pred_shards, y_shards)]
    for l in ls:
        l.backward()
    # The `True` flag allows parameters with stale gradients, which is useful
    # later (e.g., in fine-tuning BERT)
    trainer.step(labels.shape[0], ignore_stale_grad=True)
    train_loss_sum = sum([float(l.sum()) for l in ls])
    train_acc_sum = sum(d2l.accuracy(pred_shard, y_shard)
                        for pred_shard, y_shard in zip(pred_shards, y_shards))
    return train_loss_sum, train_acc_sum
```

```{.python .input}
#@tab pytorch
#@save
def train_batch_ch13(net, X, y, loss, trainer, devices):
    """Train for a minibatch with multiple GPUs (defined in Chapter 13)."""
    if isinstance(X, list):
        # Required for BERT fine-tuning (to be covered later)
        X = [x.to(devices[0]) for x in X]
    else:
        X = X.to(devices[0])
    y = y.to(devices[0])
    net.train()
    trainer.zero_grad()
    pred = net(X)
    l = loss(pred, y)
    l.sum().backward()
    trainer.step()
    train_loss_sum = l.sum()
    train_acc_sum = d2l.accuracy(pred, y)
    return train_loss_sum, train_acc_sum
```

```{.python .input}
#@tab mxnet
#@save
def train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs,
               devices=d2l.try_all_gpus(), split_f=d2l.split_batch):
    """Train a model with multiple GPUs (defined in Chapter 13)."""
    timer, num_batches = d2l.Timer(), len(train_iter)
    animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0, 1],
                            legend=['train loss', 'train acc', 'test acc'])
    for epoch in range(num_epochs):
        # Sum of training loss, sum of training accuracy, no. of examples,
        # no. of predictions
        metric = d2l.Accumulator(4)
        for i, (features, labels) in enumerate(train_iter):
            timer.start()
            l, acc = train_batch_ch13(
                net, features, labels, loss, trainer, devices, split_f)
            metric.add(l, acc, labels.shape[0], labels.size)
            timer.stop()
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                animator.add(epoch + (i + 1) / num_batches,
                             (metric[0] / metric[2], metric[1] / metric[3],
                              None))
        test_acc = d2l.evaluate_accuracy_gpus(net, test_iter, split_f)
        animator.add(epoch + 1, (None, None, test_acc))
    print(f'loss {metric[0] / metric[2]:.3f}, train acc '
          f'{metric[1] / metric[3]:.3f}, test acc {test_acc:.3f}')
    print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec on '
          f'{str(devices)}')
```

```{.python .input}
#@tab pytorch
#@save
def train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs,
               devices=d2l.try_all_gpus()):
    """Train a model with multiple GPUs (defined in Chapter 13)."""
    timer, num_batches = d2l.Timer(), len(train_iter)
    animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0, 1],
                            legend=['train loss', 'train acc', 'test acc'])
    net = nn.DataParallel(net, device_ids=devices).to(devices[0])
    for epoch in range(num_epochs):
        # Sum of training loss, sum of training accuracy, no. of examples,
        # no. of predictions
        metric = d2l.Accumulator(4)
        for i, (features, labels) in enumerate(train_iter):
            timer.start()
            l, acc = train_batch_ch13(
                net, features, labels, loss, trainer, devices)
            metric.add(l, acc, labels.shape[0], labels.numel())
            timer.stop()
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                animator.add(epoch + (i + 1) / num_batches,
                             (metric[0] / metric[2], metric[1] / metric[3],
                              None))
        test_acc = d2l.evaluate_accuracy_gpu(net, test_iter)
        animator.add(epoch + 1, (None, None, test_acc))
    print(f'loss {metric[0] / metric[2]:.3f}, train acc '
          f'{metric[1] / metric[3]:.3f}, test acc {test_acc:.3f}')
    print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec on '
          f'{str(devices)}')
```

ここで、[**画像拡張を用いてモデルを学習する `train_with_data_aug` 関数を定義します**]。  
この関数は利用可能なすべての GPU を取得し、最適化アルゴリズムとして Adam を使い、訓練データセットに画像拡張を適用し、最後に先ほど定義した `train_ch13` 関数を呼び出してモデルを学習・評価します。

```{.python .input}
#@tab mxnet
batch_size, devices, net = 256, d2l.try_all_gpus(), d2l.resnet18(10)
net.initialize(init=init.Xavier(), ctx=devices)

def train_with_data_aug(train_augs, test_augs, net, lr=0.001):
    train_iter = load_cifar10(True, train_augs, batch_size)
    test_iter = load_cifar10(False, test_augs, batch_size)
    loss = gluon.loss.SoftmaxCrossEntropyLoss()
    trainer = gluon.Trainer(net.collect_params(), 'adam',
                            {'learning_rate': lr})
    train_ch13(net, train_iter, test_iter, loss, trainer, 10, devices)
```

```{.python .input}
#@tab pytorch
batch_size, devices, net = 256, d2l.try_all_gpus(), d2l.resnet18(10, 3)
net.apply(d2l.init_cnn)

def train_with_data_aug(train_augs, test_augs, net, lr=0.001):
    train_iter = load_cifar10(True, train_augs, batch_size)
    test_iter = load_cifar10(False, test_augs, batch_size)
    loss = nn.CrossEntropyLoss(reduction="none")
    trainer = torch.optim.Adam(net.parameters(), lr=lr)
    net(next(iter(train_iter))[0])
    train_ch13(net, train_iter, test_iter, loss, trainer, 10, devices)
```

それでは、ランダム左右反転に基づく画像拡張を用いて [**モデルを学習**] してみましょう。

```{.python .input}
#@tab all
train_with_data_aug(train_augs, test_augs, net)
```

## まとめ

* 画像拡張は、既存の訓練データに基づいてランダムな画像を生成し、モデルの汎化能力を向上させます。
* 予測時に確定的な結果を得るため、通常は画像拡張を訓練例にのみ適用し、予測時にはランダム操作を伴う画像拡張を使いません。
* 深層学習フレームワークは、同時に適用できる多様な画像拡張手法を提供しています。


## 演習

1. 画像拡張を使わずにモデルを学習してください: `train_with_data_aug(test_augs, test_augs)`. 画像拡張を使う場合と使わない場合の訓練精度とテスト精度を比較してください。この比較実験は、画像拡張が過学習を緩和できるという主張を支持できますか。なぜですか。
1. CIFAR-10 データセットでのモデル学習に、複数の異なる画像拡張手法を組み合わせてください。テスト精度は向上しますか。 
1. 深層学習フレームワークのオンラインドキュメントを参照してください。ほかにどのような画像拡張手法が提供されていますか。

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/367)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1404)
:end_tab:\n