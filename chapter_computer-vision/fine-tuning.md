# ファインチューニング
:label:`sec_fine_tuning`

前の章では、Fashion-MNIST の訓練データセットに対して、わずか 60000 枚の画像だけを用いてモデルを学習する方法について説明した。また、学術分野で最も広く使われている大規模画像データセットである ImageNet についても述べた。ImageNet には 1000 種類の物体と 1000 万枚以上の画像がある。しかし、私たちが通常扱うデータセットの規模は、この 2 つのデータセットの中間にある。


画像からさまざまな種類の椅子を認識し、その後ユーザーに購入リンクを推薦したいとする。 
1 つの方法として、まず
100 種類の一般的な椅子を特定し、
各椅子について異なる角度から 1000 枚の画像を集め、
収集した画像データセット上で分類モデルを学習することが考えられる。
この椅子データセットは Fashion-MNIST データセットより大きいかもしれないが、
例の数は依然として ImageNet の
10 分の 1 未満である。
そのため、ImageNet に適した複雑なモデルは、
この椅子データセットでは過学習を起こす可能性がある。
さらに、訓練例の数が限られているため、
学習したモデルの精度が
実用上の要求を満たさないかもしれない。


上記の問題に対処するためには、
明らかな解決策として、より多くのデータを収集することが挙げられる。
しかし、データの収集とラベル付けには多くの時間と費用がかかりる。
たとえば、ImageNet データセットを収集するために、研究者たちは研究資金から数百万ドルを費やした。
現在ではデータ収集コストは大幅に下がっているが、それでもこのコストは無視できない。


別の解決策は、*転移学習* を適用して、*ソースデータセット* で学習した知識を *ターゲットデータセット* に転移することである。
たとえば、ImageNet データセットの画像の大半は椅子とは無関係であるが、このデータセットで学習したモデルは、より一般的な画像特徴を抽出できる可能性があり、エッジ、テクスチャ、形状、物体の構成を識別するのに役立ちる。
このような類似した特徴は、
椅子の認識にも
有効である可能性がある。


## 手順


この節では、転移学習における一般的な手法である *ファインチューニング* を紹介する。 :numref:`fig_finetune` に示すように、ファインチューニングは次の 4 つの手順から成る。


1. ソースデータセット（たとえば ImageNet データセット）上で、ニューラルネットワークモデル、すなわち *ソースモデル* を事前学習する。
1. 新しいニューラルネットワークモデル、すなわち *ターゲットモデル* を作成する。これは出力層を除いて、ソースモデル上のすべてのモデル設計とそのパラメータをコピーする。これらのモデルパラメータにはソースデータセットで学習した知識が含まれており、その知識はターゲットデータセットにも適用できると仮定する。また、ソースモデルの出力層はソースデータセットのラベルと密接に関係していると仮定するため、ターゲットモデルでは使用しない。
1. ターゲットモデルに出力層を追加し、その出力数をターゲットデータセットのカテゴリ数にする。そして、この層のモデルパラメータをランダムに初期化する。
1. 椅子データセットのようなターゲットデータセット上でターゲットモデルを学習する。出力層はゼロから学習され、それ以外のすべての層のパラメータはソースモデルのパラメータに基づいてファインチューニングされる。

![Fine tuning.](../img/finetune.svg)
:label:`fig_finetune`

ターゲットデータセットがソースデータセットよりはるかに小さい場合、ファインチューニングはモデルの汎化能力の向上に役立ちる。


## ホットドッグ認識

具体例として、ホットドッグ認識を通じてファインチューニングを示しよう。 
ImageNet データセットで事前学習された ResNet モデルを、
小さなデータセット上でファインチューニングする。
この小さなデータセットは、
ホットドッグを含む画像と含まない画像からなる数千枚の画像で構成されている。
ファインチューニングしたモデルを用いて、
画像からホットドッグを認識する。

```{.python .input}
#@tab mxnet
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import gluon, init, np, npx
from mxnet.gluon import nn
import os

npx.set_np()
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
from torch import nn
import torch
import torchvision
import os
```

### データセットの読み込み

[**ここで使用するホットドッグデータセットは、オンライン画像から取得したものです**]。
このデータセットは、
ホットドッグを含む正例画像 1400 枚と、
他の食品を含む同数の負例画像で構成されている。
両クラスの画像のうち 1000 枚を訓練に使用し、残りをテストに使用する。


ダウンロードしたデータセットを解凍すると、
`hotdog/train` と `hotdog/test` の 2 つのフォルダが得られる。どちらのフォルダにも `hotdog` と `not-hotdog` のサブフォルダがあり、それぞれ対応するクラスの画像が含まれている。

```{.python .input}
#@tab all
#@save
d2l.DATA_HUB['hotdog'] = (d2l.DATA_URL + 'hotdog.zip', 
                         'fba480ffa8aa7e0febbb511d181409f899b9baa5')

data_dir = d2l.download_extract('hotdog')
```

訓練データセットとテストデータセットのすべての画像ファイルをそれぞれ読み込むために、2 つのインスタンスを作成する。

```{.python .input}
#@tab mxnet
train_imgs = gluon.data.vision.ImageFolderDataset(
    os.path.join(data_dir, 'train'))
test_imgs = gluon.data.vision.ImageFolderDataset(
    os.path.join(data_dir, 'test'))
```

```{.python .input}
#@tab pytorch
train_imgs = torchvision.datasets.ImageFolder(os.path.join(data_dir, 'train'))
test_imgs = torchvision.datasets.ImageFolder(os.path.join(data_dir, 'test'))
```

最初の 8 個の正例と最後の 8 個の負例画像を以下に示す。見てわかるように、[**画像のサイズとアスペクト比はさまざまです**]。

```{.python .input}
#@tab all
hotdogs = [train_imgs[i][0] for i in range(8)]
not_hotdogs = [train_imgs[-i - 1][0] for i in range(8)]
d2l.show_images(hotdogs + not_hotdogs, 2, 8, scale=1.4);
```

訓練時には、まず画像からランダムなサイズとランダムなアスペクト比をもつ領域をランダムに切り出し、
その領域を
$224 \times 224$ の入力画像に拡大縮小する。 
テスト時には、画像の高さと幅の両方を 256 ピクセルに拡大縮小し、その後、中央の $224 \times 224$ 領域を入力として切り出す。
さらに、
3 つの RGB（赤、緑、青）色チャネルについて、
チャネルごとに値を *標準化* する。
具体的には、
各チャネルの平均値をそのチャネルの各値から引き、その結果をそのチャネルの標準偏差で割りる。

[~~Data augmentations~~]

```{.python .input}
#@tab mxnet
# Specify the means and standard deviations of the three RGB channels to
# standardize each channel
normalize = gluon.data.vision.transforms.Normalize(
    [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

train_augs = gluon.data.vision.transforms.Compose([
    gluon.data.vision.transforms.RandomResizedCrop(224),
    gluon.data.vision.transforms.RandomFlipLeftRight(),
    gluon.data.vision.transforms.ToTensor(),
    normalize])

test_augs = gluon.data.vision.transforms.Compose([
    gluon.data.vision.transforms.Resize(256),
    gluon.data.vision.transforms.CenterCrop(224),
    gluon.data.vision.transforms.ToTensor(),
    normalize])
```

```{.python .input}
#@tab pytorch
# Specify the means and standard deviations of the three RGB channels to
# standardize each channel
normalize = torchvision.transforms.Normalize(
    [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

train_augs = torchvision.transforms.Compose([
    torchvision.transforms.RandomResizedCrop(224),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.ToTensor(),
    normalize])

test_augs = torchvision.transforms.Compose([
    torchvision.transforms.Resize([256, 256]),
    torchvision.transforms.CenterCrop(224),
    torchvision.transforms.ToTensor(),
    normalize])
```

### [**モデルの定義と初期化**]

ソースモデルとして、ImageNet データセットで事前学習された ResNet-18 を使用する。ここでは、事前学習済みモデルのパラメータを自動的にダウンロードするために `pretrained=True` を指定する。 
このモデルを初めて使用する場合は、
ダウンロードのためにインターネット接続が必要である。

```{.python .input}
#@tab mxnet
pretrained_net = gluon.model_zoo.vision.resnet18_v2(pretrained=True)
```

```{.python .input}
#@tab pytorch
pretrained_net = torchvision.models.resnet18(pretrained=True)
```

:begin_tab:`mxnet`
事前学習済みのソースモデルのインスタンスには、`features` と `output` の 2 つのメンバー変数がある。前者には出力層を除くモデルのすべての層が含まれ、後者はモデルの出力層である。 
この分割の主な目的は、出力層以外のすべての層のモデルパラメータをファインチューニングしやすくすることである。ソースモデルのメンバー変数 `output` を以下に示す。
:end_tab:

:begin_tab:`pytorch`
事前学習済みのソースモデルのインスタンスには、複数の特徴抽出層と出力層 `fc` がある。
この分割の主な目的は、出力層以外のすべての層のモデルパラメータをファインチューニングしやすくすることである。ソースモデルのメンバー変数 `fc` を以下に示す。
:end_tab:

```{.python .input}
#@tab mxnet
pretrained_net.output
```

```{.python .input}
#@tab pytorch
pretrained_net.fc
```

全結合層として、ResNet の最後のグローバル平均プーリングの出力を ImageNet データセットの 1000 クラス出力に変換する。
次に、ターゲットモデルとして新しいニューラルネットワークを構築する。これは事前学習済みのソースモデルと同じ方法で定義されるが、
最終層の出力数だけが
ターゲットデータセットのクラス数（1000 ではなく）に
設定される。

以下のコードでは、ターゲットモデルインスタンス `finetune_net` の出力層より前のモデルパラメータを、ソースモデルの対応する層のモデルパラメータで初期化する。
これらのモデルパラメータは ImageNet 上での事前学習によって得られたものなので、 
有効である。
したがって、こうした事前学習済みパラメータの *ファインチューニング* には、
小さな学習率だけを使えば十分である。
一方、出力層のモデルパラメータはランダムに初期化されるため、通常はゼロから学習するためにより大きな学習率が必要である。
基準学習率を $\eta$ とすると、出力層のモデルパラメータを更新する際には $10\eta$ の学習率を用いる。

```{.python .input}
#@tab mxnet
finetune_net = gluon.model_zoo.vision.resnet18_v2(classes=2)
finetune_net.features = pretrained_net.features
finetune_net.output.initialize(init.Xavier())
# The model parameters in the output layer will be iterated using a learning
# rate ten times greater
finetune_net.output.collect_params().setattr('lr_mult', 10)
```

```{.python .input}
#@tab pytorch
finetune_net = torchvision.models.resnet18(pretrained=True)
finetune_net.fc = nn.Linear(finetune_net.fc.in_features, 2)
nn.init.xavier_uniform_(finetune_net.fc.weight);
```

### [**モデルのファインチューニング**]

まず、ファインチューニングを用いる訓練関数 `train_fine_tuning` を定義し、複数回呼び出せるようにする。

```{.python .input}
#@tab mxnet
def train_fine_tuning(net, learning_rate, batch_size=128, num_epochs=5):
    train_iter = gluon.data.DataLoader(
        train_imgs.transform_first(train_augs), batch_size, shuffle=True)
    test_iter = gluon.data.DataLoader(
        test_imgs.transform_first(test_augs), batch_size)
    devices = d2l.try_all_gpus()
    net.collect_params().reset_ctx(devices)
    net.hybridize()
    loss = gluon.loss.SoftmaxCrossEntropyLoss()
    trainer = gluon.Trainer(net.collect_params(), 'sgd', {
        'learning_rate': learning_rate, 'wd': 0.001})
    d2l.train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs,
                   devices)
```

```{.python .input}
#@tab pytorch
# If `param_group=True`, the model parameters in the output layer will be
# updated using a learning rate ten times greater
def train_fine_tuning(net, learning_rate, batch_size=128, num_epochs=5,
                      param_group=True):
    train_iter = torch.utils.data.DataLoader(torchvision.datasets.ImageFolder(
        os.path.join(data_dir, 'train'), transform=train_augs),
        batch_size=batch_size, shuffle=True)
    test_iter = torch.utils.data.DataLoader(torchvision.datasets.ImageFolder(
        os.path.join(data_dir, 'test'), transform=test_augs),
        batch_size=batch_size)
    devices = d2l.try_all_gpus()
    loss = nn.CrossEntropyLoss(reduction="none")
    if param_group:
        params_1x = [param for name, param in net.named_parameters()
             if name not in ["fc.weight", "fc.bias"]]
        trainer = torch.optim.SGD([{'params': params_1x},
                                   {'params': net.fc.parameters(),
                                    'lr': learning_rate * 10}],
                                lr=learning_rate, weight_decay=0.001)
    else:
        trainer = torch.optim.SGD(net.parameters(), lr=learning_rate,
                                  weight_decay=0.001)    
    d2l.train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs,
                   devices)
```

[**事前学習で得られたモデルパラメータをファインチューニングするために、基準学習率を小さな値に設定する**]。前述の設定に基づき、ターゲットモデルの出力層パラメータは、10 倍大きい学習率を用いてゼロから学習する。

```{.python .input}
#@tab mxnet
train_fine_tuning(finetune_net, 0.01)
```

```{.python .input}
#@tab pytorch
train_fine_tuning(finetune_net, 5e-5)
```

[**比較のために、**] 同一のモデルを定義するが、[**すべてのモデルパラメータをランダム値で初期化する**]。モデル全体をゼロから学習する必要があるため、より大きな学習率を使うことができる。

```{.python .input}
#@tab mxnet
scratch_net = gluon.model_zoo.vision.resnet18_v2(classes=2)
scratch_net.initialize(init=init.Xavier())
train_fine_tuning(scratch_net, 0.1)
```

```{.python .input}
#@tab pytorch
scratch_net = torchvision.models.resnet18()
scratch_net.fc = nn.Linear(scratch_net.fc.in_features, 2)
train_fine_tuning(scratch_net, 5e-4, param_group=False)
```

見てわかるように、ファインチューニングしたモデルは、初期パラメータ値がより有効であるため、同じエポック数ではより良い性能を示す傾向がある。


## まとめ

* 転移学習は、ソースデータセットで学習した知識をターゲットデータセットへ転移する。ファインチューニングは転移学習でよく使われる手法である。
* ターゲットモデルは、出力層を除いてソースモデルからすべてのモデル設計とそのパラメータをコピーし、ターゲットデータセットに基づいてこれらのパラメータをファインチューニングする。一方、ターゲットモデルの出力層はゼロから学習する必要がある。
* 一般に、パラメータのファインチューニングには小さな学習率を使い、出力層をゼロから学習する場合にはより大きな学習率を使える。


## 演習

1. `finetune_net` の学習率をさらに大きくしていきなさい。モデルの精度はどのように変化するか？
2. 比較実験において、`finetune_net` と `scratch_net` のハイパーパラメータをさらに調整しなさい。それでも精度に差はあるか？
3. `finetune_net` の出力層より前のパラメータをソースモデルのものに設定し、訓練中にそれらを *更新しない* ようにしなさい。モデルの精度はどのように変化するか？ 次のコードを使える。

```{.python .input}
#@tab mxnet
finetune_net.features.collect_params().setattr('grad_req', 'null')
```

```{.python .input}
#@tab pytorch
for param in finetune_net.parameters():
    param.requires_grad = False
```

4. 実際には、`ImageNet` データセットには "hotdog" クラスがある。出力層における対応する重みパラメータは、次のコードで取得できる。この重みパラメータをどのように活用できるだろうか。

```{.python .input}
#@tab mxnet
weight = pretrained_net.output.weight
hotdog_w = np.split(weight.data(), 1000, axis=0)[713]
hotdog_w.shape
```

```{.python .input}
#@tab pytorch
weight = pretrained_net.fc.weight
hotdog_w = torch.split(weight.data, 1, dim=0)[934]
hotdog_w.shape
```
