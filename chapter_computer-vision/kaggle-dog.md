# Kaggle における犬種識別（ImageNet Dogs）

この節では、Kaggle 上の犬種識別問題を実践します。[**このコンペティションの Web アドレスは https://www.kaggle.com/c/dog-breed-identification です**]

このコンペティションでは、
120 種類の犬種を識別します。
実際、
このコンペティションのデータセットは
ImageNet データセットのサブセットです。
:numref:`sec_kaggle_cifar10` における CIFAR-10 データセットの画像とは異なり、
ImageNet データセットの画像は
縦横ともにより大きく、サイズもさまざまです。
:numref:`fig_kaggle_dog` はコンペティションの Web ページの情報を示しています。結果を提出するには Kaggle アカウントが必要です。


![犬種識別コンペティションの Web サイト。コンペティションのデータセットは "Data" タブをクリックして取得できます。](../img/kaggle-dog.jpg)
:width:`400px`
:label:`fig_kaggle_dog`

```{.python .input}
#@tab mxnet
from d2l import mxnet as d2l
from mxnet import autograd, gluon, init, npx
from mxnet.gluon import nn
import os

npx.set_np()
```

```{.python .input}
#@tab pytorch
from d2l import torch as d2l
import torch
import torchvision
from torch import nn
import os
```

## データセットの取得と整理

コンペティションのデータセットは訓練セットとテストセットに分かれており、それぞれ 10222 枚と 10357 枚の JPEG 画像からなります。
各画像は 3 つの RGB（カラー）チャネルを持ちます。
訓練データセットには、
ラブラドール、プードル、ダックスフント、サモエド、ハスキー、チワワ、ヨークシャー・テリアなど、
120 種類の犬種が含まれています。


### データセットのダウンロード

Kaggle にログインした後、
:numref:`fig_kaggle_dog` に示したコンペティションの Web ページで "Data" タブをクリックし、
"Download All" ボタンを押してデータセットをダウンロードできます。
ダウンロードしたファイルを `../data` に解凍すると、データセット全体は次のパスにあります。

* ../data/dog-breed-identification/labels.csv
* ../data/dog-breed-identification/sample_submission.csv
* ../data/dog-breed-identification/train
* ../data/dog-breed-identification/test

上の構成は
:numref:`sec_kaggle_cifar10` の CIFAR-10 コンペティションの構成と
似ていることに気づいたかもしれません。そこでは `train/` と `test/` フォルダにそれぞれ訓練用とテスト用の犬画像が含まれ、`labels.csv` には
訓練画像のラベルが含まれています。
同様に、始めやすいように、上で述べた [**データセットの小さなサンプル**] を提供しています: `train_valid_test_tiny.zip`。
Kaggle コンペティション用に完全なデータセットを使う場合は、下の `demo` 変数を `False` に変更する必要があります。

```{.python .input}
#@tab all
#@save
d2l.DATA_HUB['dog_tiny'] = (d2l.DATA_URL + 'kaggle_dog_tiny.zip',
                            '0cb91d09b814ecdc07b50f31f8dcad3e81d6a86d')

# If you use the full dataset downloaded for the Kaggle competition, change
# the variable below to `False`
demo = True
if demo:
    data_dir = d2l.download_extract('dog_tiny')
else:
    data_dir = os.path.join('..', 'data', 'dog-breed-identification')
```

### [**データセットの整理**]

:numref:`sec_kaggle_cifar10` で行ったのと同様に、データセットを整理できます。つまり、
元の訓練セットから検証セットを分割し、画像をラベルごとにまとめたサブフォルダへ移動します。

以下の `reorg_dog_data` 関数は
訓練データのラベルを読み込み、検証セットを分割し、訓練セットを整理します。

```{.python .input}
#@tab all
def reorg_dog_data(data_dir, valid_ratio):
    labels = d2l.read_csv_labels(os.path.join(data_dir, 'labels.csv'))
    d2l.reorg_train_valid(data_dir, labels, valid_ratio)
    d2l.reorg_test(data_dir)


batch_size = 32 if demo else 128
valid_ratio = 0.1
reorg_dog_data(data_dir, valid_ratio)
```

## [**画像拡張**]

この犬種データセットは
ImageNet データセットのサブセットであり、
その画像は
:numref:`sec_kaggle_cifar10` の CIFAR-10 データセットの画像よりも
大きいことを思い出してください。
以下では、
比較的大きな画像に対して有用と思われる
いくつかの画像拡張操作を示します。

```{.python .input}
#@tab mxnet
transform_train = gluon.data.vision.transforms.Compose([
    # Randomly crop the image to obtain an image with an area of 0.08 to 1 of
    # the original area and height-to-width ratio between 3/4 and 4/3. Then,
    # scale the image to create a new 224 x 224 image
    gluon.data.vision.transforms.RandomResizedCrop(224, scale=(0.08, 1.0),
                                                   ratio=(3.0/4.0, 4.0/3.0)),
    gluon.data.vision.transforms.RandomFlipLeftRight(),
    # Randomly change the brightness, contrast, and saturation
    gluon.data.vision.transforms.RandomColorJitter(brightness=0.4,
                                                   contrast=0.4,
                                                   saturation=0.4),
    # Add random noise
    gluon.data.vision.transforms.RandomLighting(0.1),
    gluon.data.vision.transforms.ToTensor(),
    # Standardize each channel of the image
    gluon.data.vision.transforms.Normalize([0.485, 0.456, 0.406],
                                           [0.229, 0.224, 0.225])])
```

```{.python .input}
#@tab pytorch
transform_train = torchvision.transforms.Compose([
    # Randomly crop the image to obtain an image with an area of 0.08 to 1 of
    # the original area and height-to-width ratio between 3/4 and 4/3. Then,
    # scale the image to create a new 224 x 224 image
    torchvision.transforms.RandomResizedCrop(224, scale=(0.08, 1.0),
                                             ratio=(3.0/4.0, 4.0/3.0)),
    torchvision.transforms.RandomHorizontalFlip(),
    # Randomly change the brightness, contrast, and saturation
    torchvision.transforms.ColorJitter(brightness=0.4,
                                       contrast=0.4,
                                       saturation=0.4),
    # Add random noise
    torchvision.transforms.ToTensor(),
    # Standardize each channel of the image
    torchvision.transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])])
```

予測時には、
ランダム性のない画像前処理操作のみを使用します。

```{.python .input}
#@tab mxnet
transform_test = gluon.data.vision.transforms.Compose([
    gluon.data.vision.transforms.Resize(256),
    # Crop a 224 x 224 square area from the center of the image
    gluon.data.vision.transforms.CenterCrop(224),
    gluon.data.vision.transforms.ToTensor(),
    gluon.data.vision.transforms.Normalize([0.485, 0.456, 0.406],
                                           [0.229, 0.224, 0.225])])
```

```{.python .input}
#@tab pytorch
transform_test = torchvision.transforms.Compose([
    torchvision.transforms.Resize(256),
    # Crop a 224 x 224 square area from the center of the image
    torchvision.transforms.CenterCrop(224),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])])
```

## [**データセットの読み込み**]

:numref:`sec_kaggle_cifar10` と同様に、
生の画像ファイルからなる整理済みデータセットを読み込めます。

```{.python .input}
#@tab mxnet
train_ds, valid_ds, train_valid_ds, test_ds = [
    gluon.data.vision.ImageFolderDataset(
        os.path.join(data_dir, 'train_valid_test', folder))
    for folder in ('train', 'valid', 'train_valid', 'test')]
```

```{.python .input}
#@tab pytorch
train_ds, train_valid_ds = [torchvision.datasets.ImageFolder(
    os.path.join(data_dir, 'train_valid_test', folder),
    transform=transform_train) for folder in ['train', 'train_valid']]

valid_ds, test_ds = [torchvision.datasets.ImageFolder(
    os.path.join(data_dir, 'train_valid_test', folder),
    transform=transform_test) for folder in ['valid', 'test']]
```

以下では、データ反復子のインスタンスを
:numref:`sec_kaggle_cifar10` と同じ方法で作成します。

```{.python .input}
#@tab mxnet
train_iter, train_valid_iter = [gluon.data.DataLoader(
    dataset.transform_first(transform_train), batch_size, shuffle=True,
    last_batch='discard') for dataset in (train_ds, train_valid_ds)]

valid_iter = gluon.data.DataLoader(
    valid_ds.transform_first(transform_test), batch_size, shuffle=False,
    last_batch='discard')

test_iter = gluon.data.DataLoader(
    test_ds.transform_first(transform_test), batch_size, shuffle=False,
    last_batch='keep')
```

```{.python .input}
#@tab pytorch
train_iter, train_valid_iter = [torch.utils.data.DataLoader(
    dataset, batch_size, shuffle=True, drop_last=True)
    for dataset in (train_ds, train_valid_ds)]

valid_iter = torch.utils.data.DataLoader(valid_ds, batch_size, shuffle=False,
                                         drop_last=True)

test_iter = torch.utils.data.DataLoader(test_ds, batch_size, shuffle=False,
                                        drop_last=False)
```

## [**事前学習済みモデルのファインチューニング**]

繰り返しになりますが、
このコンペティションのデータセットは ImageNet データセットのサブセットです。
したがって、
:numref:`sec_fine_tuning`
で議論した手法を用いて、
完全な ImageNet データセットで事前学習されたモデルを選び、
そこから画像特徴を抽出して、
それをカスタムの小規模出力ネットワークに入力できます。
深層学習フレームワークの高レベル API には、
ImageNet データセットで事前学習された多様なモデルが用意されています。
ここでは、
事前学習済みの ResNet-34 モデルを選び、
このモデルの出力層の入力
（すなわち抽出された
特徴量）をそのまま再利用します。
その後、元の出力層を、
たとえば 2 層の全結合層を重ねたような、
学習可能な小さなカスタム出力ネットワークに置き換えます。
:numref:`sec_fine_tuning` の実験とは異なり、
以下では特徴抽出に用いる事前学習済みモデルを再学習しません。これにより、学習時間と
勾配を保存するためのメモリが削減されます。

完全な ImageNet データセットの 3 つの RGB チャネルの平均と標準偏差を用いて
画像を標準化したことを思い出してください。
実際、
これは ImageNet 上の事前学習済みモデルによる標準化操作とも
一致しています。

```{.python .input}
#@tab mxnet
def get_net(devices):
    finetune_net = gluon.model_zoo.vision.resnet34_v2(pretrained=True)
    # Define a new output network
    finetune_net.output_new = nn.HybridSequential(prefix='')
    finetune_net.output_new.add(nn.Dense(256, activation='relu'))
    # There are 120 output categories
    finetune_net.output_new.add(nn.Dense(120))
    # Initialize the output network
    finetune_net.output_new.initialize(init.Xavier(), ctx=devices)
    # Distribute the model parameters to the CPUs or GPUs used for computation
    finetune_net.collect_params().reset_ctx(devices)
    return finetune_net
```

```{.python .input}
#@tab pytorch
def get_net(devices):
    finetune_net = nn.Sequential()
    finetune_net.features = torchvision.models.resnet34(pretrained=True)
    # Define a new output network (there are 120 output categories)
    finetune_net.output_new = nn.Sequential(nn.Linear(1000, 256),
                                            nn.ReLU(),
                                            nn.Linear(256, 120))
    # Move the model to devices
    finetune_net = finetune_net.to(devices[0])
    # Freeze parameters of feature layers
    for param in finetune_net.features.parameters():
        param.requires_grad = False
    return finetune_net
```

[**損失を計算する前に**]、
まず事前学習済みモデルの出力層の入力、すなわち抽出された特徴量を取得します。
その後、この特徴量を小さなカスタム出力ネットワークへの入力として用い、損失を計算します。

```{.python .input}
#@tab mxnet
loss = gluon.loss.SoftmaxCrossEntropyLoss()

def evaluate_loss(data_iter, net, devices):
    l_sum, n = 0.0, 0
    for features, labels in data_iter:
        X_shards, y_shards = d2l.split_batch(features, labels, devices)
        output_features = [net.features(X_shard) for X_shard in X_shards]
        outputs = [net.output_new(feature) for feature in output_features]
        ls = [loss(output, y_shard).sum() for output, y_shard
              in zip(outputs, y_shards)]
        l_sum += sum([float(l.sum()) for l in ls])
        n += labels.size
    return l_sum / n
```

```{.python .input}
#@tab pytorch
loss = nn.CrossEntropyLoss(reduction='none')

def evaluate_loss(data_iter, net, devices):
    l_sum, n = 0.0, 0
    for features, labels in data_iter:
        features, labels = features.to(devices[0]), labels.to(devices[0])
        outputs = net(features)
        l = loss(outputs, labels)
        l_sum += l.sum()
        n += labels.numel()
    return l_sum / n
```

## [**学習関数の定義**]

検証セット上でのモデルの性能に基づいて、モデルを選択し、ハイパーパラメータを調整します。モデル学習関数 `train` は
小さなカスタム出力ネットワークのパラメータのみを反復します。

```{.python .input}
#@tab mxnet
def train(net, train_iter, valid_iter, num_epochs, lr, wd, devices, lr_period,
          lr_decay):
    # Only train the small custom output network
    trainer = gluon.Trainer(net.output_new.collect_params(), 'sgd',
                            {'learning_rate': lr, 'momentum': 0.9, 'wd': wd})
    num_batches, timer = len(train_iter), d2l.Timer()
    legend = ['train loss']
    if valid_iter is not None:
        legend.append('valid loss')
    animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs],
                            legend=legend)
    for epoch in range(num_epochs):
        metric = d2l.Accumulator(2)
        if epoch > 0 and epoch % lr_period == 0:
            trainer.set_learning_rate(trainer.learning_rate * lr_decay)
        for i, (features, labels) in enumerate(train_iter):
            timer.start()
            X_shards, y_shards = d2l.split_batch(features, labels, devices)
            output_features = [net.features(X_shard) for X_shard in X_shards]
            with autograd.record():
                outputs = [net.output_new(feature)
                           for feature in output_features]
                ls = [loss(output, y_shard).sum() for output, y_shard
                      in zip(outputs, y_shards)]
            for l in ls:
                l.backward()
            trainer.step(batch_size)
            metric.add(sum([float(l.sum()) for l in ls]), labels.shape[0])
            timer.stop()
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                animator.add(epoch + (i + 1) / num_batches,
                             (metric[0] / metric[1], None))
        if valid_iter is not None:
            valid_loss = evaluate_loss(valid_iter, net, devices)
            animator.add(epoch + 1, (None, valid_loss))
    measures = f'train loss {metric[0] / metric[1]:.3f}'
    if valid_iter is not None:
        measures += f', valid loss {valid_loss:.3f}'
    print(measures + f'\n{metric[1] * num_epochs / timer.sum():.1f}'
          f' examples/sec on {str(devices)}')
```

```{.python .input}
#@tab pytorch
def train(net, train_iter, valid_iter, num_epochs, lr, wd, devices, lr_period,
          lr_decay):
    # Only train the small custom output network
    net = nn.DataParallel(net, device_ids=devices).to(devices[0])
    trainer = torch.optim.SGD((param for param in net.parameters()
                               if param.requires_grad), lr=lr,
                              momentum=0.9, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.StepLR(trainer, lr_period, lr_decay)
    num_batches, timer = len(train_iter), d2l.Timer()
    legend = ['train loss']
    if valid_iter is not None:
        legend.append('valid loss')
    animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs],
                            legend=legend)
    for epoch in range(num_epochs):
        metric = d2l.Accumulator(2)
        for i, (features, labels) in enumerate(train_iter):
            timer.start()
            features, labels = features.to(devices[0]), labels.to(devices[0])
            trainer.zero_grad()
            output = net(features)
            l = loss(output, labels).sum()
            l.backward()
            trainer.step()
            metric.add(l, labels.shape[0])
            timer.stop()
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                animator.add(epoch + (i + 1) / num_batches,
                             (metric[0] / metric[1], None))
        measures = f'train loss {metric[0] / metric[1]:.3f}'
        if valid_iter is not None:
            valid_loss = evaluate_loss(valid_iter, net, devices)
            animator.add(epoch + 1, (None, valid_loss.detach().cpu()))
        scheduler.step()
    if valid_iter is not None:
        measures += f', valid loss {valid_loss:.3f}'
    print(measures + f'\n{metric[1] * num_epochs / timer.sum():.1f}'
          f' examples/sec on {str(devices)}')
```

## [**モデルの学習と検証**]

それではモデルを学習し、検証できます。
以下のハイパーパラメータはすべて調整可能です。
たとえば、エポック数は増やせます。`lr_period` と `lr_decay` はそれぞれ 2 と 0.9 に設定されているため、最適化アルゴリズムの学習率は 2 エポックごとに 0.9 倍になります。

```{.python .input}
#@tab mxnet
devices, num_epochs, lr, wd = d2l.try_all_gpus(), 10, 5e-3, 1e-4
lr_period, lr_decay, net = 2, 0.9, get_net(devices)
net.hybridize()
train(net, train_iter, valid_iter, num_epochs, lr, wd, devices, lr_period,
      lr_decay)
```

```{.python .input}
#@tab pytorch
devices, num_epochs, lr, wd = d2l.try_all_gpus(), 10, 1e-4, 1e-4
lr_period, lr_decay, net = 2, 0.9, get_net(devices)
train(net, train_iter, valid_iter, num_epochs, lr, wd, devices, lr_period,
      lr_decay)
```

## [**テストセットの分類**] と Kaggle への結果提出


:numref:`sec_kaggle_cifar10` の最後の手順と同様に、
最終的には、ラベル付きデータ（検証セットを含む）すべてを使ってモデルを学習し、テストセットを分類します。
分類には
学習済みのカスタム出力ネットワークを使用します。

```{.python .input}
#@tab mxnet
net = get_net(devices)
net.hybridize()
train(net, train_valid_iter, None, num_epochs, lr, wd, devices, lr_period,
      lr_decay)

preds = []
for data, label in test_iter:
    output_features = net.features(data.as_in_ctx(devices[0]))
    output = npx.softmax(net.output_new(output_features))
    preds.extend(output.asnumpy())
ids = sorted(os.listdir(
    os.path.join(data_dir, 'train_valid_test', 'test', 'unknown')))
with open('submission.csv', 'w') as f:
    f.write('id,' + ','.join(train_valid_ds.synsets) + '\n')
    for i, output in zip(ids, preds):
        f.write(i.split('.')[0] + ',' + ','.join(
            [str(num) for num in output]) + '\n')
```

```{.python .input}
#@tab pytorch
net = get_net(devices)
train(net, train_valid_iter, None, num_epochs, lr, wd, devices, lr_period,
      lr_decay)

preds = []
for data, label in test_iter:
    output = torch.nn.functional.softmax(net(data.to(devices[0])), dim=1)
    preds.extend(output.cpu().detach().numpy())
ids = sorted(os.listdir(
    os.path.join(data_dir, 'train_valid_test', 'test', 'unknown')))
with open('submission.csv', 'w') as f:
    f.write('id,' + ','.join(train_valid_ds.classes) + '\n')
    for i, output in zip(ids, preds):
        f.write(i.split('.')[0] + ',' + ','.join(
            [str(num) for num in output]) + '\n')
```

上のコードは
`submission.csv` ファイルを生成し、
:numref:`sec_kaggle_house` で説明したのと同じ方法で
Kaggle に提出できます。


## まとめ


* ImageNet データセットの画像は CIFAR-10 の画像よりも大きく、サイズもさまざまです。異なるデータセットのタスクでは、画像拡張操作を変更してよい場合があります。
* ImageNet データセットのサブセットを分類するには、完全な ImageNet データセットで事前学習されたモデルを利用して特徴を抽出し、カスタムの小規模出力ネットワークのみを学習すればよいです。これにより、計算時間とメモリ使用量を削減できます。


## 演習

1. Kaggle コンペティションの完全なデータセットを使う場合、`batch_size`（バッチサイズ）と `num_epochs`（エポック数）を増やし、他のハイパーパラメータを `lr = 0.01`, `lr_period = 10`, `lr_decay = 0.1` に設定すると、どのような結果が得られますか？
1. より深い事前学習済みモデルを使うと、より良い結果が得られますか？ハイパーパラメータはどのように調整しますか？さらに結果を改善できますか？\n
