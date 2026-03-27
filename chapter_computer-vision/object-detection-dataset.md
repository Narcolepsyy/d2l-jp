# 物体検出データセット
:label:`sec-object-detection-dataset`

物体検出の分野には、MNIST や Fashion-MNIST のような小規模データセットはない。
物体検出モデルを手早く示すために、
[**私たちは小さなデータセットを収集してラベル付けした**]。
まず、オフィスにあった自由に使えるバナナの写真を撮り、
回転やサイズの異なる 1000 枚のバナナ画像を生成した。
次に、それぞれのバナナ画像を
いくつかの背景画像上のランダムな位置に配置した。
最後に、それらの画像中のバナナに対してバウンディングボックスをラベル付けした。


## [**データセットのダウンロード**]

画像ファイルと csv ラベルファイルをすべて含むバナナ検出データセットは、インターネットから直接ダウンロードできる。

```{.python .input}
#@tab mxnet
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import gluon, image, np, npx
import os
import pandas as pd

npx.set_np()
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
import torch
import torchvision
import os
import pandas as pd
```

```{.python .input}
#@tab all
#@save
d2l.DATA_HUB['banana-detection'] = (
    d2l.DATA_URL + 'banana-detection.zip',
    '5de26c8fce5ccdea9f91267273464dc968d20d72')
```

## データセットの読み込み

以下の `read_data_bananas` 関数で[**バナナ検出データセットを読み込みます**]。
このデータセットには、
物体クラスのラベルと
左上および右下の角の
正解バウンディングボックス座標を含む csv ファイルがある。

```{.python .input}
#@tab mxnet
#@save
def read_data_bananas(is_train=True):
    """Read the banana detection dataset images and labels."""
    data_dir = d2l.download_extract('banana-detection')
    csv_fname = os.path.join(data_dir, 'bananas_train' if is_train
                             else 'bananas_val', 'label.csv')
    csv_data = pd.read_csv(csv_fname)
    csv_data = csv_data.set_index('img_name')
    images, targets = [], []
    for img_name, target in csv_data.iterrows():
        images.append(image.imread(
            os.path.join(data_dir, 'bananas_train' if is_train else
                         'bananas_val', 'images', f'{img_name}')))
        # Here `target` contains (class, upper-left x, upper-left y,
        # lower-right x, lower-right y), where all the images have the same
        # banana class (index 0)
        targets.append(list(target))
    return images, np.expand_dims(np.array(targets), 1) / 256
```

```{.python .input}
#@tab pytorch
#@save
def read_data_bananas(is_train=True):
    """Read the banana detection dataset images and labels."""
    data_dir = d2l.download_extract('banana-detection')
    csv_fname = os.path.join(data_dir, 'bananas_train' if is_train
                             else 'bananas_val', 'label.csv')
    csv_data = pd.read_csv(csv_fname)
    csv_data = csv_data.set_index('img_name')
    images, targets = [], []
    for img_name, target in csv_data.iterrows():
        images.append(torchvision.io.read_image(
            os.path.join(data_dir, 'bananas_train' if is_train else
                         'bananas_val', 'images', f'{img_name}')))
        # Here `target` contains (class, upper-left x, upper-left y,
        # lower-right x, lower-right y), where all the images have the same
        # banana class (index 0)
        targets.append(list(target))
    return images, torch.tensor(targets).unsqueeze(1) / 256
```

`read_data_bananas` 関数を使って画像とラベルを読み込むことで、
以下の `BananasDataset` クラスを用いて、
バナナ検出データセットを読み込むための[**カスタマイズした `Dataset` インスタンスを作成**]できる。

```{.python .input}
#@tab mxnet
#@save
class BananasDataset(gluon.data.Dataset):
    """A customized dataset to load the banana detection dataset."""
    def __init__(self, is_train):
        self.features, self.labels = read_data_bananas(is_train)
        print('read ' + str(len(self.features)) + (f' training examples' if
              is_train else f' validation examples'))

    def __getitem__(self, idx):
        return (self.features[idx].astype('float32').transpose(2, 0, 1),
                self.labels[idx])

    def __len__(self):
        return len(self.features)
```

```{.python .input}
#@tab pytorch
#@save
class BananasDataset(torch.utils.data.Dataset):
    """A customized dataset to load the banana detection dataset."""
    def __init__(self, is_train):
        self.features, self.labels = read_data_bananas(is_train)
        print('read ' + str(len(self.features)) + (f' training examples' if
              is_train else f' validation examples'))

    def __getitem__(self, idx):
        return (self.features[idx].float(), self.labels[idx])

    def __len__(self):
        return len(self.features)
```

最後に、
学習用とテスト用の両方について[**2 つのデータ反復子インスタンスを返す**] `load_data_bananas` 関数を定義する。
テストデータセットについては、
ランダムな順序で読み込む必要はない。

```{.python .input}
#@tab mxnet
#@save
def load_data_bananas(batch_size):
    """Load the banana detection dataset."""
    train_iter = gluon.data.DataLoader(BananasDataset(is_train=True),
                                       batch_size, shuffle=True)
    val_iter = gluon.data.DataLoader(BananasDataset(is_train=False),
                                     batch_size)
    return train_iter, val_iter
```

```{.python .input}
#@tab pytorch
#@save
def load_data_bananas(batch_size):
    """Load the banana detection dataset."""
    train_iter = torch.utils.data.DataLoader(BananasDataset(is_train=True),
                                             batch_size, shuffle=True)
    val_iter = torch.utils.data.DataLoader(BananasDataset(is_train=False),
                                           batch_size)
    return train_iter, val_iter
```

ミニバッチを 1 つ読み込み、このミニバッチ内の
画像とラベルの両方の形状を[**表示してみましょう**]。
画像ミニバッチの形状
（バッチサイズ、チャネル数、高さ、幅）は、
見覚えがあるはずである。
これは、以前の画像分類タスクと同じである。
ラベルミニバッチの形状は
（バッチサイズ、$m$、5）であり、
ここで $m$ はデータセット中のどの画像が持つバウンディングボックス数としても
あり得る最大値である。

ミニバッチでの計算はより効率的であるが、
連結によってミニバッチを作るには、すべての画像例が
同じ数のバウンディングボックスを含んでいる必要がある。
一般には、
画像ごとにバウンディングボックスの数は異なり得る。
そのため、
$m$ 個未満のバウンディングボックスしか持たない画像には、
$m$ に達するまで不正なバウンディングボックスがパディングされる。
その後、
各バウンディングボックスのラベルは長さ 5 の配列で表される。
配列の最初の要素はバウンディングボックス内の物体のクラスであり、
-1 はパディング用の不正なバウンディングボックスを示す。
配列の残り 4 要素は、
バウンディングボックスの左上隅と右下隅の
($x$, $y$) 座標値
（範囲は 0 から 1）である。
バナナデータセットでは、
各画像に 1 つのバウンディングボックスしかないため、
$m=1$ である。

```{.python .input}
#@tab all
batch_size, edge_size = 32, 256
train_iter, _ = load_data_bananas(batch_size)
batch = next(iter(train_iter))
batch[0].shape, batch[1].shape
```

## [**デモンストレーション**]

ラベル付けされた正解バウンディングボックス付きの 10 枚の画像を示しよう。
これらの画像全体で、バナナの回転、サイズ、位置が異なっていることがわかる。
もちろん、これは単純な人工データセットにすぎない。
実際には、現実世界のデータセットは通常もっと複雑である。

```{.python .input}
#@tab mxnet
imgs = (batch[0][:10].transpose(0, 2, 3, 1)) / 255
axes = d2l.show_images(imgs, 2, 5, scale=2)
for ax, label in zip(axes, batch[1][:10]):
    d2l.show_bboxes(ax, [label[0][1:5] * edge_size], colors=['w'])
```

```{.python .input}
#@tab pytorch
imgs = (batch[0][:10].permute(0, 2, 3, 1)) / 255
axes = d2l.show_images(imgs, 2, 5, scale=2)
for ax, label in zip(axes, batch[1][:10]):
    d2l.show_bboxes(ax, [label[0][1:5] * edge_size], colors=['w'])
```

## まとめ

* 私たちが収集したバナナ検出データセットは、物体検出モデルのデモに利用できる。
* 物体検出のデータ読み込みは画像分類の場合と似ている。ただし、物体検出ではラベルに正解バウンディングボックスの情報も含まれるが、画像分類にはそれがない。


## 演習

1. バナナ検出データセットに含まれる他の画像についても、正解バウンディングボックスを示してみよう。バウンディングボックスと物体の観点で、どのような違いがあるか？
1. 物体検出に対して、ランダムクロップのようなデータ拡張を適用したいとする。画像分類の場合と比べて、どのように異なるか？ ヒント: クロップした画像に物体の一部しか含まれない場合はどうなるだろうか。
