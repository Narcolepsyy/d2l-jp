{.python .input}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow', 'jax'])
```

# 画像分類データセット
:label:`sec_fashion_mnist`


画像分類で広く使われているデータセットの1つに、手書き数字の [MNISTデータセット](https://en.wikipedia.org/wiki/MNIST_database) :cite:`LeCun.Bottou.Bengio.ea.1998` がある。1990年代に公開された当時、これはほとんどの機械学習アルゴリズムにとって手強い課題であり、$28 \times 28$ ピクセル解像度の画像60,000枚（加えて10,000枚のテストデータセット）から構成されていた。参考までに言うと、1995年当時、64MBという膨大なRAMと5 MFLOPsという驚異的な性能を備えたSun SPARCStation 5は、AT&T Bell Laboratoriesにおける機械学習向けの最先端機器と見なされていた。数字認識で高い精度を達成することは、1990年代にUSPSの郵便物仕分けを自動化するうえで重要だった。LeNet-5 :cite:`LeCun.Jackel.Bottou.ea.1995` のような深層ネットワーク、変換不変性を組み込んだサポートベクターマシン :cite:`Scholkopf.Burges.Vapnik.1996`、接線距離分類器 :cite:`Simard.LeCun.Denker.ea.1998` は、いずれも1%未満の誤り率を達成できた。 

10年以上にわたり、MNISTは機械学習アルゴリズムを比較するための*基準*として機能してきた。 
ベンチマークデータセットとして長く活躍したが、
今日の基準では単純なモデルでも95%を超える分類精度を達成できるため、
強いモデルと弱いモデルを見分けるには不十分である。さらに、このデータセットでは、通常の多くの分類問題では見られないほど*非常に*高い精度が可能である。そのため、アルゴリズムの発展は、クリーンなデータセットの利点を活かせる特定の系統、たとえばアクティブセット法や境界探索型アクティブセットアルゴリズムへと偏っていった。
今日では、MNISTはベンチマークというよりも、むしろ健全性確認のためのチェックとして使われている。ImageNet :cite:`Deng.Dong.Socher.ea.2009` のほうが、はるかに
現実的な課題を提示す。残念ながら、ImageNetは大きすぎるため、この本の多くの例や図示には向いていない。というのも、例を対話的に動かせるようにするには、学習に時間がかかりすぎるからである。代わりとして、以降の節では、質的には似ているもののはるかに小さいFashion-MNIST
データセット :cite:`Xiao.Rasul.Vollgraf.2017` を扱いる。これは$28 \times 28$ ピクセル解像度の10種類の衣類画像を含んでいる。

```{.python .input}
%%tab mxnet
%matplotlib inline
import time
from d2l import mxnet as d2l
from mxnet import gluon, npx
from mxnet.gluon.data.vision import transforms
npx.set_np()

d2l.use_svg_display()
```

```{.python .input}
%%tab pytorch
%matplotlib inline
import time
from d2l import torch as d2l
import torch
import torchvision
from torchvision import transforms

d2l.use_svg_display()
```

```{.python .input}
%%tab tensorflow
%matplotlib inline
import time
from d2l import tensorflow as d2l
import tensorflow as tf

d2l.use_svg_display()
```

```{.python .input}
%%tab jax
%matplotlib inline
from d2l import jax as d2l
import jax
from jax import numpy as jnp
import numpy as np
import time
import tensorflow as tf
import tensorflow_datasets as tfds

d2l.use_svg_display()
```

## データセットの読み込み

Fashion-MNISTデータセットは非常に有用なので、主要なフレームワークはすべて、その前処理済みバージョンを提供している。組み込みのフレームワーク用ユーティリティを使って、[**ダウンロードしてメモリに読み込む**]ことができる。

```{.python .input}
%%tab mxnet
class FashionMNIST(d2l.DataModule):  #@save
    """The Fashion-MNIST dataset."""
    def __init__(self, batch_size=64, resize=(28, 28)):
        super().__init__()
        self.save_hyperparameters()
        trans = transforms.Compose([transforms.Resize(resize),
                                    transforms.ToTensor()])
        self.train = gluon.data.vision.FashionMNIST(
            train=True).transform_first(trans)
        self.val = gluon.data.vision.FashionMNIST(
            train=False).transform_first(trans)
```

```{.python .input}
%%tab pytorch
class FashionMNIST(d2l.DataModule):  #@save
    """The Fashion-MNIST dataset."""
    def __init__(self, batch_size=64, resize=(28, 28)):
        super().__init__()
        self.save_hyperparameters()
        trans = transforms.Compose([transforms.Resize(resize),
                                    transforms.ToTensor()])
        self.train = torchvision.datasets.FashionMNIST(
            root=self.root, train=True, transform=trans, download=True)
        self.val = torchvision.datasets.FashionMNIST(
            root=self.root, train=False, transform=trans, download=True)
```

```{.python .input}
%%tab tensorflow, jax
class FashionMNIST(d2l.DataModule):  #@save
    """The Fashion-MNIST dataset."""
    def __init__(self, batch_size=64, resize=(28, 28)):
        super().__init__()
        self.save_hyperparameters()
        self.train, self.val = tf.keras.datasets.fashion_mnist.load_data()
```

Fashion-MNISTは10カテゴリの画像からなり、各カテゴリにつき学習データセットには6000枚、テストデータセットには1000枚が含まれている。
*テストデータセット*はモデル性能を評価するために使われる（学習には使ってはいけない）。
したがって、学習セットとテストセットにはそれぞれ60,000枚と10,000枚の画像が含まれている。

```{.python .input}
%%tab mxnet, pytorch
data = FashionMNIST(resize=(32, 32))
len(data.train), len(data.val)
```

```{.python .input}
%%tab tensorflow, jax
data = FashionMNIST(resize=(32, 32))
len(data.train[0]), len(data.val[0])
```

上の画像はグレースケールで、$32 \times 32$ ピクセル解像度に拡大されている。これは、元のMNISTデータセットが（2値の）白黒画像で構成されていたのと似ている。ただし、現代の画像データの多くは3チャネル（赤、緑、青）を持ち、ハイパースペクトル画像では100チャネルを超えることもある（HyMapセンサーは126チャネルを持つ）。
慣例として、画像は $c \times h \times w$ のテンソルとして保存する。ここで $c$ は色チャネル数、$h$ は高さ、$w$ は幅である。

```{.python .input}
%%tab all
data.train[0][0].shape
```

データセットを可視化するためのユーティリティ関数を定義する。

Fashion-MNISTのカテゴリには、人間にとって理解しやすい名前が付いている。 
次の便利メソッドは、数値ラベルとその名前を相互に変換する。

```{.python .input}
%%tab all
@d2l.add_to_class(FashionMNIST)  #@save
def text_labels(self, indices):
    """Return text labels."""
    labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
              'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [labels[int(i)] for i in indices]
```

## ミニバッチの読み込み

学習セットとテストセットから読み込む際の手間を減らすために、
ゼロから作るのではなく、組み込みのデータイテレータを使う。
繰り返しになるが、各反復でデータイテレータは
[**サイズ `batch_size` のミニバッチを読み込みる。**]
また、学習データのイテレータではデータ例をランダムにシャッフルする。

```{.python .input}
%%tab mxnet
@d2l.add_to_class(FashionMNIST)  #@save
def get_dataloader(self, train):
    data = self.train if train else self.val
    return gluon.data.DataLoader(data, self.batch_size, shuffle=train,
                                 num_workers=self.num_workers)
```

```{.python .input}
%%tab pytorch
@d2l.add_to_class(FashionMNIST)  #@save
def get_dataloader(self, train):
    data = self.train if train else self.val
    return torch.utils.data.DataLoader(data, self.batch_size, shuffle=train,
                                       num_workers=self.num_workers)
```

```{.python .input}
%%tab tensorflow, jax
@d2l.add_to_class(FashionMNIST)  #@save
def get_dataloader(self, train):
    data = self.train if train else self.val
    process = lambda X, y: (tf.expand_dims(X, axis=3) / 255,
                            tf.cast(y, dtype='int32'))
    resize_fn = lambda X, y: (tf.image.resize_with_pad(X, *self.resize), y)
    shuffle_buf = len(data[0]) if train else 1
    if tab.selected('tensorflow'):
        return tf.data.Dataset.from_tensor_slices(process(*data)).batch(
            self.batch_size).map(resize_fn).shuffle(shuffle_buf)
    if tab.selected('jax'):
        return tfds.as_numpy(
            tf.data.Dataset.from_tensor_slices(process(*data)).batch(
                self.batch_size).map(resize_fn).shuffle(shuffle_buf))
```

これがどのように動くかを見るために、`train_dataloader` メソッドを呼び出して画像のミニバッチを読み込んでみよう。そこには64枚の画像が含まれている。

```{.python .input}
%%tab all
X, y = next(iter(data.train_dataloader()))
print(X.shape, X.dtype, y.shape, y.dtype)
```

画像の読み込みにかかる時間を見てみよう。組み込みローダーではあるが、ものすごく高速というわけではない。それでも、深層ネットワークで画像を処理するほうがかなり時間がかかるので、これで十分である。したがって、ネットワークの学習がI/Oに制約されないことが重要である。

```{.python .input}
%%tab all
tic = time.time()
for X, y in data.train_dataloader():
    continue
f'{time.time() - tic:.2f} sec'
```

## 可視化

これからFashion-MNISTデータセットを頻繁に使う。便利関数 `show_images` を使えば、画像とそれに対応するラベルを可視化できる。 
実装の詳細は省略し、ここではインターフェースだけを示す。こうしたユーティリティ関数については、どのように動くかではなく、`d2l.show_images` をどう呼び出すかだけを知っていれば十分である。

```{.python .input}
%%tab all
def show_images(imgs, num_rows, num_cols, titles=None, scale=1.5):  #@save
    """Plot a list of images."""
    raise NotImplementedError
```

実際に使ってみよう。一般に、学習に使うデータを可視化して確認するのは良い考えである。 
人間は異常を見つけるのが非常に得意なので、可視化は実験設計におけるミスや誤りに対する追加の安全策として役立ちる。以下は、学習データセットの最初のいくつかのデータ例について、[**画像とそれに対応するラベル**]（テキスト表示）である。

```{.python .input}
%%tab all
@d2l.add_to_class(FashionMNIST)  #@save
def visualize(self, batch, nrows=1, ncols=8, labels=[]):
    X, y = batch
    if not labels:
        labels = self.text_labels(y)
    if tab.selected('mxnet', 'pytorch'):
        d2l.show_images(X.squeeze(1), nrows, ncols, titles=labels)
    if tab.selected('tensorflow'):
        d2l.show_images(tf.squeeze(X), nrows, ncols, titles=labels)
    if tab.selected('jax'):
        d2l.show_images(jnp.squeeze(X), nrows, ncols, titles=labels)

batch = next(iter(data.val_dataloader()))
data.visualize(batch)
```

これで、以降の節でFashion-MNISTデータセットを扱う準備が整った。

## まとめ

これで、分類に使う少し現実的なデータセットが手に入った。Fashion-MNISTは、10カテゴリを表す画像からなる衣類分類データセットである。このデータセットは、単純な線形モデルから高度な残差ネットワークまで、さまざまなネットワーク設計を評価するために、以降の節や章で使う。画像を扱うときによく行うように、画像は（バッチサイズ、チャネル数、高さ、幅）という形状のテンソルとして読み込む。今のところ画像はグレースケールなのでチャネルは1つだけである（上の可視化では見やすさのために疑似カラーのパレットを使っている）。 

最後に、データイテレータは効率的な性能を実現するうえで重要な要素である。たとえば、画像の解凍、動画のトランスコード、その他の前処理を効率よく行うためにGPUを使うこともある。可能な限り、高性能計算を活用するよく実装されたデータイテレータに頼るべきである。そうすることで、学習ループが遅くなるのを避けられる。


## 演習

1. `batch_size` を小さくする（たとえば1にする）と、読み込み性能に影響するか？
1. データイテレータの性能は重要である。現在の実装は十分に高速だと思うか？改善するためのさまざまな方法を調べよ。ボトルネックがどこにあるかを調べるために、システムプロファイラを使っよ。
1. フレームワークのオンラインAPIドキュメントを確認せよ。ほかにどのようなデータセットがあるか？
