# 物体検出とバウンディングボックス
:label:`sec_bbox`


前の節（たとえば :numref:`sec_alexnet`-- :numref:`sec_googlenet`）では、
画像分類のためのさまざまなモデルを紹介した。
画像分類タスクでは、
画像中に主要な物体がただ *1つ* だけ存在すると仮定し、そのカテゴリを認識することだけに注目する。
しかし、関心のある画像にはしばしば *複数* の物体が含まれる。
その場合、私たちはそれらのカテゴリだけでなく、画像内での具体的な位置も知りたいのである。
コンピュータビジョンでは、このようなタスクを *物体検出*（object detection、または *物体認識*）と呼ぶ。

物体検出は、
多くの分野で広く応用されている。
たとえば、自動運転では、
撮影した動画画像から車両、歩行者、道路、障害物の位置を検出することで、
走行経路を計画する必要がある。
また、
ロボットはこの技術を用いて、
環境内を移動する間に関心のある物体を検出し、その位置を特定することがある。
さらに、
セキュリティシステムでは、
侵入者や爆弾のような異常な物体を検出する必要があるかもしれない。

次のいくつかの節では、
物体検出のためのいくつかの深層学習手法を紹介する。
まずは、物体の *位置*（または *場所*）の紹介から始める。

```{.python .input}
#@tab mxnet
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import image, npx, np

npx.set_np()
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
import torch
```

```{.python .input}
#@tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
import tensorflow as tf
```

この節で使用するサンプル画像を読み込みる。画像の左側に犬、右側に猫がいることがわかる。
これらがこの画像における2つの主要な物体である。

```{.python .input}
#@tab mxnet
d2l.set_figsize()
img = image.imread('../img/catdog.jpg').asnumpy()
d2l.plt.imshow(img);
```

```{.python .input}
#@tab pytorch, tensorflow
d2l.set_figsize()
img = d2l.plt.imread('../img/catdog.jpg')
d2l.plt.imshow(img);
```

## バウンディングボックス


物体検出では、
通常、物体の空間的位置を表すために *バウンディングボックス* を用いる。
バウンディングボックスは長方形であり、長方形の左上隅の $x$ 座標と $y$ 座標、および右下隅の同様の座標によって決まりる。 
もう1つの一般的なバウンディングボックス表現は、バウンディングボックス中心の $(x, y)$ 座標と、ボックスの幅および高さである。

[**ここでは、これら**][**2つの表現**]の間を変換する関数を定義する。
`box_corner_to_center` は2隅表現から中心・幅・高さ表現へ変換し、
`box_center_to_corner` はその逆を行う。
入力引数 `boxes` は形状 ($n$, 4) の2次元テンソルである必要があり、ここで $n$ はバウンディングボックスの数である。

```{.python .input}
#@tab all
#@save
def box_corner_to_center(boxes):
    """Convert from (upper-left, lower-right) to (center, width, height)."""
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    w = x2 - x1
    h = y2 - y1
    boxes = d2l.stack((cx, cy, w, h), axis=-1)
    return boxes

#@save
def box_center_to_corner(boxes):
    """Convert from (center, width, height) to (upper-left, lower-right)."""
    cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    x1 = cx - 0.5 * w
    y1 = cy - 0.5 * h
    x2 = cx + 0.5 * w
    y2 = cy + 0.5 * h
    boxes = d2l.stack((x1, y1, x2, y2), axis=-1)
    return boxes
```

座標情報に基づいて、
画像中の犬と猫のバウンディングボックスを[**定義する**]。
画像における座標の原点は画像の左上隅であり、右方向と下方向がそれぞれ $x$ 軸と $y$ 軸の正の方向である。

```{.python .input}
#@tab all
# Here `bbox` is the abbreviation for bounding box
dog_bbox, cat_bbox = [60.0, 45.0, 378.0, 516.0], [400.0, 112.0, 655.0, 493.0]
```

2回変換することで、2つのバウンディングボックス変換関数の正しさを確認できる。

```{.python .input}
#@tab all
boxes = d2l.tensor((dog_bbox, cat_bbox))
box_center_to_corner(box_corner_to_center(boxes)) == boxes
```

正確に描けているか確認するために、画像に[**バウンディングボックスを描画してみましょう**]。
描画する前に、補助関数 `bbox_to_rect` を定義する。これはバウンディングボックスを `matplotlib` パッケージのバウンディングボックス形式で表す。

```{.python .input}
#@tab all
#@save
def bbox_to_rect(bbox, color):
    """Convert bounding box to matplotlib format."""
    # Convert the bounding box (upper-left x, upper-left y, lower-right x,
    # lower-right y) format to the matplotlib format: ((upper-left x,
    # upper-left y), width, height)
    return d2l.plt.Rectangle(
        xy=(bbox[0], bbox[1]), width=bbox[2]-bbox[0], height=bbox[3]-bbox[1],
        fill=False, edgecolor=color, linewidth=2)
```

画像にバウンディングボックスを追加すると、
2つの物体の主な輪郭が、基本的に2つのボックスの内側に収まっていることがわかる。

```{.python .input}
#@tab all
fig = d2l.plt.imshow(img)
fig.axes.add_patch(bbox_to_rect(dog_bbox, 'blue'))
fig.axes.add_patch(bbox_to_rect(cat_bbox, 'red'));
```

## まとめ

* 物体検出では、画像中の関心のあるすべての物体を認識するだけでなく、それらの位置も認識する。位置は一般に長方形のバウンディングボックスで表される。
* よく使われる2種類のバウンディングボックス表現の間で変換できる。

## 演習

1. 別の画像を見つけて、物体を含むバウンディングボックスをラベル付けしてみなさい。バウンディングボックスのラベル付けとカテゴリのラベル付けを比較すると、通常どちらに時間がかかりますか？
1. `box_corner_to_center` と `box_center_to_corner` の入力引数 `boxes` の最内次元が常に 4 であるのはなぜですか？
