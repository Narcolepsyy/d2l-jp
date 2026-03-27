# アンカーボックス
:label:`sec_anchor`


物体検出アルゴリズムは通常、
入力画像内の多数の領域をサンプリングし、それらの領域に注目対象の物体が含まれているかを判定し、さらに領域の境界を調整して、物体の
*真のバウンディングボックス*
をより正確に予測します。
モデルによって、領域のサンプリング方法は
さまざまです。 
ここではそのような方法の一つを紹介します。
これは、各ピクセルを中心として、異なるスケールとアスペクト比をもつ複数のバウンディングボックスを生成します。 
これらのバウンディングボックスを *アンカーボックス* と呼びます。
:numref:`sec_ssd` では、アンカーボックスに基づく物体検出モデルを設計します。

まず、出力をより簡潔にするために、
表示精度を少し変更しましょう。

```{.python .input}
#@tab mxnet
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import gluon, image, np, npx

np.set_printoptions(2)  # 表示精度を簡略化
npx.set_np()
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
import torch

torch.set_printoptions(2)  # 表示精度を簡略化
```

## 複数のアンカーボックスの生成

入力画像の高さが $h$、幅が $w$ であるとします。 
画像の各ピクセルを中心として、異なる形状のアンカーボックスを生成します。
*スケール* を $s\in (0, 1]$、
*アスペクト比*（幅と高さの比）を $r > 0$ とします。 
すると、[**アンカーボックスの幅と高さはそれぞれ $ws\sqrt{r}$ と $hs/\sqrt{r}$ になります。**]
中心位置が与えられれば、幅と高さが既知のアンカーボックスは一意に定まることに注意してください。

異なる形状の複数のアンカーボックスを生成するために、
一連のスケール
$s_1,\ldots, s_n$ と 
一連のアスペクト比 $r_1,\ldots, r_m$ を設定します。
各ピクセルを中心として、これらのスケールとアスペクト比の全組合せを用いると、入力画像には合計 $whnm$ 個のアンカーボックスが生成されます。これらのアンカーボックスはすべての
真のバウンディングボックスを覆えるかもしれませんが、計算量が容易に高くなりすぎます。
実際には、
[**$s_1$ または $r_1$ を含む組合せだけを考慮する**] ことができます。

[**$$(s_1, r_1), (s_1, r_2), \ldots, (s_1, r_m), (s_2, r_1), (s_3, r_1), \ldots, (s_n, r_1).$$**]

つまり、同じピクセルを中心とするアンカーボックスの数は $n+m-1$ です。入力画像全体では、合計 $wh(n+m-1)$ 個のアンカーボックスを生成します。

上記のアンカーボックス生成方法は、次の `multibox_prior` 関数で実装されています。入力画像、スケールのリスト、アスペクト比のリストを指定すると、この関数はすべてのアンカーボックスを返します。

```{.python .input}
#@tab mxnet
#@save
def multibox_prior(data, sizes, ratios):
    """Generate anchor boxes with different shapes centered on each pixel."""
    in_height, in_width = data.shape[-2:]
    device, num_sizes, num_ratios = data.ctx, len(sizes), len(ratios)
    boxes_per_pixel = (num_sizes + num_ratios - 1)
    size_tensor = d2l.tensor(sizes, ctx=device)
    ratio_tensor = d2l.tensor(ratios, ctx=device)
    # Offsets are required to move the anchor to the center of a pixel. Since
    # a pixel has height=1 and width=1, we choose to offset our centers by 0.5
    offset_h, offset_w = 0.5, 0.5
    steps_h = 1.0 / in_height  # Scaled steps in y-axis
    steps_w = 1.0 / in_width  # Scaled steps in x-axis

    # Generate all center points for the anchor boxes
    center_h = (d2l.arange(in_height, ctx=device) + offset_h) * steps_h
    center_w = (d2l.arange(in_width, ctx=device) + offset_w) * steps_w
    shift_x, shift_y = d2l.meshgrid(center_w, center_h)
    shift_x, shift_y = shift_x.reshape(-1), shift_y.reshape(-1)

    # Generate `boxes_per_pixel` number of heights and widths that are later
    # used to create anchor box corner coordinates (xmin, xmax, ymin, ymax)
    w = np.concatenate((size_tensor * np.sqrt(ratio_tensor[0]),
                        sizes[0] * np.sqrt(ratio_tensor[1:]))) \
                        * in_height / in_width  # Handle rectangular inputs
    h = np.concatenate((size_tensor / np.sqrt(ratio_tensor[0]),
                        sizes[0] / np.sqrt(ratio_tensor[1:])))
    # Divide by 2 to get half height and half width
    anchor_manipulations = np.tile(np.stack((-w, -h, w, h)).T,
                                   (in_height * in_width, 1)) / 2

    # Each center point will have `boxes_per_pixel` number of anchor boxes, so
    # generate a grid of all anchor box centers with `boxes_per_pixel` repeats
    out_grid = d2l.stack([shift_x, shift_y, shift_x, shift_y],
                         axis=1).repeat(boxes_per_pixel, axis=0)
    output = out_grid + anchor_manipulations
    return np.expand_dims(output, axis=0)
```

```{.python .input}
#@tab pytorch
#@save
def multibox_prior(data, sizes, ratios):
    """Generate anchor boxes with different shapes centered on each pixel."""
    in_height, in_width = data.shape[-2:]
    device, num_sizes, num_ratios = data.device, len(sizes), len(ratios)
    boxes_per_pixel = (num_sizes + num_ratios - 1)
    size_tensor = d2l.tensor(sizes, device=device)
    ratio_tensor = d2l.tensor(ratios, device=device)
    # Offsets are required to move the anchor to the center of a pixel. Since
    # a pixel has height=1 and width=1, we choose to offset our centers by 0.5
    offset_h, offset_w = 0.5, 0.5
    steps_h = 1.0 / in_height  # Scaled steps in y axis
    steps_w = 1.0 / in_width  # Scaled steps in x axis

    # Generate all center points for the anchor boxes
    center_h = (torch.arange(in_height, device=device) + offset_h) * steps_h
    center_w = (torch.arange(in_width, device=device) + offset_w) * steps_w
    shift_y, shift_x = torch.meshgrid(center_h, center_w, indexing='ij')
    shift_y, shift_x = shift_y.reshape(-1), shift_x.reshape(-1)

    # Generate `boxes_per_pixel` number of heights and widths that are later
    # used to create anchor box corner coordinates (xmin, xmax, ymin, ymax)
    w = torch.cat((size_tensor * torch.sqrt(ratio_tensor[0]),
                   sizes[0] * torch.sqrt(ratio_tensor[1:])))\
                   * in_height / in_width  # Handle rectangular inputs
    h = torch.cat((size_tensor / torch.sqrt(ratio_tensor[0]),
                   sizes[0] / torch.sqrt(ratio_tensor[1:])))
    # Divide by 2 to get half height and half width
    anchor_manipulations = torch.stack((-w, -h, w, h)).T.repeat(
                                        in_height * in_width, 1) / 2

    # Each center point will have `boxes_per_pixel` number of anchor boxes, so
    # generate a grid of all anchor box centers with `boxes_per_pixel` repeats
    out_grid = torch.stack([shift_x, shift_y, shift_x, shift_y],
                dim=1).repeat_interleave(boxes_per_pixel, dim=0)
    output = out_grid + anchor_manipulations
    return output.unsqueeze(0)
```

返されるアンカーボックス変数 `Y` の[**形状**]は、
(batch size, number of anchor boxes, 4) です。

```{.python .input}
#@tab mxnet
img = image.imread('../img/catdog.jpg').asnumpy()
h, w = img.shape[:2]

print(h, w)
X = np.random.uniform(size=(1, 3, h, w))  # Construct input data
Y = multibox_prior(X, sizes=[0.75, 0.5, 0.25], ratios=[1, 2, 0.5])
Y.shape
```

```{.python .input}
#@tab pytorch
img = d2l.plt.imread('../img/catdog.jpg')
h, w = img.shape[:2]

print(h, w)
X = torch.rand(size=(1, 3, h, w))  # Construct input data
Y = multibox_prior(X, sizes=[0.75, 0.5, 0.25], ratios=[1, 2, 0.5])
Y.shape
```

アンカーボックス変数 `Y` の形状を (image height, image width, number of anchor boxes centered on the same pixel, 4) に変更すると、
指定したピクセル位置を中心とするすべてのアンカーボックスを得ることができます。
以下では、
(250, 250) を中心とする最初のアンカーボックスを[**参照します**]。これは4つの要素を持ちます。すなわち、アンカーボックスの左上隅の $(x, y)$ 軸座標と右下隅の $(x, y)$ 軸座標です。
両軸の座標値は、
それぞれ画像の幅と高さで割られています。

```{.python .input}
#@tab all
boxes = Y.reshape(h, w, 5, 4)
boxes[250, 250, 0, :]
```

画像内の1つのピクセルを中心とするすべてのアンカーボックスを[**表示する**]ために、
複数のバウンディングボックスを画像上に描画する次の `show_bboxes` 関数を定義します。

```{.python .input}
#@tab all
#@save
def show_bboxes(axes, bboxes, labels=None, colors=None):
    """Show bounding boxes."""

    def make_list(obj, default_values=None):
        if obj is None:
            obj = default_values
        elif not isinstance(obj, (list, tuple)):
            obj = [obj]
        return obj

    labels = make_list(labels)
    colors = make_list(colors, ['b', 'g', 'r', 'm', 'c'])
    for i, bbox in enumerate(bboxes):
        color = colors[i % len(colors)]
        rect = d2l.bbox_to_rect(d2l.numpy(bbox), color)
        axes.add_patch(rect)
        if labels and len(labels) > i:
            text_color = 'k' if color == 'w' else 'w'
            axes.text(rect.xy[0], rect.xy[1], labels[i],
                      va='center', ha='center', fontsize=9, color=text_color,
                      bbox=dict(facecolor=color, lw=0))
```

先ほど見たように、変数 `boxes` における $x$ 軸と $y$ 軸の座標値は、それぞれ画像の幅と高さで割られています。
アンカーボックスを描画する際には、
元の座標値に戻す必要があります。
そのため、以下で変数 `bbox_scale` を定義します。 
これで、画像内の (250, 250) を中心とするすべてのアンカーボックスを描画できます。
ご覧のとおり、スケール0.75、アスペクト比1の青いアンカーボックスは、画像中の犬をうまく
囲んでいます。

```{.python .input}
#@tab all
d2l.set_figsize()
bbox_scale = d2l.tensor((w, h, w, h))
fig = d2l.plt.imshow(img)
show_bboxes(fig.axes, boxes[250, 250, :, :] * bbox_scale,
            ['s=0.75, r=1', 's=0.5, r=1', 's=0.25, r=1', 's=0.75, r=2',
             's=0.75, r=0.5'])
```

## [**Intersection over Union (IoU)**]

先ほど、アンカーボックスが画像中の犬を「うまく」囲んでいると述べました。
物体の真のバウンディングボックスが分かっている場合、この「うまく」はどのように定量化できるでしょうか。
直感的には、
アンカーボックスと真のバウンディングボックスの類似度を測れます。
*Jaccard index* は2つの集合の類似度を測ることができます。集合 $\mathcal{A}$ と $\mathcal{B}$ に対して、その Jaccard index は、共通部分の大きさを和集合の大きさで割ったものです。

$$J(\mathcal{A},\mathcal{B}) = \frac{\left|\mathcal{A} \cap \mathcal{B}\right|}{\left| \mathcal{A} \cup \mathcal{B}\right|}.$$


実際には、任意のバウンディングボックスのピクセル領域をピクセルの集合として考えることができます。 
このようにして、2つのバウンディングボックスの類似度を、それらのピクセル集合の Jaccard index によって測定できます。2つのバウンディングボックスについて、通常はその Jaccard index を *intersection over union*（*IoU*）と呼びます。これは、 :numref:`fig_iou` に示すように、共通部分の面積を和集合の面積で割った比です。
IoU の範囲は 0 から 1 です。
0 は2つのバウンディングボックスがまったく重なっていないことを意味し、
1 は2つのバウンディングボックスが等しいことを示します。

![IoU is the ratio of the intersection area to the union area of two bounding boxes.](../img/iou.svg)
:label:`fig_iou`

この節の残りでは、IoU を用いてアンカーボックスと真のバウンディングボックスの類似度、および異なるアンカーボックス同士の類似度を測ります。
2つのアンカーまたはバウンディングボックスのリストが与えられたとき、
次の `box_iou` は、これら2つのリスト間の全組合せについて IoU を計算します。

```{.python .input}
#@tab mxnet
#@save
def box_iou(boxes1, boxes2):
    """Compute pairwise IoU across two lists of anchor or bounding boxes."""
    box_area = lambda boxes: ((boxes[:, 2] - boxes[:, 0]) *
                              (boxes[:, 3] - boxes[:, 1]))
    # Shape of `boxes1`, `boxes2`, `areas1`, `areas2`: (no. of boxes1, 4),
    # (no. of boxes2, 4), (no. of boxes1,), (no. of boxes2,)
    areas1 = box_area(boxes1)
    areas2 = box_area(boxes2)
    # Shape of `inter_upperlefts`, `inter_lowerrights`, `inters`: (no. of
    # boxes1, no. of boxes2, 2)
    inter_upperlefts = np.maximum(boxes1[:, None, :2], boxes2[:, :2])
    inter_lowerrights = np.minimum(boxes1[:, None, 2:], boxes2[:, 2:])
    inters = (inter_lowerrights - inter_upperlefts).clip(min=0)
    # Shape of `inter_areas` and `union_areas`: (no. of boxes1, no. of boxes2)
    inter_areas = inters[:, :, 0] * inters[:, :, 1]
    union_areas = areas1[:, None] + areas2 - inter_areas
    return inter_areas / union_areas
```

```{.python .input}
#@tab pytorch
#@save
def box_iou(boxes1, boxes2):
    """Compute pairwise IoU across two lists of anchor or bounding boxes."""
    box_area = lambda boxes: ((boxes[:, 2] - boxes[:, 0]) *
                              (boxes[:, 3] - boxes[:, 1]))
    # Shape of `boxes1`, `boxes2`, `areas1`, `areas2`: (no. of boxes1, 4),
    # (no. of boxes2, 4), (no. of boxes1,), (no. of boxes2,)
    areas1 = box_area(boxes1)
    areas2 = box_area(boxes2)
    # Shape of `inter_upperlefts`, `inter_lowerrights`, `inters`: (no. of
    # boxes1, no. of boxes2, 2)
    inter_upperlefts = torch.max(boxes1[:, None, :2], boxes2[:, :2])
    inter_lowerrights = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])
    inters = (inter_lowerrights - inter_upperlefts).clamp(min=0)
    # Shape of `inter_areas` and `union_areas`: (no. of boxes1, no. of boxes2)
    inter_areas = inters[:, :, 0] * inters[:, :, 1]
    union_areas = areas1[:, None] + areas2 - inter_areas
    return inter_areas / union_areas
```

## 学習データにおけるアンカーボックスのラベル付け
:label:`subsec_labeling-anchor-boxes`


学習データセットでは、
各アンカーボックスを1つの学習例とみなします。
物体検出モデルを学習するためには、
各アンカーボックスに対して *クラス* ラベルと *オフセット* ラベルが必要です。
前者はアンカーボックスに対応する物体のクラスであり、
後者はアンカーボックスに対する真のバウンディングボックスのオフセットです。
予測時には、
各画像について
複数のアンカーボックスを生成し、
すべてのアンカーボックスについてクラスとオフセットを予測し、
予測されたオフセットに従って位置を調整して予測バウンディングボックスを得て、
最後に一定の条件を満たす
予測バウンディングボックスだけを出力します。


ご存じのように、物体検出の学習データセットには、
*真のバウンディングボックス* の位置と、それらに囲まれた物体のクラスのラベルが付いています。
生成された任意の *アンカーボックス* にラベルを付けるには、
そのアンカーボックスに最も近い *割り当てられた* 真のバウンディングボックスのラベル付き位置とクラスを参照します。
以下では、
アンカーボックスに最も近い真のバウンディングボックスを割り当てるアルゴリズムを説明します。 

### [**真のバウンディングボックスをアンカーボックスに割り当てる**]

画像が与えられたとき、
アンカーボックスを $A_1, A_2, \ldots, A_{n_a}$、真のバウンディングボックスを $B_1, B_2, \ldots, B_{n_b}$ とし、ここで $n_a \geq n_b$ とします。
行列 $\mathbf{X} \in \mathbb{R}^{n_a \times n_b}$ を定義し、その $i^\textrm{th}$ 行 $j^\textrm{th}$ 列の要素 $x_{ij}$ をアンカーボックス $A_i$ と真のバウンディングボックス $B_j$ の IoU とします。アルゴリズムは次の手順からなります。

1. 行列 $\mathbf{X}$ の最大要素を見つけ、その行と列のインデックスをそれぞれ $i_1$ と $j_1$ とします。すると、真のバウンディングボックス $B_{j_1}$ はアンカーボックス $A_{i_1}$ に割り当てられます。これは、すべてのアンカーボックスと真のバウンディングボックスの組の中で $A_{i_1}$ と $B_{j_1}$ が最も近いからであり、直感的です。最初の割り当ての後、行列 $\mathbf{X}$ の ${i_1}^\textrm{th}$ 行と ${j_1}^\textrm{th}$ 列のすべての要素を破棄します。 
1. 行列 $\mathbf{X}$ に残っている要素の最大値を見つけ、その行と列のインデックスをそれぞれ $i_2$ と $j_2$ とします。真のバウンディングボックス $B_{j_2}$ をアンカーボックス $A_{i_2}$ に割り当て、行列 $\mathbf{X}$ の ${i_2}^\textrm{th}$ 行と ${j_2}^\textrm{th}$ 列のすべての要素を破棄します。
1. この時点で、行列 $\mathbf{X}$ の2つの行と2つの列の要素が破棄されています。行列 $\mathbf{X}$ の $n_b$ 列すべての要素が破棄されるまで続けます。この時点で、$n_b$ 個のアンカーボックスそれぞれに真のバウンディングボックスが割り当てられています。
1. 残りの $n_a - n_b$ 個のアンカーボックスだけを走査します。たとえば、任意のアンカーボックス $A_i$ について、行列 $\mathbf{X}$ の $i^\textrm{th}$ 行全体で $A_i$ と最も IoU が大きい真のバウンディングボックス $B_j$ を見つけ、この IoU があらかじめ定めたしきい値より大きい場合にのみ $B_j$ を $A_i$ に割り当てます。

上記のアルゴリズムを具体例で説明しましょう。
:numref:`fig_anchor_label`（左）に示すように、行列 $\mathbf{X}$ の最大値が $x_{23}$ だとすると、真のバウンディングボックス $B_3$ をアンカーボックス $A_2$ に割り当てます。
次に、行2と列3のすべての要素を破棄し、残りの要素（網掛け部分）の中で最大の $x_{71}$ を見つけ、真のバウンディングボックス $B_1$ をアンカーボックス $A_7$ に割り当てます。 
次に、 :numref:`fig_anchor_label`（中央）に示すように、行7と列1のすべての要素を破棄し、残りの要素（網掛け部分）の中で最大の $x_{54}$ を見つけ、真のバウンディングボックス $B_4$ をアンカーボックス $A_5$ に割り当てます。 
最後に、 :numref:`fig_anchor_label`（右）に示すように、行5と列4のすべての要素を破棄し、残りの要素（網掛け部分）の中で最大の $x_{92}$ を見つけ、真のバウンディングボックス $B_2$ をアンカーボックス $A_9$ に割り当てます。
その後は、残りのアンカーボックス $A_1, A_3, A_4, A_6, A_8$ を走査し、しきい値に従って真のバウンディングボックスを割り当てるかどうかを決めるだけです。

![Assigning ground-truth bounding boxes to anchor boxes.](../img/anchor-label.svg)
:label:`fig_anchor_label`

このアルゴリズムは、次の `assign_anchor_to_bbox` 関数で実装されています。

```{.python .input}
#@tab mxnet
#@save
def assign_anchor_to_bbox(ground_truth, anchors, device, iou_threshold=0.5):
    """Assign closest ground-truth bounding boxes to anchor boxes."""
    num_anchors, num_gt_boxes = anchors.shape[0], ground_truth.shape[0]
    # Element x_ij in the i-th row and j-th column is the IoU of the anchor
    # box i and the ground-truth bounding box j
    jaccard = box_iou(anchors, ground_truth)
    # Initialize the tensor to hold the assigned ground-truth bounding box for
    # each anchor
    anchors_bbox_map = np.full((num_anchors,), -1, dtype=np.int32, ctx=device)
    # Assign ground-truth bounding boxes according to the threshold
    max_ious, indices = np.max(jaccard, axis=1), np.argmax(jaccard, axis=1)
    anc_i = np.nonzero(max_ious >= iou_threshold)[0]
    box_j = indices[max_ious >= iou_threshold]
    anchors_bbox_map[anc_i] = box_j
    col_discard = np.full((num_anchors,), -1)
    row_discard = np.full((num_gt_boxes,), -1)
    for _ in range(num_gt_boxes):
        max_idx = np.argmax(jaccard)  # Find the largest IoU
        box_idx = (max_idx % num_gt_boxes).astype('int32')
        anc_idx = (max_idx / num_gt_boxes).astype('int32')
        anchors_bbox_map[anc_idx] = box_idx
        jaccard[:, box_idx] = col_discard
        jaccard[anc_idx, :] = row_discard
    return anchors_bbox_map
```

```{.python .input}
#@tab pytorch
#@save
def assign_anchor_to_bbox(ground_truth, anchors, device, iou_threshold=0.5):
    """Assign closest ground-truth bounding boxes to anchor boxes."""
    num_anchors, num_gt_boxes = anchors.shape[0], ground_truth.shape[0]
    # Element x_ij in the i-th row and j-th column is the IoU of the anchor
    # box i and the ground-truth bounding box j
    jaccard = box_iou(anchors, ground_truth)
    # Initialize the tensor to hold the assigned ground-truth bounding box for
    # each anchor
    anchors_bbox_map = torch.full((num_anchors,), -1, dtype=torch.long,
                                  device=device)
    # Assign ground-truth bounding boxes according to the threshold
    max_ious, indices = torch.max(jaccard, dim=1)
    anc_i = torch.nonzero(max_ious >= iou_threshold).reshape(-1)
    box_j = indices[max_ious >= iou_threshold]
    anchors_bbox_map[anc_i] = box_j
    col_discard = torch.full((num_anchors,), -1)
    row_discard = torch.full((num_gt_boxes,), -1)
    for _ in range(num_gt_boxes):
        max_idx = torch.argmax(jaccard)  # Find the largest IoU
        box_idx = (max_idx % num_gt_boxes).long()
        anc_idx = (max_idx / num_gt_boxes).long()
        anchors_bbox_map[anc_idx] = box_idx
        jaccard[:, box_idx] = col_discard
        jaccard[anc_idx, :] = row_discard
    return anchors_bbox_map
```

### クラスとオフセットのラベル付け

これで、各アンカーボックスのクラスとオフセットにラベルを付けることができます。アンカーボックス $A$ に
真のバウンディングボックス $B$ が割り当てられたとします。 
一方で、
アンカーボックス $A$ のクラスは
$B$ のクラスとしてラベル付けされます。
他方で、
アンカーボックス $A$ のオフセットは、
$B$ と $A$ の中心座標の相対位置と、
これら2つのボックスの相対サイズに基づいてラベル付けされます。
データセット内のさまざまなボックスの位置とサイズが異なるため、
これらの相対位置とサイズに変換を施すことで、
より一様に分布したオフセットを得られ、
学習しやすくなります。
ここでは一般的な変換を説明します。
[**$A$ と $B$ の中心座標をそれぞれ $(x_a, y_a)$ と $(x_b, y_b)$、
幅を $w_a$ と $w_b$、
高さを $h_a$ と $h_b$ とします。 
このとき、$A$ のオフセットは次のようにラベル付けできます。

$$\left( \frac{ \frac{x_b - x_a}{w_a} - \mu_x }{\sigma_x},
\frac{ \frac{y_b - y_a}{h_a} - \mu_y }{\sigma_y},
\frac{ \log \frac{w_b}{w_a} - \mu_w }{\sigma_w},
\frac{ \log \frac{h_b}{h_a} - \mu_h }{\sigma_h}\right),$$
**]
ここで定数のデフォルト値は $\mu_x = \mu_y = \mu_w = \mu_h = 0, \sigma_x=\sigma_y=0.1$, および $\sigma_w=\sigma_h=0.2$ です。
この変換は以下の `offset_boxes` 関数で実装されています。

```{.python .input}
#@tab all
#@save
def offset_boxes(anchors, assigned_bb, eps=1e-6):
    """Transform for anchor box offsets."""
    c_anc = d2l.box_corner_to_center(anchors)
    c_assigned_bb = d2l.box_corner_to_center(assigned_bb)
    offset_xy = 10 * (c_assigned_bb[:, :2] - c_anc[:, :2]) / c_anc[:, 2:]
    offset_wh = 5 * d2l.log(eps + c_assigned_bb[:, 2:] / c_anc[:, 2:])
    offset = d2l.concat([offset_xy, offset_wh], axis=1)
    return offset
```

アンカーボックスに真のバウンディングボックスが割り当てられていない場合、そのアンカーボックスのクラスは単に「背景」とラベル付けします。
クラスが背景であるアンカーボックスはしばしば *負例* アンカーボックスと呼ばれ、
それ以外は *正例* アンカーボックスと呼ばれます。
以下の `multibox_target` 関数を実装して、
真のバウンディングボックス（`labels` 引数）を用いて、アンカーボックス（`anchors` 引数）の[**クラスとオフセットにラベルを付けます**]。
この関数では背景クラスを0とし、新しいクラスの整数インデックスは1から始めます。

```{.python .input}
#@tab mxnet
#@save
def multibox_target(anchors, labels):
    """Label anchor boxes using ground-truth bounding boxes."""
    batch_size, anchors = labels.shape[0], anchors.squeeze(0)
    batch_offset, batch_mask, batch_class_labels = [], [], []
    device, num_anchors = anchors.ctx, anchors.shape[0]
    for i in range(batch_size):
        label = labels[i, :, :]
        anchors_bbox_map = assign_anchor_to_bbox(
            label[:, 1:], anchors, device)
        bbox_mask = np.tile((np.expand_dims((anchors_bbox_map >= 0),
                                            axis=-1)), (1, 4)).astype('int32')
        # Initialize class labels and assigned bounding box coordinates with
        # zeros
        class_labels = d2l.zeros(num_anchors, dtype=np.int32, ctx=device)
        assigned_bb = d2l.zeros((num_anchors, 4), dtype=np.float32,
                                ctx=device)
        # Label classes of anchor boxes using their assigned ground-truth
        # bounding boxes. If an anchor box is not assigned any, we label its
        # class as background (the value remains zero)
        indices_true = np.nonzero(anchors_bbox_map >= 0)[0]
        bb_idx = anchors_bbox_map[indices_true]
        class_labels[indices_true] = label[bb_idx, 0].astype('int32') + 1
        assigned_bb[indices_true] = label[bb_idx, 1:]
        # Offset transformation
        offset = offset_boxes(anchors, assigned_bb) * bbox_mask
        batch_offset.append(offset.reshape(-1))
        batch_mask.append(bbox_mask.reshape(-1))
        batch_class_labels.append(class_labels)
    bbox_offset = d2l.stack(batch_offset)
    bbox_mask = d2l.stack(batch_mask)
    class_labels = d2l.stack(batch_class_labels)
    return (bbox_offset, bbox_mask, class_labels)
```

```{.python .input}
#@tab pytorch
#@save
def multibox_target(anchors, labels):
    """Label anchor boxes using ground-truth bounding boxes."""
    batch_size, anchors = labels.shape[0], anchors.squeeze(0)
    batch_offset, batch_mask, batch_class_labels = [], [], []
    device, num_anchors = anchors.device, anchors.shape[0]
    for i in range(batch_size):
        label = labels[i, :, :]
        anchors_bbox_map = assign_anchor_to_bbox(
            label[:, 1:], anchors, device)
        bbox_mask = ((anchors_bbox_map >= 0).float().unsqueeze(-1)).repeat(
            1, 4)
        # Initialize class labels and assigned bounding box coordinates with
        # zeros
        class_labels = torch.zeros(num_anchors, dtype=torch.long,
                                   device=device)
        assigned_bb = torch.zeros((num_anchors, 4), dtype=torch.float32,
                                  device=device)
        # Label classes of anchor boxes using their assigned ground-truth
        # bounding boxes. If an anchor box is not assigned any, we label its
        # class as background (the value remains zero)
        indices_true = torch.nonzero(anchors_bbox_map >= 0)
        bb_idx = anchors_bbox_map[indices_true]
        class_labels[indices_true] = label[bb_idx, 0].long() + 1
        assigned_bb[indices_true] = label[bb_idx, 1:]
        # Offset transformation
        offset = offset_boxes(anchors, assigned_bb) * bbox_mask
        batch_offset.append(offset.reshape(-1))
        batch_mask.append(bbox_mask.reshape(-1))
        batch_class_labels.append(class_labels)
    bbox_offset = torch.stack(batch_offset)
    bbox_mask = torch.stack(batch_mask)
    class_labels = torch.stack(batch_class_labels)
    return (bbox_offset, bbox_mask, class_labels)
```

### 例

具体例を通してアンカーボックスのラベル付けを説明しましょう。
読み込んだ画像中の犬と猫に対する真のバウンディングボックスを定義します。ここで最初の要素はクラス（犬は0、猫は1）で、残りの4要素は左上隅と右下隅の $(x, y)$ 軸座標（範囲は0から1）です。 
また、左上隅と右下隅の座標を用いてラベル付けする5つのアンカーボックス
$A_0, \ldots, A_4$（インデックスは0から始まる）を構成します。
そして、これらの真のバウンディングボックスと
アンカーボックスを画像上に[**描画します。**]

```{.python .input}
#@tab all
ground_truth = d2l.tensor([[0, 0.1, 0.08, 0.52, 0.92],
                         [1, 0.55, 0.2, 0.9, 0.88]])
anchors = d2l.tensor([[0, 0.1, 0.2, 0.3], [0.15, 0.2, 0.4, 0.4],
                    [0.63, 0.05, 0.88, 0.98], [0.66, 0.45, 0.8, 0.8],
                    [0.57, 0.3, 0.92, 0.9]])

fig = d2l.plt.imshow(img)
show_bboxes(fig.axes, ground_truth[:, 1:] * bbox_scale, ['dog', 'cat'], 'k')
show_bboxes(fig.axes, anchors * bbox_scale, ['0', '1', '2', '3', '4']);
```

上で定義した `multibox_target` 関数を用いると、
犬と猫の真のバウンディングボックスに基づいて、これらのアンカーボックスの[**クラスとオフセットにラベル付け**]できます。
この例では、背景、犬、猫のクラスのインデックスはそれぞれ0、1、2です。 
以下では、アンカーボックスと真のバウンディングボックスの例のために次元を追加します。

```{.python .input}
#@tab mxnet
labels = multibox_target(np.expand_dims(anchors, axis=0),
                         np.expand_dims(ground_truth, axis=0))
```

```{.python .input}
#@tab pytorch
labels = multibox_target(anchors.unsqueeze(dim=0),
                         ground_truth.unsqueeze(dim=0))
```

返される結果は3つの項目からなり、いずれもテンソル形式です。
3番目の項目には、入力アンカーボックスのラベル付きクラスが含まれます。

以下では、画像内のアンカーボックスと真のバウンディングボックスの位置に基づいて、返されたクラスラベルを解析しましょう。
まず、すべてのアンカーボックスと真のバウンディングボックスの組の中で、
アンカーボックス $A_4$ と猫の真のバウンディングボックスの IoU が最大です。 
したがって、$A_4$ のクラスは猫としてラベル付けされます。
$A_4$ または猫の真のバウンディングボックスを含む組を除くと、残りの中では
アンカーボックス $A_1$ と犬の真のバウンディングボックスの組の IoU が最大です。
したがって、$A_1$ のクラスは犬としてラベル付けされます。
次に、残りの3つの未ラベルのアンカーボックス $A_0$, $A_2$, $A_3$ を走査する必要があります。
$A_0$ については、
IoU が最大となる真のバウンディングボックスのクラスは犬ですが、IoU はあらかじめ定めたしきい値（0.5）を下回るため、クラスは背景としてラベル付けされます。
$A_2$ については、
IoU が最大となる真のバウンディングボックスのクラスは猫で、IoU はしきい値を超えるため、クラスは猫としてラベル付けされます。
$A_3$ については、
IoU が最大となる真のバウンディングボックスのクラスは猫ですが、その値はしきい値を下回るため、クラスは背景としてラベル付けされます。

```{.python .input}
#@tab all
labels[2]
```

2番目に返される項目は、形状 (batch size, four times the number of anchor boxes) のマスク変数です。
マスク変数の4要素ごとに、
各アンカーボックスの4つのオフセット値に対応します。
背景検出は考慮しないので、
この負例クラスのオフセットは目的関数に影響すべきではありません。
要素ごとの乗算を通じて、マスク変数の0は、目的関数を計算する前に負例クラスのオフセットを除外します。

```{.python .input}
#@tab all
labels[1]
```

1番目に返される項目には、各アンカーボックスに対してラベル付けされた4つのオフセット値が含まれます。
負例クラスのアンカーボックスのオフセットは0としてラベル付けされることに注意してください。

```{.python .input}
#@tab all
labels[0]
```

## 非最大抑制によるバウンディングボックスの予測
:label:`subsec_predicting-bounding-boxes-nms`

予測時には、
画像に対して複数のアンカーボックスを生成し、それぞれについてクラスとオフセットを予測します。
したがって、
アンカーボックスとその予測オフセットに基づいて
*予測バウンディングボックス* が得られます。
以下では、アンカーボックスと
オフセット予測を入力として受け取り、[**逆オフセット変換を適用して
予測バウンディングボックスの座標を返す**] `offset_inverse` 関数を実装します。

```{.python .input}
#@tab all
#@save
def offset_inverse(anchors, offset_preds):
    """Predict bounding boxes based on anchor boxes with predicted offsets."""
    anc = d2l.box_corner_to_center(anchors)
    pred_bbox_xy = (offset_preds[:, :2] * anc[:, 2:] / 10) + anc[:, :2]
    pred_bbox_wh = d2l.exp(offset_preds[:, 2:] / 5) * anc[:, 2:]
    pred_bbox = d2l.concat((pred_bbox_xy, pred_bbox_wh), axis=1)
    predicted_bbox = d2l.box_center_to_corner(pred_bbox)
    return predicted_bbox
```

アンカーボックスが多数あると、
同じ物体を囲む非常に似た（大きく重なった）予測バウンディングボックスが
多数出力される可能性があります。
出力を簡潔にするために、
*非最大抑制*（NMS）を用いて、同じ物体に属する似た予測バウンディングボックスをまとめることができます。

非最大抑制の仕組みは次のとおりです。
予測バウンディングボックス $B$ について、
物体検出モデルは各クラスの予測確率を計算します。
最大の予測確率を $p$ とすると、
この確率に対応するクラスが $B$ の予測クラスです。
特に、この確率 $p$ を予測バウンディングボックス $B$ の *confidence*（スコア）と呼びます。
同じ画像上で、
背景以外のすべての予測バウンディングボックスを confidence の降順に並べてリスト $L$ を作ります。
その後、並べ替えたリスト $L$ を次の手順で操作します。

1. 最も confidence の高い予測バウンディングボックス $B_1$ を基準として選び、$B_1$ との IoU があらかじめ定めたしきい値 $\epsilon$ を超える、基準以外の予測バウンディングボックスをすべて $L$ から取り除きます。この時点で、$L$ は最も confidence の高い予測バウンディングボックスを保持し、それに似すぎている他のものを削除します。要するに、*最大でない* confidence スコアをもつものは *抑制* されます。
1. 2番目に confidence の高い予測バウンディングボックス $B_2$ を別の基準として選び、$B_2$ との IoU が $\epsilon$ を超える、基準以外の予測バウンディングボックスをすべて $L$ から取り除きます。
1. リスト $L$ のすべての予測バウンディングボックスが基準として使われるまで、上記の処理を繰り返します。この時点で、$L$ に含まれる任意の2つの予測バウンディングボックスの IoU はしきい値 $\epsilon$ 未満です。したがって、どの2つも互いに似すぎていません。 
1. リスト $L$ に含まれるすべての予測バウンディングボックスを出力します。

[**次の `nms` 関数は confidence スコアを降順に並べ、そのインデックスを返します。**]

```{.python .input}
#@tab mxnet
#@save
def nms(boxes, scores, iou_threshold):
    """Sort confidence scores of predicted bounding boxes."""
    B = scores.argsort()[::-1]
    keep = []  # Indices of predicted bounding boxes that will be kept
    while B.size > 0:
        i = B[0]
        keep.append(i)
        if B.size == 1: break
        iou = box_iou(boxes[i, :].reshape(-1, 4),
                      boxes[B[1:], :].reshape(-1, 4)).reshape(-1)
        inds = np.nonzero(iou <= iou_threshold)[0]
        B = B[inds + 1]
    return np.array(keep, dtype=np.int32, ctx=boxes.ctx)
```

```{.python .input}
#@tab pytorch
#@save
def nms(boxes, scores, iou_threshold):
    """Sort confidence scores of predicted bounding boxes."""
    B = torch.argsort(scores, dim=-1, descending=True)
    keep = []  # Indices of predicted bounding boxes that will be kept
    while B.numel() > 0:
        i = B[0]
        keep.append(i)
        if B.numel() == 1: break
        iou = box_iou(boxes[i, :].reshape(-1, 4),
                      boxes[B[1:], :].reshape(-1, 4)).reshape(-1)
        inds = torch.nonzero(iou <= iou_threshold).reshape(-1)
        B = B[inds + 1]
    return d2l.tensor(keep, device=boxes.device)
```

以下の `multibox_detection` を定義して、[**予測バウンディングボックスに非最大抑制を適用**]します。
実装が少し複雑に見えても心配はいりません。実装の直後に、具体例を用いて動作を示します。

```{.python .input}
#@tab mxnet
#@save
def multibox_detection(cls_probs, offset_preds, anchors, nms_threshold=0.5,
                       pos_threshold=0.009999999):
    """Predict bounding boxes using non-maximum suppression."""
    device, batch_size = cls_probs.ctx, cls_probs.shape[0]
    anchors = np.squeeze(anchors, axis=0)
    num_classes, num_anchors = cls_probs.shape[1], cls_probs.shape[2]
    out = []
    for i in range(batch_size):
        cls_prob, offset_pred = cls_probs[i], offset_preds[i].reshape(-1, 4)
        conf, class_id = np.max(cls_prob[1:], 0), np.argmax(cls_prob[1:], 0)
        predicted_bb = offset_inverse(anchors, offset_pred)
        keep = nms(predicted_bb, conf, nms_threshold)
        # Find all non-`keep` indices and set the class to background
        all_idx = np.arange(num_anchors, dtype=np.int32, ctx=device)
        combined = d2l.concat((keep, all_idx))
        unique, counts = np.unique(combined, return_counts=True)
        non_keep = unique[counts == 1]
        all_id_sorted = d2l.concat((keep, non_keep))
        class_id[non_keep] = -1
        class_id = class_id[all_id_sorted].astype('float32')
        conf, predicted_bb = conf[all_id_sorted], predicted_bb[all_id_sorted]
        # Here `pos_threshold` is a threshold for positive (non-background)
        # predictions
        below_min_idx = (conf < pos_threshold)
        class_id[below_min_idx] = -1
        conf[below_min_idx] = 1 - conf[below_min_idx]
        pred_info = d2l.concat((np.expand_dims(class_id, axis=1),
                                np.expand_dims(conf, axis=1),
                                predicted_bb), axis=1)
        out.append(pred_info)
    return d2l.stack(out)
```

```{.python .input}
#@tab pytorch
#@save
def multibox_detection(cls_probs, offset_preds, anchors, nms_threshold=0.5,
                       pos_threshold=0.009999999):
    """Predict bounding boxes using non-maximum suppression."""
    device, batch_size = cls_probs.device, cls_probs.shape[0]
    anchors = anchors.squeeze(0)
    num_classes, num_anchors = cls_probs.shape[1], cls_probs.shape[2]
    out = []
    for i in range(batch_size):
        cls_prob, offset_pred = cls_probs[i], offset_preds[i].reshape(-1, 4)
        conf, class_id = torch.max(cls_prob[1:], 0)
        predicted_bb = offset_inverse(anchors, offset_pred)
        keep = nms(predicted_bb, conf, nms_threshold)
        # Find all non-`keep` indices and set the class to background
        all_idx = torch.arange(num_anchors, dtype=torch.long, device=device)
        combined = torch.cat((keep, all_idx))
        uniques, counts = combined.unique(return_counts=True)
        non_keep = uniques[counts == 1]
        all_id_sorted = torch.cat((keep, non_keep))
        class_id[non_keep] = -1
        class_id = class_id[all_id_sorted]
        conf, predicted_bb = conf[all_id_sorted], predicted_bb[all_id_sorted]
        # Here `pos_threshold` is a threshold for positive (non-background)
        # predictions
        below_min_idx = (conf < pos_threshold)
        class_id[below_min_idx] = -1
        conf[below_min_idx] = 1 - conf[below_min_idx]
        pred_info = torch.cat((class_id.unsqueeze(1),
                               conf.unsqueeze(1),
                               predicted_bb), dim=1)
        out.append(pred_info)
    return d2l.stack(out)
```

それでは、4つのアンカーボックスを用いた具体例に[**上記の実装を適用**]してみましょう。
簡単のため、
予測オフセットはすべて0と仮定します。
これは、予測バウンディングボックスがアンカーボックスと同じであることを意味します。 
背景、犬、猫の各クラスについて、それぞれの予測確率も定義します。

```{.python .input}
#@tab all
anchors = d2l.tensor([[0.1, 0.08, 0.52, 0.92], [0.08, 0.2, 0.56, 0.95],
                      [0.15, 0.3, 0.62, 0.91], [0.55, 0.2, 0.9, 0.88]])
offset_preds = d2l.tensor([0] * d2l.size(anchors))
cls_probs = d2l.tensor([[0] * 4,  # Predicted background likelihood 
                      [0.9, 0.8, 0.7, 0.1],  # Predicted dog likelihood 
                      [0.1, 0.2, 0.3, 0.9]])  # Predicted cat likelihood
```

これらの予測バウンディングボックスとその confidence を画像上に[**描画できます。**]

```{.python .input}
#@tab all
fig = d2l.plt.imshow(img)
show_bboxes(fig.axes, anchors * bbox_scale,
            ['dog=0.9', 'dog=0.8', 'dog=0.7', 'cat=0.9'])
```

ここで `multibox_detection` 関数を呼び出して、
しきい値を0.5に設定した非最大抑制を実行できます。
なお、テンソル入力には
例の次元を追加しています。

[**返される結果の形状**]は
(batch size, number of anchor boxes, 6) です。
最内側の次元の6要素は、
同じ予測バウンディングボックスに対する出力情報を与えます。
1番目の要素は予測クラスのインデックスで、0から始まります（0は犬、1は猫です）。値 -1 は背景または非最大抑制による除去を表します。
2番目の要素は予測バウンディングボックスの confidence です。
残りの4要素は、それぞれ予測バウンディングボックスの左上隅と右下隅の $(x, y)$ 軸座標（範囲は0から1）です。

```{.python .input}
#@tab mxnet
output = multibox_detection(np.expand_dims(cls_probs, axis=0),
                            np.expand_dims(offset_preds, axis=0),
                            np.expand_dims(anchors, axis=0),
                            nms_threshold=0.5)
output
```

```{.python .input}
#@tab pytorch
output = multibox_detection(cls_probs.unsqueeze(dim=0),
                            offset_preds.unsqueeze(dim=0),
                            anchors.unsqueeze(dim=0),
                            nms_threshold=0.5)
output
```

クラス -1 の予測バウンディングボックスを取り除いた後、
非最大抑制によって保持された最終的な予測バウンディングボックスを[**出力できます**]。

```{.python .input}
#@tab all
fig = d2l.plt.imshow(img)
for i in d2l.numpy(output[0]):
    if i[0] == -1:
        continue
    label = ('dog=', 'cat=')[int(i[0])] + str(i[1])
    show_bboxes(fig.axes, [d2l.tensor(i[2:]) * bbox_scale], label)
```

実際には、非最大抑制を行う前に、confidence の低い予測バウンディングボックスを取り除くことで、このアルゴリズムの計算量を減らせます。
また、非最大抑制の出力に後処理を施し、たとえば
最終出力ではより高い confidence をもつ結果だけを保持することもできます。


## まとめ

* 画像の各ピクセルを中心として、異なる形状のアンカーボックスを生成する。
* Intersection over union（IoU）は Jaccard index とも呼ばれ、2つのバウンディングボックスの類似度を測る。これは共通部分の面積を和集合の面積で割った比である。
* 学習データセットでは、各アンカーボックスに対して2種類のラベルが必要である。1つはアンカーボックスに対応する物体のクラス、もう1つはアンカーボックスに対する真のバウンディングボックスのオフセットである。
* 予測時には、非最大抑制（NMS）を用いて似た予測バウンディングボックスを取り除き、出力を簡潔にできる。


## 演習

1. `multibox_prior` 関数の `sizes` と `ratios` の値を変更してみてください。生成されるアンカーボックスはどう変わりますか。
1. IoU が 0.5 となる2つのバウンディングボックスを構成し、可視化してください。それらはどのように重なっていますか。
1. :numref:`subsec_labeling-anchor-boxes` と :numref:`subsec_predicting-bounding-boxes-nms` の `anchors` 変数を変更してください。結果はどう変わりますか。
1. 非最大抑制は、予測バウンディングボックスを *取り除く* ことで抑制する貪欲法です。取り除かれたものの中に、実は有用なものが含まれている可能性はありますか。*ソフトに* 抑制するようにこのアルゴリズムをどのように修正できますか。Soft-NMS :cite:`Bodla.Singh.Chellappa.ea.2017` を参照してもよいです。
1. 手作業で設計するのではなく、非最大抑制を学習することは可能でしょうか。
