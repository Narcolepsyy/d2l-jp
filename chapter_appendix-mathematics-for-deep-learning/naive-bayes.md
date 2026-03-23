# ナイーブベイズ
:label:`sec_naive_bayes`

これまでの節を通して、確率論と確率変数について学んできました。この理論を実際に役立てるために、*ナイーブベイズ*分類器を導入しましょう。これは確率の基本原理だけを用いて、数字の分類を可能にします。

学習とは、仮定を置くことにほかなりません。これまで見たことのない新しいデータ例を分類したいなら、どのデータ例が互いに似ているかについて何らかの仮定を置かなければなりません。ナイーブベイズ分類器は、広く使われている非常に明快なアルゴリズムで、計算を簡単にするために、すべての特徴が互いに独立であると仮定します。この節では、このモデルを画像中の文字認識に適用します。

```{.python .input}
#@tab mxnet
%matplotlib inline
from d2l import mxnet as d2l
import math
from mxnet import gluon, np, npx
npx.set_np()
d2l.use_svg_display()
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
import math
import torch
import torchvision
d2l.use_svg_display()
```

```{.python .input}
#@tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
import math
import tensorflow as tf
d2l.use_svg_display()
```

## 光学文字認識

MNIST :cite:`LeCun.Bottou.Bengio.ea.1998` は、広く使われているデータセットの一つです。これは学習用に60,000枚、検証用に10,000枚の画像を含みます。各画像には0から9までの手書き数字が1つ含まれています。課題は、それぞれの画像を対応する数字に分類することです。

Gluon は `data.vision` モジュールに `MNIST` クラスを提供しており、
データセットをインターネットから自動的に取得できます。
その後は、Gluon はすでにダウンロード済みのローカルコピーを使用します。
学習セットかテストセットかは、パラメータ `train` の値をそれぞれ `True` または `False` に設定することで指定します。
各画像は幅と高さがともに $28$ のグレースケール画像で、形状は ($28$,$28$,$1$) です。最後のチャネル次元を取り除くために、カスタマイズした変換を使います。さらに、このデータセットでは各画素が符号なし $8$ ビット整数で表されています。問題を簡単にするために、これらを2値特徴に量子化します。

```{.python .input}
#@tab mxnet
def transform(data, label):
    return np.floor(data.astype('float32') / 128).squeeze(axis=-1), label

mnist_train = gluon.data.vision.MNIST(train=True, transform=transform)
mnist_test = gluon.data.vision.MNIST(train=False, transform=transform)
```

```{.python .input}
#@tab pytorch
data_transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    lambda x: torch.floor(x * 255 / 128).squeeze(dim=0)
])

mnist_train = torchvision.datasets.MNIST(
    root='./temp', train=True, transform=data_transform, download=True)
mnist_test = torchvision.datasets.MNIST(
    root='./temp', train=False, transform=data_transform, download=True)
```

```{.python .input}
#@tab tensorflow
((train_images, train_labels), (
    test_images, test_labels)) = tf.keras.datasets.mnist.load_data()

# Original pixel values of MNIST range from 0-255 (as the digits are stored as
# uint8). For this section, pixel values that are greater than 128 (in the
# original image) are converted to 1 and values that are less than 128 are
# converted to 0. See section 18.9.2 and 18.9.3 for why
train_images = tf.floor(tf.constant(train_images / 128, dtype = tf.float32))
test_images = tf.floor(tf.constant(test_images / 128, dtype = tf.float32))

train_labels = tf.constant(train_labels, dtype = tf.int32)
test_labels = tf.constant(test_labels, dtype = tf.int32)
```

画像と対応するラベルを含む、特定の例にアクセスできます。

```{.python .input}
#@tab mxnet
image, label = mnist_train[2]
image.shape, label
```

```{.python .input}
#@tab pytorch
image, label = mnist_train[2]
image.shape, label
```

```{.python .input}
#@tab tensorflow
image, label = train_images[2], train_labels[2]
image.shape, label.numpy()
```

ここで変数 `image` に格納されている例は、縦横ともに $28$ ピクセルの画像に対応しています。

```{.python .input}
#@tab all
image.shape, image.dtype
```

コードでは、各画像のラベルをスカラーとして保存しています。その型は32ビット整数です。

```{.python .input}
#@tab mxnet
label, type(label), label.dtype
```

```{.python .input}
#@tab pytorch
label, type(label)
```

```{.python .input}
#@tab tensorflow
label.numpy(), label.dtype
```

複数の例に同時にアクセスすることもできます。

```{.python .input}
#@tab mxnet
images, labels = mnist_train[10:38]
images.shape, labels.shape
```

```{.python .input}
#@tab pytorch
images = torch.stack([mnist_train[i][0] for i in range(10, 38)], dim=0)
labels = torch.tensor([mnist_train[i][1] for i in range(10, 38)])
images.shape, labels.shape
```

```{.python .input}
#@tab tensorflow
images = tf.stack([train_images[i] for i in range(10, 38)], axis=0)
labels = tf.constant([train_labels[i].numpy() for i in range(10, 38)])
images.shape, labels.shape
```

これらの例を可視化してみましょう。

```{.python .input}
#@tab all
d2l.show_images(images, 2, 9);
```

## 分類のための確率モデル

分類タスクでは、例をあるカテゴリに写像します。ここでの例はグレースケールの $28\times 28$ 画像で、カテゴリは数字です。（より詳しい説明は :numref:`sec_softmax` を参照してください。）
分類タスクを表現する自然な方法の一つは、確率的な問いとして考えることです。すなわち、特徴（つまり画像の画素）が与えられたとき、最もありそうなラベルは何か、という問いです。例の特徴を $\mathbf x\in\mathbb R^d$、ラベルを $y\in\mathbb R$ と表します。ここで特徴は画像の画素であり、2次元画像をベクトルに変形すれば $d=28^2=784$ となります。ラベルは数字です。
特徴が与えられたときのラベルの確率は $p(y  \mid  \mathbf{x})$ です。もしこれらの確率、つまりこの例では $y=0, \ldots,9$ に対する $p(y  \mid  \mathbf{x})$ を計算できるなら、分類器は次の式で与えられる予測 $\hat{y}$ を出力します。

$$\hat{y} = \mathrm{argmax} \> p(y  \mid  \mathbf{x}).$$

残念ながら、これを行うには $\mathbf{x} = x_1, ..., x_d$ のすべての値に対して $p(y  \mid  \mathbf{x})$ を推定しなければなりません。各特徴が $2$ つの値のどちらかを取るとしましょう。たとえば、特徴 $x_1 = 1$ はある文書に単語 apple が現れることを意味し、$x_1 = 0$ は現れないことを意味するとします。もしそのような2値特徴が $30$ 個あれば、入力ベクトル $\mathbf{x}$ の $2^{30}$（10億以上！）通りの可能な値のどれに対しても分類できるように準備しておく必要があることになります。

さらに、どこに学習があるのでしょうか。対応するラベルを予測するためにあり得るすべての例を見なければならないなら、私たちは本当にパターンを学習しているのではなく、単にデータセットを記憶しているだけです。

## ナイーブベイズ分類器

幸いなことに、条件付き独立についていくつかの仮定を置くことで、帰納バイアスを導入し、比較的少数の学習例から一般化できるモデルを構築できます。まずベイズの定理を使って、分類器を次のように表しましょう。

$$\hat{y} = \mathrm{argmax}_y \> p(y  \mid  \mathbf{x}) = \mathrm{argmax}_y \> \frac{p( \mathbf{x}  \mid  y) p(y)}{p(\mathbf{x})}.$$

分母は正規化項 $p(\mathbf{x})$ であり、ラベル $y$ の値には依存しないことに注意してください。その結果、異なる $y$ の値に対して分子を比較することだけを考えれば十分です。たとえ分母の計算が困難であっても、分子を評価できる限り、それを無視して構いません。幸いなことに、正規化定数を復元したくなったとしても可能です。$\sum_y p(y  \mid  \mathbf{x}) = 1$ なので、正規化項は常に復元できます。

では、$p( \mathbf{x}  \mid  y)$ に注目しましょう。確率の連鎖律を用いると、$p( \mathbf{x}  \mid  y)$ を次のように表せます。

$$p(x_1  \mid y) \cdot p(x_2  \mid  x_1, y) \cdot ... \cdot p( x_d  \mid  x_1, ..., x_{d-1}, y).$$

これだけでは、まだ先に進めません。依然としておよそ $2^d$ 個のパラメータを推定しなければなりません。しかし、*ラベルが与えられたとき、特徴同士は条件付き独立である* と仮定すれば、この項は $\prod_i p(x_i  \mid  y)$ に簡単化され、予測器は

$$\hat{y} = \mathrm{argmax}_y \> \prod_{i=1}^d p(x_i  \mid  y) p(y).$$

となります。

すべての $i$ と $y$ に対して $p(x_i=1  \mid  y)$ を推定し、その値を $P_{xy}[i, y]$ に保存できるとします。ここで $P_{xy}$ は $d\times n$ 行列で、$n$ はクラス数、$y\in\{1, \ldots, n\}$ です。すると、これを使って $p(x_i = 0 \mid y)$ も推定できます。すなわち、

$$
p(x_i = t_i \mid y) =
\begin{cases}
    P_{xy}[i, y] & \textrm{for } t_i=1 ;\\
    1 - P_{xy}[i, y] & \textrm{for } t_i = 0 .
\end{cases}
$$

さらに、各 $y$ に対して $p(y)$ を推定して $P_y[y]$ に保存します。ここで $P_y$ は長さ $n$ のベクトルです。すると、新しい例 $\mathbf t = (t_1, t_2, \ldots, t_d)$ に対して、次を計算できます。

$$\begin{aligned}\hat{y} &= \mathrm{argmax}_ y \ p(y)\prod_{i=1}^d   p(x_t = t_i \mid y) \\ &= \mathrm{argmax}_y \ P_y[y]\prod_{i=1}^d \ P_{xy}[i, y]^{t_i}\, \left(1 - P_{xy}[i, y]\right)^{1-t_i}\end{aligned}$$
:eqlabel:`eq_naive_bayes_estimation`

任意の $y$ に対してです。したがって、条件付き独立という仮定により、モデルの複雑さは特徴数に対する指数的依存 $\mathcal{O}(2^dn)$ から線形依存 $\mathcal{O}(dn)$ へと減少しました。


## 学習

問題は、$P_{xy}$ と $P_y$ が分からないことです。したがって、まずいくつかの学習データからそれらの値を推定する必要があります。これがモデルの*学習*です。$P_y$ の推定はそれほど難しくありません。扱うクラスは $10$ 個しかないので、各数字の出現回数 $n_y$ を数え、全データ数 $n$ で割ればよいのです。たとえば、数字8が $n_8 = 5,800$ 回現れ、画像の総数が $n = 60,000$ 枚なら、確率推定は $p(y=8) = 0.0967$ です。

```{.python .input}
#@tab mxnet
X, Y = mnist_train[:]  # All training examples

n_y = np.zeros((10))
for y in range(10):
    n_y[y] = (Y == y).sum()
P_y = n_y / n_y.sum()
P_y
```

```{.python .input}
#@tab pytorch
X = torch.stack([mnist_train[i][0] for i in range(len(mnist_train))], dim=0)
Y = torch.tensor([mnist_train[i][1] for i in range(len(mnist_train))])

n_y = torch.zeros(10)
for y in range(10):
    n_y[y] = (Y == y).sum()
P_y = n_y / n_y.sum()
P_y
```

```{.python .input}
#@tab tensorflow
X = train_images
Y = train_labels

n_y = tf.Variable(tf.zeros(10))
for y in range(10):
    n_y[y].assign(tf.reduce_sum(tf.cast(Y == y, tf.float32)))
P_y = n_y / tf.reduce_sum(n_y)
P_y
```

次に、少し難しい $P_{xy}$ に進みます。白黒画像を選んだので、$p(x_i  \mid  y)$ は、クラス $y$ に対して画素 $i$ がオンになる確率を表します。先ほどと同様に、事象が起こる回数 $n_{iy}$ を数え、それを $y$ の総出現回数、すなわち $n_y$ で割ればよいのです。しかし、少し厄介な点があります。ある画素が決して黒にならないことがあり得るのです（たとえば、きちんと切り抜かれた画像では、角の画素は常に白かもしれません）。統計学者がこの問題に対処する便利な方法は、すべての出現回数に擬似カウントを加えることです。したがって、$n_{iy}$ の代わりに $n_{iy}+1$ を使い、$n_y$ の代わりに $n_{y}+2$ を使います（画素 $i$ が取りうる値は2つ、つまり黒か白のどちらかだからです）。これは *ラプラス平滑化* とも呼ばれます。これは場当たり的に見えるかもしれませんが、ベータ二項モデルによるベイズ的観点から動機づけることができます。

```{.python .input}
#@tab mxnet
n_x = np.zeros((10, 28, 28))
for y in range(10):
    n_x[y] = np.array(X.asnumpy()[Y.asnumpy() == y].sum(axis=0))
P_xy = (n_x + 1) / (n_y + 2).reshape(10, 1, 1)

d2l.show_images(P_xy, 2, 5);
```

```{.python .input}
#@tab pytorch
n_x = torch.zeros((10, 28, 28))
for y in range(10):
    n_x[y] = torch.tensor(X.numpy()[Y.numpy() == y].sum(axis=0))
P_xy = (n_x + 1) / (n_y + 2).reshape(10, 1, 1)

d2l.show_images(P_xy, 2, 5);
```

```{.python .input}
#@tab tensorflow
n_x = tf.Variable(tf.zeros((10, 28, 28)))
for y in range(10):
    n_x[y].assign(tf.cast(tf.reduce_sum(
        X.numpy()[Y.numpy() == y], axis=0), tf.float32))
P_xy = (n_x + 1) / tf.reshape((n_y + 2), (10, 1, 1))

d2l.show_images(P_xy, 2, 5);
```

これらの $10\times 28\times 28$ 個の確率（各クラスの各画素ごと）を可視化すると、何となく数字らしいものが見えてきます。

これで :eqref:`eq_naive_bayes_estimation` を使って新しい画像を予測できます。$\mathbf x$ が与えられたとき、次の関数は各 $y$ に対する $p(\mathbf x \mid y)p(y)$ を計算します。

```{.python .input}
#@tab mxnet
def bayes_pred(x):
    x = np.expand_dims(x, axis=0)  # (28, 28) -> (1, 28, 28)
    p_xy = P_xy * x + (1 - P_xy)*(1 - x)
    p_xy = p_xy.reshape(10, -1).prod(axis=1)  # p(x|y)
    return np.array(p_xy) * P_y

image, label = mnist_test[0]
bayes_pred(image)
```

```{.python .input}
#@tab pytorch
def bayes_pred(x):
    x = x.unsqueeze(0)  # (28, 28) -> (1, 28, 28)
    p_xy = P_xy * x + (1 - P_xy)*(1 - x)
    p_xy = p_xy.reshape(10, -1).prod(dim=1)  # p(x|y)
    return p_xy * P_y

image, label = mnist_test[0]
bayes_pred(image)
```

```{.python .input}
#@tab tensorflow
def bayes_pred(x):
    x = tf.expand_dims(x, axis=0)  # (28, 28) -> (1, 28, 28)
    p_xy = P_xy * x + (1 - P_xy)*(1 - x)
    p_xy = tf.math.reduce_prod(tf.reshape(p_xy, (10, -1)), axis=1)  # p(x|y)
    return p_xy * P_y

image, label = train_images[0], train_labels[0]
bayes_pred(image)
```

これはひどく失敗しました！その理由を調べるために、画素ごとの確率を見てみましょう。これらは通常 $0.001$ から $1$ の間の数です。私たちはそれらを784個掛け合わせています。ここで重要なのは、これらの数をコンピュータ上で計算しているため、指数部の範囲が有限であるということです。起こるのは *数値アンダーフロー* で、小さな数をすべて掛け合わせるとさらに小さくなり、最終的には0に丸め込まれてしまいます。これは :numref:`sec_maximum_likelihood` で理論的な問題として議論しましたが、ここではその現象が実際に明確に見えます。

その節で述べたように、$\log a b = \log a + \log b$ という性質を使って、対数の和を取るようにすれば解決できます。
$ a $ と $ b $ の両方が小さい数であっても、対数値は適切な範囲に収まるはずです。

```{.python .input}
#@tab mxnet
a = 0.1
print('underflow:', a**784)
print('logarithm is normal:', 784*math.log(a))
```

```{.python .input}
#@tab pytorch
a = 0.1
print('underflow:', a**784)
print('logarithm is normal:', 784*math.log(a))
```

```{.python .input}
#@tab tensorflow
a = 0.1
print('underflow:', a**784)
print('logarithm is normal:', 784*tf.math.log(a).numpy())
```

対数は単調増加関数なので、:eqref:`eq_naive_bayes_estimation` を次のように書き換えられます。

$$ \hat{y} = \mathrm{argmax}_y \ \log P_y[y] + \sum_{i=1}^d \Big[t_i\log P_{xy}[x_i, y] + (1-t_i) \log (1 - P_{xy}[x_i, y]) \Big].$$

次の安定版を実装できます。

```{.python .input}
#@tab mxnet
log_P_xy = np.log(P_xy)
log_P_xy_neg = np.log(1 - P_xy)
log_P_y = np.log(P_y)

def bayes_pred_stable(x):
    x = np.expand_dims(x, axis=0)  # (28, 28) -> (1, 28, 28)
    p_xy = log_P_xy * x + log_P_xy_neg * (1 - x)
    p_xy = p_xy.reshape(10, -1).sum(axis=1)  # p(x|y)
    return p_xy + log_P_y

py = bayes_pred_stable(image)
py
```

```{.python .input}
#@tab pytorch
log_P_xy = torch.log(P_xy)
log_P_xy_neg = torch.log(1 - P_xy)
log_P_y = torch.log(P_y)

def bayes_pred_stable(x):
    x = x.unsqueeze(0)  # (28, 28) -> (1, 28, 28)
    p_xy = log_P_xy * x + log_P_xy_neg * (1 - x)
    p_xy = p_xy.reshape(10, -1).sum(dim=1)  # p(x|y)
    return p_xy + log_P_y

py = bayes_pred_stable(image)
py
```

```{.python .input}
#@tab tensorflow
log_P_xy = tf.math.log(P_xy)
log_P_xy_neg = tf.math.log(1 - P_xy)
log_P_y = tf.math.log(P_y)

def bayes_pred_stable(x):
    x = tf.expand_dims(x, axis=0)  # (28, 28) -> (1, 28, 28)
    p_xy = log_P_xy * x + log_P_xy_neg * (1 - x)
    p_xy = tf.math.reduce_sum(tf.reshape(p_xy, (10, -1)), axis=1)  # p(x|y)
    return p_xy + log_P_y

py = bayes_pred_stable(image)
py
```

これで予測が正しいかどうか確認できます。

```{.python .input}
#@tab mxnet
# Convert label which is a scalar tensor of int32 dtype to a Python scalar
# integer for comparison
py.argmax(axis=0) == int(label)
```

```{.python .input}
#@tab pytorch
py.argmax(dim=0) == label
```

```{.python .input}
#@tab tensorflow
tf.argmax(py, axis=0, output_type = tf.int32) == label
```

いくつかの検証例を予測してみると、ベイズ分類器がかなりうまく機能していることが分かります。

```{.python .input}
#@tab mxnet
def predict(X):
    return [bayes_pred_stable(x).argmax(axis=0).astype(np.int32) for x in X]

X, y = mnist_test[:18]
preds = predict(X)
d2l.show_images(X, 2, 9, titles=[str(d) for d in preds]);
```

```{.python .input}
#@tab pytorch
def predict(X):
    return [bayes_pred_stable(x).argmax(dim=0).type(torch.int32).item()
            for x in X]

X = torch.stack([mnist_test[i][0] for i in range(18)], dim=0)
y = torch.tensor([mnist_test[i][1] for i in range(18)])
preds = predict(X)
d2l.show_images(X, 2, 9, titles=[str(d) for d in preds]);
```

```{.python .input}
#@tab tensorflow
def predict(X):
    return [tf.argmax(
        bayes_pred_stable(x), axis=0, output_type = tf.int32).numpy()
            for x in X]

X = tf.stack([train_images[i] for i in range(10, 38)], axis=0)
y = tf.constant([train_labels[i].numpy() for i in range(10, 38)])
preds = predict(X)
d2l.show_images(X, 2, 9, titles=[str(d) for d in preds]);
```

最後に、分類器全体の精度を計算しましょう。

```{.python .input}
#@tab mxnet
X, y = mnist_test[:]
preds = np.array(predict(X), dtype=np.int32)
float((preds == y).sum()) / len(y)  # Validation accuracy
```

```{.python .input}
#@tab pytorch
X = torch.stack([mnist_test[i][0] for i in range(len(mnist_test))], dim=0)
y = torch.tensor([mnist_test[i][1] for i in range(len(mnist_test))])
preds = torch.tensor(predict(X), dtype=torch.int32)
float((preds == y).sum()) / len(y)  # Validation accuracy
```

```{.python .input}
#@tab tensorflow
X = test_images
y = test_labels
preds = tf.constant(predict(X), dtype=tf.int32)
# Validation accuracy
tf.reduce_sum(tf.cast(preds == y, tf.float32)).numpy() / len(y)
```

現代の深層ネットワークは、$0.01$ 未満の誤り率を達成します。比較的性能が低いのは、モデルで置いた統計的仮定が誤っているからです。つまり、各画素がラベルのみに依存して*独立に*生成されると仮定したのです。これは人間が数字を書く方法とは明らかに異なり、この誤った仮定が、あまりにもナイーブな（ベイズ）分類器の失敗につながりました。

## まとめ
* ベイズの規則を用いると、観測されたすべての特徴が独立であると仮定することで分類器を作れる。
* この分類器は、ラベルと画素値の組合せの出現回数を数えることで、データセット上で学習できる。
* この分類器は、スパム検出のようなタスクにおいて何十年もの間、標準的な手法だった。

## 演習
1. データセット $[[0,0], [0,1], [1,0], [1,1]]$ を考え、ラベルを2つの要素の XOR で与えた $[0,1,1,0]$ とします。このデータセット上に構築したナイーブベイズ分類器の確率はどうなりますか。点を正しく分類できますか。できない場合、どの仮定が破られていますか。
1. 確率を推定する際にラプラス平滑化を使わず、テスト時に学習で一度も観測されなかった値を含むデータ例が来たとします。このときモデルは何を出力するでしょうか。
1. ナイーブベイズ分類器はベイジアンネットワークの特別な例であり、確率変数の依存関係がグラフ構造で表現されます。この節の範囲を超える完全な理論については :citet:`Koller.Friedman.2009` を参照してくださいが、XOR モデルにおいて2つの入力変数の間の明示的な依存を許すと、なぜ成功する分類器を作れるのか説明してください。

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/418)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1100)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/1101)
:end_tab:\n