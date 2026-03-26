# 類似度によるアテンションプーリング

:label:`sec_attention-pooling`

ここまででアテンション機構の主要な構成要素を導入したので、今度はそれらをかなり古典的な設定、すなわちカーネル密度推定による回帰と分類 :cite:`Nadaraya.1964,Watson.1964` に用いてみましょう。この寄り道は単に追加の背景を与えるだけです。完全に任意であり、必要なら飛ばして構いません。  
Nadaraya--Watson 推定量の本質は、クエリ $\mathbf{q}$ とキー $\mathbf{k}$ を結び付ける何らかの類似度カーネル $\alpha(\mathbf{q}, \mathbf{k})$ にあります。代表的なカーネルには次のようなものがあります。

$$\begin{aligned}
\alpha(\mathbf{q}, \mathbf{k}) & = \exp\left(-\frac{1}{2} \|\mathbf{q} - \mathbf{k}\|^2 \right) && \textrm{Gaussian;} \\
\alpha(\mathbf{q}, \mathbf{k}) & = 1 \textrm{ if } \|\mathbf{q} - \mathbf{k}\| \leq 1 && \textrm{Boxcar;} \\
\alpha(\mathbf{q}, \mathbf{k}) & = \mathop{\mathrm{max}}\left(0, 1 - \|\mathbf{q} - \mathbf{k}\|\right) && \textrm{Epanechikov.}
\end{aligned}
$$

他にも多くの選択肢があります。より詳しい概説と、カーネルの選択がカーネル密度推定、しばしば *Parzen Windows* とも呼ばれるもの :cite:`parzen1957consistent` とどう関係するかについては、[Wikipedia の記事](https://en.wikipedia.org/wiki/Kernel_(statistics)) を参照してください。これらのカーネルはいずれもヒューリスティックであり、調整可能です。たとえば、幅は全体としてだけでなく、各座標ごとにも調整できます。いずれにせよ、どれも回帰と分類の両方に対して次の式を導きます。

$$f(\mathbf{q}) = \sum_i \mathbf{v}_i \frac{\alpha(\mathbf{q}, \mathbf{k}_i)}{\sum_j \alpha(\mathbf{q}, \mathbf{k}_j)}.$$

特徴量とラベルの観測 $(\mathbf{x}_i, y_i)$ を用いる（スカラー）回帰の場合、$\mathbf{v}_i = y_i$ はスカラー、$\mathbf{k}_i = \mathbf{x}_i$ はベクトルであり、クエリ $\mathbf{q}$ は $f$ を評価すべき新しい位置を表します。（多クラス）分類の場合は、$y_i$ の one-hot エンコーディングを用いて $\mathbf{v}_i$ を得ます。この推定量の便利な性質の一つは、学習を必要としないことです。さらに、データ量の増加に応じてカーネルを適切に狭めれば、この手法は整合的であり :cite:`mack1982weak`、すなわち統計的に最適な解のいずれかに収束します。まずはいくつかのカーネルを見てみましょう。

```{.python .input}
%load_ext d2lbook.tab
tab.interact_select('mxnet', 'pytorch', 'tensorflow', 'jax')
```

```{.python .input}
%%tab mxnet
from d2l import mxnet as d2l
from mxnet import autograd, gluon, np, npx
from mxnet.gluon import nn
npx.set_np()
d2l.use_svg_display()
```

```{.python .input}
%%tab pytorch
from d2l import torch as d2l
import torch
from torch import nn
from torch.nn import functional as F
import numpy as np

d2l.use_svg_display()
```

```{.python .input}
%%tab tensorflow
from d2l import tensorflow as d2l
import tensorflow as tf
import numpy as np

d2l.use_svg_display()
```

```{.python .input}
%%tab jax
from d2l import jax as d2l
import jax
from jax import numpy as jnp
from flax import linen as nn
```

## [**カーネルとデータ**]

この節で定義するすべてのカーネル $\alpha(\mathbf{k}, \mathbf{q})$ は *平行移動および回転に不変* です。つまり、$\mathbf{k}$ と $\mathbf{q}$ を同じように平行移動・回転させても、$\alpha$ の値は変わりません。簡単のため、ここではスカラー引数 $k, q \in \mathbb{R}$ を取り、キー $k = 0$ を原点として選びます。すると次のようになります。

```{.python .input}
%%tab all
# Define some kernels
def gaussian(x):
    return d2l.exp(-x**2 / 2)

def boxcar(x):
    return d2l.abs(x) < 1.0

def constant(x):
    return 1.0 + 0 * x
 
if tab.selected('pytorch'):
    def epanechikov(x):
        return torch.max(1 - d2l.abs(x), torch.zeros_like(x))
if tab.selected('mxnet'):
    def epanechikov(x):
        return np.maximum(1 - d2l.abs(x), 0)
if tab.selected('tensorflow'):
    def epanechikov(x):
        return tf.maximum(1 - d2l.abs(x), 0)
if tab.selected('jax'):
    def epanechikov(x):
        return jnp.maximum(1 - d2l.abs(x), 0)
```

```{.python .input}
%%tab all
fig, axes = d2l.plt.subplots(1, 4, sharey=True, figsize=(12, 3))

kernels = (gaussian, boxcar, constant, epanechikov)
names = ('Gaussian', 'Boxcar', 'Constant', 'Epanechikov')
x = d2l.arange(-2.5, 2.5, 0.1)
for kernel, name, ax in zip(kernels, names, axes):
    if tab.selected('pytorch', 'mxnet', 'tensorflow'):
        ax.plot(d2l.numpy(x), d2l.numpy(kernel(x)))
    if tab.selected('jax'):
        ax.plot(x, kernel(x))
    ax.set_xlabel(name)

d2l.plt.show()
```

異なるカーネルは、範囲と滑らかさに関する異なる概念に対応します。たとえば、boxcar カーネルは距離 $1$（あるいは別に定義したハイパーパラメータ）以内の観測値にしか注目せず、しかもそれを無差別に行います。  

Nadaraya--Watson 推定を実際に見てみるために、訓練データを定義しましょう。以下では次の依存関係を用います。

$$y_i = 2\sin(x_i) + x_i + \epsilon,$$

ここで $\epsilon$ は平均 0、分散 1 の正規分布から生成されます。40 個の訓練例をサンプルします。

```{.python .input}
%%tab all
def f(x):
    return 2 * d2l.sin(x) + x

n = 40
if tab.selected('pytorch'):
    x_train, _ = torch.sort(d2l.rand(n) * 5)
    y_train = f(x_train) + d2l.randn(n)
if tab.selected('mxnet'):
    x_train = np.sort(d2l.rand(n) * 5, axis=None)
    y_train = f(x_train) + d2l.randn(n)
if tab.selected('tensorflow'):
    x_train = tf.sort(d2l.rand((n,1)) * 5, 0)
    y_train = f(x_train) + d2l.normal((n, 1))
if tab.selected('jax'):
    x_train = jnp.sort(jax.random.uniform(d2l.get_key(), (n,)) * 5)
    y_train = f(x_train) + jax.random.normal(d2l.get_key(), (n,))
x_val = d2l.arange(0, 5, 0.1)
y_val = f(x_val)
```

## [**Nadaraya--Watson 回帰によるアテンションプーリング**]

データとカーネルがそろったので、あとはカーネル回帰の推定値を計算する関数だけです。なお、簡単な診断を行うために相対的なカーネル重みも得たいので、まず訓練特徴（共変量）`x_train` とすべての検証特徴 `x_val` の間のカーネルを計算します。これにより行列が得られ、それを正規化します。これを訓練ラベル `y_train` と掛け合わせると推定値が得られます。

:eqref:`eq_attention_pooling` のアテンションプーリングを思い出してください。各検証特徴をクエリとし、各訓練特徴--ラベルの組をキー--値のペアとみなします。その結果、正規化された相対カーネル重み（以下の `attention_w`）が *アテンション重み* になります。

```{.python .input}
%%tab all
def nadaraya_watson(x_train, y_train, x_val, kernel):
    dists = d2l.reshape(x_train, (-1, 1)) - d2l.reshape(x_val, (1, -1))
    # Each column/row corresponds to each query/key
    k = d2l.astype(kernel(dists), d2l.float32)
    # Normalization over keys for each query
    attention_w = k / d2l.reduce_sum(k, 0)
    if tab.selected('pytorch'):
        y_hat = y_train@attention_w
    if tab.selected('mxnet'):
        y_hat = np.dot(y_train, attention_w)
    if tab.selected('tensorflow'):
        y_hat = d2l.transpose(d2l.transpose(y_train)@attention_w)
    if tab.selected('jax'):
        y_hat = y_train@attention_w
    return y_hat, attention_w
```

異なるカーネルがどのような推定を生み出すか見てみましょう。

```{.python .input}
%%tab all
def plot(x_train, y_train, x_val, y_val, kernels, names, attention=False):
    fig, axes = d2l.plt.subplots(1, 4, sharey=True, figsize=(12, 3))
    for kernel, name, ax in zip(kernels, names, axes):
        y_hat, attention_w = nadaraya_watson(x_train, y_train, x_val, kernel)
        if attention:
            if tab.selected('pytorch', 'mxnet', 'tensorflow'):
                pcm = ax.imshow(d2l.numpy(attention_w), cmap='Reds')
            if tab.selected('jax'):
                pcm = ax.imshow(attention_w, cmap='Reds')
        else:
            ax.plot(x_val, y_hat)
            ax.plot(x_val, y_val, 'm--')
            ax.plot(x_train, y_train, 'o', alpha=0.5);
        ax.set_xlabel(name)
        if not attention:
            ax.legend(['y_hat', 'y'])
    if attention:
        fig.colorbar(pcm, ax=axes, shrink=0.7)
```

```{.python .input}
%%tab all
plot(x_train, y_train, x_val, y_val, kernels, names)
```

まず目につくのは、3 つの非自明なカーネル（Gaussian、Boxcar、Epanechikov）が、真の関数からそれほど離れていない、かなり実用的な推定を与えていることです。自明な推定 $f(x) = \frac{1}{n} \sum_i y_i$ に帰着する constant カーネルだけが、かなり非現実的な結果を生みます。アテンションの重み付けをもう少し詳しく見てみましょう。

```{.python .input}
%%tab all
plot(x_train, y_train, x_val, y_val, kernels, names, attention=True)
```

この可視化から、Gaussian、Boxcar、Epanechikov の推定が非常によく似ている理由がはっきり分かります。結局のところ、カーネルの関数形は異なっていても、非常によく似たアテンション重みから導かれているからです。ここで疑問になるのは、これが常に成り立つのかということです。  

## [**アテンションプーリングの適応**]

Gaussian カーネルを別の幅のものに置き換えることができます。つまり、
$\alpha(\mathbf{q}, \mathbf{k}) = \exp\left(-\frac{1}{2 \sigma^2} \|\mathbf{q} - \mathbf{k}\|^2 \right)$
を使い、$\sigma^2$ でカーネルの幅を決めることができます。これが結果に影響するか見てみましょう。

```{.python .input}
%%tab all
sigmas = (0.1, 0.2, 0.5, 1)
names = ['Sigma ' + str(sigma) for sigma in sigmas]

def gaussian_with_width(sigma): 
    return (lambda x: d2l.exp(-x**2 / (2*sigma**2)))

kernels = [gaussian_with_width(sigma) for sigma in sigmas]
plot(x_train, y_train, x_val, y_val, kernels, names)
```

明らかに、カーネルが狭いほど推定は滑らかでなくなります。同時に、局所的な変動にはよりよく適応します。対応するアテンション重みを見てみましょう。

```{.python .input}
%%tab all
plot(x_train, y_train, x_val, y_val, kernels, names, attention=True)
```

予想どおり、カーネルが狭いほど、大きなアテンション重みを持つ範囲も狭くなります。また、同じ幅を選ぶことが必ずしも理想的ではないことも明らかです。実際、:citet:`Silverman86` は局所密度に依存するヒューリスティックを提案しました。このような「工夫」は他にも多数提案されています。たとえば、:citet:`norelli2022asif` は、クロスモーダルな画像・テキスト表現を設計するために、類似した最近傍補間技術を用いました。  

この方法が半世紀以上前のものであるにもかかわらず、なぜここまで詳しく扱うのか、不思議に思う読者もいるかもしれません。第一に、これは現代のアテンション機構の最も初期の先駆けの一つだからです。第二に、可視化に非常に適しています。第三に、そして同じくらい重要ですが、手作りのアテンション機構の限界を示してくれます。より良い戦略は、クエリとキーの表現を学習することで機構そのものを *学習する* ことです。これが次の節で取り組む内容です。


## まとめ

Nadaraya--Watson カーネル回帰は、現在のアテンション機構の初期の先駆けです。  
分類にも回帰にも、ほとんどあるいはまったく学習や調整を必要とせず、そのまま使えます。  
アテンション重みは、クエリとキーの類似度（または距離）と、どれだけ類似した観測が利用可能かに応じて割り当てられます。  

## 演習

1. Parzen windows による密度推定は $\hat{p}(\mathbf{x}) = \frac{1}{n} \sum_i k(\mathbf{x}, \mathbf{x}_i)$ で与えられます。二値分類に対して、Parzen windows から得られる関数 $\hat{p}(\mathbf{x}, y=1) - \hat{p}(\mathbf{x}, y=-1)$ が Nadaraya--Watson 分類と等価であることを証明しなさい。 
1. Nadaraya--Watson 回帰において、カーネル幅の良い値を学習するための確率的勾配降下法を実装しなさい。 
    1. 上の推定値をそのまま使って $(f(\mathbf{x_i}) - y_i)^2$ を直接最小化するとどうなりますか？ ヒント: $y_i$ は $f$ を計算するために使われる項の一部です。
    1. $f(\mathbf{x}_i)$ の推定から $(\mathbf{x}_i, y_i)$ を除外し、カーネル幅について最適化しなさい。それでも過学習は観測されますか？
1. すべての $\mathbf{x}$ が単位球面上にある、すなわちすべて $\|\mathbf{x}\| = 1$ を満たすと仮定します。指数関数内の $\|\mathbf{x} - \mathbf{x}_i\|^2$ の項を簡単化できますか？ ヒント: これは後で、ドット積アテンションと非常に密接に関係していることが分かります。 
1. :citet:`mack1982weak` が Nadaraya--Watson 推定の整合性を証明したことを思い出してください。データが増えるにつれて、アテンション機構のスケールをどのくらいの速さで小さくすべきでしょうか？ 答えの直感も述べなさい。これはデータの次元に依存しますか？ どのように依存しますか？\n
