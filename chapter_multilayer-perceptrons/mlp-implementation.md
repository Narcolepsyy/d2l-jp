{.python .input}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow', 'jax'])
```

# 多層パーセプトロンの実装
:label:`sec_mlp-implementation`

多層パーセプトロン（MLP）は、単純な線形モデルよりも実装がそれほど複雑になるわけではありません。重要な概念上の違いは、複数の層を連結するようになったことです。

```{.python .input}
%%tab mxnet
from d2l import mxnet as d2l
from mxnet import np, npx
from mxnet.gluon import nn
npx.set_np()
```

```{.python .input}
%%tab pytorch
from d2l import torch as d2l
import torch
from torch import nn
```

```{.python .input}
%%tab tensorflow
from d2l import tensorflow as d2l
import tensorflow as tf
```

```{.python .input}
%%tab jax
from d2l import jax as d2l
from flax import linen as nn
import jax
from jax import numpy as jnp
```

## ゼロからの実装

まずは、こうしたネットワークをゼロから実装してみましょう。

### モデルパラメータの初期化

Fashion-MNIST には 10 クラスがあり、
各画像は $28 \times 28 = 784$
個のグレースケール画素値の格子から成ります。
これまでと同様に、ここではひとまず画素間の空間構造は無視するので、
784 個の入力特徴量と 10 クラスをもつ分類データセットと考えられます。
まずは、[**1 つの隠れ層と 256 個の隠れユニットをもつ MLP を実装**]しましょう。
層の数もその幅も調整可能です
（これらはハイパーパラメータと見なされます）。
通常、層の幅は 2 の大きな冪で割り切れるように選びます。
これは、ハードウェアにおけるメモリの割り当てとアドレス指定の仕組みにより、計算効率がよいからです。

ここでも、パラメータは複数のテンソルで表します。
*各層ごとに*、1 つの重み行列と 1 つのバイアスベクトルを保持しなければならないことに注意してください。
いつものように、これらのパラメータに関する損失の勾配のためのメモリも確保します。

:begin_tab:`mxnet`
以下のコードでは、まずパラメータを定義して初期化し、その後で勾配追跡を有効にします。
:end_tab:

:begin_tab:`pytorch`
以下のコードでは `nn.Parameter` を使って、
クラス属性を `autograd` によって追跡されるパラメータとして自動的に登録します（:numref:`sec_autograd`）。
:end_tab:

:begin_tab:`tensorflow`
以下のコードでは `tf.Variable` を使ってモデルパラメータを定義します。
:end_tab:

:begin_tab:`jax`
以下のコードでは `flax.linen.Module.param` を使ってモデルパラメータを定義します。
:end_tab:

```{.python .input}
%%tab mxnet
class MLPScratch(d2l.Classifier):
    def __init__(self, num_inputs, num_outputs, num_hiddens, lr, sigma=0.01):
        super().__init__()
        self.save_hyperparameters()
        self.W1 = np.random.randn(num_inputs, num_hiddens) * sigma
        self.b1 = np.zeros(num_hiddens)
        self.W2 = np.random.randn(num_hiddens, num_outputs) * sigma
        self.b2 = np.zeros(num_outputs)
        for param in self.get_scratch_params():
            param.attach_grad()
```

```{.python .input}
%%tab pytorch
class MLPScratch(d2l.Classifier):
    def __init__(self, num_inputs, num_outputs, num_hiddens, lr, sigma=0.01):
        super().__init__()
        self.save_hyperparameters()
        self.W1 = nn.Parameter(torch.randn(num_inputs, num_hiddens) * sigma)
        self.b1 = nn.Parameter(torch.zeros(num_hiddens))
        self.W2 = nn.Parameter(torch.randn(num_hiddens, num_outputs) * sigma)
        self.b2 = nn.Parameter(torch.zeros(num_outputs))
```

```{.python .input}
%%tab tensorflow
class MLPScratch(d2l.Classifier):
    def __init__(self, num_inputs, num_outputs, num_hiddens, lr, sigma=0.01):
        super().__init__()
        self.save_hyperparameters()
        self.W1 = tf.Variable(
            tf.random.normal((num_inputs, num_hiddens)) * sigma)
        self.b1 = tf.Variable(tf.zeros(num_hiddens))
        self.W2 = tf.Variable(
            tf.random.normal((num_hiddens, num_outputs)) * sigma)
        self.b2 = tf.Variable(tf.zeros(num_outputs))
```

```{.python .input}
%%tab jax
class MLPScratch(d2l.Classifier):
    num_inputs: int
    num_outputs: int
    num_hiddens: int
    lr: float
    sigma: float = 0.01

    def setup(self):
        self.W1 = self.param('W1', nn.initializers.normal(self.sigma),
                             (self.num_inputs, self.num_hiddens))
        self.b1 = self.param('b1', nn.initializers.zeros, self.num_hiddens)
        self.W2 = self.param('W2', nn.initializers.normal(self.sigma),
                             (self.num_hiddens, self.num_outputs))
        self.b2 = self.param('b2', nn.initializers.zeros, self.num_outputs)
```

### モデル

すべてがどのように動くのかを確実に理解するために、
組み込みの `relu` 関数を直接呼び出すのではなく、
[**ReLU 活性化関数を自分で実装**]します。

```{.python .input}
%%tab mxnet
def relu(X):
    return np.maximum(X, 0)
```

```{.python .input}
%%tab pytorch
def relu(X):
    a = torch.zeros_like(X)
    return torch.max(X, a)
```

```{.python .input}
%%tab tensorflow
def relu(X):
    return tf.math.maximum(X, 0)
```

```{.python .input}
%%tab jax
def relu(X):
    return jnp.maximum(X, 0)
```

空間構造を無視するので、
各 2 次元画像を `reshape` して
長さ `num_inputs` の平坦なベクトルに変換します。
最後に、わずか数行のコードで [**モデルを実装**] します。フレームワーク組み込みの autograd を使うので、これだけで十分です。

```{.python .input}
%%tab all
@d2l.add_to_class(MLPScratch)
def forward(self, X):
    X = d2l.reshape(X, (-1, self.num_inputs))
    H = relu(d2l.matmul(X, self.W1) + self.b1)
    return d2l.matmul(H, self.W2) + self.b2
```

### 学習

幸いなことに、[**MLP の学習ループはソフトマックス回帰の場合とまったく同じです。**] モデル、データ、トレーナーを定義し、最後にモデルとデータに対して `fit` メソッドを呼び出します。

```{.python .input}
%%tab all
model = MLPScratch(num_inputs=784, num_outputs=10, num_hiddens=256, lr=0.1)
data = d2l.FashionMNIST(batch_size=256)
trainer = d2l.Trainer(max_epochs=10)
trainer.fit(model, data)
```

## 簡潔な実装

予想どおり、高水準 API に頼れば、MLP はさらに簡潔に実装できます。

### モデル

ソフトマックス回帰の簡潔な実装
（:numref:`sec_softmax_concise`）と比べると、
違いは、以前は *1 つ* だけ追加していた全結合層を、
ここでは *2 つ* 追加する点だけです。
1 つ目が[**隠れ層**]で、
2 つ目が出力層です。

```{.python .input}
%%tab mxnet
class MLP(d2l.Classifier):
    def __init__(self, num_outputs, num_hiddens, lr):
        super().__init__()
        self.save_hyperparameters()
        self.net = nn.Sequential()
        self.net.add(nn.Dense(num_hiddens, activation='relu'),
                     nn.Dense(num_outputs))
        self.net.initialize()
```

```{.python .input}
%%tab pytorch
class MLP(d2l.Classifier):
    def __init__(self, num_outputs, num_hiddens, lr):
        super().__init__()
        self.save_hyperparameters()
        self.net = nn.Sequential(nn.Flatten(), nn.LazyLinear(num_hiddens),
                                 nn.ReLU(), nn.LazyLinear(num_outputs))
```

```{.python .input}
%%tab tensorflow
class MLP(d2l.Classifier):
    def __init__(self, num_outputs, num_hiddens, lr):
        super().__init__()
        self.save_hyperparameters()
        self.net = tf.keras.models.Sequential([
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(num_hiddens, activation='relu'),
            tf.keras.layers.Dense(num_outputs)])
```

```{.python .input}
%%tab jax
class MLP(d2l.Classifier):
    num_outputs: int
    num_hiddens: int
    lr: float

    @nn.compact
    def __call__(self, X):
        X = X.reshape((X.shape[0], -1))  # Flatten
        X = nn.Dense(self.num_hiddens)(X)
        X = nn.relu(X)
        X = nn.Dense(self.num_outputs)(X)
        return X
```

以前は、モデルのパラメータを使って入力を変換する `forward` メソッドを定義していました。
これらの操作は本質的にはパイプラインです。
つまり、入力を受け取り、
変換（たとえば、
重みとの行列積の後にバイアスを加える）を適用し、
その変換の出力を次の変換への入力として
繰り返し使います。
しかし、ここでは
`forward` メソッドが定義されていないことに気づいたかもしれません。
実際には、`MLP` は `Module` クラス（:numref:`subsec_oo-design-models`）から `forward` メソッドを継承し、
単に `self.net(X)`（`X` は入力）を呼び出します。
そして `self.net` は `Sequential` クラスによって
変換の列として定義されています。
`Sequential` クラスは順伝播の過程を抽象化し、
私たちが変換そのものに集中できるようにします。
`Sequential` クラスの動作については、 :numref:`subsec_model-construction-sequential` でさらに説明します。


### 学習

[**学習ループ**] は、ソフトマックス回帰を実装したときとまったく同じです。
このモジュール性により、
モデルアーキテクチャに関する事項と
それ以外の考慮事項を分離できます。

```{.python .input}
%%tab all
model = MLP(num_outputs=10, num_hiddens=256, lr=0.1)
trainer.fit(model, data)
```

## まとめ

深層ネットワークの設計にさらに慣れてきた今では、深層ネットワークを 1 層から複数層へと拡張することは、それほど大きな難題ではなくなりました。特に、学習アルゴリズムとデータローダーは再利用できます。ただし、MLP をゼロから実装するのはそれでも煩雑です。モデルパラメータに名前を付けて管理する必要があるため、モデルの拡張が難しくなります。たとえば、42 層目と 43 層目の間に別の層を挿入したいとしましょう。その場合、順番に名前を付け直す覚悟がない限り、それは 42b 層のようなものになってしまうかもしれません。さらに、ネットワークをゼロから実装すると、フレームワークが意味のある性能最適化を行うのははるかに難しくなります。

それでも、完全結合の深層ネットワークがニューラルネットワークモデリングの第一選択だった 1980 年代後半の最先端に、あなたは到達しました。次の概念的なステップでは、画像を扱います。その前に、いくつかの統計の基礎と、モデルを効率よく計算する方法についての詳細を復習する必要があります。


## 演習

1. 隠れユニット数 `num_hiddens` を変えて、その数がモデルの精度にどう影響するかをプロットしてください。最適なこのハイパーパラメータの値は何ですか。
1. 隠れ層を追加して、結果にどう影響するか試してください。
1. 1 個のニューロンしかない隠れ層を挿入するのが悪い考えなのはなぜですか。何がうまくいかなくなる可能性がありますか。
1. 学習率を変えると結果はどう変わりますか。他のすべてのパラメータを固定したとき、どの学習率が最良の結果を与えますか。それはエポック数とどう関係しますか。
1. 学習率、エポック数、隠れ層の数、各隠れ層の隠れユニット数を含む、すべてのハイパーパラメータを同時に最適化してみましょう。
    1. それらすべてを最適化したとき、得られる最良の結果は何ですか。
    1. 複数のハイパーパラメータを扱うのがはるかに難しいのはなぜですか。
    1. 複数のパラメータを同時に最適化するための効率的な戦略を説明してください。
1. 難しい問題に対して、フレームワーク版とゼロからの実装の速度を比較してください。ネットワークの複雑さによってどう変わりますか。
1. 適切に整列した行列と、整列していない行列について、テンソル--行列積の速度を測定してください。たとえば、次元が 1024、1025、1026、1028、1032 の行列でテストしてください。
    1. これは GPU と CPU の間でどう変わりますか。
    1. CPU と GPU のメモリバス幅を求めてください。
1. さまざまな活性化関数を試してください。どれが最もよく機能しますか。
1. ネットワークの重み初期化には違いがありますか。それは重要ですか。

:begin_tab:`mxnet`
[議論](https://discuss.d2l.ai/t/92)
:end_tab:

:begin_tab:`pytorch`
[議論](https://discuss.d2l.ai/t/93)
:end_tab:

:begin_tab:`tensorflow`
[議論](https://discuss.d2l.ai/t/227)
:end_tab:

:begin_tab:`jax`
[議論](https://discuss.d2l.ai/t/17985)
:end_tab:\n