{.python .input}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow', 'jax'])
```

# 層とモジュール
:label:`sec_model_construction`

ニューラルネットワークを最初に導入したとき、
私たちは単一の出力をもつ線形モデルに焦点を当てました。
ここでは、モデル全体がたった1つのニューロンだけで構成されています。
単一のニューロンは、
(i) ある入力集合を受け取り、
(ii) それに対応するスカラー出力を生成し、
(iii) 関心のある目的関数を最適化するために更新可能な関連パラメータの集合をもつ、
ことに注意してください。
その後、複数の出力をもつネットワークについて考え始めると、
ベクトル化された算術を活用して、
ニューロンの層全体を表現できるようになりました。
個々のニューロンと同様に、
層も (i) 入力の集合を受け取り、
(ii) 対応する出力を生成し、
(iii) 調整可能なパラメータの集合によって記述されます。
softmax回帰を扱ったときには、
1つの層そのものがモデルでした。
しかし、その後MLPを導入した後でも、
モデルは同じ基本構造を保っていると考えることができました。

興味深いことに、MLPでは、
モデル全体とその構成要素である層の両方が
この構造を共有しています。
モデル全体は生の入力（特徴量）を受け取り、
出力（予測）を生成し、
パラメータ（すべての構成層のパラメータを合わせたもの）を持ちます。
同様に、各個別の層は入力（前の層から供給される）を受け取り、
出力（次の層への入力）を生成し、
さらに、次の層から逆向きに流れてくる信号に応じて更新される
調整可能なパラメータの集合を持ちます。


ニューロン、層、モデルで
十分な抽象化が得られているように思えるかもしれませんが、
実際には、個々の層より大きく、
モデル全体よりは小さい構成要素について
述べると便利なことがよくあります。
たとえば、コンピュータビジョンで非常に人気のある
ResNet-152アーキテクチャは、
数百もの層を持っています。
これらの層は、繰り返し現れる*層のグループ*のパターンから構成されています。こうしたネットワークを1層ずつ実装するのは面倒になりがちです。
この懸念は単なる仮説ではありません。こうした
設計パターンは実際によく見られます。
上で述べたResNetアーキテクチャは、
認識と検出の両方で2015年のImageNetおよびCOCOのコンピュータビジョン競技会を制し :cite:`He.Zhang.Ren.ea.2016`
、今なお多くの視覚タスクで定番のアーキテクチャです。
層がさまざまな繰り返しパターンで配置された
同様のアーキテクチャは、
自然言語処理や音声を含む他の分野でも
今や至るところに見られます。

こうした複雑なネットワークを実装するために、
ニューラルネットワークの*モジュール*という概念を導入します。
モジュールは単一の層、
複数の層からなる構成要素、
あるいはモデル全体そのものを表すことができます！
モジュール抽象化を用いる利点の1つは、
それらをより大きな成果物へと組み合わせられることです。
しかも、その組み合わせはしばしば再帰的に行えます。これは :numref:`fig_blocks` に示されています。任意の複雑さをもつモジュールを必要に応じて生成するコードを定義することで、
驚くほど簡潔なコードで
複雑なニューラルネットワークを実装できます。

![複数の層がモジュールにまとめられ、より大きなモデルの繰り返しパターンを形成する。](../img/blocks.svg)
:label:`fig_blocks`


プログラミングの観点では、モジュールは*クラス*として表現されます。
そのサブクラスはすべて、入力を出力へ変換する
順伝播メソッドを定義しなければならず、
必要なパラメータを保存しなければなりません。
なお、モジュールの中には
パラメータをまったく必要としないものもあります。
最後に、モジュールは勾配を計算するための
逆伝播メソッドを備えていなければなりません。
幸いなことに、自動微分
(:numref:`sec_autograd` で導入)
によって提供される裏方の魔法のおかげで、
自分自身のモジュールを定義するときに
私たちが気にする必要があるのは
パラメータと順伝播メソッドだけです。

```{.python .input}
%%tab mxnet
from mxnet import np, npx
from mxnet.gluon import nn
npx.set_np()
```

```{.python .input}
%%tab pytorch
import torch
from torch import nn
from torch.nn import functional as F
```

```{.python .input}
%%tab tensorflow
import tensorflow as tf
```

```{.python .input}
%%tab jax
from typing import List
from d2l import jax as d2l
from flax import linen as nn
import jax
from jax import numpy as jnp
```

[**まず、MLPを実装するために使ったコードを再確認しましょう**]
(:numref:`sec_mlp`)。
次のコードは、
256ユニットとReLU活性化をもつ全結合隠れ層1つと、
10ユニットをもつ全結合出力層1つ（活性化関数なし）からなるネットワークを生成します。

```{.python .input}
%%tab mxnet
net = nn.Sequential()
net.add(nn.Dense(256, activation='relu'))
net.add(nn.Dense(10))
net.initialize()

X = np.random.uniform(size=(2, 20))
net(X).shape
```

```{.python .input}
%%tab pytorch
net = nn.Sequential(nn.LazyLinear(256), nn.ReLU(), nn.LazyLinear(10))

X = torch.rand(2, 20)
net(X).shape
```

```{.python .input}
%%tab tensorflow
net = tf.keras.models.Sequential([
    tf.keras.layers.Dense(256, activation=tf.nn.relu),
    tf.keras.layers.Dense(10),
])

X = tf.random.uniform((2, 20))
net(X).shape
```

```{.python .input}
%%tab jax
net = nn.Sequential([nn.Dense(256), nn.relu, nn.Dense(10)])

# get_key is a d2l saved function returning jax.random.PRNGKey(random_seed)
X = jax.random.uniform(d2l.get_key(), (2, 20))
params = net.init(d2l.get_key(), X)
net.apply(params, X).shape
```

:begin_tab:`mxnet`
この例では、`nn.Sequential` をインスタンス化し、
返されたオブジェクトを `net` 変数に代入することで
モデルを構築しました。
次に、その `add` メソッドを繰り返し呼び出し、
実行される順序で層を追加していきます。
要するに、`nn.Sequential` は Gluon における*モジュール*を表すクラスである
特別な種類の `Block` を定義しています。
これは構成要素である `Block` の順序付きリストを保持します。
`add` メソッドは、単に各 `Block` をそのリストに順次追加しやすくしているだけです。
各層は `Dense` クラスのインスタンスであり、
それ自体が `Block` のサブクラスであることに注意してください。
順伝播（`forward`）メソッドも非常に単純です。
リスト内の各 `Block` を連結し、
それぞれの出力を次の入力として渡していきます。
なお、これまで私たちはモデルの出力を得るために
`net(X)` という構文でモデルを呼び出してきました。
これは実際には `net.forward(X)` の省略形にすぎず、
`Block` クラスの `__call__` メソッドによって実現される
洗練されたPythonのトリックです。
:end_tab:

:begin_tab:`pytorch`
この例では、`nn.Sequential` をインスタンス化し、実行される順序で層を引数として渡すことで
モデルを構築しました。
要するに、[**`nn.Sequential` は特別な種類の `Module` を定義する**]
ものであり、PyTorch におけるモジュールを表すクラスです。
これは構成要素である `Module` の順序付きリストを保持します。
2つの全結合層はいずれも `Linear` クラスのインスタンスであり、
それ自体が `Module` のサブクラスであることに注意してください。
順伝播（`forward`）メソッドも非常に単純です。
リスト内の各モジュールを連結し、
それぞれの出力を次の入力として渡していきます。
なお、これまで私たちはモデルの出力を得るために
`net(X)` という構文でモデルを呼び出してきました。
これは実際には `net.__call__(X)` の省略形にすぎません。
:end_tab:

:begin_tab:`tensorflow`
この例では、`keras.models.Sequential` をインスタンス化し、実行される順序で層を引数として渡すことで
モデルを構築しました。
要するに、`Sequential` は特別な種類の `keras.Model` を定義するものであり、
Keras におけるモジュールを表すクラスです。
これは構成要素である `Model` の順序付きリストを保持します。
2つの全結合層はいずれも `Dense` クラスのインスタンスであり、
それ自体が `Model` のサブクラスであることに注意してください。
順伝播（`call`）メソッドも非常に単純です。
リスト内の各モジュールを連結し、
それぞれの出力を次の入力として渡していきます。
なお、これまで私たちはモデルの出力を得るために
`net(X)` という構文でモデルを呼び出してきました。
これは実際には `net.call(X)` の省略形であり、
モジュールクラスの `__call__` メソッドによって実現される
洗練されたPythonのトリックです。
:end_tab:

## [**カスタムモジュール**]

モジュールの仕組みを理解する最も簡単な方法は、
自分で1つ実装してみることかもしれません。
その前に、
各モジュールが提供しなければならない基本機能を
簡単にまとめておきます。


1. 順伝播メソッドの引数として入力データを受け取る。
1. 順伝播メソッドが値を返すことで出力を生成する。出力の形状は入力と異なっていてもよい。たとえば、上のモデルの最初の全結合層は任意次元の入力を受け取るが、256次元の出力を返す。
1. 出力の入力に関する勾配を計算する。これは逆伝播メソッドを通じてアクセスできる。通常、これは自動的に行われる。
1. 順伝播計算を実行するために必要なパラメータを保存し、それらへアクセスできるようにする。
1. 必要に応じてモデルパラメータを初期化する。


次のスニペットでは、
隠れユニット256個の隠れ層1つと
10次元の出力層1つをもつMLPに対応するモジュールを
ゼロから記述します。
以下の `MLP` クラスは、モジュールを表すクラスを継承していることに注意してください。
親クラスのメソッドに大きく依存し、
独自に実装するのはコンストラクタ（Pythonでは `__init__` メソッド）と順伝播メソッドだけです。

```{.python .input}
%%tab mxnet
class MLP(nn.Block):
    def __init__(self):
        # Call the constructor of the MLP parent class nn.Block to perform
        # the necessary initialization
        super().__init__()
        self.hidden = nn.Dense(256, activation='relu')
        self.out = nn.Dense(10)

    # Define the forward propagation of the model, that is, how to return the
    # required model output based on the input X
    def forward(self, X):
        return self.out(self.hidden(X))
```

```{.python .input}
%%tab pytorch
class MLP(nn.Module):
    def __init__(self):
        # Call the constructor of the parent class nn.Module to perform
        # the necessary initialization
        super().__init__()
        self.hidden = nn.LazyLinear(256)
        self.out = nn.LazyLinear(10)

    # Define the forward propagation of the model, that is, how to return the
    # required model output based on the input X
    def forward(self, X):
        return self.out(F.relu(self.hidden(X)))
```

```{.python .input}
%%tab tensorflow
class MLP(tf.keras.Model):
    def __init__(self):
        # Call the constructor of the parent class tf.keras.Model to perform
        # the necessary initialization
        super().__init__()
        self.hidden = tf.keras.layers.Dense(units=256, activation=tf.nn.relu)
        self.out = tf.keras.layers.Dense(units=10)

    # Define the forward propagation of the model, that is, how to return the
    # required model output based on the input X
    def call(self, X):
        return self.out(self.hidden((X)))
```

```{.python .input}
%%tab jax
class MLP(nn.Module):
    def setup(self):
        # Define the layers
        self.hidden = nn.Dense(256)
        self.out = nn.Dense(10)

    # Define the forward propagation of the model, that is, how to return the
    # required model output based on the input X
    def __call__(self, X):
        return self.out(nn.relu(self.hidden(X)))
```

まず順伝播メソッドに注目しましょう。
これは `X` を入力として受け取り、
活性化関数を適用した隠れ表現を計算し、
ロジットを出力することがわかります。
この `MLP` 実装では、
両方の層がインスタンス変数です。
これが妥当である理由を理解するために、
`net1` と `net2` という2つのMLPをインスタンス化し、
それぞれを異なるデータで学習させる場面を想像してください。
当然、それらは
2つの異なる学習済みモデルを表すはずです。

私たちはコンストラクタの中で
MLPの層を[**インスタンス化し**]
[**その後、順伝播メソッドが呼ばれるたびにこれらの層を呼び出します**]。
いくつか重要な点に注意してください。
まず、カスタマイズした `__init__` メソッドは
`super().__init__()` を通じて親クラスの `__init__` メソッドを呼び出しており、
これにより、多くのモジュールに共通する
定型コードを何度も書き直す手間を省いています。
次に、2つの全結合層をインスタンス化し、
それらを `self.hidden` と `self.out` に代入しています。
新しい層を実装しない限り、
逆伝播メソッドやパラメータ初期化について
心配する必要はありません。
これらのメソッドはシステムが自動的に生成します。
試してみましょう。

```{.python .input}
%%tab pytorch, mxnet, tensorflow
net = MLP()
if tab.selected('mxnet'):
    net.initialize()
net(X).shape
```

```{.python .input}
%%tab jax
net = MLP()
params = net.init(d2l.get_key(), X)
net.apply(params, X).shape
```

モジュール抽象化の大きな利点は、その柔軟性です。
モジュールをサブクラス化して、
層（たとえば全結合層クラス）、
モデル全体（上の `MLP` クラスのようなもの）、
あるいは中間的な複雑さをもつさまざまな構成要素を作れます。
この柔軟性は、
今後の章を通じて活用していきます。
たとえば、
畳み込みニューラルネットワークを扱うときなどです。


## [**Sequentialモジュール**]
:label:`subsec_model-construction-sequential`

ここで `Sequential` クラスの仕組みを
もう少し詳しく見てみましょう。
`Sequential` は他のモジュールを
数珠つなぎにするために設計されていたことを思い出してください。
独自の簡略版 `MySequential` を作るには、
次の2つの重要なメソッドを定義すれば十分です。

1. モジュールを1つずつリストに追加するメソッド。
1. 追加された順序と同じ順でモジュールの連鎖を入力に通す順伝播メソッド。

次の `MySequential` クラスは、デフォルトの `Sequential` クラスと同じ機能を提供します。

```{.python .input}
%%tab mxnet
class MySequential(nn.Block):
    def add(self, block):
        # Here, block is an instance of a Block subclass, and we assume that
        # it has a unique name. We save it in the member variable _children of
        # the Block class, and its type is OrderedDict. When the MySequential
        # instance calls the initialize method, the system automatically
        # initializes all members of _children
        self._children[block.name] = block

    def forward(self, X):
        # OrderedDict guarantees that members will be traversed in the order
        # they were added
        for block in self._children.values():
            X = block(X)
        return X
```

```{.python .input}
%%tab pytorch
class MySequential(nn.Module):
    def __init__(self, *args):
        super().__init__()
        for idx, module in enumerate(args):
            self.add_module(str(idx), module)

    def forward(self, X):
        for module in self.children():            
            X = module(X)
        return X
```

```{.python .input}
%%tab tensorflow
class MySequential(tf.keras.Model):
    def __init__(self, *args):
        super().__init__()
        self.modules = args

    def call(self, X):
        for module in self.modules:
            X = module(X)
        return X
```

```{.python .input}
%%tab jax
class MySequential(nn.Module):
    modules: List

    def __call__(self, X):
        for module in self.modules:
            X = module(X)
        return X
```

:begin_tab:`mxnet`
`add` メソッドは、単一のブロックを順序付き辞書 `_children` に追加します。
なぜすべてのGluon `Block` が `_children` 属性を持ち、
なぜそれを単なるPythonのリストではなく使うのか
不思議に思うかもしれません。
要するに、`_children` の最大の利点は、
ブロックのパラメータ初期化の際に、
Gluon が `_children` 辞書の中を見て、
パラメータを初期化する必要のある
サブブロックを見つけられることです。
:end_tab:

:begin_tab:`pytorch`
`__init__` メソッドでは、`add_modules` メソッドを呼び出してすべてのモジュールを追加します。これらのモジュールは後で `children` メソッドからアクセスできます。
このようにしてシステムは追加されたモジュールを把握し、
各モジュールのパラメータを適切に初期化します。
:end_tab:

`MySequential` の順伝播メソッドが呼び出されると、
追加された各モジュールが
追加された順序で実行されます。
これで、`MySequential` クラスを使って
MLPを再実装できます。

```{.python .input}
%%tab mxnet
net = MySequential()
net.add(nn.Dense(256, activation='relu'))
net.add(nn.Dense(10))
net.initialize()
net(X).shape
```

```{.python .input}
%%tab pytorch
net = MySequential(nn.LazyLinear(256), nn.ReLU(), nn.LazyLinear(10))
net(X).shape
```

```{.python .input}
%%tab tensorflow
net = MySequential(
    tf.keras.layers.Dense(units=256, activation=tf.nn.relu),
    tf.keras.layers.Dense(10))
net(X).shape
```

```{.python .input}
%%tab jax
net = MySequential([nn.Dense(256), nn.relu, nn.Dense(10)])
params = net.init(d2l.get_key(), X)
net.apply(params, X).shape
```

`MySequential` のこの使い方は、
以前 `Sequential` クラスについて書いたコード
(:numref:`sec_mlp` で説明) とまったく同じであることに注意してください。


## [**順伝播メソッド内でコードを実行する**]

`Sequential` クラスはモデル構築を簡単にし、
自分でクラスを定義しなくても
新しいアーキテクチャを組み立てられるようにしてくれます。
しかし、すべてのアーキテクチャが単純な数珠つなぎとは限りません。
より高い柔軟性が必要な場合には、
独自のブロックを定義したくなります。
たとえば、順伝播メソッドの中で
Pythonの制御フローを実行したいことがあるでしょう。
さらに、あらかじめ定義されたニューラルネットワーク層に頼るだけでなく、
任意の数学演算を行いたいこともあります。

これまでのところ、
ネットワーク内のすべての演算は
ネットワークの活性化値と
そのパラメータに対して作用してきました。
しかし時には、
前の層の結果でも更新可能なパラメータでもない項を
組み込みたいことがあります。
これらを*定数パラメータ*と呼びます。
たとえば、
$f(\mathbf{x},\mathbf{w}) = c \cdot \mathbf{w}^\top \mathbf{x}$ という関数を計算する層が欲しいとしましょう。
ここで $\mathbf{x}$ は入力、$\mathbf{w}$ はパラメータ、
$c$ は最適化中に更新されない指定の定数です。
これを次のように `FixedHiddenMLP` クラスとして実装します。

```{.python .input}
%%tab mxnet
class FixedHiddenMLP(nn.Block):
    def __init__(self):
        super().__init__()
        # Random weight parameters created with the get_constant method
        # are not updated during training (i.e., constant parameters)
        self.rand_weight = self.params.get_constant(
            'rand_weight', np.random.uniform(size=(20, 20)))
        self.dense = nn.Dense(20, activation='relu')

    def forward(self, X):
        X = self.dense(X)
        # Use the created constant parameters, as well as the relu and dot
        # functions
        X = npx.relu(np.dot(X, self.rand_weight.data()) + 1)
        # Reuse the fully connected layer. This is equivalent to sharing
        # parameters with two fully connected layers
        X = self.dense(X)
        # Control flow
        while np.abs(X).sum() > 1:
            X /= 2
        return X.sum()
```

```{.python .input}
%%tab pytorch
class FixedHiddenMLP(nn.Module):
    def __init__(self):
        super().__init__()
        # Random weight parameters that will not compute gradients and
        # therefore keep constant during training
        self.rand_weight = torch.rand((20, 20))
        self.linear = nn.LazyLinear(20)

    def forward(self, X):
        X = self.linear(X)        
        X = F.relu(X @ self.rand_weight + 1)
        # Reuse the fully connected layer. This is equivalent to sharing
        # parameters with two fully connected layers
        X = self.linear(X)
        # Control flow
        while X.abs().sum() > 1:
            X /= 2
        return X.sum()
```

```{.python .input}
%%tab tensorflow
class FixedHiddenMLP(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.flatten = tf.keras.layers.Flatten()
        # Random weight parameters created with tf.constant are not updated
        # during training (i.e., constant parameters)
        self.rand_weight = tf.constant(tf.random.uniform((20, 20)))
        self.dense = tf.keras.layers.Dense(20, activation=tf.nn.relu)

    def call(self, inputs):
        X = self.flatten(inputs)
        # Use the created constant parameters, as well as the relu and
        # matmul functions
        X = tf.nn.relu(tf.matmul(X, self.rand_weight) + 1)
        # Reuse the fully connected layer. This is equivalent to sharing
        # parameters with two fully connected layers
        X = self.dense(X)
        # Control flow
        while tf.reduce_sum(tf.math.abs(X)) > 1:
            X /= 2
        return tf.reduce_sum(X)
```

```{.python .input}
%%tab jax
class FixedHiddenMLP(nn.Module):
    # Random weight parameters that will not compute gradients and
    # therefore keep constant during training
    rand_weight: jnp.array = jax.random.uniform(d2l.get_key(), (20, 20))

    def setup(self):
        self.dense = nn.Dense(20)

    def __call__(self, X):
        X = self.dense(X)
        X = nn.relu(X @ self.rand_weight + 1)
        # Reuse the fully connected layer. This is equivalent to sharing
        # parameters with two fully connected layers
        X = self.dense(X)
        # Control flow
        while jnp.abs(X).sum() > 1:
            X /= 2
        return X.sum()
```

このモデルでは、
重み（`self.rand_weight`）が
インスタンス化時にランダムに初期化され、その後は定数となる
隠れ層を実装しています。
この重みはモデルパラメータではないため、
逆伝播によって更新されることはありません。
その後、ネットワークはこの「固定」層の出力を
全結合層へ通します。

出力を返す前に、
私たちのモデルは少し変わったことをしました。
$\ell_1$ ノルムが1より大きいかどうかを条件にした while ループを回し、
条件を満たすまで出力ベクトルを2で割り続けました。
最後に、`X` の各要素の和を返しました。
私たちの知る限り、標準的なニューラルネットワークで
この操作を行うものはありません。
この特定の操作が実世界のどんなタスクにも
役立つとは限りません。
ここでの目的は、任意のコードを
ニューラルネットワーク計算の流れに
組み込む方法を示すことだけです。

```{.python .input}
%%tab pytorch, mxnet, tensorflow
net = FixedHiddenMLP()
if tab.selected('mxnet'):
    net.initialize()
net(X)
```

```{.python .input}
%%tab jax
net = FixedHiddenMLP()
params = net.init(d2l.get_key(), X)
net.apply(params, X)
```

さまざまなモジュールの組み立て方を
[**組み合わせて使うこともできます。**]
次の例では、モジュールを
いくつか創造的な方法で入れ子にしています。

```{.python .input}
%%tab mxnet
class NestMLP(nn.Block):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.net = nn.Sequential()
        self.net.add(nn.Dense(64, activation='relu'),
                     nn.Dense(32, activation='relu'))
        self.dense = nn.Dense(16, activation='relu')

    def forward(self, X):
        return self.dense(self.net(X))

chimera = nn.Sequential()
chimera.add(NestMLP(), nn.Dense(20), FixedHiddenMLP())
chimera.initialize()
chimera(X)
```

```{.python .input}
%%tab pytorch
class NestMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.LazyLinear(64), nn.ReLU(),
                                 nn.LazyLinear(32), nn.ReLU())
        self.linear = nn.LazyLinear(16)

    def forward(self, X):
        return self.linear(self.net(X))

chimera = nn.Sequential(NestMLP(), nn.LazyLinear(20), FixedHiddenMLP())
chimera(X)
```

```{.python .input}
%%tab tensorflow
class NestMLP(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.net = tf.keras.Sequential()
        self.net.add(tf.keras.layers.Dense(64, activation=tf.nn.relu))
        self.net.add(tf.keras.layers.Dense(32, activation=tf.nn.relu))
        self.dense = tf.keras.layers.Dense(16, activation=tf.nn.relu)

    def call(self, inputs):
        return self.dense(self.net(inputs))

chimera = tf.keras.Sequential()
chimera.add(NestMLP())
chimera.add(tf.keras.layers.Dense(20))
chimera.add(FixedHiddenMLP())
chimera(X)
```

```{.python .input}
%%tab jax
class NestMLP(nn.Module):
    def setup(self):
        self.net = nn.Sequential([nn.Dense(64), nn.relu,
                                  nn.Dense(32), nn.relu])
        self.dense = nn.Dense(16)

    def __call__(self, X):
        return self.dense(self.net(X))


chimera = nn.Sequential([NestMLP(), nn.Dense(20), FixedHiddenMLP()])
params = chimera.init(d2l.get_key(), X)
chimera.apply(params, X)
```

## 要約

個々の層はモジュールになり得ます。
多くの層が1つのモジュールを構成できます。
多くのモジュールが1つのモジュールを構成できます。

モジュールはコードを含むことができます。
モジュールは、パラメータ初期化や逆伝播を含む多くの雑務を引き受けてくれます。
層やモジュールの順次連結は `Sequential` モジュールによって処理されます。


## 演習

1. `MySequential` をPythonのリストにモジュールを保存するよう変更すると、どのような問題が起こりますか？
1. 2つのモジュール、たとえば `net1` と `net2` を引数に取り、順伝播で両方のネットワークの連結出力を返すモジュールを実装してください。これは*並列モジュール*とも呼ばれます。
1. 同じネットワークの複数インスタンスを連結したいとします。同じモジュールの複数インスタンスを生成するファクトリ関数を実装し、それを使ってより大きなネットワークを構築してください。\n
