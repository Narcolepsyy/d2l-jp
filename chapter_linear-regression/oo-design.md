{.python .input}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow', 'jax'])
```

# 実装のためのオブジェクト指向設計
:label:`sec_oo-design`

線形回帰の導入では、
データ、モデル、損失関数、
最適化アルゴリズムを含む
さまざまな構成要素を順に見てきた。
実際、
線形回帰は
機械学習モデルの中でも最も単純なものの一つである。
しかし、その学習には、
この本の他のモデルでも必要となる多くの同じ構成要素が使われる。
したがって、
実装の詳細に入る前に、
ここで使ういくつかのAPIを
設計しておく価値がある。
深層学習の構成要素を
オブジェクトとして扱えば、
それらのオブジェクトと相互作用を定義する
クラスを作ることから始められる。
この実装のためのオブジェクト指向設計は、
説明を大幅に整理するだけでなく、実際のプロジェクトでも有用にお使いいただけるだろう。


[PyTorch Lightning](https://www.pytorchlightning.ai/) のようなオープンソースライブラリに着想を得て、
大まかには次の3つのクラスを用意したいと考える。
(i) `Module` はモデル、損失、最適化手法を含む。
(ii) `DataModule` は学習用と検証用のデータローダーを提供する。
(iii) これら2つのクラスは `Trainer` クラスによって統合され、さまざまなハードウェアプラットフォーム上でモデルを学習できるようにする。
この本のコードの大部分は `Module` と `DataModule` を拡張したものである。`Trainer` クラスに触れるのは、GPU、CPU、並列学習、最適化アルゴリズムを扱うときだけである。

```{.python .input}
%%tab mxnet
import time
import numpy as np
from d2l import mxnet as d2l
from mxnet.gluon import nn
```

```{.python .input}
%%tab pytorch
import time
import numpy as np
from d2l import torch as d2l
import torch
from torch import nn
```

```{.python .input}
%%tab tensorflow
import time
import numpy as np
from d2l import tensorflow as d2l
import tensorflow as tf
```

```{.python .input}
%%tab jax
from dataclasses import field
from d2l import jax as d2l
from flax import linen as nn
from flax.training import train_state
from jax import numpy as jnp
import numpy as np
import jax
import time
from typing import Any
```

## ユーティリティ
:label:`oo-design-utilities`

Jupyterノートブックでオブジェクト指向プログラミングを簡単にするために、いくつかのユーティリティが必要である。課題の一つは、クラス定義がかなり長いコードブロックになりがちなことである。ノートブックの読みやすさを保つには、説明を挟みながら短いコード片を並べる必要があるが、これはPythonライブラリで一般的なプログラミングスタイルとは相性がよくない。最初の
ユーティリティ関数は、クラスが作成された*後*に、その関数をクラスのメソッドとして登録できるようにする。実際、クラスのインスタンスを作成した*後*であっても可能である。これにより、クラスの実装を複数のコードブロックに分割できる。

```{.python .input}
%%tab all
def add_to_class(Class):  #@save
    """Register functions as methods in created class."""
    def wrapper(obj):
        setattr(Class, obj.__name__, obj)
    return wrapper
```

使い方を簡単に見てみよう。`do` というメソッドを持つクラス `A` を実装したいとする。`A` と `do` のコードを同じコードブロックに書く代わりに、まずクラス `A` を宣言してインスタンス `a` を作成できる。

```{.python .input}
%%tab all
class A:
    def __init__(self):
        self.b = 1

a = A()
```

次に、通常どおり `do` メソッドを定義するが、`A` クラスのスコープ内では定義しない。その代わり、このメソッドを `add_to_class` でデコレートし、引数としてクラス `A` を渡す。こうすることで、このメソッドは、あたかも `A` の定義の一部として含まれていたかのように、`A` のメンバー変数へアクセスできる。インスタンス `a` に対して呼び出すとどうなるか見てみよう。

```{.python .input}
%%tab all
@add_to_class(A)
def do(self):
    print('Class attribute "b" is', self.b)

a.do()
```

2つ目のユーティリティは、クラスの `__init__` メソッドのすべての引数をクラス属性として保存するクラスである。これにより、追加のコードを書かずにコンストラクタの呼び出しシグネチャを暗黙的に拡張できる。

```{.python .input}
%%tab all
class HyperParameters:  #@save
    """The base class of hyperparameters."""
    def save_hyperparameters(self, ignore=[]):
        raise NotImplemented
```

その実装は :numref:`sec_utils` に回す。使うには、`HyperParameters` を継承し、`__init__` メソッド内で `save_hyperparameters` を呼び出すクラスを定義する。

```{.python .input}
%%tab all
# Call the fully implemented HyperParameters class saved in d2l
class B(d2l.HyperParameters):
    def __init__(self, a, b, c):
        self.save_hyperparameters(ignore=['c'])
        print('self.a =', self.a, 'self.b =', self.b)
        print('There is no self.c =', not hasattr(self, 'c'))

b = B(a=1, b=2, c=3)
```

最後のユーティリティは、実験の進行状況をその場で対話的に描画できるようにするものである。はるかに強力で複雑な [TensorBoard](https://www.tensorflow.org/tensorboard) に敬意を表して、これを `ProgressBoard` と名付ける。実装は :numref:`sec_utils` に回す。ここでは、まず動作だけ見てみよう。

`draw` メソッドは、図中に点 `(x, y)` を描画し、`label` を凡例に指定する。オプションの `every_n` は、図に $1/n$ 個の点だけを表示することで線を滑らかにする。それらの値は、元の図における $n$ 個の近傍点から平均される。

```{.python .input}
%%tab all
class ProgressBoard(d2l.HyperParameters):  #@save
    """The board that plots data points in animation."""
    def __init__(self, xlabel=None, ylabel=None, xlim=None,
                 ylim=None, xscale='linear', yscale='linear',
                 ls=['-', '--', '-.', ':'], colors=['C0', 'C1', 'C2', 'C3'],
                 fig=None, axes=None, figsize=(3.5, 2.5), display=True):
        self.save_hyperparameters()

    def draw(self, x, y, label, every_n=1):
        raise NotImplemented
```

次の例では、異なる滑らかさで `sin` と `cos` を描画する。このコードブロックを実行すると、線がアニメーションのように伸びていくのが見えるはずである。

```{.python .input}
%%tab all
board = d2l.ProgressBoard('x')
for x in np.arange(0, 10, 0.1):
    board.draw(x, np.sin(x), 'sin', every_n=2)
    board.draw(x, np.cos(x), 'cos', every_n=10)
```

## モデル
:label:`subsec_oo-design-models`

`Module` クラスは、これから実装するすべてのモデルの基底クラスである。少なくとも3つのメソッドが必要である。1つ目の `__init__` は学習可能なパラメータを保存し、`training_step` メソッドはデータバッチを受け取って損失値を返す。最後に、`configure_optimizers` は学習可能なパラメータを更新するために使う最適化手法、またはそのリストを返す。必要に応じて、評価指標を報告するための `validation_step` を定義できる。
出力の計算コードを別の `forward` メソッドに分けておくと、再利用しやすくなることがある。

:begin_tab:`jax`
Python 3.7 で [dataclasses](https://docs.python.org/3/library/dataclasses.html)
が導入されてから、`@dataclass` を付けたクラスには `__init__` や `__repr__` のような魔法メソッドが自動的に追加される。メンバー変数は型注釈を使って定義する。Flax のすべてのモジュールは Python 3.7 の dataclass である。
:end_tab:

```{.python .input}
%%tab pytorch
class Module(d2l.nn_Module, d2l.HyperParameters):  #@save
    """The base class of models."""
    def __init__(self, plot_train_per_epoch=2, plot_valid_per_epoch=1):
        super().__init__()
        self.save_hyperparameters()
        self.board = ProgressBoard()

    def loss(self, y_hat, y):
        raise NotImplementedError

    def forward(self, X):
        assert hasattr(self, 'net'), 'Neural network is defined'
        return self.net(X)

    def plot(self, key, value, train):
        """Plot a point in animation."""
        assert hasattr(self, 'trainer'), 'Trainer is not inited'
        self.board.xlabel = 'epoch'
        if train:
            x = self.trainer.train_batch_idx / \
                self.trainer.num_train_batches
            n = self.trainer.num_train_batches / \
                self.plot_train_per_epoch
        else:
            x = self.trainer.epoch + 1
            n = self.trainer.num_val_batches / \
                self.plot_valid_per_epoch
        self.board.draw(x, d2l.numpy(d2l.to(value, d2l.cpu())),
                        ('train_' if train else 'val_') + key,
                        every_n=int(n))

    def training_step(self, batch):
        l = self.loss(self(*batch[:-1]), batch[-1])
        self.plot('loss', l, train=True)
        return l

    def validation_step(self, batch):
        l = self.loss(self(*batch[:-1]), batch[-1])
        self.plot('loss', l, train=False)

    def configure_optimizers(self):
        raise NotImplementedError
```

```{.python .input}
%%tab mxnet, tensorflow, jax
class Module(d2l.nn_Module, d2l.HyperParameters):  #@save
    """The base class of models."""
    if tab.selected('mxnet', 'tensorflow'):
        def __init__(self, plot_train_per_epoch=2, plot_valid_per_epoch=1):
            super().__init__()
            self.save_hyperparameters()
            self.board = ProgressBoard()
        if tab.selected('tensorflow'):
            self.training = None

    if tab.selected('jax'):
        # No need for save_hyperparam when using Python dataclass
        plot_train_per_epoch: int = field(default=2, init=False)
        plot_valid_per_epoch: int = field(default=1, init=False)
        # Use default_factory to make sure new plots are generated on each run
        board: ProgressBoard = field(default_factory=lambda: ProgressBoard(),
                                     init=False)

    def loss(self, y_hat, y):
        raise NotImplementedError

    if tab.selected('mxnet', 'tensorflow'):
        def forward(self, X):
            assert hasattr(self, 'net'), 'Neural network is defined'
            return self.net(X)

    if tab.selected('tensorflow'):
        def call(self, X, *args, **kwargs):
            if kwargs and "training" in kwargs:
                self.training = kwargs['training']
            return self.forward(X, *args)

    if tab.selected('jax'):
        # JAX & Flax do not have a forward-method-like syntax. Flax uses setup
        # and built-in __call__ magic methods for forward pass. Adding here
        # for consistency
        def forward(self, X, *args, **kwargs):
            assert hasattr(self, 'net'), 'Neural network is defined'
            return self.net(X, *args, **kwargs)

        def __call__(self, X, *args, **kwargs):
            return self.forward(X, *args, **kwargs)

    def plot(self, key, value, train):
        """Plot a point in animation."""
        assert hasattr(self, 'trainer'), 'Trainer is not inited'
        self.board.xlabel = 'epoch'
        if train:
            x = self.trainer.train_batch_idx / \
                self.trainer.num_train_batches
            n = self.trainer.num_train_batches / \
                self.plot_train_per_epoch
        else:
            x = self.trainer.epoch + 1
            n = self.trainer.num_val_batches / \
                self.plot_valid_per_epoch
        if tab.selected('mxnet', 'tensorflow'):
            self.board.draw(x, d2l.numpy(value), (
                'train_' if train else 'val_') + key, every_n=int(n))
        if tab.selected('jax'):
            self.board.draw(x, d2l.to(value, d2l.cpu()),
                            ('train_' if train else 'val_') + key,
                            every_n=int(n))

    if tab.selected('mxnet', 'tensorflow'):
        def training_step(self, batch):
            l = self.loss(self(*batch[:-1]), batch[-1])
            self.plot('loss', l, train=True)
            return l

        def validation_step(self, batch):
            l = self.loss(self(*batch[:-1]), batch[-1])
            self.plot('loss', l, train=False)

    if tab.selected('jax'):
        def training_step(self, params, batch, state):
            l, grads = jax.value_and_grad(self.loss)(params, batch[:-1],
                                                     batch[-1], state)
            self.plot("loss", l, train=True)
            return l, grads

        def validation_step(self, params, batch, state):
            l = self.loss(params, batch[:-1], batch[-1], state)
            self.plot('loss', l, train=False)
        
        def apply_init(self, dummy_input, key):
            """To be defined later in :numref:`sec_lazy_init`"""
            raise NotImplementedError

    def configure_optimizers(self):
        raise NotImplementedError
```

:begin_tab:`mxnet`
`Module` が Gluon のニューラルネットワークの基底クラスである `nn.Block` のサブクラスであることに気づくかもしれない。
これはニューラルネットワークを扱うための便利な機能を提供する。たとえば、`forward(self, X)` のような `forward` メソッドを定義すると、インスタンス `a` に対して `a(X)` と書くだけでこのメソッドを呼び出せる。これは、組み込みの `__call__` メソッドが `forward` を呼び出すためである。`nn.Block` についての詳細や例は :numref:`sec_model_construction` を参照されたい。
:end_tab:

:begin_tab:`pytorch`
`Module` が PyTorch のニューラルネットワークの基底クラスである `nn.Module` のサブクラスであることに気づくかもしれない。
これはニューラルネットワークを扱うための便利な機能を提供する。たとえば、`forward(self, X)` のような `forward` メソッドを定義すると、インスタンス `a` に対して `a(X)` と書くだけでこのメソッドを呼び出せる。これは、組み込みの `__call__` メソッドが `forward` を呼び出すためである。`nn.Module` についての詳細や例は :numref:`sec_model_construction` を参照されたい。
:end_tab:

:begin_tab:`tensorflow`
`Module` が TensorFlow のニューラルネットワークの基底クラスである `tf.keras.Model` のサブクラスであることに気づくかもしれない。
これはニューラルネットワークを扱うための便利な機能を提供する。たとえば、組み込みの `__call__` メソッドは `call` メソッドを呼び出す。ここでは、`call` を `forward` メソッドに転送し、その引数をクラス属性として保存している。これは、コードを他のフレームワークの実装により近づけるためである。
:end_tab:

:begin_tab:`jax`
`Module` が Flax のニューラルネットワークの基底クラスである `linen.Module` のサブクラスであることに気づくかもしれない。
これはニューラルネットワークを扱うための便利な機能を提供する。たとえば、モデルパラメータを扱い、コードを簡潔にする `nn.compact` デコレータを提供し、`__call__` メソッドなどを呼び出す。
ここでも `__call__` を `forward` メソッドに転送している。これは、コードを他のフレームワークの実装により近づけるためである。
:end_tab:

##  データ
:label:`oo-design-data`

`DataModule` クラスはデータのための基底クラスである。かなり頻繁に、`__init__` メソッドはデータの準備に使われる。必要ならダウンロードや前処理も含まれる。`train_dataloader` は学習データセット用のデータローダーを返す。データローダーは、使われるたびにデータバッチを1つ返す（Pythonの）ジェネレータである。このバッチは `Module` の `training_step` メソッドに渡され、損失を計算する。検証データセット用のローダーを返す `val_dataloader` も任意で用意できる。こちらも同様に動作するが、`Module` の `validation_step` メソッドに渡すデータバッチを返す。

```{.python .input}
%%tab all
class DataModule(d2l.HyperParameters):  #@save
    """The base class of data."""
    if tab.selected('mxnet', 'pytorch'):
        def __init__(self, root='../data', num_workers=4):
            self.save_hyperparameters()

    if tab.selected('tensorflow', 'jax'):
        def __init__(self, root='../data'):
            self.save_hyperparameters()

    def get_dataloader(self, train):
        raise NotImplementedError

    def train_dataloader(self):
        return self.get_dataloader(train=True)

    def val_dataloader(self):
        return self.get_dataloader(train=False)
```

## 学習
:label:`oo-design-training`

:begin_tab:`pytorch, mxnet, tensorflow`
`Trainer` クラスは、`DataModule` で指定されたデータを使って `Module` クラスの学習可能パラメータを学習する。中心となるメソッドは `fit` で、2つの引数を受け取る。`model` は `Module` のインスタンス、`data` は `DataModule` のインスタンスである。その後、データセット全体を `max_epochs` 回繰り返してモデルを学習する。これまでと同様、このメソッドの実装は後の章に回す。
:end_tab:

:begin_tab:`jax`
`Trainer` クラスは、`DataModule` で指定されたデータを使って学習可能パラメータ `params` を学習する。中心となるメソッドは `fit` で、3つの引数を受け取る。`model` は `Module` のインスタンス、`data` は `DataModule` のインスタンス、`key` は JAX の `PRNGKeyArray` である。ここではインターフェースを簡単にするために `key` 引数を任意にしているが、JAX と Flax では、モデルパラメータをルートキーで常に渡して初期化することが推奨される。その後、データセット全体を `max_epochs` 回繰り返してモデルを学習する。これまでと同様、このメソッドの実装は後の章に回す。
:end_tab:

```{.python .input}
%%tab all
class Trainer(d2l.HyperParameters):  #@save
    """The base class for training models with data."""
    def __init__(self, max_epochs, num_gpus=0, gradient_clip_val=0):
        self.save_hyperparameters()
        assert num_gpus == 0, 'No GPU support yet'

    def prepare_data(self, data):
        self.train_dataloader = data.train_dataloader()
        self.val_dataloader = data.val_dataloader()
        self.num_train_batches = len(self.train_dataloader)
        self.num_val_batches = (len(self.val_dataloader)
                                if self.val_dataloader is not None else 0)

    def prepare_model(self, model):
        model.trainer = self
        model.board.xlim = [0, self.max_epochs]
        self.model = model

    if tab.selected('pytorch', 'mxnet', 'tensorflow'):
        def fit(self, model, data):
            self.prepare_data(data)
            self.prepare_model(model)
            self.optim = model.configure_optimizers()
            self.epoch = 0
            self.train_batch_idx = 0
            self.val_batch_idx = 0
            for self.epoch in range(self.max_epochs):
                self.fit_epoch()

    if tab.selected('jax'):
        def fit(self, model, data, key=None):
            self.prepare_data(data)
            self.prepare_model(model)
            self.optim = model.configure_optimizers()

            if key is None:
                root_key = d2l.get_key()
            else:
                root_key = key
            params_key, dropout_key = jax.random.split(root_key)
            key = {'params': params_key, 'dropout': dropout_key}

            dummy_input = next(iter(self.train_dataloader))[:-1]
            variables = model.apply_init(dummy_input, key=key)
            params = variables['params']

            if 'batch_stats' in variables.keys():
                # Here batch_stats will be used later (e.g., for batch norm)
                batch_stats = variables['batch_stats']
            else:
                batch_stats = {}

            # Flax uses optax under the hood for a single state obj TrainState.
            # More will be discussed later in the dropout and batch
            # normalization section
            class TrainState(train_state.TrainState):
                batch_stats: Any
                dropout_rng: jax.random.PRNGKeyArray

            self.state = TrainState.create(apply_fn=model.apply,
                                           params=params,
                                           batch_stats=batch_stats,
                                           dropout_rng=dropout_key,
                                           tx=model.configure_optimizers())
            self.epoch = 0
            self.train_batch_idx = 0
            self.val_batch_idx = 0
            for self.epoch in range(self.max_epochs):
                self.fit_epoch()

    def fit_epoch(self):
        raise NotImplementedError
```

## 要約

深層学習の今後の実装に向けた
オブジェクト指向設計を強調するために、
上のクラス群は、オブジェクトが
どのようにデータを保存し、互いにやり取りするかを示しているにすぎない。
この本の残りでは、
`@add_to_class` などを通じて、
これらのクラスの実装をさらに充実させていく。
さらに、
これらの完全実装済みクラスは
[D2Lライブラリ](https://github.com/d2l-ai/d2l-en/tree/master/d2l) に保存されており、
深層学習の構造化モデリングを容易にする*軽量ツールキット*である。
特に、プロジェクト間で多くの構成要素をほとんど変更せずに再利用できるようにする。たとえば、最適化手法だけ、モデルだけ、データセットだけを置き換えることができる。
この程度のモジュール性は、本書全体を通して簡潔さと単純さの面で大きな効果をもたらす（そのためにこれを追加した）。そして、あなた自身のプロジェクトでも同じ効果を発揮するだろう。 


## 演習

1. [D2Lライブラリ](https://github.com/d2l-ai/d2l-en/tree/master/d2l) に保存されている上記クラスの完全実装を見つけよ。深層学習モデリングにもう少し慣れたら、実装を詳しく読むことを強く勧める。
1. `B` クラスの `save_hyperparameters` 文を削除せよ。それでも `self.a` と `self.b` を表示できるか？ 任意: `HyperParameters` クラスの完全実装まで読み進めたなら、なぜそうなるのか説明できるか？
