{.python .input  n=1}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow', 'jax'])
```

# 線形回帰の簡潔な実装
:label:`sec_linear_concise`

深層学習は、この10年で一種のカンブリア爆発を経験してきました。
技術、応用、アルゴリズムの数は、過去数十年の進歩をはるかに上回っています。 
これは複数の要因が幸運にも組み合わさった結果であり、
その一つが、いくつかのオープンソース深層学習フレームワークが提供する強力で無料のツールです。
Theano :cite:`Bergstra.Breuleux.Bastien.ea.2010`,
DistBelief :cite:`Dean.Corrado.Monga.ea.2012`,
および Caffe :cite:`Jia.Shelhamer.Donahue.ea.2014`
は、広く採用されたそのようなモデルの
第一世代を代表していると言えるでしょう。
Lisp風のプログラミング体験を提供した
SN2 (Simulateur Neuristique) :cite:`Bottou.Le-Cun.1988`
のような初期の（先駆的な）研究とは対照的に、
現代のフレームワークは自動微分と
Python の利便性を提供します。
これらのフレームワークにより、勾配ベース学習アルゴリズムの実装における
反復的な作業を自動化し、モジュール化できます。

:numref:`sec_linear_scratch` では、
(i) データ保存と線形代数のためのテンソル、
および (ii) 勾配計算のための自動微分
だけに依拠しました。
実際には、データイテレータ、損失関数、最適化器、
ニューラルネットワーク層は
非常に一般的であるため、現代のライブラリはこれらの構成要素も
私たちの代わりに実装してくれます。
この節では、深層学習フレームワークの
(**高レベル API を使って**)
:numref:`sec_linear_scratch` の線形回帰モデルを
(**簡潔に実装する方法を示します**)。

```{.python .input}
%%tab mxnet
from d2l import mxnet as d2l
from mxnet import autograd, gluon, init, np, npx
from mxnet.gluon import nn
npx.set_np()
```

```{.python .input}
%%tab pytorch
from d2l import torch as d2l
import numpy as np
import torch
from torch import nn
```

```{.python .input}
%%tab tensorflow
from d2l import tensorflow as d2l
import numpy as np
import tensorflow as tf
```

```{.python .input}
%%tab jax
from d2l import jax as d2l
from flax import linen as nn
import jax
from jax import numpy as jnp
import optax
```

## モデルの定義

:numref:`sec_linear_scratch` で
線形回帰をスクラッチから実装したときには、
モデルパラメータを明示的に定義し、
基本的な線形代数演算を使って出力を生成する計算を
コード化しました。
これは*知っておくべき*ことです。
しかし、モデルがより複雑になり、
しかもそれをほぼ毎日行わなければならなくなると、
助けがあることのありがたさが分かるでしょう。
状況は、ブログをスクラッチから自作するのに似ています。
一度や二度ならやりがいがあり、学びもありますが、
車輪の再発明に1か月費やすようでは、
優秀な Web 開発者とは言えません。

標準的な演算については、
[**フレームワークにあらかじめ定義された層を使う**]ことができ、
実装を気にするよりも、
モデルを構成する層そのものに集中できます。
:numref:`fig_single_neuron` で説明した
単層ネットワークのアーキテクチャを思い出してください。
この層は *全結合* と呼ばれます。
なぜなら、その入力の各要素が
行列—ベクトル積によって
各出力に接続されているからです。

:begin_tab:`mxnet`
Gluon では、全結合層は `Dense` クラスで定義されます。
ここでは単一のスカラー出力だけを生成したいので、
その数を 1 に設定します。
なお、利便性のために、
Gluon では各層の入力形状を指定する必要はありません。
したがって、この線形層に何個の入力が入るかを
Gluon に伝える必要はありません。
最初にモデルへデータを通すとき、
たとえば後で `net(X)` を実行したときに、
Gluon は各層への入力数を自動的に推論し、
それに応じた正しいモデルを生成します。
この仕組みについては後で詳しく説明します。
:end_tab:

:begin_tab:`pytorch`
PyTorch では、全結合層は `Linear` クラスと `LazyLinear` クラス（バージョン 1.8.0 以降で利用可能）で定義されます。 
後者は、ユーザーが*単に*
出力次元だけを指定できるようにしますが、
前者ではそれに加えて、
この層に何個の入力が入るかも指定する必要があります。
入力形状の指定は不便であり、（畳み込み層のように）
非自明な計算を要することがあります。
そのため、簡潔さのために、可能な限り
このような「遅延」層を使います。 
:end_tab:

:begin_tab:`tensorflow`
Keras では、全結合層は `Dense` クラスで定義されます。
ここでは単一のスカラー出力だけを生成したいので、
その数を 1 に設定します。
なお、利便性のために、
Keras では各層の入力形状を指定する必要はありません。
この線形層に何個の入力が入るかを
Keras に伝える必要はありません。
最初にモデルへデータを通そうとしたとき、
たとえば後で `net(X)` を実行したときに、
Keras は各層への入力数を自動的に推論します。
この仕組みについては後で詳しく説明します。
:end_tab:

```{.python .input}
%%tab pytorch, mxnet, tensorflow
class LinearRegression(d2l.Module):  #@save
    """The linear regression model implemented with high-level APIs."""
    def __init__(self, lr):
        super().__init__()
        self.save_hyperparameters()
        if tab.selected('mxnet'):
            self.net = nn.Dense(1)
            self.net.initialize(init.Normal(sigma=0.01))
        if tab.selected('tensorflow'):
            initializer = tf.initializers.RandomNormal(stddev=0.01)
            self.net = tf.keras.layers.Dense(1, kernel_initializer=initializer)
        if tab.selected('pytorch'):
            self.net = nn.LazyLinear(1)
            self.net.weight.data.normal_(0, 0.01)
            self.net.bias.data.fill_(0)
```

```{.python .input}
%%tab jax
class LinearRegression(d2l.Module):  #@save
    """The linear regression model implemented with high-level APIs."""
    lr: float

    def setup(self):
        self.net = nn.Dense(1, kernel_init=nn.initializers.normal(0.01))
```

`forward` メソッドでは、あらかじめ定義された層の組み込み `__call__` メソッドを呼び出して出力を計算するだけです。

```{.python .input}
%%tab pytorch, mxnet, tensorflow
@d2l.add_to_class(LinearRegression)  #@save
def forward(self, X):
    return self.net(X)
```

```{.python .input}
%%tab jax
@d2l.add_to_class(LinearRegression)  #@save
def forward(self, X):
    return self.net(X)
```

## 損失関数の定義

:begin_tab:`mxnet`
`loss` モジュールには、多くの有用な損失関数が定義されています。
速度と利便性のために、自前で実装することはせず、
代わりに組み込みの `loss.L2Loss` を選びます。
これが返す `loss` は
各サンプルごとの二乗誤差なので、
ミニバッチ全体で損失を平均するために `mean` を使います。
:end_tab:

:begin_tab:`pytorch`
[**`MSELoss` クラスは平均二乗誤差を計算します（:eqref:`eq_mse` の $1/2$ 因子は含みません）。**]
デフォルトでは、`MSELoss` はサンプル全体の平均損失を返します。
自前で実装するよりも高速で（しかも使いやすいです）。
:end_tab:

:begin_tab:`tensorflow`
`MeanSquaredError` クラスは平均二乗誤差を計算します（:eqref:`eq_mse` の $1/2$ 因子は含みません）。
デフォルトでは、サンプル全体の平均損失を返します。
:end_tab:

```{.python .input}
%%tab pytorch, mxnet, tensorflow
@d2l.add_to_class(LinearRegression)  #@save
def loss(self, y_hat, y):
    if tab.selected('mxnet'):
        fn = gluon.loss.L2Loss()
        return fn(y_hat, y).mean()
    if tab.selected('pytorch'):
        fn = nn.MSELoss()
        return fn(y_hat, y)
    if tab.selected('tensorflow'):
        fn = tf.keras.losses.MeanSquaredError()
        return fn(y, y_hat)
```

```{.python .input}
%%tab jax
@d2l.add_to_class(LinearRegression)  #@save
def loss(self, params, X, y, state):
    y_hat = state.apply_fn({'params': params}, *X)
    return d2l.reduce_mean(optax.l2_loss(y_hat, y))
```

## 最適化アルゴリズムの定義

:begin_tab:`mxnet`
ミニバッチ SGD はニューラルネットワークを最適化するための標準的な手法であり、
そのため Gluon は `Trainer` クラスを通じて、
このアルゴリズムのいくつかの変種をサポートしています。
Gluon の `Trainer` クラスは最適化アルゴリズムそのものを表すのに対し、
:numref:`sec_oo-design` で作成した `Trainer` クラスは
学習メソッド、すなわち最適化器を繰り返し呼び出して
モデルパラメータを更新する処理を含みます。
`Trainer` をインスタンス化するときには、
最適化対象のパラメータ（`net.collect_params()` を通じて
モデル `net` から取得可能）、
使用したい最適化アルゴリズム（`sgd`）、
および最適化アルゴリズムに必要な
ハイパーパラメータの辞書を指定します。
:end_tab:

:begin_tab:`pytorch`
ミニバッチ SGD はニューラルネットワークを最適化するための標準的な手法であり、
そのため PyTorch は `optim` モジュールで
このアルゴリズムのいくつかの変種をサポートしています。
(**`SGD` インスタンスを生成するときには、**)
最適化対象のパラメータ（モデルの `self.parameters()` から取得可能）と、
最適化アルゴリズムに必要な学習率（`self.lr`）を指定します。
:end_tab:

:begin_tab:`tensorflow`
ミニバッチ SGD はニューラルネットワークを最適化するための標準的な手法であり、
そのため Keras は `optimizers` モジュールで
このアルゴリズムのいくつかの変種をサポートしています。
:end_tab:

```{.python .input}
%%tab all
@d2l.add_to_class(LinearRegression)  #@save
def configure_optimizers(self):
    if tab.selected('mxnet'):
        return gluon.Trainer(self.collect_params(),
                             'sgd', {'learning_rate': self.lr})
    if tab.selected('pytorch'):
        return torch.optim.SGD(self.parameters(), self.lr)
    if tab.selected('tensorflow'):
        return tf.keras.optimizers.SGD(self.lr)
    if tab.selected('jax'):
        return optax.sgd(self.lr)
```

## 学習

深層学習フレームワークの高レベル API を通してモデルを表現すると、
必要なコード行数が少なくなることに気づいたかもしれません。
パラメータを個別に割り当てたり、
損失関数を定義したり、
ミニバッチ SGD を実装したりする必要はありませんでした。
より複雑なモデルを扱い始めると、
高レベル API の利点はさらに大きくなります。

基本要素がすべて揃ったので、
[**学習ループ自体はスクラッチ実装したものと同じです。**]
したがって、`fit` メソッド（:numref:`oo-design-training` で導入）を呼び出すだけで、
:numref:`sec_linear_scratch` の `fit_epoch` メソッドの実装に依存して、
モデルを学習できます。

```{.python .input}
%%tab all
model = LinearRegression(lr=0.03)
data = d2l.SyntheticRegressionData(w=d2l.tensor([2, -3.4]), b=4.2)
trainer = d2l.Trainer(max_epochs=3)
trainer.fit(model, data)
```

以下では、
[**有限データで学習して得られたモデルパラメータと
実際のパラメータを比較します**]。
パラメータにアクセスするには、
必要な層の重みとバイアスにアクセスします。
スクラッチ実装の場合と同様に、
推定されたパラメータが真の値に近いことに注意してください。

```{.python .input}
%%tab pytorch, mxnet, tensorflow
@d2l.add_to_class(LinearRegression)  #@save
def get_w_b(self):
    if tab.selected('mxnet'):
        return (self.net.weight.data(), self.net.bias.data())
    if tab.selected('pytorch'):
        return (self.net.weight.data, self.net.bias.data)
    if tab.selected('tensorflow'):
        return (self.get_weights()[0], self.get_weights()[1])

w, b = model.get_w_b()
```

```{.python .input}
%%tab jax
@d2l.add_to_class(LinearRegression)  #@save
def get_w_b(self, state):
    net = state.params['net']
    return net['kernel'], net['bias']

w, b = model.get_w_b(trainer.state)
```

```{.python .input}
print(f'error in estimating w: {data.w - d2l.reshape(w, data.w.shape)}')
print(f'error in estimating b: {data.b - b}')
```

## まとめ

この節では、本書で初めて、
MXNet :cite:`Chen.Li.Li.ea.2015`、
JAX :cite:`Frostig.Johnson.Leary.2018`、
PyTorch :cite:`Paszke.Gross.Massa.ea.2019`、
および Tensorflow :cite:`Abadi.Barham.Chen.ea.2016`
のような現代の深層学習フレームワークがもたらす利便性を活用した
深層ネットワークの実装を行いました。
データの読み込み、層の定義、
損失関数、最適化器、学習ループには
フレームワークのデフォルトを使いました。
フレームワークが必要な機能をすべて提供しているなら、
通常はそれらを使うのがよいでしょう。
これらの構成要素のライブラリ実装は
性能のために大きく最適化されており、
信頼性のために適切にテストされている傾向があるからです。
同時に、これらのモジュールは*直接実装できる*ことも
忘れないようにしてください。
これは特に、モデル開発の最前線で生きたいと願う
意欲的な研究者にとって重要です。
そこでは、現在のどのライブラリにも存在しえない
新しい構成要素を発明することになるからです。

:begin_tab:`mxnet`
Gluon では、`data` モジュールがデータ処理のためのツールを提供し、
`nn` モジュールが多数のニューラルネットワーク層を定義し、
`loss` モジュールが多くの一般的な損失関数を定義します。
さらに、`initializer` により
多様なパラメータ初期化方法にアクセスできます。
ユーザーにとって便利なことに、
次元とストレージは自動的に推論されます。
この遅延初期化の結果として、
パラメータがインスタンス化（および初期化）される前に
アクセスしようとしてはいけません。
:end_tab:

:begin_tab:`pytorch`
PyTorch では、`data` モジュールがデータ処理のためのツールを提供し、
`nn` モジュールが多数のニューラルネットワーク層と一般的な損失関数を定義します。
末尾が `_` のメソッドで値を置き換えることで、
パラメータを初期化できます。
ネットワークの入力次元を指定する必要があることに注意してください。
今のところは単純ですが、多数の層を持つ複雑なネットワークを設計したいときには、
大きな波及効果を持つ可能性があります。
これらのネットワークをどのようにパラメータ化するかを慎重に考える必要があり、
それによって移植性を確保できます。
:end_tab:

:begin_tab:`tensorflow`
TensorFlow では、`data` モジュールがデータ処理のためのツールを提供し、
`keras` モジュールが多数のニューラルネットワーク層と一般的な損失関数を定義します。
さらに、`initializers` モジュールが
モデルパラメータの初期化方法をさまざまに提供します。
ネットワークの次元とストレージは自動的に推論されます
（ただし、初期化される前にパラメータへアクセスしようとしないよう注意してください）。
:end_tab:

## 演習

1. ミニバッチ上の損失の総和を使う代わりに、ミニバッチ上の損失の平均を使うようにした場合、学習率はどのように変更する必要がありますか？
1. フレームワークのドキュメントを確認して、どの損失関数が提供されているかを見てください。特に、二乗損失を Huber のロバスト損失関数に置き換えてください。すなわち、次の損失関数を使います。
   $$l(y,y') = \begin{cases}|y-y'| -\frac{\sigma}{2} & \textrm{ if } |y-y'| > \sigma \\ \frac{1}{2 \sigma} (y-y')^2 & \textrm{ otherwise}\end{cases}$$
1. モデルの重みの勾配にはどのようにアクセスしますか？
1. 学習率とエポック数を変えると、解にどのような影響がありますか？改善し続けますか？
1. 生成するデータ量を変えると、解はどのように変わりますか？
    1. データ量の関数として、$\hat{\mathbf{w}} - \mathbf{w}$ と $\hat{b} - b$ の推定誤差をプロットしてください。ヒント: データ量は線形ではなく対数的に増やします。つまり、1000, 2000, ..., 10,000 ではなく、5, 10, 20, 50, ..., 10,000 とします。
    2. なぜヒントの提案が適切なのですか？


:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/44)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/45)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/204)
:end_tab:

:begin_tab:`jax`
[Discussions](https://discuss.d2l.ai/t/17977)
:end_tab:\n