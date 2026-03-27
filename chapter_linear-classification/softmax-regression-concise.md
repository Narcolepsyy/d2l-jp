{.python .input  n=1}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow', 'jax'])
```

# Softmax回帰の簡潔な実装
:label:`sec_softmax_concise`



高水準の深層学習フレームワークが
線形回帰の実装を容易にしたのと同様に
(:numref:`sec_linear_concise` を参照)、
ここでも同様に便利です。

```{.python .input}
%%tab mxnet
from d2l import mxnet as d2l
from mxnet import gluon, init, npx
from mxnet.gluon import nn
npx.set_np()
```

```{.python .input}
%%tab pytorch
from d2l import torch as d2l
import torch
from torch import nn
from torch.nn import functional as F
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
from functools import partial
import jax
from jax import numpy as jnp
import optax
```

## モデルの定義

:numref:`sec_linear_concise` と同様に、
組み込み層を使って
全結合層を構成します。
その後、組み込みの `__call__` メソッドが、ネットワークを入力に適用する必要があるたびに `forward` を呼び出します。

:begin_tab:`mxnet`
入力 `X` は4階テンソルですが、
組み込みの `Dense` 層は
第1軸の次元を変えずに `X` を自動的に2階テンソルへ変換します。
:end_tab:

:begin_tab:`pytorch`
`Flatten` 層を使って、4階テンソル `X` を2階テンソルに変換します。
第1軸の次元は変えません。

:end_tab:

:begin_tab:`tensorflow`
`Flatten` 層を使って、4階テンソル `X`
を第1軸の次元を変えずに変換します。
:end_tab:

:begin_tab:`jax`
Flaxでは、`@nn.compact` デコレータを使うことで、より簡潔にネットワーククラスを記述できます。`@nn.compact` を使うと、dataclass 内で標準的な `setup` メソッドを定義することなく、単一の「順伝播」メソッドの中にネットワークのロジックをすべて書けます。
:end_tab:

```{.python .input}
%%tab pytorch
class SoftmaxRegression(d2l.Classifier):  #@save
    """The softmax regression model."""
    def __init__(self, num_outputs, lr):
        super().__init__()
        self.save_hyperparameters()
        self.net = nn.Sequential(nn.Flatten(),
                                 nn.LazyLinear(num_outputs))

    def forward(self, X):
        return self.net(X)
```

```{.python .input}
%%tab mxnet, tensorflow
class SoftmaxRegression(d2l.Classifier):  #@save
    """The softmax regression model."""
    def __init__(self, num_outputs, lr):
        super().__init__()
        self.save_hyperparameters()
        if tab.selected('mxnet'):
            self.net = nn.Dense(num_outputs)
            self.net.initialize()
        if tab.selected('tensorflow'):
            self.net = tf.keras.models.Sequential()
            self.net.add(tf.keras.layers.Flatten())
            self.net.add(tf.keras.layers.Dense(num_outputs))

    def forward(self, X):
        return self.net(X)
```

```{.python .input}
%%tab jax
class SoftmaxRegression(d2l.Classifier):  #@save
    num_outputs: int
    lr: float

    @nn.compact
    def __call__(self, X):
        X = X.reshape((X.shape[0], -1))  # Flatten
        X = nn.Dense(self.num_outputs)(X)
        return X
```

## Softmaxの再考
:label:`subsec_softmax-implementation-revisited`

:numref:`sec_softmax_scratch` では、モデルの出力を計算し、
クロスエントロピー損失を適用しました。数学的にはこれはまったく
妥当ですが、指数計算における数値的なアンダーフローとオーバーフローのため、
計算上は危険です。

softmax関数は
$\hat y_j = \frac{\exp(o_j)}{\sum_k \exp(o_k)}$
によって確率を計算することを思い出してください。
もし $o_k$ のいくつかが非常に大きい、つまり絶対値の大きな正の値であれば、
$\exp(o_k)$ は特定のデータ型で表現できる最大値よりも大きくなるかもしれません。
これを *オーバーフロー* と呼びます。同様に、
引数がすべて非常に大きな負の値であれば、*アンダーフロー* が起こります。
たとえば、単精度浮動小数点数はおおよそ
$10^{-38}$ から $10^{38}$ の範囲をカバーします。したがって、$\mathbf{o}$ の最大項が
$[-90, 90]$ の範囲外にあると、結果は安定しません。
この問題を回避する方法は、すべての要素から
$\bar{o} \stackrel{\textrm{def}}{=} \max_k o_k$ を引くことです。

$$
\hat y_j = \frac{\exp o_j}{\sum_k \exp o_k} =
\frac{\exp(o_j - \bar{o}) \exp \bar{o}}{\sum_k \exp (o_k - \bar{o}) \exp \bar{o}} =
\frac{\exp(o_j - \bar{o})}{\sum_k \exp (o_k - \bar{o})}.
$$

構成上、すべての $j$ について $o_j - \bar{o} \leq 0$ であることがわかります。したがって、$q$ クラス
分類問題では、分母は区間 $[1, q]$ に収まります。さらに、
分子は1を超えないため、数値オーバーフローを防げます。数値アンダーフローは
$\exp(o_j - \bar{o})$ が数値的に $0$ と評価されるときにのみ起こります。それでも、
少し先で $\log \hat{y}_j$ を $\log 0$ として計算しようとすると問題が生じるかもしれません。
特に、逆伝播では、
忌まわしい `NaN`（Not a Number）の結果が画面いっぱいに
現れる事態に直面するかもしれません。

幸いなことに、指数関数を計算しているにもかかわらず、
最終的にはその対数を取る（クロスエントロピー損失を計算するとき）ことを
意図しているため、救われます。
softmaxとクロスエントロピーを組み合わせることで、
数値安定性の問題を完全に回避できます。次が成り立ちます。

$$
\log \hat{y}_j =
\log \frac{\exp(o_j - \bar{o})}{\sum_k \exp (o_k - \bar{o})} =
o_j - \bar{o} - \log \sum_k \exp (o_k - \bar{o}).
$$

これにより、オーバーフローとアンダーフローの両方を避けられます。
モデルの出力確率を評価したい場合に備えて、
従来のsoftmax関数も手元に置いておきたいところです。
しかし、新しい損失関数にsoftmax確率を渡す代わりに、
単に
[**ロジットを渡し、クロスエントロピー損失関数の内部でsoftmaxとその対数を
一度に計算する**]ことで、
["LogSumExp trick"](https://en.wikipedia.org/wiki/LogSumExp) のような賢い処理を行えます。

```{.python .input}
%%tab pytorch, mxnet, tensorflow
@d2l.add_to_class(d2l.Classifier)  #@save
def loss(self, Y_hat, Y, averaged=True):
    Y_hat = d2l.reshape(Y_hat, (-1, Y_hat.shape[-1]))
    Y = d2l.reshape(Y, (-1,))
    if tab.selected('mxnet'):
        fn = gluon.loss.SoftmaxCrossEntropyLoss()
        l = fn(Y_hat, Y)
        return l.mean() if averaged else l
    if tab.selected('pytorch'):
        return F.cross_entropy(
            Y_hat, Y, reduction='mean' if averaged else 'none')
    if tab.selected('tensorflow'):
        fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        return fn(Y, Y_hat)
```

```{.python .input}
%%tab jax
@d2l.add_to_class(d2l.Classifier)  #@save
@partial(jax.jit, static_argnums=(0, 5))
def loss(self, params, X, Y, state, averaged=True):
    # To be used later (e.g., for batch norm)
    Y_hat = state.apply_fn({'params': params}, *X,
                           mutable=False, rngs=None)
    Y_hat = d2l.reshape(Y_hat, (-1, Y_hat.shape[-1]))
    Y = d2l.reshape(Y, (-1,))
    fn = optax.softmax_cross_entropy_with_integer_labels
    # The returned empty dictionary is a placeholder for auxiliary data,
    # which will be used later (e.g., for batch norm)
    return (fn(Y_hat, Y).mean(), {}) if averaged else (fn(Y_hat, Y), {})
```

## 学習

次にモデルを学習します。Fashion-MNIST画像を、784次元の特徴ベクトルに平坦化して用います。

```{.python .input}
%%tab all
data = d2l.FashionMNIST(batch_size=256)
model = SoftmaxRegression(num_outputs=10, lr=0.1)
trainer = d2l.Trainer(max_epochs=10)
trainer.fit(model, data)
```

これまでと同様に、このアルゴリズムは
かなり正確な解に収束します。
今回は以前よりも少ないコード行数でそれが実現できます。


## まとめ

高水準APIは、数値安定性のような潜在的に危険な側面を
利用者からうまく隠してくれるので非常に便利です。さらに、
ごく少ないコード行数で簡潔にモデルを設計できるようにしてくれます。これは
祝福であると同時に呪いでもあります。明らかな利点は、
統計の授業を一度も受けたことのないエンジニアでさえも
非常に利用しやすくなることです（実際、彼らはこの本の想定読者の一部です）。
しかし、鋭い部分を隠すことには代償もあります。自分で新しく異なる構成要素を追加しようという
意欲が削がれやすいのです。というのも、それを行うための身体的な記憶が
ほとんど身につかないからです。さらに、フレームワークの保護用の
緩衝材がすべての例外ケースを完全には覆いきれないときに、
それを*修正*することも難しくなります。これもやはり、
慣れの不足によるものです。

そのため、以下に続く多くの実装については、
素朴な版と洗練された版の両方を確認することを強く勧めます。理解しやすさを重視していますが、
それでも実装は通常かなり高性能です（畳み込みはここでの大きな例外です）。
私たちの意図は、あなたがフレームワークでは得られない新しいものを発明したときに、
それを土台として発展させられるようにすることです。


## 演習

1. 深層学習では、FP64倍精度（非常にまれに使われる）、
FP32単精度、BFLOAT16（圧縮表現に適している）、FP16（非常に不安定）、
TF32（NVIDIAの新しい形式）、INT8など、多くの異なる数値形式が使われます。
数値アンダーフローやオーバーフローを起こさない指数関数の引数の最小値と最大値を求めなさい。
1. INT8は、$1$ から $255$ までの非ゼロ数からなる非常に制約の厳しい形式です。より多くのビットを使わずに、その動的範囲をどのように拡張できますか？通常の乗算と加算はそのまま使えますか？
1. 学習のエポック数を増やしなさい。しばらくすると検証精度が下がるのはなぜでしょうか？それをどう修正できますか？
1. 学習率を増やすと何が起こりますか？いくつかの学習率について損失曲線を比較しなさい。どれがよりうまく機能しますか？それはいつですか？
