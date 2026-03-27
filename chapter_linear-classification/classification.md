{.python .input}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow', 'jax'])
```

# 分類の基礎モデル
:label:`sec_classification`

回帰の場合、スクラッチ実装とフレームワークの機能を用いた簡潔な実装がかなり似ていることに気づいたかもしれない。分類でも同じことが言える。この本の多くのモデルは分類を扱うため、この設定を特に支援する機能を追加しておく価値がある。この節では、今後のコードを簡潔にするために、分類モデル用の基底クラスを提供する。

```{.python .input}
%%tab mxnet
from d2l import mxnet as d2l
from mxnet import autograd, np, npx, gluon
npx.set_np()
```

```{.python .input}
%%tab pytorch
from d2l import torch as d2l
import torch
```

```{.python .input}
%%tab tensorflow
from d2l import tensorflow as d2l
import tensorflow as tf
```

```{.python .input}
%%tab jax
from d2l import jax as d2l
from functools import partial
from jax import numpy as jnp
import jax
import optax
```

## `Classifier` クラス

:begin_tab:`pytorch, mxnet, tensorflow`
以下で `Classifier` クラスを定義する。`validation_step` では、検証バッチに対する損失値と分類精度の両方を報告する。`num_val_batches` バッチごとに 1 回更新を描画する。これにより、検証データ全体に対する平均損失と精度を生成できるという利点がある。最後のバッチの例数が少ない場合、これらの平均値は厳密には正しくないが、コードを簡潔に保つためにこの小さな違いは無視する。
:end_tab:


:begin_tab:`jax`
以下で `Classifier` クラスを定義する。`validation_step` では、検証バッチに対する損失値と分類精度の両方を報告する。`num_val_batches` バッチごとに 1 回更新を描画する。これにより、検証データ全体に対する平均損失と精度を生成できるという利点がある。最後のバッチの例数が少ない場合、これらの平均値は厳密には正しくないが、コードを簡潔に保つためにこの小さな違いは無視する。

また、JAX では `training_step` メソッドも再定義する。これは、後で `Classifier` を継承するすべてのモデルの損失が補助データを返すようになるためである。この補助データはバッチ正規化を用いるモデル（:numref:`sec_batch_norm` で説明）に利用できるが、それ以外の場合でも、損失が補助データを表すプレースホルダ（空の辞書）を返すようにする。
:end_tab:

```{.python .input}
%%tab pytorch, mxnet, tensorflow
class Classifier(d2l.Module):  #@save
    """分類モデルの基底クラス。"""
    def validation_step(self, batch):
        Y_hat = self(*batch[:-1])
        self.plot('loss', self.loss(Y_hat, batch[-1]), train=False)
        self.plot('acc', self.accuracy(Y_hat, batch[-1]), train=False)
```

```{.python .input}
%%tab jax
class Classifier(d2l.Module):  #@save
    """分類モデルの基底クラス。"""
    def training_step(self, params, batch, state):
        # ここで value はタプルである。BatchNorm 層を持つモデルでは
        # 損失が補助データを返す必要があるため
        value, grads = jax.value_and_grad(
            self.loss, has_aux=True)(params, batch[:-1], batch[-1], state)
        l, _ = value
        self.plot("loss", l, train=True)
        return value, grads

    def validation_step(self, params, batch, state):
        # 2つ目の戻り値は破棄する。これは BatchNorm 層を持つモデルの
        # 学習に使われる。損失も補助データを返すためである
        l, _ = self.loss(params, batch[:-1], batch[-1], state)
        self.plot('loss', l, train=False)
        self.plot('acc', self.accuracy(params, batch[:-1], batch[-1], state),
                  train=False)
```

デフォルトでは、線形回帰の文脈で行ったのと同様に、ミニバッチ上で動作する確率的勾配降下法オプティマイザを使う。

```{.python .input}
%%tab mxnet
@d2l.add_to_class(d2l.Module)  #@save
def configure_optimizers(self):
    params = self.parameters()
    if isinstance(params, list):
        return d2l.SGD(params, self.lr)
    return gluon.Trainer(params, 'sgd', {'learning_rate': self.lr})
```

```{.python .input}
%%tab pytorch
@d2l.add_to_class(d2l.Module)  #@save
def configure_optimizers(self):
    return torch.optim.SGD(self.parameters(), lr=self.lr)
```

```{.python .input}
%%tab tensorflow
@d2l.add_to_class(d2l.Module)  #@save
def configure_optimizers(self):
    return tf.keras.optimizers.SGD(self.lr)
```

```{.python .input}
%%tab jax
@d2l.add_to_class(d2l.Module)  #@save
def configure_optimizers(self):
    return optax.sgd(self.lr)
```

## 精度

予測された確率分布 `y_hat` が与えられたとき、ハードな予測を出力しなければならない場合には、通常、予測確率が最も高いクラスを選ぶ。実際、多くのアプリケーションでは何らかの選択を行う必要がある。たとえば Gmail は、メールを "Primary"、"Social"、"Updates"、"Forums"、または "Spam" に分類しなければならない。内部的には確率を推定しているかもしれないが、最終的にはクラスの中から 1 つを選ばなければならない。

予測がラベルクラス `y` と一致しているとき、それらは正解である。分類精度は、すべての予測のうち正しいものの割合である。精度を直接最適化するのは難しい場合があるが（微分不可能であるため）、しばしば私たちが最も重視する性能指標である。ベンチマークにおいては、しばしば *最も* 重視される指標である。そのため、分類器を学習するときにはほぼ常に精度を報告する。

精度は次のように計算する。まず、`y_hat` が行列である場合、2 次元目に各クラスの予測スコアが格納されていると仮定する。`argmax` を用いて、各行で最大の要素のインデックスから予測クラスを得る。次に、[**予測クラスと正解 `y` を要素ごとに比較する。**]
等価演算子 `==` はデータ型に敏感なので、`y_hat` のデータ型を `y` に合わせて変換する。
その結果は、0（偽）と 1（真）の要素を含むテンソルになる。
それらを合計すると、正しい予測の数が得られる。

```{.python .input  n=9}
%%tab pytorch, mxnet, tensorflow
@d2l.add_to_class(Classifier)  #@save
def accuracy(self, Y_hat, Y, averaged=True):
    """正しい予測の数を計算する。"""
    Y_hat = d2l.reshape(Y_hat, (-1, Y_hat.shape[-1]))
    preds = d2l.astype(d2l.argmax(Y_hat, axis=1), Y.dtype)
    compare = d2l.astype(preds == d2l.reshape(Y, -1), d2l.float32)
    return d2l.reduce_mean(compare) if averaged else compare
```

```{.python .input  n=9}
%%tab jax
@d2l.add_to_class(Classifier)  #@save
@partial(jax.jit, static_argnums=(0, 5))
def accuracy(self, params, X, Y, state, averaged=True):
    """正しい予測の数を計算する。"""
    Y_hat = state.apply_fn({'params': params,
                            'batch_stats': state.batch_stats},  # BatchNorm のみ
                           *X)
    Y_hat = d2l.reshape(Y_hat, (-1, Y_hat.shape[-1]))
    preds = d2l.astype(d2l.argmax(Y_hat, axis=1), Y.dtype)
    compare = d2l.astype(preds == d2l.reshape(Y, -1), d2l.float32)
    return d2l.reduce_mean(compare) if averaged else compare
```

```{.python .input  n=10}
%%tab mxnet

@d2l.add_to_class(d2l.Module)  #@save
def get_scratch_params(self):
    params = []
    for attr in dir(self):
        a = getattr(self, attr)
        if isinstance(a, np.ndarray):
            params.append(a)
        if isinstance(a, d2l.Module):
            params.extend(a.get_scratch_params())
    return params

@d2l.add_to_class(d2l.Module)  #@save
def parameters(self):
    params = self.collect_params()
    return params if isinstance(params, gluon.parameter.ParameterDict) and len(
        params.keys()) else self.get_scratch_params()
```

## 要約

分類は十分に一般的な問題であるため、専用の便利関数を用意する価値がある。分類において中心的に重要なのは、分類器の *精度* である。しばしば私たちが主に気にするのは精度であるが、統計的・計算的な理由から、分類器はさまざまな他の目的を最適化するように学習される。しかし、学習中にどの損失関数を最小化したとしても、分類器の精度を経験的に評価するための便利なメソッドがあると有用である。 


## 演習

1. 検証損失を $L_\textrm{v}$ とし、この節の損失関数による平均化で計算される、その簡便だが粗い推定値を $L_\textrm{v}^\textrm{q}$ とする。最後に、最後のミニバッチ上の損失を $l_\textrm{v}^\textrm{b}$ とする。$L_\textrm{v}$ を $L_\textrm{v}^\textrm{q}$、$l_\textrm{v}^\textrm{b}$、およびサンプルサイズとミニバッチサイズで表せ。
1. この簡便だが粗い推定値 $L_\textrm{v}^\textrm{q}$ が不偏であることを示せ。すなわち、$E[L_\textrm{v}] = E[L_\textrm{v}^\textrm{q}]$ を示せ。それでもなお、なぜ $L_\textrm{v}$ を使いたいのか。
1. 多クラス分類の損失が与えられたとき、$y$ を見たときに $y'$ を推定する際の罰則を $l(y,y')$ とし、確率 $p(y \mid x)$ が与えられているとする。$y'$ の最適な選択の規則を定式化せよ。ヒント: $l$ と $p(y \mid x)$ を用いて期待損失を表せ。
