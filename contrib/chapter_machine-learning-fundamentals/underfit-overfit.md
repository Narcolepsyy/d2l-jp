{.python .input  n=1}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow'])
```

# 過少適合と過学習 
:label:`sec_polynomial`

この節では、これまでに見てきた概念のいくつかを試してみます。話を簡単にするために、玩具例として多項式回帰を用います。

```{.python .input  n=3}
%%tab mxnet
from d2l import mxnet as d2l
from mxnet import gluon, np, npx
from mxnet.gluon import nn
import math
npx.set_np()
```

```{.python .input  n=4}
%%tab pytorch
from d2l import torch as d2l
import torch
from torch import nn
import math
```

```{.python .input  n=5}
%%tab tensorflow
from d2l import tensorflow as d2l
import tensorflow as tf
import math
```

### データセットの生成

まずデータが必要です。$x$ が与えられたとき、訓練データとテストデータのラベルを生成するために、[**次の3次多項式を用います**]。

(**$$y = 5 + 1.2x - 3.4\frac{x^2}{2!} + 5.6 \frac{x^3}{3!} + \epsilon \text{ where }
\epsilon \sim \mathcal{N}(0, 0.1^2).$$**)

ノイズ項 $\epsilon$ は、平均0、標準偏差0.1の正規分布に従います。
最適化では、一般に勾配や損失が非常に大きな値になることを避けたいものです。
そのため、*特徴量* は $x^i$ から $\frac{x^i}{i!}$ にスケーリングし直しています。
これにより、指数 $i$ が大きい場合でも非常に大きな値を避けられます。
訓練セットとテストセットそれぞれについて、100個のサンプルを生成します。

```{.python .input  n=6}
%%tab all
class Data(d2l.DataModule):
    def __init__(self, num_train, num_val, num_inputs, batch_size):
        self.save_hyperparameters()        
        p, n = max(3, self.num_inputs), num_train + num_val
        w = d2l.tensor([1.2, -3.4, 5.6] + [0]*(p-3))
        if tab.selected('mxnet') or tab.selected('pytorch'):
            x = d2l.randn(n, 1)
            noise = d2l.randn(n, 1) * 0.1
        if tab.selected('tensorflow'):
            x = d2l.normal((n, 1))
            noise = d2l.normal((n, 1)) * 0.1
        X = d2l.concat([x ** (i+1) / math.gamma(i+2) for i in range(p)], 1)
        self.y = d2l.matmul(X, d2l.reshape(w, (-1, 1))) + noise
        self.X = X[:,:num_inputs]
        
    def get_dataloader(self, train):
        i = slice(0, self.num_train) if train else slice(self.num_train, None)
        return self.get_tensorloader([self.X, self.y], train, i)
```

ここでも、`poly_features` に格納される単項式はガンマ関数でスケーリングされています。
ここで、$\Gamma(n)=(n-1)!$ です。
生成されたデータセットの[**最初の2サンプルを見てみましょう**]。
値1は厳密には特徴量であり、すなわちバイアスに対応する定数特徴量です。

### [**3次多項式関数の当てはめ（通常）**]

まず、データ生成関数と同じ次数である3次多項式関数を用いてみます。
結果は、このモデルの訓練損失とテスト損失の両方を効果的に減少させられることを示しています。
学習されたモデルパラメータも、真の値 $w = [1.2, -3.4, 5.6], b=5$ に近くなります。

```{.python .input  n=7}
%%tab all
def train(p):
    if tab.selected('mxnet') or tab.selected('tensorflow'):
        model = d2l.LinearRegression(lr=0.01)
    if tab.selected('pytorch'):
        model = d2l.LinearRegression(p, lr=0.01)
    model.board.ylim = [1, 1e2]
    data = Data(200, 200, p, 20)
    trainer = d2l.Trainer(max_epochs=10)
    trainer.fit(model, data)
    print(model.get_w_b())
    
train(p=3)
```

### [**線形関数の当てはめ（過少適合）**]

線形関数の当てはめをもう一度見てみましょう。
初期のエポックで損失が下がった後は、
このモデルの訓練損失をさらに下げることが難しくなります。
最後のエポックの反復が完了した後でも、
訓練損失は依然として高いままです。
非線形なパターン（ここでは3次多項式関数のようなもの）を当てはめるために用いると、
線形モデルは過少適合しやすくなります。

```{.python .input  n=8}
%%tab all
train(p=1)
```

### [**高次多項式関数の当てはめ（過学習）**]

では、次数が高すぎる多項式を使ってモデルを訓練してみましょう。
ここでは、高次の係数が0に近い値を持つべきだと学習するには、データが不十分です。
その結果、過度に複雑なモデルは、
訓練データ中のノイズの影響を受けやすくなってしまいます。
訓練損失は効果的に減少できるものの、
テスト損失は依然としてはるかに高いままです。
これは、複雑なモデルがデータに過学習していることを示しています。

```{.python .input  n=9}
%%tab all
train(p=10)
```

次の節では、過学習の問題と、
重み減衰やドロップアウトなど、それに対処する方法について引き続き議論します。


## 要約

* 一般化誤差は訓練誤差から推定できないため、訓練誤差を単に最小化するだけでは、必ずしも一般化誤差の低下を意味しません。機械学習モデルでは、一般化誤差を最小にするために、過学習を防ぐよう注意する必要があります。
* 検証集合は、使いすぎない限り、モデル選択に利用できます。
* 過少適合とは、モデルが訓練誤差を十分に下げられないことを意味します。訓練誤差が検証誤差よりはるかに低いとき、過学習が起きています。
* 適切な複雑さのモデルを選び、不十分な訓練サンプルを使わないようにすべきです。


## 演習

1. 多項式回帰問題を厳密に解けますか？ ヒント: 線形代数を使ってください。
1. 多項式のモデル選択について考えなさい:
    1. 訓練損失とモデルの複雑さ（多項式の次数）の関係をプロットしなさい。何が観察できますか？ 訓練損失を0にするには、どの次数の多項式が必要ですか？
    1. この場合のテスト損失をプロットしなさい。
    1. データ量の関数として同じプロットを生成しなさい。
1. 多項式特徴量 $x^i$ の正規化（$1/i!$）を取り除くとどうなりますか？ 別の方法でこれを修正できますか？
1. 一般化誤差が0になることを期待できる場合はありますか？

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/96)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/97)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/234)
:end_tab:\n