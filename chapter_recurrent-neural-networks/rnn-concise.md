# リカレントニューラルネットワークの簡潔な実装
:label:`sec_rnn-concise`

ほとんどのスクラッチ実装と同様に、
:numref:`sec_rnn-scratch` は
各コンポーネントがどのように動作するかを理解できるように設計されていました。
しかし、日常的にRNNを使うときや
本番コードを書くときには、
実装時間
（一般的なモデルや関数に対するライブラリコードを提供してくれるため）
と計算時間
（これらのライブラリ実装を徹底的に最適化してくれるため）
の両方を削減できるライブラリに、より頼りたくなるでしょう。
この節では、
深層学習フレームワークが提供する高水準APIを用いて、
同じ言語モデルをより効率的に実装する方法を示します。
まずはこれまでと同様に、
*タイムマシン* データセットを読み込みます。

```{.python .input}
%load_ext d2lbook.tab
tab.interact_select('mxnet', 'pytorch', 'tensorflow', 'jax')
```

```{.python .input}
%%tab mxnet
from d2l import mxnet as d2l
from mxnet import np, npx
from mxnet.gluon import nn, rnn
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
from jax import numpy as jnp
```

## [**モデルの定義**]

以下のクラスを、
高水準APIで実装されたRNNを用いて定義します。

:begin_tab:`mxnet`
具体的には、隠れ状態を初期化するために、
メンバーメソッド `begin_state` を呼び出します。
これは、ミニバッチ内の各サンプルに対する初期隠れ状態を含むリストを返します。
その形状は
（隠れ層数、バッチサイズ、隠れユニット数）
です。
後で導入するいくつかのモデル
（たとえば長短期記憶）では、
このリストには他の情報も含まれます。
:end_tab:

:begin_tab:`jax`
現時点では、Flaxは通常のRNNを簡潔に実装するためのRNNCellを提供していません。
Flaxの `linen` API では、LSTMやGRUのような
より高度なRNNの変種は利用できます。
:end_tab:

```{.python .input}
%%tab mxnet
class RNN(d2l.Module):  #@save
    """高水準APIで実装されたRNNモデル。"""
    def __init__(self, num_hiddens):
        super().__init__()
        self.save_hyperparameters()        
        self.rnn = rnn.RNN(num_hiddens)
        
    def forward(self, inputs, H=None):
        if H is None:
            H, = self.rnn.begin_state(inputs.shape[1], ctx=inputs.ctx)
        outputs, (H, ) = self.rnn(inputs, (H, ))
        return outputs, H
```

```{.python .input}
%%tab pytorch
class RNN(d2l.Module):  #@save
    """高水準APIで実装されたRNNモデル。"""
    def __init__(self, num_inputs, num_hiddens):
        super().__init__()
        self.save_hyperparameters()
        self.rnn = nn.RNN(num_inputs, num_hiddens)
        
    def forward(self, inputs, H=None):
        return self.rnn(inputs, H)
```

```{.python .input}
%%tab tensorflow
class RNN(d2l.Module):  #@save
    """高水準APIで実装されたRNNモデル。"""
    def __init__(self, num_hiddens):
        super().__init__()
        self.save_hyperparameters()            
        self.rnn = tf.keras.layers.SimpleRNN(
            num_hiddens, return_sequences=True, return_state=True,
            time_major=True)
        
    def forward(self, inputs, H=None):
        outputs, H = self.rnn(inputs, H)
        return outputs, H
```

```{.python .input}
%%tab jax
class RNN(nn.Module):  #@save
    """高水準APIで実装されたRNNモデル。"""
    num_hiddens: int

    @nn.compact
    def __call__(self, inputs, H=None):
        raise NotImplementedError
```

:numref:`sec_rnn-scratch` の `RNNLMScratch` クラスを継承して、
次の `RNNLM` クラスはRNNベースの完全な言語モデルを定義します。
別個の全結合出力層を作成する必要があることに注意してください。

```{.python .input}
%%tab pytorch
class RNNLM(d2l.RNNLMScratch):  #@save
    """高水準APIで実装されたRNNベースの言語モデル。"""
    def init_params(self):
        self.linear = nn.LazyLinear(self.vocab_size)
        
    def output_layer(self, hiddens):
        return d2l.swapaxes(self.linear(hiddens), 0, 1)
```

```{.python .input}
%%tab mxnet, tensorflow
class RNNLM(d2l.RNNLMScratch):  #@save
    """高水準APIで実装されたRNNベースの言語モデル。"""
    def init_params(self):
        if tab.selected('mxnet'):
            self.linear = nn.Dense(self.vocab_size, flatten=False)
            self.initialize()
        if tab.selected('tensorflow'):
            self.linear = tf.keras.layers.Dense(self.vocab_size)
        
    def output_layer(self, hiddens):
        if tab.selected('mxnet'):
            return d2l.swapaxes(self.linear(hiddens), 0, 1)        
        if tab.selected('tensorflow'):
            return d2l.transpose(self.linear(hiddens), (1, 0, 2))
```

```{.python .input}
%%tab jax
class RNNLM(d2l.RNNLMScratch):  #@save
    """高水準APIで実装されたRNNベースの言語モデル。"""
    training: bool = True

    def setup(self):
        self.linear = nn.Dense(self.vocab_size)

    def output_layer(self, hiddens):
        return d2l.swapaxes(self.linear(hiddens), 0, 1)

    def forward(self, X, state=None):
        embs = self.one_hot(X)
        rnn_outputs, _ = self.rnn(embs, state, self.training)
        return self.output_layer(rnn_outputs)
```

## 学習と予測

モデルを学習する前に、[**ランダムな重みで初期化されたモデルを使って予測してみましょう。**]
ネットワークはまだ学習されていないので、
意味をなさない予測を生成します。

```{.python .input}
%%tab pytorch, mxnet, tensorflow
data = d2l.TimeMachine(batch_size=1024, num_steps=32)
if tab.selected('mxnet', 'tensorflow'):
    rnn = RNN(num_hiddens=32)
if tab.selected('pytorch'):
    rnn = RNN(num_inputs=len(data.vocab), num_hiddens=32)
model = RNNLM(rnn, vocab_size=len(data.vocab), lr=1)
model.predict('it has', 20, data.vocab)
```

次に、高水準APIを活用して、[**モデルを学習します**]。

```{.python .input}
%%tab pytorch, mxnet, tensorflow
if tab.selected('mxnet', 'pytorch'):
    trainer = d2l.Trainer(max_epochs=100, gradient_clip_val=1, num_gpus=1)
if tab.selected('tensorflow'):
    with d2l.try_gpu():
        trainer = d2l.Trainer(max_epochs=100, gradient_clip_val=1)
trainer.fit(model, data)
```

:numref:`sec_rnn-scratch` と比べると、
このモデルは同程度の困惑度を達成しますが、
最適化された実装のおかげでより高速に動作します。
これまでと同様に、指定した接頭辞文字列に続く予測トークンを生成できます。

```{.python .input}
%%tab mxnet, pytorch
model.predict('it has', 20, data.vocab, d2l.try_gpu())
```

```{.python .input}
%%tab tensorflow
model.predict('it has', 20, data.vocab)
```

## まとめ

深層学習フレームワークの高水準APIは、標準的なRNNの実装を提供します。
これらのライブラリを使えば、標準モデルを再実装するために時間を浪費せずに済みます。
さらに、
フレームワークの実装はしばしば高度に最適化されているため、
スクラッチ実装と比べて
大幅な（計算）性能向上が得られます。

## 演習

1. 高水準APIを使ってRNNモデルを過学習させることはできますか？
1. :numref:`sec_sequence` の自己回帰モデルをRNNを用いて実装してください。

:begin_tab:`mxnet`
[議論](https://discuss.d2l.ai/t/335)
:end_tab:

:begin_tab:`pytorch`
[議論](https://discuss.d2l.ai/t/1053)
:end_tab:

:begin_tab:`tensorflow`
[議論](https://discuss.d2l.ai/t/2211)
:end_tab:

:begin_tab:`jax`
[議論](https://discuss.d2l.ai/t/18015)
:end_tab:\n