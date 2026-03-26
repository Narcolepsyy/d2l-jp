# 深層再帰ニューラルネットワーク

:label:`sec_deep_rnn`

これまで、系列入力、単一の隠れRNN層、および出力層からなるネットワークの定義に焦点を当ててきました。  
任意の時刻の入力とそれに対応する出力の間に隠れ層が1つしかないにもかかわらず、これらのネットワークには深さがあると言えます。  
最初の時刻の入力は、最終時刻 $T$ の出力（しばしば100ステップや1000ステップも後）に影響を与えることができます。  
これらの入力は、最終出力に到達するまでに再帰層を $T$ 回通過します。  
しかし、私たちはしばしば、ある時刻における入力と同じ時刻における出力の間の複雑な関係を表現する能力も保持したいと考えます。  
そのため、RNNは時間方向だけでなく、入力から出力への方向にも深いように構成されることがよくあります。  
これは、MLPや深層CNNの開発で既に出会ってきた、まさに深さの概念です。


この種の深いRNNを構築する標準的な方法は驚くほど単純です。RNNを上に積み重ねればよいのです。  
長さ $T$ の系列が与えられると、最初のRNNは長さ $T$ の出力系列を生成します。  
これらは次のRNN層への入力を構成します。  
この短い節では、この設計パターンを示し、そのようなスタック型RNNをどのように実装するかの簡単な例を示します。  
以下の :numref:`fig_deep_rnn` では、$L$ 個の隠れ層を持つ深いRNNを示します。  
各隠れ状態は系列入力を受け取り、系列出力を生成します。  
さらに、各時刻における任意のRNNセル（:numref:`fig_deep_rnn` の白い箱）は、同じ層の前時刻の値と、同じ時刻の前の層の値の両方に依存します。 

![深いRNNのアーキテクチャ。](../img/deep-rnn.svg)
:label:`fig_deep_rnn`

形式的には、時刻 $t$ においてミニバッチ入力 $\mathbf{X}_t \in \mathbb{R}^{n \times d}$（例の数 $=n$、各例の入力数 $=d$）があるとします。  
同じ時刻において、$l^\textrm{th}$ 隠れ層（$l=1,\ldots,L$）の隠れ状態を $\mathbf{H}_t^{(l)} \in \mathbb{R}^{n \times h}$（隠れユニット数 $=h$）とし、出力層変数を $\mathbf{O}_t \in \mathbb{R}^{n \times q}$（出力数 $=q$）とします。  
$\mathbf{H}_t^{(0)} = \mathbf{X}_t$ とおくと、活性化関数 $\phi_l$ を用いる $l^\textrm{th}$ 隠れ層の隠れ状態は次のように計算されます。

$$\mathbf{H}_t^{(l)} = \phi_l(\mathbf{H}_t^{(l-1)} \mathbf{W}_{\textrm{xh}}^{(l)} + \mathbf{H}_{t-1}^{(l)} \mathbf{W}_{\textrm{hh}}^{(l)}  + \mathbf{b}_\textrm{h}^{(l)}),$$
:eqlabel:`eq_deep_rnn_H`

ここで、重み $\mathbf{W}_{\textrm{xh}}^{(l)} \in \mathbb{R}^{h \times h}$ と $\mathbf{W}_{\textrm{hh}}^{(l)} \in \mathbb{R}^{h \times h}$、およびバイアス $\mathbf{b}_\textrm{h}^{(l)} \in \mathbb{R}^{1 \times h}$ は、$l^\textrm{th}$ 隠れ層のモデルパラメータです。

最後に、出力層の計算は最終の $L^\textrm{th}$ 隠れ層の隠れ状態のみに基づきます。

$$\mathbf{O}_t = \mathbf{H}_t^{(L)} \mathbf{W}_{\textrm{hq}} + \mathbf{b}_\textrm{q},$$

ここで、重み $\mathbf{W}_{\textrm{hq}} \in \mathbb{R}^{h \times q}$ とバイアス $\mathbf{b}_\textrm{q} \in \mathbb{R}^{1 \times q}$ は出力層のモデルパラメータです。

MLPと同様に、隠れ層の数 $L$ と隠れユニット数 $h$ は調整可能なハイパーパラメータです。  
一般的なRNN層の幅（$h$）は $(64, 2056)$ の範囲にあり、一般的な深さ（$L$）は $(1, 8)$ の範囲にあります。  
さらに、:eqref:`eq_deep_rnn_H` の隠れ状態計算をLSTMやGRUのものに置き換えることで、深いゲート付きRNNを容易に得ることができます。

```{.python .input}
%load_ext d2lbook.tab
tab.interact_select('mxnet', 'pytorch', 'tensorflow', 'jax')
```

```{.python .input}
%%tab mxnet
from d2l import mxnet as d2l
from mxnet import np, npx
from mxnet.gluon import rnn
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

ゼロから多層RNNを実装するには、各層をそれぞれ独自の学習可能パラメータを持つ `RNNScratch` インスタンスとして扱えばよいです。

```{.python .input}
%%tab mxnet, tensorflow
class StackedRNNScratch(d2l.Module):
    def __init__(self, num_inputs, num_hiddens, num_layers, sigma=0.01):
        super().__init__()
        self.save_hyperparameters()
        self.rnns = [d2l.RNNScratch(num_inputs if i==0 else num_hiddens,
                                    num_hiddens, sigma)
                     for i in range(num_layers)]
```

```{.python .input}
%%tab pytorch
class StackedRNNScratch(d2l.Module):
    def __init__(self, num_inputs, num_hiddens, num_layers, sigma=0.01):
        super().__init__()
        self.save_hyperparameters()
        self.rnns = nn.Sequential(*[d2l.RNNScratch(
            num_inputs if i==0 else num_hiddens, num_hiddens, sigma)
                                    for i in range(num_layers)])
```

```{.python .input}
%%tab jax
class StackedRNNScratch(d2l.Module):
    num_inputs: int
    num_hiddens: int
    num_layers: int
    sigma: float = 0.01

    def setup(self):
        self.rnns = [d2l.RNNScratch(self.num_inputs if i==0 else self.num_hiddens,
                                    self.num_hiddens, self.sigma)
                     for i in range(self.num_layers)]
```

多層の順伝播計算は、単純に層ごとに順伝播を行うだけです。

```{.python .input}
%%tab all
@d2l.add_to_class(StackedRNNScratch)
def forward(self, inputs, Hs=None):
    outputs = inputs
    if Hs is None: Hs = [None] * self.num_layers
    for i in range(self.num_layers):
        outputs, Hs[i] = self.rnns[i](outputs, Hs[i])
        outputs = d2l.stack(outputs, 0)
    return outputs, Hs
```

例として、*The Time Machine* データセット（:numref:`sec_rnn-scratch` と同じ）上で深いGRUモデルを学習します。  
簡単のため、層数は2に設定します。

```{.python .input}
%%tab all
data = d2l.TimeMachine(batch_size=1024, num_steps=32)
if tab.selected('mxnet', 'pytorch', 'jax'):
    rnn_block = StackedRNNScratch(num_inputs=len(data.vocab),
                                  num_hiddens=32, num_layers=2)
    model = d2l.RNNLMScratch(rnn_block, vocab_size=len(data.vocab), lr=2)
    trainer = d2l.Trainer(max_epochs=100, gradient_clip_val=1, num_gpus=1)
if tab.selected('tensorflow'):
    with d2l.try_gpu():
        rnn_block = StackedRNNScratch(num_inputs=len(data.vocab),
                                  num_hiddens=32, num_layers=2)
        model = d2l.RNNLMScratch(rnn_block, vocab_size=len(data.vocab), lr=2)
    trainer = d2l.Trainer(max_epochs=100, gradient_clip_val=1)
trainer.fit(model, data)
```

## 簡潔な実装

:begin_tab:`pytorch, mxnet, tensorflow`
幸いなことに、RNNの複数層を実装するために必要な多くの実務的な詳細は、高レベルAPIで容易に利用できます。  
ここでの簡潔な実装では、そのような組み込み機能を使用します。  
このコードは、 :numref:`sec_gru` で以前に使ったものを一般化したもので、デフォルトの1層ではなく、層数を明示的に指定できるようにしています。
:end_tab:

:begin_tab:`jax`
FlaxはRNNの実装において最小主義的なアプローチを取っています。RNNの層数を定義したり、dropoutと組み合わせたりする機能は、標準では用意されていません。  
ここでの簡潔な実装では、すべての組み込み機能を使い、その上に `num_layers` と `dropout` の機能を追加します。  
このコードは、 :numref:`sec_gru` で以前に使ったものを一般化したもので、単一層のデフォルトではなく、層数を明示的に指定できるようにしています。
:end_tab:

```{.python .input}
%%tab mxnet
class GRU(d2l.RNN):  #@save
    """The multilayer GRU model."""
    def __init__(self, num_hiddens, num_layers, dropout=0):
        d2l.Module.__init__(self)
        self.save_hyperparameters()
        self.rnn = rnn.GRU(num_hiddens, num_layers, dropout=dropout)
```

```{.python .input}
%%tab pytorch
class GRU(d2l.RNN):  #@save
    """The multilayer GRU model."""
    def __init__(self, num_inputs, num_hiddens, num_layers, dropout=0):
        d2l.Module.__init__(self)
        self.save_hyperparameters()
        self.rnn = nn.GRU(num_inputs, num_hiddens, num_layers,
                          dropout=dropout)
```

```{.python .input}
%%tab tensorflow
class GRU(d2l.RNN):  #@save
    """The multilayer GRU model."""
    def __init__(self, num_hiddens, num_layers, dropout=0):
        d2l.Module.__init__(self)
        self.save_hyperparameters()
        gru_cells = [tf.keras.layers.GRUCell(num_hiddens, dropout=dropout)
                     for _ in range(num_layers)]
        self.rnn = tf.keras.layers.RNN(gru_cells, return_sequences=True,
                                       return_state=True, time_major=True)

    def forward(self, X, state=None):
        outputs, *state = self.rnn(X, state)
        return outputs, state
```

```{.python .input}
%%tab jax
class GRU(d2l.RNN):  #@save
    """The multilayer GRU model."""
    num_hiddens: int
    num_layers: int
    dropout: float = 0

    @nn.compact
    def __call__(self, X, state=None, training=False):
        outputs = X
        new_state = []
        if state is None:
            batch_size = X.shape[1]
            state = [nn.GRUCell.initialize_carry(jax.random.PRNGKey(0),
                    (batch_size,), self.num_hiddens)] * self.num_layers

        GRU = nn.scan(nn.GRUCell, variable_broadcast="params",
                      in_axes=0, out_axes=0, split_rngs={"params": False})

        # 最後のGRU層を除く各GRU層の後にdropout層を導入する
        for i in range(self.num_layers - 1):
            layer_i_state, X = GRU()(state[i], outputs)
            new_state.append(layer_i_state)
            X = nn.Dropout(self.dropout, deterministic=not training)(X)

        # dropoutなしの最終GRU層
        out_state, X = GRU()(state[-1], X)
        new_state.append(out_state)
        return X, jnp.array(new_state)
```

ハイパーパラメータの選択などのアーキテクチャ上の決定は、 :numref:`sec_gru` のものと非常によく似ています。  
異なるトークンの数、すなわち `vocab_size` と同じ数の入力と出力を選びます。  
隠れユニット数は引き続き32です。  
唯一の違いは、ここでは [**`num_layers` の値を指定することで、単純ではない数の隠れ層を選択する**] ことです。

```{.python .input}
%%tab mxnet
gru = GRU(num_hiddens=32, num_layers=2)
model = d2l.RNNLM(gru, vocab_size=len(data.vocab), lr=2)

# 実行には1時間以上かかる（MXNet側の修正待ち）
# trainer.fit(model, data)
# model.predict('it has', 20, data.vocab, d2l.try_gpu())
```

```{.python .input}
%%tab pytorch, tensorflow, jax
if tab.selected('tensorflow', 'jax'):
    gru = GRU(num_hiddens=32, num_layers=2)
if tab.selected('pytorch'):
    gru = GRU(num_inputs=len(data.vocab), num_hiddens=32, num_layers=2)
if tab.selected('pytorch', 'jax'):
    model = d2l.RNNLM(gru, vocab_size=len(data.vocab), lr=2)
if tab.selected('tensorflow'):
    with d2l.try_gpu():
        model = d2l.RNNLM(gru, vocab_size=len(data.vocab), lr=2)
trainer.fit(model, data)
```

```{.python .input}
%%tab pytorch
model.predict('it has', 20, data.vocab, d2l.try_gpu())
```

```{.python .input}
%%tab tensorflow
model.predict('it has', 20, data.vocab)
```

```{.python .input}
%%tab jax
model.predict('it has', 20, data.vocab, trainer.state.params)
```

## まとめ

深いRNNでは、隠れ状態情報は現在の層の次の時刻と、次の層の現在の時刻へと受け渡されます。  
LSTM、GRU、あるいは通常のRNNなど、深いRNNにはさまざまな種類があります。  
便利なことに、これらのモデルはすべて、深層学習フレームワークの高レベルAPIの一部として利用できます。  
モデルの初期化には注意が必要です。  
全体として、深いRNNは適切に収束させるために、学習率やクリッピングなど、かなりの作業を必要とします。

## 演習

1. GRUをLSTMに置き換え、精度と学習速度を比較せよ。
1. 学習データを増やして複数の本を含めるようにせよ。困惑度をどこまで下げられるか。
1. テキストをモデル化する際に、異なる著者のソースを組み合わせたいと思うか。なぜそれがよい考えなのか。何が問題になりうるか。\n
