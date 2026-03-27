# 双方向リカレントニューラルネットワーク
:label:`sec_bi_rnn`

これまで、系列学習タスクの作業例として言語モデルを扱ってきた。
そこでは、系列内のそれ以前のすべてのトークンが与えられたときに、次のトークンを予測することを目指す。
この状況では、左側の文脈のみに条件づければよく、そのため標準的な RNN の一方向の連鎖が適しているように見える。
しかし、他にも多くの系列学習タスクがあり、そこでは各時刻での予測を左側と右側の両方の文脈に条件づけることがまったく問題ない。
たとえば、品詞判定を考えてみよう。
ある単語に対応する品詞を評価するときに、なぜ両方向の文脈を考慮しないのだろうか。

もう1つよくあるタスクとして、実際に関心のあるタスクでモデルをファインチューニングする前の事前学習として有用なことが多いものに、テキスト文書中のランダムなトークンをマスクし、欠落したトークンの値を予測するよう系列モデルを学習させる、というものがある。
空欄の後に何が続くかによって、欠落トークンの有力候補は大きく変わることに注意されたい。

* I am `___`.
* I am `___` hungry.
* I am `___` hungry, and I can eat half a pig.

最初の文では、「happy」が有力な候補に見える。
2つ目の文では「not」と「very」のどちらもありえそうであるが、
3つ目の文では「not」は適切ではなさそうである。


幸いなことに、単純な手法で任意の一方向 RNN を双方向 RNN に変換できる :cite:`Schuster.Paliwal.1997`。
同じ入力に対して、逆方向に連結された2つの一方向 RNN 層を実装するだけです（:numref:`fig_birnn`）。
最初の RNN 層では、最初の入力は $\mathbf{x}_1$ で最後の入力は $\mathbf{x}_T$ であるが、2つ目の RNN 層では、最初の入力は $\mathbf{x}_T$ で最後の入力は $\mathbf{x}_1$ である。
この双方向 RNN 層の出力を得るには、2つの基礎となる一方向 RNN 層の対応する出力を単純に連結する。


![双方向 RNN のアーキテクチャ。](../img/birnn.svg)
:label:`fig_birnn`


形式的には、任意の時刻 $t$ について、ミニバッチ入力 $\mathbf{X}_t \in \mathbb{R}^{n \times d}$
（例の数 $=n$、各例における入力数 $=d$）
を考え、隠れ層の活性化関数を $\phi$ とする。
双方向アーキテクチャでは、この時刻における順方向と逆方向の隠れ状態はそれぞれ
$\overrightarrow{\mathbf{H}}_t  \in \mathbb{R}^{n \times h}$ と
$\overleftarrow{\mathbf{H}}_t \in \mathbb{R}^{n \times h}$ である。
ここで $h$ は隠れユニット数である。
順方向と逆方向の隠れ状態の更新は次のようになる。


$$
\begin{aligned}
\overrightarrow{\mathbf{H}}_t &= \phi(\mathbf{X}_t \mathbf{W}_{\textrm{xh}}^{(f)} + \overrightarrow{\mathbf{H}}_{t-1} \mathbf{W}_{\textrm{hh}}^{(f)}  + \mathbf{b}_\textrm{h}^{(f)}),\\
\overleftarrow{\mathbf{H}}_t &= \phi(\mathbf{X}_t \mathbf{W}_{\textrm{xh}}^{(b)} + \overleftarrow{\mathbf{H}}_{t+1} \mathbf{W}_{\textrm{hh}}^{(b)}  + \mathbf{b}_\textrm{h}^{(b)}),
\end{aligned}
$$

ここで、重み $\mathbf{W}_{\textrm{xh}}^{(f)} \in \mathbb{R}^{d \times h}, \mathbf{W}_{\textrm{hh}}^{(f)} \in \mathbb{R}^{h \times h}, \mathbf{W}_{\textrm{xh}}^{(b)} \in \mathbb{R}^{d \times h}, \textrm{ and } \mathbf{W}_{\textrm{hh}}^{(b)} \in \mathbb{R}^{h \times h}$、およびバイアス $\mathbf{b}_\textrm{h}^{(f)} \in \mathbb{R}^{1 \times h}$ と $\mathbf{b}_\textrm{h}^{(b)} \in \mathbb{R}^{1 \times h}$ はすべてモデルパラメータである。

次に、順方向と逆方向の隠れ状態
$\overrightarrow{\mathbf{H}}_t$ と $\overleftarrow{\mathbf{H}}_t$
を連結して、出力層に入力する隠れ状態 $\mathbf{H}_t \in \mathbb{R}^{n \times 2h}$ を得る。
複数の隠れ層を持つ深い双方向 RNN では、このような情報は次の双方向層への *入力* として渡される。
最後に、出力層は出力
$\mathbf{O}_t \in \mathbb{R}^{n \times q}$（出力数 $=q$）を計算する。

$$\mathbf{O}_t = \mathbf{H}_t \mathbf{W}_{\textrm{hq}} + \mathbf{b}_\textrm{q}.$$

ここで、重み行列 $\mathbf{W}_{\textrm{hq}} \in \mathbb{R}^{2h \times q}$
とバイアス $\mathbf{b}_\textrm{q} \in \mathbb{R}^{1 \times q}$
は出力層のモデルパラメータである。
厳密には、2つの方向で異なる数の隠れユニットを持たせることもできるが、
実際にはこの設計が選ばれることはほとんどない。
それでは、双方向 RNN の簡単な実装を示す。

```{.python .input}
%load_ext d2lbook.tab
tab.interact_select('mxnet', 'pytorch', 'tensorflow', 'jax')
```

```{.python .input}
%%tab mxnet
from d2l import mxnet as d2l
from mxnet import npx, np
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
from jax import numpy as jnp
```

## ゼロからの実装

双方向 RNN をゼロから実装したい場合は、別々の学習可能パラメータを持つ2つの一方向 `RNNScratch` インスタンスを含めればよいである。

```{.python .input}
%%tab pytorch, mxnet, tensorflow
class BiRNNScratch(d2l.Module):
    def __init__(self, num_inputs, num_hiddens, sigma=0.01):
        super().__init__()
        self.save_hyperparameters()
        self.f_rnn = d2l.RNNScratch(num_inputs, num_hiddens, sigma)
        self.b_rnn = d2l.RNNScratch(num_inputs, num_hiddens, sigma)
        self.num_hiddens *= 2  # The output dimension will be doubled
```

```{.python .input}
%%tab jax
class BiRNNScratch(d2l.Module):
    num_inputs: int
    num_hiddens: int
    sigma: float = 0.01

    def setup(self):
        self.f_rnn = d2l.RNNScratch(num_inputs, num_hiddens, sigma)
        self.b_rnn = d2l.RNNScratch(num_inputs, num_hiddens, sigma)
        self.num_hiddens *= 2  # The output dimension will be doubled
```

順方向と逆方向の RNN の状態は別々に更新され、
これら2つの RNN の出力は連結される。

```{.python .input}
%%tab all
@d2l.add_to_class(BiRNNScratch)
def forward(self, inputs, Hs=None):
    f_H, b_H = Hs if Hs is not None else (None, None)
    f_outputs, f_H = self.f_rnn(inputs, f_H)
    b_outputs, b_H = self.b_rnn(reversed(inputs), b_H)
    outputs = [d2l.concat((f, b), -1) for f, b in zip(
        f_outputs, reversed(b_outputs))]
    return outputs, (f_H, b_H)
```

## 簡潔な実装

:begin_tab:`pytorch, mxnet, tensorflow`
高水準 API を使えば、双方向 RNN をより簡潔に実装できる。
ここでは GRU モデルを例として取り上げる。
:end_tab:

:begin_tab:`jax`
Flax API には RNN 層がないため、`bidirectional` 引数という概念もない。双方向層が必要な場合は、ゼロからの実装で示したように、入力を手動で逆順にする必要がある。
:end_tab:

```{.python .input}
%%tab mxnet, pytorch
class BiGRU(d2l.RNN):
    def __init__(self, num_inputs, num_hiddens):
        d2l.Module.__init__(self)
        self.save_hyperparameters()
        if tab.selected('mxnet'):
            self.rnn = rnn.GRU(num_hiddens, bidirectional=True)
        if tab.selected('pytorch'):
            self.rnn = nn.GRU(num_inputs, num_hiddens, bidirectional=True)
        self.num_hiddens *= 2
```

## まとめ

双方向 RNN では、各時刻の隠れ状態は、現在の時刻より前と後のデータの両方によって同時に決まりる。双方向 RNN は主に、系列の符号化や、双方向の文脈が与えられた観測の推定に有用である。双方向 RNN は、長い勾配連鎖のために学習コストが非常に高くなる。

## 演習

1. 異なる方向で異なる数の隠れユニットを使う場合、$\mathbf{H}_t$ の形状はどのように変わるか。
1. 複数の隠れ層を持つ双方向 RNN を設計せよ。
1. 自然言語では多義性が一般的である。たとえば、単語 "bank" は “i went to the bank to deposit cash” と “i went to the bank to sit down” という文脈で異なる意味を持つ。文脈系列と単語が与えられたときに、その単語の正しい文脈におけるベクトル表現を返すようなニューラルネットワークモデルをどのように設計できるか？ 多義性の扱いにはどのような種類のニューラルアーキテクチャが好まれるか。
