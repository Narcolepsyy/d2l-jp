# 再帰型ニューラルネットワーク
:label:`sec_rnn`


:numref:`sec_language-model` では、言語モデリングのためのマルコフモデルと $n$-gram を説明しました。そこでは、時刻 $t$ におけるトークン $x_t$ の条件付き確率は、直前の $n-1$ 個のトークンのみに依存します。
時刻 $t-(n-1)$ より前のトークンが $x_t$ に及ぼしうる影響を取り込みたい場合は、
$n$ を大きくする必要がある。
しかし、その場合、モデルパラメータの数もそれに伴って指数的に増加する。というのも、語彙集合 $\mathcal{V}$ に対して $|\mathcal{V}|^n$ 個の数を保持しなければならないからである。
したがって、$P(x_t \mid x_{t-1}, \ldots, x_{1})$ を直接モデル化するよりも、潜在変数モデルを用いる方が望ましいである。

$$P(x_t \mid x_{t-1}, \ldots, x_1) \approx P(x_t \mid h_{t-1}),$$

ここで $h_{t-1}$ は、時刻 $t-1$ までの系列情報を保持する *隠れ状態* である。
一般に、
任意の時刻 $t$ における隠れ状態は、現在の入力 $x_{t}$ と前の隠れ状態 $h_{t-1}$ の両方に基づいて計算できる。

$$h_t = f(x_{t}, h_{t-1}).$$
:eqlabel:`eq_ht_xt`

:eqref:`eq_ht_xt` において十分に強力な関数 $f$ を用いれば、この潜在変数モデルは近似ではありません。結局のところ、$h_t$ はこれまでに観測したすべてのデータを単に保持していてもよいからです。
しかし、その場合、計算と記憶の両方が高コストになる可能性がある。

:numref:`chap_perceptrons` で、隠れユニットを持つ隠れ層について説明したことを思い出してください。
ここで重要なのは、
隠れ層と隠れ状態は、まったく異なる概念だということである。
隠れ層は、説明したように、入力から出力へ至る経路の途中で外から見えない層である。
一方、隠れ状態は技術的には、ある時点で行う処理に対する *入力* であり、
それは過去の時刻のデータを見て初めて計算できる。

*再帰型ニューラルネットワーク*（RNN）は、隠れ状態を持つニューラルネットワークである。RNN モデルを導入する前に、まず :numref:`sec_mlp` で導入した MLP モデルを振り返りよう。

```{.python .input}
%load_ext d2lbook.tab
tab.interact_select('mxnet', 'pytorch', 'tensorflow', 'jax')
```

```{.python .input}
%%tab mxnet
from d2l import mxnet as d2l
from mxnet import np, npx
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
import jax
from jax import numpy as jnp
```

## 隠れ状態を持たないニューラルネットワーク

単一の隠れ層を持つ MLP を見てみよう。
隠れ層の活性化関数を $\phi$ とする。
バッチサイズが $n$、入力次元が $d$ のミニバッチの例 $\mathbf{X} \in \mathbb{R}^{n \times d}$ が与えられたとき、隠れ層の出力 $\mathbf{H} \in \mathbb{R}^{n \times h}$ は次のように計算される。

$$\mathbf{H} = \phi(\mathbf{X} \mathbf{W}_{\textrm{xh}} + \mathbf{b}_\textrm{h}).$$
:eqlabel:`rnn_h_without_state`

:eqref:`rnn_h_without_state` では、隠れ層に対して重みパラメータ $\mathbf{W}_{\textrm{xh}} \in \mathbb{R}^{d \times h}$、バイアスパラメータ $\mathbf{b}_\textrm{h} \in \mathbb{R}^{1 \times h}$、および隠れユニット数 $h$ を用いています。
このため、加算の際にはブロードキャスト（:numref:`subsec_broadcasting` を参照）を適用する。
次に、隠れ層の出力 $\mathbf{H}$ を出力層の入力として用いる。出力層は次式で与えられる。

$$\mathbf{O} = \mathbf{H} \mathbf{W}_{\textrm{hq}} + \mathbf{b}_\textrm{q},$$

ここで $\mathbf{O} \in \mathbb{R}^{n \times q}$ は出力変数、$\mathbf{W}_{\textrm{hq}} \in \mathbb{R}^{h \times q}$ は重みパラメータ、$\mathbf{b}_\textrm{q} \in \mathbb{R}^{1 \times q}$ は出力層のバイアスパラメータである。分類問題であれば、$\mathrm{softmax}(\mathbf{O})$ を用いて出力カテゴリの確率分布を計算できる。

これは :numref:`sec_sequence` で以前に解いた回帰問題と完全に同様なので、詳細は省略する。
要するに、特徴とラベルのペアをランダムに取り出し、自動微分と確率的勾配降下法によってネットワークのパラメータを学習できるということである。

## 隠れ状態を持つ再帰型ニューラルネットワーク
:label:`subsec_rnn_w_hidden_states`

隠れ状態がある場合は、事情がまったく異なる。構造をもう少し詳しく見てみよう。

時刻 $t$ において、ミニバッチの入力
$\mathbf{X}_t \in \mathbb{R}^{n \times d}$
があると仮定する。
言い換えると、
$n$ 個の系列例からなるミニバッチについて、
$\mathbf{X}_t$ の各行は系列中の時刻 $t$ における 1 つの例に対応する。
次に、
時刻 $t$ の隠れ層出力を $\mathbf{H}_t  \in \mathbb{R}^{n \times h}$ と表す。
MLP とは異なり、ここでは前の時刻の隠れ層出力 $\mathbf{H}_{t-1}$ を保持し、前の時刻の隠れ層出力を現在の時刻でどのように使うかを表す新しい重みパラメータ $\mathbf{W}_{\textrm{hh}} \in \mathbb{R}^{h \times h}$ を導入する。具体的には、現在の時刻の隠れ層出力の計算は、現在の時刻の入力と前の時刻の隠れ層出力の両方によって決まりる。

$$\mathbf{H}_t = \phi(\mathbf{X}_t \mathbf{W}_{\textrm{xh}} + \mathbf{H}_{t-1} \mathbf{W}_{\textrm{hh}}  + \mathbf{b}_\textrm{h}).$$
:eqlabel:`rnn_h_with_state`

:eqref:`rnn_h_without_state` と比べると、:eqref:`rnn_h_with_state` では項 $\mathbf{H}_{t-1} \mathbf{W}_{\textrm{hh}}$ が 1 つ追加されており、そのため :eqref:`eq_ht_xt` を具体化しています。
隣接する時刻の隠れ層出力 $\mathbf{H}_t$ と $\mathbf{H}_{t-1}$ の関係から、
これらの変数が、現在の時刻までの系列の履歴情報を保持していることがわかる。これは、ニューラルネットワークの現在の時刻における状態、あるいは記憶のようなものである。したがって、このような隠れ層出力は *隠れ状態* と呼ばれる。
隠れ状態は、前の時刻の定義を現在の時刻でそのまま用いるため、:eqref:`rnn_h_with_state` の計算は *再帰的* である。したがって、先ほど述べたように、再帰的計算に基づく隠れ状態を持つニューラルネットワークは *再帰型ニューラルネットワーク* と呼ばれる。
RNN において
:eqref:`rnn_h_with_state`
の計算を行う層は、*再帰層* と呼ばれる。


RNN の構成方法にはさまざまなものがある。
:eqref:`rnn_h_with_state` で定義される隠れ状態を持つものは非常に一般的です。
時刻 $t$ に対して、
出力層の出力は MLP の計算と同様である。

$$\mathbf{O}_t = \mathbf{H}_t \mathbf{W}_{\textrm{hq}} + \mathbf{b}_\textrm{q}.$$

RNN のパラメータには、隠れ層の重み $\mathbf{W}_{\textrm{xh}} \in \mathbb{R}^{d \times h}, \mathbf{W}_{\textrm{hh}} \in \mathbb{R}^{h \times h}$、
バイアス $\mathbf{b}_\textrm{h} \in \mathbb{R}^{1 \times h}$、
および出力層の重み $\mathbf{W}_{\textrm{hq}} \in \mathbb{R}^{h \times q}$
とバイアス $\mathbf{b}_\textrm{q} \in \mathbb{R}^{1 \times q}$
が含まれる。
特筆すべき点として、
異なる時刻であっても、
RNN は常にこれらのモデルパラメータを使用する。
したがって、RNN のパラメータ化コストは
時刻数が増えても増加しません。

:numref:`fig_rnn` は、3 つの隣接する時刻における RNN の計算ロジックを示しています。
任意の時刻 $t$ において、
隠れ状態の計算は次のように扱える。
(i) 現在の時刻 $t$ の入力 $\mathbf{X}_t$ と前の時刻 $t-1$ の隠れ状態 $\mathbf{H}_{t-1}$ を連結する。
(ii) その連結結果を活性化関数 $\phi$ を持つ全結合層に入力する。
このような全結合層の出力が、現在の時刻 $t$ の隠れ状態 $\mathbf{H}_t$ である。
この場合、
モデルパラメータは :eqref:`rnn_h_with_state` にある $\mathbf{W}_{\textrm{xh}}$ と $\mathbf{W}_{\textrm{hh}}$ の連結、およびバイアス $\mathbf{b}_\textrm{h}$ である。
現在の時刻 $t$ の隠れ状態 $\mathbf{H}_t$ は、次の時刻 $t+1$ の隠れ状態 $\mathbf{H}_{t+1}$ の計算に関与する。
さらに、
$\mathbf{H}_t$ は全結合の出力層にも入力され、
現在の時刻 $t$ の出力 $\mathbf{O}_t$ を計算する。

![隠れ状態を持つ RNN。](../img/rnn.svg)
:label:`fig_rnn`

先ほど、隠れ状態のための $\mathbf{X}_t \mathbf{W}_{\textrm{xh}} + \mathbf{H}_{t-1} \mathbf{W}_{\textrm{hh}}$ の計算は、
$\mathbf{X}_t$ と $\mathbf{H}_{t-1}$ の連結と、
$\mathbf{W}_{\textrm{xh}}$ と $\mathbf{W}_{\textrm{hh}}$ の連結との
行列積に等しいと述べました。
これは数学的に証明できるが、
以下では簡単なコード片で示すだけにする。
まず、
形状がそれぞれ (3, 1), (1, 4), (3, 4), (4, 4) の行列 `X`, `W_xh`, `H`, `W_hh` を定義する。
`X` に `W_xh` を掛け、`H` に `W_hh` を掛け、その 2 つの積を加えると、
形状 (3, 4) の行列が得られる。

```{.python .input}
%%tab mxnet, pytorch
X, W_xh = d2l.randn(3, 1), d2l.randn(1, 4)
H, W_hh = d2l.randn(3, 4), d2l.randn(4, 4)
d2l.matmul(X, W_xh) + d2l.matmul(H, W_hh)
```

```{.python .input}
%%tab tensorflow
X, W_xh = d2l.normal((3, 1)), d2l.normal((1, 4))
H, W_hh = d2l.normal((3, 4)), d2l.normal((4, 4))
d2l.matmul(X, W_xh) + d2l.matmul(H, W_hh)
```

```{.python .input}
%%tab jax
X, W_xh = jax.random.normal(d2l.get_key(), (3, 1)), jax.random.normal(
                                                        d2l.get_key(), (1, 4))
H, W_hh = jax.random.normal(d2l.get_key(), (3, 4)), jax.random.normal(
                                                        d2l.get_key(), (4, 4))
d2l.matmul(X, W_xh) + d2l.matmul(H, W_hh)
```

次に、行方向（axis 1）に沿って行列 `X` と `H` を連結し、
列方向（axis 0）に沿って行列
`W_xh` と `W_hh` を連結する。
これら 2 つの連結結果は、
それぞれ形状 (3, 5) と (5, 4) の行列になる。
この 2 つの連結した行列を掛け合わせると、
上と同じ形状 (3, 4) の出力行列が得られる。

```{.python .input}
%%tab all
d2l.matmul(d2l.concat((X, H), 1), d2l.concat((W_xh, W_hh), 0))
```

## RNN に基づく文字レベル言語モデル

:numref:`sec_language-model` での言語モデリングでは、
現在および過去のトークンに基づいて
次のトークンを予測することを目指した。
そのため、元の系列を 1 トークンずらしたものを目標（ラベル）として用いる。
:citet:`Bengio.Ducharme.Vincent.ea.2003` は、言語モデリングにニューラルネットワークを用いることを最初に提案しました。
以下では、RNN を用いて言語モデルを構築する方法を示す。
ミニバッチサイズを 1 とし、テキスト系列を "machine" とする。
後続の節での学習を簡単にするため、
テキストを単語ではなく文字にトークン化し、
*文字レベル言語モデル* を考える。
:numref:`fig_rnn_train` は、文字レベル言語モデリングのために RNN を通じて現在および過去の文字に基づいて次の文字を予測する方法を示しています。

![RNN に基づく文字レベル言語モデル。入力系列と目標系列はそれぞれ "machin" と "achine" である。](../img/rnn-train.svg)
:label:`fig_rnn_train`

学習過程では、
各時刻の出力層からの出力に対して softmax 演算を行い、その後、交差エントロピー損失を用いてモデル出力と目標との誤差を計算する。
隠れ層における隠れ状態の再帰的計算のため、 :numref:`fig_rnn_train` の時刻 3 における出力 $\mathbf{O}_3$ は、テキスト系列 "m", "a", "c" によって決まりる。系列の次の文字は学習データ中では "h" なので、時刻 3 の損失は、特徴系列 "m", "a", "c" に基づいて生成された次の文字の確率分布と、この時刻の目標 "h" に依存する。

実際には、各トークンは $d$ 次元ベクトルで表され、バッチサイズ $n>1$ を用いる。したがって、時刻 $t$ における入力 $\mathbf X_t$ は $n\times d$ 行列となり、これは :numref:`subsec_rnn_w_hidden_states` で説明した内容と同じである。

以下の節では、文字レベル言語モデルのための RNN を実装する。


## まとめ

隠れ状態に対して再帰的計算を用いるニューラルネットワークを、再帰型ニューラルネットワーク（RNN）と呼びる。
RNN の隠れ状態は、現在の時刻までの系列の履歴情報を捉える可能である。再帰的計算を用いることで、RNN のモデルパラメータ数は時刻数が増えても増加しません。応用として、RNN は文字レベル言語モデルの構築に利用できる。


## 演習

1. RNN を用いてテキスト系列中の次の文字を予測する場合、任意の出力に必要な次元はどれくらいですか。
1. なぜ RNN は、テキスト系列中のある時刻におけるトークンの条件付き確率を、それ以前のすべてのトークンに基づいて表現できるのですか。
1. 長い系列を逆伝播すると、勾配はどうなるか。
1. この節で説明した言語モデルに関連する問題にはどのようなものがあるか。
