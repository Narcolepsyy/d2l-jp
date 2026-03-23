{.python .input}
%load_ext d2lbook.tab
tab.interact_select('mxnet', 'pytorch', 'tensorflow', 'jax')
```

# 自己注意機構と位置エンコーディング
:label:`sec_self-attention-and-positional-encoding`

深層学習では、系列をエンコードするためにCNNやRNNをよく用います。
ここで注意機構を念頭に置くと、
トークン列を注意機構に入力し、
各ステップで各トークンがそれぞれ独自のクエリ、キー、値を持つ
と考えることができます。
ここで、次の層におけるあるトークンの表現の値を計算するとき、
そのトークンは（クエリベクトルを介して）他の任意のトークンに
（キー ベクトルに基づいて一致を取りながら）注意を向けることができます。
クエリとキーの適合度スコアの全体を用いることで、
各トークンについて、他のトークンに対する適切な重み付き和を構成し、
表現を計算できます。
各トークンが互いのトークンに注意を向けるため
（デコーダのステップがエンコーダのステップに注意を向ける場合とは異なり）、
このようなアーキテクチャは通常 *自己注意* モデル :cite:`Lin.Feng.Santos.ea.2017,Vaswani.Shazeer.Parmar.ea.2017` と呼ばれ、
別の文脈では *intra-attention* モデル :cite:`Cheng.Dong.Lapata.2016,Parikh.Tackstrom.Das.ea.2016,Paulus.Xiong.Socher.2017` とも呼ばれます。
この節では、系列の順序に関する追加情報も含めた、自己注意を用いる系列エンコーディングについて議論します。

```{.python .input}
%%tab mxnet
from d2l import mxnet as d2l
import math
from mxnet import autograd, np, npx
from mxnet.gluon import nn
npx.set_np()
```

```{.python .input}
%%tab pytorch
from d2l import torch as d2l
import math
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
from jax import numpy as jnp
import jax
```

## [**自己注意**]

入力トークン列
$\mathbf{x}_1, \ldots, \mathbf{x}_n$ があり、任意の $\mathbf{x}_i \in \mathbb{R}^d$（$1 \leq i \leq n$）とします。
その自己注意の出力は
同じ長さの系列
$\mathbf{y}_1, \ldots, \mathbf{y}_n$
であり、ここで

$$\mathbf{y}_i = f(\mathbf{x}_i, (\mathbf{x}_1, \mathbf{x}_1), \ldots, (\mathbf{x}_n, \mathbf{x}_n)) \in \mathbb{R}^d$$

は
:eqref:`eq_attention_pooling`
における注意プーリングの定義に従います。
マルチヘッド注意を用いると、
次のコード片は
形状が（バッチサイズ、時間ステップ数またはトークン列長、$d$）のテンソルに対する自己注意を計算します。
出力テンソルは同じ形状を持ちます。

```{.python .input}
%%tab pytorch
num_hiddens, num_heads = 100, 5
attention = d2l.MultiHeadAttention(num_hiddens, num_heads, 0.5)
batch_size, num_queries, valid_lens = 2, 4, d2l.tensor([3, 2])
X = d2l.ones((batch_size, num_queries, num_hiddens))
d2l.check_shape(attention(X, X, X, valid_lens),
                (batch_size, num_queries, num_hiddens))
```

```{.python .input}
%%tab mxnet
num_hiddens, num_heads = 100, 5
attention = d2l.MultiHeadAttention(num_hiddens, num_heads, 0.5)
attention.initialize()
```

```{.python .input}
%%tab jax
num_hiddens, num_heads = 100, 5
attention = d2l.MultiHeadAttention(num_hiddens, num_heads, 0.5)
```

```{.python .input}
%%tab tensorflow
num_hiddens, num_heads = 100, 5
attention = d2l.MultiHeadAttention(num_hiddens, num_hiddens, num_hiddens,
                                   num_hiddens, num_heads, 0.5)
```

```{.python .input}
%%tab mxnet
batch_size, num_queries, valid_lens = 2, 4, d2l.tensor([3, 2])
X = d2l.ones((batch_size, num_queries, num_hiddens))
d2l.check_shape(attention(X, X, X, valid_lens),
                (batch_size, num_queries, num_hiddens))
```

```{.python .input}
%%tab tensorflow
batch_size, num_queries, valid_lens = 2, 4, tf.constant([3, 2])
X = tf.ones((batch_size, num_queries, num_hiddens))
d2l.check_shape(attention(X, X, X, valid_lens, training=False),
                (batch_size, num_queries, num_hiddens))
```

```{.python .input}
%%tab jax
batch_size, num_queries, valid_lens = 2, 4, d2l.tensor([3, 2])
X = d2l.ones((batch_size, num_queries, num_hiddens))
d2l.check_shape(attention.init_with_output(d2l.get_key(), X, X, X, valid_lens,
                                           training=False)[0][0],
                (batch_size, num_queries, num_hiddens))
```

## CNN、RNN、自己注意の比較
:label:`subsec_cnn-rnn-self-attention`

$n$ 個のトークンからなる系列を
同じ長さの別の系列へ写像するアーキテクチャを比較しましょう。
ここで各入力トークンまたは出力トークンは
$d$ 次元ベクトルで表されます。
具体的には、
CNN、RNN、自己注意を考えます。
それらの
計算量、
逐次演算、
最大経路長を比較します。
逐次演算は並列計算を妨げる一方で、
系列位置の任意の組み合わせ間の経路が短いほど、
系列内の長距離依存関係を学習しやすくなります :cite:`Hochreiter.Bengio.Frasconi.ea.2001`。


![CNN（パディングトークンは省略）、RNN、自己注意アーキテクチャの比較。](../img/cnn-rnn-self-attention.svg)
:label:`fig_cnn-rnn-self-attention`



テキスト系列を「一次元画像」とみなしてみましょう。同様に、一次元CNNはテキスト中の $n$-gram のような局所特徴を処理できます。
長さ $n$ の系列を考え、
カーネルサイズが $k$、
入力チャネル数と出力チャネル数がともに $d$ の畳み込み層を考えます。
この畳み込み層の計算量は $\mathcal{O}(knd^2)$ です。
:numref:`fig_cnn-rnn-self-attention` が示すように、
CNNは階層的であるため、
逐次演算は $\mathcal{O}(1)$ で済み、
最大経路長は $\mathcal{O}(n/k)$ です。
たとえば、:numref:`fig_cnn-rnn-self-attention` では、
$\mathbf{x}_1$ と $\mathbf{x}_5$ は
カーネルサイズ 3 の2層CNNの受容野内にあります。

RNNの隠れ状態を更新するとき、
$d \times d$ の重み行列と
$d$ 次元の隠れ状態の乗算の計算量は
$\mathcal{O}(d^2)$ です。
系列長が $n$ なので、
再帰層の計算量は
$\mathcal{O}(nd^2)$ です。
:numref:`fig_cnn-rnn-self-attention` によれば、
並列化できない逐次演算が $\mathcal{O}(n)$ 回あり、
最大経路長も $\mathcal{O}(n)$ です。

自己注意では、
クエリ、キー、値はすべて
$n \times d$ 行列です。
:eqref:`eq_softmax_QK_V` のスケールド・ドット積注意を考えると、
$n \times d$ 行列に
$d \times n$ 行列を掛け、
その後、出力の $n \times n$ 行列に
$n \times d$ 行列を掛けます。
その結果、
自己注意の計算量は
$\mathcal{O}(n^2d)$ になります。
:numref:`fig_cnn-rnn-self-attention` からわかるように、
各トークンは自己注意を通じて
他の任意のトークンに直接接続されています。
したがって、
計算は $\mathcal{O}(1)$ の逐次演算で並列に行え、
最大経路長も $\mathcal{O}(1)$ です。

要するに、
CNNと自己注意はいずれも並列計算の恩恵を受け、
自己注意は最大経路長が最も短いです。
しかし、系列長に対して二次の計算量を持つため、
自己注意は非常に長い系列に対しては
極めて遅くなります。





## [**位置エンコーディング**]
:label:`subsec_positional-encoding`


RNNが系列のトークンを
1つずつ再帰的に処理するのに対し、
自己注意は
逐次演算を捨てて
並列計算を優先します。
ただし、自己注意だけでは
系列の順序は保持されません。
入力系列がどの順序で到着したかを
モデルが知っていることが本当に重要な場合、
どうすればよいでしょうか。

トークンの順序に関する情報を保持するための
主流の方法は、
各トークンに関連付けられた追加入力として
それをモデルに表現することです。
これらの入力は *位置エンコーディング* と呼ばれ、
学習可能なものにも、あらかじめ固定されたものにもできます。
ここでは、正弦関数と余弦関数に基づく
固定位置エンコーディングの簡単な方式を説明します :cite:`Vaswani.Shazeer.Parmar.ea.2017`。

入力表現
$\mathbf{X} \in \mathbb{R}^{n \times d}$
が系列中の $n$ 個のトークンの
$d$ 次元埋め込みを含むとします。
位置エンコーディングは
同じ形状の位置埋め込み行列
$\mathbf{P} \in \mathbb{R}^{n \times d}$ を用いて
$\mathbf{X} + \mathbf{P}$
を出力します。
その $i^\textrm{th}$ 行
および $(2j)^\textrm{th}$
または $(2j + 1)^\textrm{th}$ 列の要素は

$$\begin{aligned} p_{i, 2j} &= \sin\left(\frac{i}{10000^{2j/d}}\right),\\p_{i, 2j+1} &= \cos\left(\frac{i}{10000^{2j/d}}\right).\end{aligned}$$
:eqlabel:`eq_positional-encoding-def`

一見すると、
この三角関数を使った設計は奇妙に見えます。
この設計の理由を説明する前に、
まず次の `PositionalEncoding` クラスで実装してみましょう。

```{.python .input}
%%tab mxnet
class PositionalEncoding(nn.Block):  #@save
    """Positional encoding."""
    def __init__(self, num_hiddens, dropout, max_len=1000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        # Create a long enough P
        self.P = d2l.zeros((1, max_len, num_hiddens))
        X = d2l.arange(max_len).reshape(-1, 1) / np.power(
            10000, np.arange(0, num_hiddens, 2) / num_hiddens)
        self.P[:, :, 0::2] = np.sin(X)
        self.P[:, :, 1::2] = np.cos(X)

    def forward(self, X):
        X = X + self.P[:, :X.shape[1], :].as_in_ctx(X.ctx)
        return self.dropout(X)
```

```{.python .input}
%%tab pytorch
class PositionalEncoding(nn.Module):  #@save
    """Positional encoding."""
    def __init__(self, num_hiddens, dropout, max_len=1000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        # Create a long enough P
        self.P = d2l.zeros((1, max_len, num_hiddens))
        X = d2l.arange(max_len, dtype=torch.float32).reshape(
            -1, 1) / torch.pow(10000, torch.arange(
            0, num_hiddens, 2, dtype=torch.float32) / num_hiddens)
        self.P[:, :, 0::2] = torch.sin(X)
        self.P[:, :, 1::2] = torch.cos(X)

    def forward(self, X):
        X = X + self.P[:, :X.shape[1], :].to(X.device)
        return self.dropout(X)
```

```{.python .input}
%%tab tensorflow
class PositionalEncoding(tf.keras.layers.Layer):  #@save
    """Positional encoding."""
    def __init__(self, num_hiddens, dropout, max_len=1000):
        super().__init__()
        self.dropout = tf.keras.layers.Dropout(dropout)
        # Create a long enough P
        self.P = np.zeros((1, max_len, num_hiddens))
        X = np.arange(max_len, dtype=np.float32).reshape(
            -1,1)/np.power(10000, np.arange(
            0, num_hiddens, 2, dtype=np.float32) / num_hiddens)
        self.P[:, :, 0::2] = np.sin(X)
        self.P[:, :, 1::2] = np.cos(X)
        
    def call(self, X, **kwargs):
        X = X + self.P[:, :X.shape[1], :]
        return self.dropout(X, **kwargs)
```

```{.python .input}
%%tab jax
class PositionalEncoding(nn.Module):  #@save
    """Positional encoding."""
    num_hiddens: int
    dropout: float
    max_len: int = 1000

    def setup(self):
        # Create a long enough P
        self.P = d2l.zeros((1, self.max_len, self.num_hiddens))
        X = d2l.arange(self.max_len, dtype=jnp.float32).reshape(
            -1, 1) / jnp.power(10000, jnp.arange(
            0, self.num_hiddens, 2, dtype=jnp.float32) / self.num_hiddens)
        self.P = self.P.at[:, :, 0::2].set(jnp.sin(X))
        self.P = self.P.at[:, :, 1::2].set(jnp.cos(X))

    @nn.compact
    def __call__(self, X, training=False):
        # Flax sow API is used to capture intermediate variables
        self.sow('intermediates', 'P', self.P)
        X = X + self.P[:, :X.shape[1], :]
        return nn.Dropout(self.dropout)(X, deterministic=not training)
```

位置埋め込み行列 $\mathbf{P}$ では、
[**行は系列内の位置に対応し、
列は異なる位置エンコーディング次元を表します**]。
以下の例では、
位置埋め込み行列の
$6^{\textrm{th}}$ 列と $7^{\textrm{th}}$ 列は、
$8^{\textrm{th}}$ 列と $9^{\textrm{th}}$ 列よりも
高い周波数を持つことがわかります。
$6^{\textrm{th}}$ 列と $7^{\textrm{th}}$ 列
（$8^{\textrm{th}}$ 列と $9^{\textrm{th}}$ 列も同様）の間のオフセットは、
正弦関数と余弦関数を交互に用いていることに由来します。

```{.python .input}
%%tab mxnet
encoding_dim, num_steps = 32, 60
pos_encoding = PositionalEncoding(encoding_dim, 0)
pos_encoding.initialize()
X = pos_encoding(np.zeros((1, num_steps, encoding_dim)))
P = pos_encoding.P[:, :X.shape[1], :]
d2l.plot(d2l.arange(num_steps), P[0, :, 6:10].T, xlabel='Row (position)',
         figsize=(6, 2.5), legend=["Col %d" % d for d in d2l.arange(6, 10)])
```

```{.python .input}
%%tab pytorch
encoding_dim, num_steps = 32, 60
pos_encoding = PositionalEncoding(encoding_dim, 0)
X = pos_encoding(d2l.zeros((1, num_steps, encoding_dim)))
P = pos_encoding.P[:, :X.shape[1], :]
d2l.plot(d2l.arange(num_steps), P[0, :, 6:10].T, xlabel='Row (position)',
         figsize=(6, 2.5), legend=["Col %d" % d for d in d2l.arange(6, 10)])
```

```{.python .input}
%%tab tensorflow
encoding_dim, num_steps = 32, 60
pos_encoding = PositionalEncoding(encoding_dim, 0)
X = pos_encoding(tf.zeros((1, num_steps, encoding_dim)), training=False)
P = pos_encoding.P[:, :X.shape[1], :]
d2l.plot(np.arange(num_steps), P[0, :, 6:10].T, xlabel='Row (position)',
         figsize=(6, 2.5), legend=["Col %d" % d for d in np.arange(6, 10)])
```

```{.python .input}
%%tab jax
encoding_dim, num_steps = 32, 60
pos_encoding = PositionalEncoding(encoding_dim, 0)
params = pos_encoding.init(d2l.get_key(), d2l.zeros((1, num_steps, encoding_dim)))
X, inter_vars = pos_encoding.apply(params, d2l.zeros((1, num_steps, encoding_dim)),
                                   mutable='intermediates')
P = inter_vars['intermediates']['P'][0]  # retrieve intermediate value P
P = P[:, :X.shape[1], :]
d2l.plot(d2l.arange(num_steps), P[0, :, 6:10].T, xlabel='Row (position)',
         figsize=(6, 2.5), legend=["Col %d" % d for d in d2l.arange(6, 10)])
```

### 絶対的な位置情報

エンコーディング次元に沿って周波数が単調に減少することが
絶対的な位置情報とどのように関係するかを見るために、
$0, 1, \ldots, 7$ の[**2進表現**]を出力してみましょう。
ご覧のように、最下位ビット、下から2番目のビット、
下から3番目のビットは、それぞれ1つおき、2つおき、4つおきに変化します。

```{.python .input}
%%tab all
for i in range(8):
    print(f'{i} in binary is {i:>03b}')
```

2進表現では、上位ビットほど下位ビットよりも周波数が低くなります。
同様に、下のヒートマップが示すように、
[**位置エンコーディングは三角関数を用いて
エンコーディング次元に沿って周波数を減少させます**]。
出力は浮動小数点数なので、
このような連続表現は
2進表現よりも
空間効率が高いです。

```{.python .input}
%%tab mxnet
P = np.expand_dims(np.expand_dims(P[0, :, :], 0), 0)
d2l.show_heatmaps(P, xlabel='Column (encoding dimension)',
                  ylabel='Row (position)', figsize=(3.5, 4), cmap='Blues')
```

```{.python .input}
%%tab pytorch
P = P[0, :, :].unsqueeze(0).unsqueeze(0)
d2l.show_heatmaps(P, xlabel='Column (encoding dimension)',
                  ylabel='Row (position)', figsize=(3.5, 4), cmap='Blues')
```

```{.python .input}
%%tab tensorflow
P = tf.expand_dims(tf.expand_dims(P[0, :, :], axis=0), axis=0)
d2l.show_heatmaps(P, xlabel='Column (encoding dimension)',
                  ylabel='Row (position)', figsize=(3.5, 4), cmap='Blues')
```

```{.python .input}
%%tab jax
P = jnp.expand_dims(jnp.expand_dims(P[0, :, :], axis=0), axis=0)
d2l.show_heatmaps(P, xlabel='Column (encoding dimension)',
                  ylabel='Row (position)', figsize=(3.5, 4), cmap='Blues')
```

### 相対的な位置情報

絶対的な位置情報を捉えることに加えて、
上記の位置エンコーディングは
モデルが相対位置に基づいて容易に注意を学習することも可能にします。
これは、
任意の固定位置オフセット $\delta$ に対して、
位置 $i + \delta$ における位置エンコーディングが
位置 $i$ におけるものの線形射影として表せるからです。


この射影は
数学的に説明できます。
$\omega_j = 1/10000^{2j/d}$ とおくと、
:eqref:`eq_positional-encoding-def`
における任意の $(p_{i, 2j}, p_{i, 2j+1})$
の組は、
任意の固定オフセット $\delta$ に対して
$(p_{i+\delta, 2j}, p_{i+\delta, 2j+1})$
へ線形射影できます。

$$\begin{aligned}
\begin{bmatrix} \cos(\delta \omega_j) & \sin(\delta \omega_j) \\  -\sin(\delta \omega_j) & \cos(\delta \omega_j) \\ \end{bmatrix}
\begin{bmatrix} p_{i, 2j} \\  p_{i, 2j+1} \\ \end{bmatrix}
=&\begin{bmatrix} \cos(\delta \omega_j) \sin(i \omega_j) + \sin(\delta \omega_j) \cos(i \omega_j) \\  -\sin(\delta \omega_j) \sin(i \omega_j) + \cos(\delta \omega_j) \cos(i \omega_j) \\ \end{bmatrix}\\
=&\begin{bmatrix} \sin\left((i+\delta) \omega_j\right) \\  \cos\left((i+\delta) \omega_j\right) \\ \end{bmatrix}\\
=& 
\begin{bmatrix} p_{i+\delta, 2j} \\  p_{i+\delta, 2j+1} \\ \end{bmatrix},
\end{aligned}$$

ここで、$2\times 2$ の射影行列は
どの位置インデックス $i$ にも依存しません。

## まとめ

自己注意では、クエリ、キー、値はすべて同じ場所から来ます。
CNNと自己注意はいずれも並列計算の恩恵を受け、
自己注意は最大経路長が最も短いです。
しかし、系列長に対して二次の計算量を持つため、
自己注意は非常に長い系列に対しては
極めて遅くなります。
系列順序の情報を使うには、
入力表現に位置エンコーディングを加えることで、
絶対的または相対的な位置情報を注入できます。

## 演習

1. 位置エンコーディングを用いた自己注意層を積み重ねることで系列を表現する深いアーキテクチャを設計するとします。どのような問題が起こりうるでしょうか？
1. 学習可能な位置エンコーディング手法を設計できますか？
1. 自己注意で比較されるクエリとキーの間の異なるオフセットに応じて、異なる学習済み埋め込みを割り当てることはできますか？ ヒント：相対位置埋め込みを参照してください :cite:`shaw2018self,huang2018music`。

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/1651)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1652)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/3870)
:end_tab:

:begin_tab:`jax`
[Discussions](https://discuss.d2l.ai/t/18030)
:end_tab:\n