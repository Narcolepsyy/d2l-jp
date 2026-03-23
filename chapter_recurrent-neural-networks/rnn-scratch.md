# ゼロからの再帰ニューラルネットワークの実装
:label:`sec_rnn-scratch`

これで、RNNをゼロから実装する準備が整いました。
特に、このRNNを
文字レベルの言語モデルとして機能するように学習させます
(:numref:`sec_rnn` を参照)。
また、H. G. ウェルズの *The Time Machine* の全文からなるコーパスで学習し、
:numref:`sec_text-sequence` で概説した
データ処理手順に従います。
まずデータセットを読み込みます。

```{.python .input}
%load_ext d2lbook.tab
tab.interact_select('mxnet', 'pytorch', 'tensorflow', 'jax')
```

```{.python .input  n=2}
%%tab mxnet
%matplotlib inline
from d2l import mxnet as d2l
import math
from mxnet import autograd, gluon, np, npx
npx.set_np()
```

```{.python .input}
%%tab pytorch
%matplotlib inline
from d2l import torch as d2l
import math
import torch
from torch import nn
from torch.nn import functional as F
```

```{.python .input}
%%tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
import math
import tensorflow as tf
```

```{.python .input  n=5}
%%tab jax
%matplotlib inline
from d2l import jax as d2l
from flax import linen as nn
import jax
from jax import numpy as jnp
import math
```

## RNNモデル

まず、RNNモデルを実装するためのクラスを定義します
(:numref:`subsec_rnn_w_hidden_states`)。
隠れユニット数 `num_hiddens` は
調整可能なハイパーパラメータであることに注意してください。

```{.python .input}
%%tab pytorch, mxnet, tensorflow
class RNNScratch(d2l.Module):  #@save
    """The RNN model implemented from scratch."""
    def __init__(self, num_inputs, num_hiddens, sigma=0.01):
        super().__init__()
        self.save_hyperparameters()
        if tab.selected('mxnet'):
            self.W_xh = d2l.randn(num_inputs, num_hiddens) * sigma
            self.W_hh = d2l.randn(
                num_hiddens, num_hiddens) * sigma
            self.b_h = d2l.zeros(num_hiddens)
        if tab.selected('pytorch'):
            self.W_xh = nn.Parameter(
                d2l.randn(num_inputs, num_hiddens) * sigma)
            self.W_hh = nn.Parameter(
                d2l.randn(num_hiddens, num_hiddens) * sigma)
            self.b_h = nn.Parameter(d2l.zeros(num_hiddens))
        if tab.selected('tensorflow'):
            self.W_xh = tf.Variable(d2l.normal(
                (num_inputs, num_hiddens)) * sigma)
            self.W_hh = tf.Variable(d2l.normal(
                (num_hiddens, num_hiddens)) * sigma)
            self.b_h = tf.Variable(d2l.zeros(num_hiddens))
```

```{.python .input  n=7}
%%tab jax
class RNNScratch(nn.Module):  #@save
    """The RNN model implemented from scratch."""
    num_inputs: int
    num_hiddens: int
    sigma: float = 0.01

    def setup(self):
        self.W_xh = self.param('W_xh', nn.initializers.normal(self.sigma),
                               (self.num_inputs, self.num_hiddens))
        self.W_hh = self.param('W_hh', nn.initializers.normal(self.sigma),
                               (self.num_hiddens, self.num_hiddens))
        self.b_h = self.param('b_h', nn.initializers.zeros, (self.num_hiddens))
```

[**以下の `forward` メソッドは、現在の入力と前の時刻のモデル状態が与えられたときに、
任意の時刻における出力と隠れ状態をどのように計算するかを定義します。**]
RNNモデルは `inputs` の最外側の次元に沿ってループし、
隠れ状態を1時刻ずつ更新することに注意してください。
ここでのモデルは $\tanh$ 活性化関数を使います
(:numref:`subsec_tanh`)。

```{.python .input}
%%tab pytorch, mxnet, tensorflow
@d2l.add_to_class(RNNScratch)  #@save
def forward(self, inputs, state=None):
    if state is None:
        # Initial state with shape: (batch_size, num_hiddens)
        if tab.selected('mxnet'):
            state = d2l.zeros((inputs.shape[1], self.num_hiddens),
                              ctx=inputs.ctx)
        if tab.selected('pytorch'):
            state = d2l.zeros((inputs.shape[1], self.num_hiddens),
                              device=inputs.device)
        if tab.selected('tensorflow'):
            state = d2l.zeros((inputs.shape[1], self.num_hiddens))
    else:
        state, = state
        if tab.selected('tensorflow'):
            state = d2l.reshape(state, (-1, self.num_hiddens))
    outputs = []
    for X in inputs:  # Shape of inputs: (num_steps, batch_size, num_inputs) 
        state = d2l.tanh(d2l.matmul(X, self.W_xh) +
                         d2l.matmul(state, self.W_hh) + self.b_h)
        outputs.append(state)
    return outputs, state
```

```{.python .input  n=9}
%%tab jax
@d2l.add_to_class(RNNScratch)  #@save
def __call__(self, inputs, state=None):
    if state is not None:
        state, = state
    outputs = []
    for X in inputs:  # Shape of inputs: (num_steps, batch_size, num_inputs) 
        state = d2l.tanh(d2l.matmul(X, self.W_xh) + (
            d2l.matmul(state, self.W_hh) if state is not None else 0)
                         + self.b_h)
        outputs.append(state)
    return outputs, state
```

RNNモデルに入力シーケンスのミニバッチを次のように与えることができます。

```{.python .input}
%%tab pytorch, mxnet, tensorflow
batch_size, num_inputs, num_hiddens, num_steps = 2, 16, 32, 100
rnn = RNNScratch(num_inputs, num_hiddens)
X = d2l.ones((num_steps, batch_size, num_inputs))
outputs, state = rnn(X)
```

```{.python .input  n=11}
%%tab jax
batch_size, num_inputs, num_hiddens, num_steps = 2, 16, 32, 100
rnn = RNNScratch(num_inputs, num_hiddens)
X = d2l.ones((num_steps, batch_size, num_inputs))
(outputs, state), _ = rnn.init_with_output(d2l.get_key(), X)
```

RNNモデルが正しい形状の結果を出力し、
隠れ状態の次元が変わらないことを確認しましょう。

```{.python .input}
%%tab all
def check_len(a, n):  #@save
    """Check the length of a list."""
    assert len(a) == n, f'list\'s length {len(a)} != expected length {n}'
    
def check_shape(a, shape):  #@save
    """Check the shape of a tensor."""
    assert a.shape == shape, \
            f'tensor\'s shape {a.shape} != expected shape {shape}'

check_len(outputs, num_steps)
check_shape(outputs[0], (batch_size, num_hiddens))
check_shape(state, (batch_size, num_hiddens))
```

## RNNベースの言語モデル

以下の `RNNLMScratch` クラスは、
RNNベースの言語モデルを定義します。
ここでは `__init__` メソッドの `rnn` 引数を通して
RNNを渡します。
言語モデルを学習するとき、入力と出力は
同じ語彙から来ます。
したがって、それらの次元は同じであり、
語彙サイズに等しくなります。
モデルの評価には困惑度を使うことに注意してください。
:numref:`subsec_perplexity` で述べたように、これにより
長さの異なる系列を比較できます。

```{.python .input}
%%tab pytorch
class RNNLMScratch(d2l.Classifier):  #@save
    """The RNN-based language model implemented from scratch."""
    def __init__(self, rnn, vocab_size, lr=0.01):
        super().__init__()
        self.save_hyperparameters()
        self.init_params()
        
    def init_params(self):
        self.W_hq = nn.Parameter(
            d2l.randn(
                self.rnn.num_hiddens, self.vocab_size) * self.rnn.sigma)
        self.b_q = nn.Parameter(d2l.zeros(self.vocab_size)) 

    def training_step(self, batch):
        l = self.loss(self(*batch[:-1]), batch[-1])
        self.plot('ppl', d2l.exp(l), train=True)
        return l
        
    def validation_step(self, batch):
        l = self.loss(self(*batch[:-1]), batch[-1])
        self.plot('ppl', d2l.exp(l), train=False)
```

```{.python .input}
%%tab mxnet, tensorflow
class RNNLMScratch(d2l.Classifier):  #@save
    """The RNN-based language model implemented from scratch."""
    def __init__(self, rnn, vocab_size, lr=0.01):
        super().__init__()
        self.save_hyperparameters()
        self.init_params()
        
    def init_params(self):
        if tab.selected('mxnet'):
            self.W_hq = d2l.randn(
                self.rnn.num_hiddens, self.vocab_size) * self.rnn.sigma
            self.b_q = d2l.zeros(self.vocab_size)        
            for param in self.get_scratch_params():
                param.attach_grad()
        if tab.selected('tensorflow'):
            self.W_hq = tf.Variable(d2l.normal(
                (self.rnn.num_hiddens, self.vocab_size)) * self.rnn.sigma)
            self.b_q = tf.Variable(d2l.zeros(self.vocab_size))
        
    def training_step(self, batch):
        l = self.loss(self(*batch[:-1]), batch[-1])
        self.plot('ppl', d2l.exp(l), train=True)
        return l
        
    def validation_step(self, batch):
        l = self.loss(self(*batch[:-1]), batch[-1])
        self.plot('ppl', d2l.exp(l), train=False)
```

```{.python .input  n=14}
%%tab jax
class RNNLMScratch(d2l.Classifier):  #@save
    """The RNN-based language model implemented from scratch."""
    rnn: nn.Module
    vocab_size: int
    lr: float = 0.01

    def setup(self):
        self.W_hq = self.param('W_hq', nn.initializers.normal(self.rnn.sigma),
                               (self.rnn.num_hiddens, self.vocab_size))
        self.b_q = self.param('b_q', nn.initializers.zeros, (self.vocab_size))

    def training_step(self, params, batch, state):
        value, grads = jax.value_and_grad(
            self.loss, has_aux=True)(params, batch[:-1], batch[-1], state)
        l, _ = value
        self.plot('ppl', d2l.exp(l), train=True)
        return value, grads

    def validation_step(self, params, batch, state):
        l, _ = self.loss(params, batch[:-1], batch[-1], state)
        self.plot('ppl', d2l.exp(l), train=False)
```

### [**ワンホットエンコーディング**]

各トークンは、対応する単語/文字/単語片が語彙内のどの位置にあるかを示す数値インデックスで表されることを思い出してください。
各時刻に1つの入力ノードだけを持つニューラルネットワークを構築し、
そのインデックスをスカラー値として入力することを考えたくなるかもしれません。
これは、価格や温度のような数値入力を扱う場合にはうまくいきます。
そのような場合、十分に近い任意の2つの値は
同様に扱うべきだからです。
しかし、これはここではあまり意味をなしません。
語彙の45番目と46番目の単語はたまたま "their" と "said" ですが、
その意味はまったく似ていません。

このようなカテゴリデータを扱うとき、
最も一般的な戦略は各項目を *ワンホットエンコーディング* で表すことです
(:numref:`subsec_classification-problem` を参照)。
ワンホットエンコーディングとは、長さが語彙サイズ $N$ で与えられるベクトルであり、
トークンに対応する要素だけが $1$ に設定され、
それ以外の要素はすべて $0$ に設定されます。
たとえば、語彙に5個の要素があるなら、
インデックス0と2に対応するワンホットベクトルは次のようになります。

```{.python .input}
%%tab mxnet
npx.one_hot(np.array([0, 2]), 5)
```

```{.python .input}
%%tab pytorch
F.one_hot(torch.tensor([0, 2]), 5)
```

```{.python .input}
%%tab tensorflow
tf.one_hot(tf.constant([0, 2]), 5)
```

```{.python .input  n=18}
%%tab jax
jax.nn.one_hot(jnp.array([0, 2]), 5)
```

(**各反復でサンプリングされるミニバッチは、(バッチサイズ, 時間ステップ数) の形を取ります。
各入力をワンホットベクトルとして表現すると、
各ミニバッチは3次元テンソルとみなせます。
その第3軸方向の長さは語彙サイズ (`len(vocab)`) で与えられます。**)
入力を転置して、
(時間ステップ数, バッチサイズ, 語彙サイズ) の形の出力を得ることがよくあります。
これにより、ミニバッチの隠れ状態を時刻ごとに更新するために、
最外側の次元に沿ってより便利にループできます
（たとえば、上の `forward` メソッドのように）。

```{.python .input}
%%tab all
@d2l.add_to_class(RNNLMScratch)  #@save
def one_hot(self, X):    
    # Output shape: (num_steps, batch_size, vocab_size)    
    if tab.selected('mxnet'):
        return npx.one_hot(X.T, self.vocab_size)
    if tab.selected('pytorch'):
        return F.one_hot(X.T, self.vocab_size).type(torch.float32)
    if tab.selected('tensorflow'):
        return tf.one_hot(tf.transpose(X), self.vocab_size)
    if tab.selected('jax'):
        return jax.nn.one_hot(X.T, self.vocab_size)
```

### RNN出力の変換

言語モデルは全結合出力層を使って、
各時刻のRNN出力をトークン予測へ変換します。

```{.python .input}
%%tab all
@d2l.add_to_class(RNNLMScratch)  #@save
def output_layer(self, rnn_outputs):
    outputs = [d2l.matmul(H, self.W_hq) + self.b_q for H in rnn_outputs]
    return d2l.stack(outputs, 1)

@d2l.add_to_class(RNNLMScratch)  #@save
def forward(self, X, state=None):
    embs = self.one_hot(X)
    rnn_outputs, _ = self.rnn(embs, state)
    return self.output_layer(rnn_outputs)
```

[**順伝播計算が正しい形状の出力を生成するか確認しましょう。**]

```{.python .input}
%%tab pytorch, mxnet, tensorflow
model = RNNLMScratch(rnn, num_inputs)
outputs = model(d2l.ones((batch_size, num_steps), dtype=d2l.int64))
check_shape(outputs, (batch_size, num_steps, num_inputs))
```

```{.python .input  n=23}
%%tab jax
model = RNNLMScratch(rnn, num_inputs)
outputs, _ = model.init_with_output(d2l.get_key(),
                                    d2l.ones((batch_size, num_steps),
                                             dtype=d2l.int32))
check_shape(outputs, (batch_size, num_steps, num_inputs))
```

## [**勾配クリッピング**]


すでに、ニューラルネットワークを「深い」と考えるとき、
単一の時刻の中でも入力と出力の間に多くの層があるという意味で深いことに慣れているでしょうが、
系列の長さは新たな深さの概念を導入します。
入力から出力への方向でネットワークを通過することに加えて、
最初の時刻の入力は、モデルの最終時刻での出力に影響を与えるために、
時刻に沿って $T$ 層からなる連鎖を通過しなければなりません。
逆方向から見ると、各反復で時間方向に逆伝播を行うため、
長さ $\mathcal{O}(T)$ の行列積の連鎖が生じます。
:numref:`sec_numerical_stability` で述べたように、
これは数値的不安定性を引き起こし、
重み行列の性質に応じて勾配が爆発したり消失したりします。 

消失勾配と爆発勾配への対処は、RNNを設計するうえでの根本的な問題であり、
現代のニューラルネットワークアーキテクチャにおける最大級の進歩のいくつかを生み出す原動力となってきました。
次の章では、消失勾配問題の緩和を目指して設計された
特殊なアーキテクチャについて説明します。
しかし、現代のRNNであっても、
爆発勾配に悩まされることは少なくありません。
洗練されてはいないものの広く使われている解決策の1つは、
勾配を単純にクリップして、
結果として得られる「クリップされた」勾配の値を小さくすることです。 


一般に、勾配降下法で何らかの目的関数を最適化するとき、
たとえばベクトル $\mathbf{x}$ のような注目するパラメータを反復的に更新しますが、
その際には負の勾配 $\mathbf{g}$ の方向へ押し進めます
（確率的勾配降下法では、この勾配をランダムにサンプリングしたミニバッチ上で計算します）。
たとえば、学習率 $\eta > 0$ のとき、
各更新は
$\mathbf{x} \gets \mathbf{x} - \eta \mathbf{g}$ の形を取ります。
さらに、目的関数 $f$ が十分に滑らかであると仮定しましょう。
形式的には、目的関数が定数 $L$ の *Lipschitz連続* であるといい、
任意の $\mathbf{x}$ と $\mathbf{y}$ に対して

$$|f(\mathbf{x}) - f(\mathbf{y})| \leq L \|\mathbf{x} - \mathbf{y}\|.$$

ご覧のとおり、パラメータベクトルから $\eta \mathbf{g}$ を引いて更新するとき、
目的関数の値の変化は、学習率、勾配のノルム、および $L$ に次のように依存します。

$$|f(\mathbf{x}) - f(\mathbf{x} - \eta\mathbf{g})| \leq L \eta\|\mathbf{g}\|.$$

言い換えると、目的関数は $L \eta \|\mathbf{g}\|$ を超えて変化することはありません。
この上界が小さいことは、良いことにも悪いことにもなりえます。
欠点としては、目的関数の値を減らせる速度を制限してしまうことです。
一方で、1回の勾配ステップでどれだけ大きく失敗しうるかを抑えるという利点があります。


勾配が爆発するとは、
$\|\mathbf{g}\|$ が過度に大きくなることを意味します。
この最悪の場合、1回の勾配ステップで非常に大きな損害を与えてしまい、
数千回の学習反復で得られた進歩をすべて打ち消してしまうことさえあります。
勾配が非常に大きくなりうると、
ニューラルネットワークの学習はしばしば発散し、
目的関数の値を減らせなくなります。
また別の場合には、最終的には収束するものの、
損失の大きなスパイクのために不安定になります。


$L \eta \|\mathbf{g}\|$ の大きさを抑える1つの方法は、
学習率 $\eta$ を非常に小さな値に縮小することです。
この利点は、更新にバイアスを導入しないことです。
しかし、大きな勾配が起こるのが *まれ* でしかない場合はどうでしょうか。
この思い切った対策は、まれな爆発勾配イベントに対処するためだけに、
すべてのステップでの進歩を遅くしてしまいます。
よく使われる代替案は、*勾配クリッピング* のヒューリスティックを採用し、
次のように勾配 $\mathbf{g}$ を半径 $\theta$ の球へ射影することです。

(**$$\mathbf{g} \leftarrow \min\left(1, \frac{\theta}{\|\mathbf{g}\|}\right) \mathbf{g}.$$**)

これにより、勾配ノルムが $\theta$ を超えないことが保証され、
更新後の勾配が元の $\mathbf{g}$ の方向と完全に整列したままであることも保証されます。
また、任意のミニバッチ（およびその中の任意のサンプル）が
パラメータベクトルに及ぼしうる影響を制限するという望ましい副作用もあります。
これにより、モデルにある程度の頑健性が与えられます。
はっきり言えば、これはハックです。
勾配クリッピングは、常に真の勾配に従っているわけではないことを意味し、
考えられる副作用を解析的に理解するのは困難です。
しかし、非常に有用なハックであり、
ほとんどの深層学習フレームワークにおけるRNN実装で広く採用されています。


以下では、勾配をクリップするメソッドを定義します。
これは `d2l.Trainer` クラスの `fit_epoch` メソッドから呼び出されます
(:numref:`sec_linear_scratch` を参照)。
勾配ノルムを計算するときは、
すべてのモデルパラメータを連結して、
1つの巨大なパラメータベクトルとして扱っていることに注意してください。

```{.python .input}
%%tab mxnet
@d2l.add_to_class(d2l.Trainer)  #@save
def clip_gradients(self, grad_clip_val, model):
    params = model.parameters()
    if not isinstance(params, list):
        params = [p.data() for p in params.values()]    
    norm = math.sqrt(sum((p.grad ** 2).sum() for p in params))
    if norm > grad_clip_val:
        for param in params:
            param.grad[:] *= grad_clip_val / norm
```

```{.python .input}
%%tab pytorch
@d2l.add_to_class(d2l.Trainer)  #@save
def clip_gradients(self, grad_clip_val, model):
    params = [p for p in model.parameters() if p.requires_grad]
    norm = torch.sqrt(sum(torch.sum((p.grad ** 2)) for p in params))
    if norm > grad_clip_val:
        for param in params:
            param.grad[:] *= grad_clip_val / norm
```

```{.python .input}
%%tab tensorflow
@d2l.add_to_class(d2l.Trainer)  #@save
def clip_gradients(self, grad_clip_val, grads):
    grad_clip_val = tf.constant(grad_clip_val, dtype=tf.float32)
    new_grads = [tf.convert_to_tensor(grad) if isinstance(
        grad, tf.IndexedSlices) else grad for grad in grads]    
    norm = tf.math.sqrt(sum((tf.reduce_sum(grad ** 2)) for grad in new_grads))
    if tf.greater(norm, grad_clip_val):
        for i, grad in enumerate(new_grads):
            new_grads[i] = grad * grad_clip_val / norm
        return new_grads
    return grads
```

```{.python .input  n=27}
%%tab jax
@d2l.add_to_class(d2l.Trainer)  #@save
def clip_gradients(self, grad_clip_val, grads):
    grad_leaves, _ = jax.tree_util.tree_flatten(grads)
    norm = jnp.sqrt(sum(jnp.vdot(x, x) for x in grad_leaves))
    clip = lambda grad: jnp.where(norm < grad_clip_val,
                                  grad, grad * (grad_clip_val / norm))
    return jax.tree_util.tree_map(clip, grads)
```

## 学習

*The Time Machine* データセット (`data`) を使って、
ゼロから実装したRNN (`rnn`) に基づく
文字レベルの言語モデル (`model`) を学習します。
まず勾配を計算し、
次にそれらをクリップし、
最後にクリップされた勾配を使って
モデルパラメータを更新することに注意してください。

```{.python .input}
%%tab all
data = d2l.TimeMachine(batch_size=1024, num_steps=32)
if tab.selected('mxnet', 'pytorch', 'jax'):
    rnn = RNNScratch(num_inputs=len(data.vocab), num_hiddens=32)
    model = RNNLMScratch(rnn, vocab_size=len(data.vocab), lr=1)
    trainer = d2l.Trainer(max_epochs=100, gradient_clip_val=1, num_gpus=1)
if tab.selected('tensorflow'):
    with d2l.try_gpu():
        rnn = RNNScratch(num_inputs=len(data.vocab), num_hiddens=32)
        model = RNNLMScratch(rnn, vocab_size=len(data.vocab), lr=1)
    trainer = d2l.Trainer(max_epochs=100, gradient_clip_val=1)
trainer.fit(model, data)
```

## デコード

言語モデルが学習されると、
次のトークンを予測するだけでなく、
それ以降の各トークンを順に予測し続けることができます。
その際、直前に予測したトークンを
次の入力であるかのように扱います。
文書の先頭から始めるかのように
テキストを生成したいだけの場合もあります。
しかし、多くの場合は、
ユーザーが与えたプレフィックスに条件づけて
言語モデルを使うと便利です。
たとえば、検索エンジンのオートコンプリート機能を開発していたり、
メール作成を支援したりする場合には、
ユーザーがここまで入力した内容（プレフィックス）を与え、
その続きとしてありそうなテキストを生成したいでしょう。


[**以下の `predict` メソッドは、
ユーザーが与えた `prefix` を取り込んだあと、
1文字ずつ継続を生成します**]。
`prefix` の文字をループするとき、
隠れ状態を次の時刻へ渡し続けますが、
出力は生成しません。
これを *ウォームアップ* 期間と呼びます。
プレフィックスを取り込んだ後は、
以降の文字の出力を開始する準備が整います。
それぞれの文字は次の時刻の入力として
モデルにフィードバックされます。

```{.python .input}
%%tab pytorch, mxnet, tensorflow
@d2l.add_to_class(RNNLMScratch)  #@save
def predict(self, prefix, num_preds, vocab, device=None):
    state, outputs = None, [vocab[prefix[0]]]
    for i in range(len(prefix) + num_preds - 1):
        if tab.selected('mxnet'):
            X = d2l.tensor([[outputs[-1]]], ctx=device)
        if tab.selected('pytorch'):
            X = d2l.tensor([[outputs[-1]]], device=device)
        if tab.selected('tensorflow'):
            X = d2l.tensor([[outputs[-1]]])
        embs = self.one_hot(X)
        rnn_outputs, state = self.rnn(embs, state)
        if i < len(prefix) - 1:  # Warm-up period
            outputs.append(vocab[prefix[i + 1]])
        else:  # Predict num_preds steps
            Y = self.output_layer(rnn_outputs)
            outputs.append(int(d2l.reshape(d2l.argmax(Y, axis=2), 1)))
    return ''.join([vocab.idx_to_token[i] for i in outputs])
```

```{.python .input}
%%tab jax
@d2l.add_to_class(RNNLMScratch)  #@save
def predict(self, prefix, num_preds, vocab, params):
    state, outputs = None, [vocab[prefix[0]]]
    for i in range(len(prefix) + num_preds - 1):
        X = d2l.tensor([[outputs[-1]]])
        embs = self.one_hot(X)
        rnn_outputs, state = self.rnn.apply({'params': params['rnn']},
                                            embs, state)
        if i < len(prefix) - 1:  # Warm-up period
            outputs.append(vocab[prefix[i + 1]])
        else:  # Predict num_preds steps
            Y = self.apply({'params': params}, rnn_outputs,
                           method=self.output_layer)
            outputs.append(int(d2l.reshape(d2l.argmax(Y, axis=2), 1)))
    return ''.join([vocab.idx_to_token[i] for i in outputs])
```

以下では、プレフィックスを指定して
20文字を追加生成させます。

```{.python .input}
%%tab mxnet, pytorch
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

上のRNNモデルをゼロから実装することは学習になりますが、便利ではありません。
次の節では、標準的なアーキテクチャを使ってRNNを簡単に構築し、
高度に最適化されたライブラリ関数に頼ることで性能向上を得る方法を見ます。


## まとめ

ユーザーが与えたテキストのプレフィックスに続くテキストを生成するように、
RNNベースの言語モデルを学習できます。 
単純なRNN言語モデルは、入力エンコーディング、RNNによるモデリング、出力生成から構成されます。
学習中、勾配クリッピングは爆発勾配の問題を緩和できますが、消失勾配の問題は解決しません。
実験では、単純なRNN言語モデルを実装し、文字レベルでトークン化したテキスト系列に対して
勾配クリッピング付きで学習させました。
プレフィックスを条件にすることで、言語モデルを使って
ありそうな続きのテキストを生成でき、これはオートコンプリート機能など多くの応用で有用です。


## 演習

1. 実装した言語モデルは、*The Time Machine* の最初のトークンまでのすべての過去トークンに基づいて次のトークンを予測しますか？
1. 予測に使われる履歴の長さを制御するハイパーパラメータはどれですか？
1. ワンホットエンコーディングが、各対象に対して異なる埋め込みを選ぶことと等価であることを示してください。
1. ハイパーパラメータ（たとえば、エポック数、隠れユニット数、ミニバッチ内の時間ステップ数、学習率）を調整して困惑度を改善してください。この単純なアーキテクチャのままで、どこまで下げられますか？
1. ワンホットエンコーディングを学習可能な埋め込みに置き換えてください。これにより性能は向上しますか？
1. *The Time Machine* で学習したこの言語モデルが、H. G. ウェルズの他の本、たとえば *The War of the Worlds* に対してどの程度うまく機能するかを調べる実験を行ってください。
1. 別の実験として、このモデルの困惑度を他の著者による本で評価してください。
1. 予測方法を修正して、最もありそうな次の文字を選ぶのではなく、サンプリングを使うようにしてください。
    * 何が起こりますか？
    * たとえば、$q(x_t \mid x_{t-1}, \ldots, x_1) \propto P(x_t \mid x_{t-1}, \ldots, x_1)^\alpha$ で $\alpha > 1$ としてサンプリングすることで、モデルをよりありそうな出力に偏らせてください。
1. この節のコードを勾配クリッピングなしで実行してください。何が起こりますか？
1. この節で使った活性化関数を ReLU に置き換え、この節の実験を繰り返してください。まだ勾配クリッピングは必要でしょうか？ なぜですか？

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/336)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/486)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/1052)
:end_tab:

:begin_tab:`jax`
[Discussions](https://discuss.d2l.ai/t/18014)
:end_tab:\n