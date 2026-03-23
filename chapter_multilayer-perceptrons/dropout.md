{.python .input}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow', 'jax'])
```

# ドロップアウト
:label:`sec_dropout`


まず、優れた予測モデルに何を期待するかを簡単に考えてみましょう。
私たちは、未知のデータに対して良い性能を発揮してほしいと考えます。
古典的な汎化理論は、訓練性能とテスト性能の差を縮めるには、
単純なモデルを目指すべきだと示唆しています。
単純さは、次元数が少ないという形で現れます。
これは、:numref:`sec_generalization_basics` で線形モデルの単項式基底関数を議論したときに見ました。
さらに、:numref:`sec_weight_decay` で重み減衰（$\ell_2$ 正則化）を議論したときに見たように、
パラメータの（逆）ノルムも単純さの有用な尺度を表します。
単純さのもう1つの有用な概念は滑らかさ、すなわち、関数が入力の小さな変化に敏感でないことです。
たとえば、画像を分類するときには、画素に多少のランダムノイズを加えても
ほとんど問題ないはずだと期待します。

:citet:`Bishop.1995` は、入力ノイズを加えて学習することが
チホノフ正則化と等価であることを証明し、この考えを形式化しました。
この研究は、関数が滑らかである（したがって単純である）ことの要請と、
入力の摂動に対して頑健であることの要請との間に、明確な数学的つながりを与えました。

その後、:citet:`Srivastava.Hinton.Krizhevsky.ea.2014` は、
Bishop の考えをネットワークの内部層にも適用する巧妙なアイデアを開発しました。
*ドロップアウト* と呼ばれるこのアイデアは、順伝播の各内部層を計算する際にノイズを注入するもので、
ニューラルネットワークの学習における標準的な手法となっています。
この手法が *dropout* と呼ばれるのは、学習中に実際にいくつかのニューロンを
*drop out*（脱落）させるからです。
学習の各反復で、標準的なドロップアウトは、次の層を計算する前に
各層のノードの一定割合をゼロにします。

明確にしておくと、Bishop とのつながりは私たちが独自に付けた説明です。
ドロップアウトの元論文は、性生殖との驚くべき類推を通じて直感を与えています。
著者らは、ニューラルネットワークの過学習は、
各層が前の層の特定の活性化パターンに依存している状態として特徴づけられると主張し、
この状態を *co-adaptation* と呼んでいます。
彼らによれば、ドロップアウトは co-adaptation を壊し、
性生殖が co-adapted な遺伝子を壊すとされるのと同じだといいます。
この理論の正当化は確かに議論の余地がありますが、
ドロップアウト手法そのものは長く使われ続けており、
さまざまな形式のドロップアウトがほとんどの深層学習ライブラリに実装されています。 


重要な課題は、このノイズをどのように注入するかです。
1つの考え方は、各層の期待値が---他の層を固定したときに---ノイズがない場合に取る値と等しくなるように、
*不偏* な方法で注入することです。
Bishop の研究では、線形モデルの入力にガウスノイズを加えました。
各学習反復で、平均0の分布からサンプルしたノイズ
$\epsilon \sim \mathcal{N}(0,\sigma^2)$ を入力 $\mathbf{x}$ に加え、
摂動された点 $\mathbf{x}' = \mathbf{x} + \epsilon$ を得ました。
期待値では、$E[\mathbf{x}'] = \mathbf{x}$ です。

標準的なドロップアウト正則化では、各層のノードの一定割合をゼロにし、
その後、保持された（ドロップアウトされなかった）ノードの割合で正規化することで、
各層を *補正* します。
言い換えると、
*ドロップアウト確率* $p$ に対して、
各中間活性化 $h$ は次のようにランダム変数 $h'$ に置き換えられます。

$$
\begin{aligned}
h' =
\begin{cases}
    0 & \textrm{ with probability } p \\
    \frac{h}{1-p} & \textrm{ otherwise}
\end{cases}
\end{aligned}
$$

設計上、期待値は変わらず、すなわち $E[h'] = h$ です。

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
from functools import partial
import jax
from jax import numpy as jnp
import optax
```

## 実践におけるドロップアウト

:numref:`fig_mlp` の、隠れ層1つと5個の隠れユニットを持つ MLP を思い出してください。
隠れ層にドロップアウトを適用し、各隠れユニットを確率 $p$ でゼロにすると、
その結果は元のニューロンの部分集合だけを含むネットワークと見なせます。
:numref:`fig_dropout2` では、$h_2$ と $h_5$ が取り除かれています。
その結果、出力の計算はもはや $h_2$ や $h_5$ に依存せず、
逆伝播を行うときにはそれぞれの勾配も消えます。
このようにして、出力層の計算が $h_1, \ldots, h_5$ のどれか1つの要素に
過度に依存することを防げます。

![ドロップアウト前後の MLP。](../img/dropout2.svg)
:label:`fig_dropout2`

通常、テスト時にはドロップアウトを無効にします。
学習済みモデルと新しい例が与えられたときには、どのノードもドロップアウトしないため、
正規化も不要です。
ただし、例外もあります。
一部の研究者は、ニューラルネットワークの予測の *不確実性* を推定するヒューリスティックとして、
テスト時にもドロップアウトを使います。
もし多くの異なるドロップアウト出力で予測が一致するなら、
そのネットワークはより確信を持っていると言えるかもしれません。

## ゼロからの実装

単一層のドロップアウト関数を実装するには、
層の次元数と同じだけのベルヌーイ（二値）確率変数からサンプルを生成する必要があります。
この確率変数は、確率 $1-p$ で値 $1$（保持）を、確率 $p$ で値 $0$（ドロップ）を取ります。
これを実装する簡単な方法は、まず一様分布 $U[0, 1]$ からサンプルを生成することです。
そして、対応するサンプルが $p$ より大きいノードを保持し、それ以外をドロップします。

以下のコードでは、（**テンソル入力 `X` の要素を確率 `dropout` でドロップする `dropout_layer` 関数を実装し**）、
上で説明したように残りを再スケーリングします。
つまり、生き残った要素を `1.0-dropout` で割ります。

```{.python .input}
%%tab mxnet
def dropout_layer(X, dropout):
    assert 0 <= dropout <= 1
    if dropout == 1: return np.zeros_like(X)
    mask = np.random.uniform(0, 1, X.shape) > dropout
    return mask.astype(np.float32) * X / (1.0 - dropout)
```

```{.python .input}
%%tab pytorch
def dropout_layer(X, dropout):
    assert 0 <= dropout <= 1
    if dropout == 1: return torch.zeros_like(X)
    mask = (torch.rand(X.shape) > dropout).float()
    return mask * X / (1.0 - dropout)
```

```{.python .input}
%%tab tensorflow
def dropout_layer(X, dropout):
    assert 0 <= dropout <= 1
    if dropout == 1: return tf.zeros_like(X)
    mask = tf.random.uniform(
        shape=tf.shape(X), minval=0, maxval=1) < 1 - dropout
    return tf.cast(mask, dtype=tf.float32) * X / (1.0 - dropout)
```

```{.python .input}
%%tab jax
def dropout_layer(X, dropout, key=d2l.get_key()):
    assert 0 <= dropout <= 1
    if dropout == 1: return jnp.zeros_like(X)
    mask = jax.random.uniform(key, X.shape) > dropout
    return jnp.asarray(mask, dtype=jnp.float32) * X / (1.0 - dropout)
```

いくつかの例で `dropout_layer` 関数を[**試してみましょう**]。
以下のコードでは、
入力 `X` をそれぞれ確率 0、0.5、1 のドロップアウト操作に通します。

```{.python .input}
%%tab all
if tab.selected('mxnet'):
    X = np.arange(16).reshape(2, 8)
if tab.selected('pytorch'):
    X = torch.arange(16, dtype = torch.float32).reshape((2, 8))
if tab.selected('tensorflow'):
    X = tf.reshape(tf.range(16, dtype=tf.float32), (2, 8))
if tab.selected('jax'):
    X = jnp.arange(16, dtype=jnp.float32).reshape(2, 8)
print('dropout_p = 0:', dropout_layer(X, 0))
print('dropout_p = 0.5:', dropout_layer(X, 0.5))
print('dropout_p = 1:', dropout_layer(X, 1))
```

### モデルの定義

以下のモデルは、各隠れ層の出力（活性化関数の後）にドロップアウトを適用します。
層ごとに別々のドロップアウト確率を設定できます。
一般的には、入力層に近いほどドロップアウト確率を低く設定します。
ドロップアウトは学習時にのみ有効になるようにします。

```{.python .input}
%%tab mxnet
class DropoutMLPScratch(d2l.Classifier):
    def __init__(self, num_outputs, num_hiddens_1, num_hiddens_2,
                 dropout_1, dropout_2, lr):
        super().__init__()
        self.save_hyperparameters()
        self.lin1 = nn.Dense(num_hiddens_1, activation='relu')
        self.lin2 = nn.Dense(num_hiddens_2, activation='relu')
        self.lin3 = nn.Dense(num_outputs)
        self.initialize()

    def forward(self, X):
        H1 = self.lin1(X)
        if autograd.is_training():
            H1 = dropout_layer(H1, self.dropout_1)
        H2 = self.lin2(H1)
        if autograd.is_training():
            H2 = dropout_layer(H2, self.dropout_2)
        return self.lin3(H2)
```

```{.python .input}
%%tab pytorch
class DropoutMLPScratch(d2l.Classifier):
    def __init__(self, num_outputs, num_hiddens_1, num_hiddens_2,
                 dropout_1, dropout_2, lr):
        super().__init__()
        self.save_hyperparameters()
        self.lin1 = nn.LazyLinear(num_hiddens_1)
        self.lin2 = nn.LazyLinear(num_hiddens_2)
        self.lin3 = nn.LazyLinear(num_outputs)
        self.relu = nn.ReLU()

    def forward(self, X):
        H1 = self.relu(self.lin1(X.reshape((X.shape[0], -1))))
        if self.training:  
            H1 = dropout_layer(H1, self.dropout_1)
        H2 = self.relu(self.lin2(H1))
        if self.training:
            H2 = dropout_layer(H2, self.dropout_2)
        return self.lin3(H2)
```

```{.python .input}
%%tab tensorflow
class DropoutMLPScratch(d2l.Classifier):
    def __init__(self, num_outputs, num_hiddens_1, num_hiddens_2,
                 dropout_1, dropout_2, lr):
        super().__init__()
        self.save_hyperparameters()
        self.lin1 = tf.keras.layers.Dense(num_hiddens_1, activation='relu')
        self.lin2 = tf.keras.layers.Dense(num_hiddens_2, activation='relu')
        self.lin3 = tf.keras.layers.Dense(num_outputs)

    def forward(self, X):
        H1 = self.lin1(tf.reshape(X, (X.shape[0], -1)))
        if self.training:
            H1 = dropout_layer(H1, self.dropout_1)
        H2 = self.lin2(H1)
        if self.training:
            H2 = dropout_layer(H2, self.dropout_2)
        return self.lin3(H2)
```

```{.python .input}
%%tab jax
class DropoutMLPScratch(d2l.Classifier):
    num_hiddens_1: int
    num_hiddens_2: int
    num_outputs: int
    dropout_1: float
    dropout_2: float
    lr: float
    training: bool = True

    def setup(self):
        self.lin1 = nn.Dense(self.num_hiddens_1)
        self.lin2 = nn.Dense(self.num_hiddens_2)
        self.lin3 = nn.Dense(self.num_outputs)
        self.relu = nn.relu

    def forward(self, X):
        H1 = self.relu(self.lin1(X.reshape(X.shape[0], -1)))
        if self.training:
            H1 = dropout_layer(H1, self.dropout_1)
        H2 = self.relu(self.lin2(H1))
        if self.training:
            H2 = dropout_layer(H2, self.dropout_2)
        return self.lin3(H2)
```

### [**学習**]

以下は、先に説明した MLP の学習と同様です。

```{.python .input}
%%tab all
hparams = {'num_outputs':10, 'num_hiddens_1':256, 'num_hiddens_2':256,
           'dropout_1':0.5, 'dropout_2':0.5, 'lr':0.1}
model = DropoutMLPScratch(**hparams)
data = d2l.FashionMNIST(batch_size=256)
trainer = d2l.Trainer(max_epochs=10)
trainer.fit(model, data)
```

## [**簡潔な実装**]

高水準 API を使う場合、各全結合層の後に `Dropout` 層を追加し、
コンストラクタの唯一の引数としてドロップアウト確率を渡すだけで済みます。
学習中、`Dropout` 層は指定されたドロップアウト確率に従って、
前の層の出力（あるいは同値に、次の層への入力）をランダムにドロップします。
学習モードでないときは、`Dropout` 層は単にデータをそのまま通してテストします。

```{.python .input}
%%tab mxnet
class DropoutMLP(d2l.Classifier):
    def __init__(self, num_outputs, num_hiddens_1, num_hiddens_2,
                 dropout_1, dropout_2, lr):
        super().__init__()
        self.save_hyperparameters()
        self.net = nn.Sequential()
        self.net.add(nn.Dense(num_hiddens_1, activation="relu"),
                     nn.Dropout(dropout_1),
                     nn.Dense(num_hiddens_2, activation="relu"),
                     nn.Dropout(dropout_2),
                     nn.Dense(num_outputs))
        self.net.initialize()
```

```{.python .input}
%%tab pytorch
class DropoutMLP(d2l.Classifier):
    def __init__(self, num_outputs, num_hiddens_1, num_hiddens_2,
                 dropout_1, dropout_2, lr):
        super().__init__()
        self.save_hyperparameters()
        self.net = nn.Sequential(
            nn.Flatten(), nn.LazyLinear(num_hiddens_1), nn.ReLU(), 
            nn.Dropout(dropout_1), nn.LazyLinear(num_hiddens_2), nn.ReLU(), 
            nn.Dropout(dropout_2), nn.LazyLinear(num_outputs))
```

```{.python .input}
%%tab tensorflow
class DropoutMLP(d2l.Classifier):
    def __init__(self, num_outputs, num_hiddens_1, num_hiddens_2,
                 dropout_1, dropout_2, lr):
        super().__init__()
        self.save_hyperparameters()
        self.net = tf.keras.models.Sequential([
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(num_hiddens_1, activation=tf.nn.relu),
            tf.keras.layers.Dropout(dropout_1),
            tf.keras.layers.Dense(num_hiddens_2, activation=tf.nn.relu),
            tf.keras.layers.Dropout(dropout_2),
            tf.keras.layers.Dense(num_outputs)])
```

```{.python .input}
%%tab jax
class DropoutMLP(d2l.Classifier):
    num_hiddens_1: int
    num_hiddens_2: int
    num_outputs: int
    dropout_1: float
    dropout_2: float
    lr: float
    training: bool = True

    @nn.compact
    def __call__(self, X):
        x = nn.relu(nn.Dense(self.num_hiddens_1)(X.reshape((X.shape[0], -1))))
        x = nn.Dropout(self.dropout_1, deterministic=not self.training)(x)
        x = nn.relu(nn.Dense(self.num_hiddens_2)(x))
        x = nn.Dropout(self.dropout_2, deterministic=not self.training)(x)
        return nn.Dense(self.num_outputs)(x)
```

:begin_tab:`jax`
ドロップアウト層を含むネットワークで `Module.apply()` を使う場合、PRNGKey が必要になるため、
損失関数を再定義する必要があることに注意してください。
また、この RNG シードは明示的に `dropout` と名付ける必要があります。
このキーは Flax の `dropout` 層によって内部でランダムなドロップアウトマスクを生成するために使われます。
学習ループの各エポックで一意の `dropout_rng` キーを使うことが重要です。
そうしないと、生成されるドロップアウトマスクは確率的にならず、エポックごとに異なるものになりません。
この `dropout_rng` は
:numref:`oo-design-training` で定義した `d2l.Trainer` クラスの `TrainState` オブジェクトに属性として保存でき、
各エポックで新しい `dropout_rng` に置き換えられます。
これは、:numref:`sec_linear_scratch` で定義した `fit_epoch` メソッドですでに処理済みです。
:end_tab:

```{.python .input}
%%tab jax
@d2l.add_to_class(d2l.Classifier)  #@save
@partial(jax.jit, static_argnums=(0, 5))
def loss(self, params, X, Y, state, averaged=True):
    Y_hat = state.apply_fn({'params': params}, *X,
                           mutable=False,  # To be used later (e.g., batch norm)
                           rngs={'dropout': state.dropout_rng})
    Y_hat = d2l.reshape(Y_hat, (-1, Y_hat.shape[-1]))
    Y = d2l.reshape(Y, (-1,))
    fn = optax.softmax_cross_entropy_with_integer_labels
    # The returned empty dictionary is a placeholder for auxiliary data,
    # which will be used later (e.g., for batch norm)
    return (fn(Y_hat, Y).mean(), {}) if averaged else (fn(Y_hat, Y), {})
```

次に、[**モデルを学習します**]。

```{.python .input}
%%tab all
model = DropoutMLP(**hparams)
trainer.fit(model, data)
```

## まとめ

次元数や重みベクトルの大きさを制御することに加えて、ドロップアウトは過学習を避けるためのもう1つの手段です。
多くの場合、これらの手段は組み合わせて使われます。
ドロップアウトは学習時にのみ使われることに注意してください。
これは、活性化 $h$ を期待値が $h$ のランダム変数に置き換えます。


## 演習

1. 1層目と2層目のドロップアウト確率を変えるとどうなりますか？ 特に、両方の層の確率を入れ替えるとどうなりますか？ これらの質問に答える実験を設計し、結果を定量的に述べ、定性的な要点をまとめなさい。
1. エポック数を増やし、ドロップアウトを使った場合と使わない場合の結果を比較しなさい。
1. ドロップアウトを適用した場合としない場合で、各隠れ層の活性化の分散はどれくらいですか？ この量が時間とともにどのように変化するかを、両方のモデルについてプロットしなさい。
1. なぜドロップアウトは通常テスト時には使われないのですか？
1. この節のモデルを例に、ドロップアウトと重み減衰の効果を比較しなさい。ドロップアウトと重み減衰を同時に使うとどうなりますか？ 効果は加算的ですか？ それとも逓減しますか（あるいは悪化しますか）？ 互いに打ち消し合いますか？
1. 活性化ではなく、重み行列の個々の重みにドロップアウトを適用するとどうなりますか？
1. 標準的なドロップアウト手法とは異なる、各層にランダムノイズを注入する別の手法を考案しなさい。Fashion-MNIST データセットで（固定アーキテクチャに対して）ドロップアウトを上回る方法を開発できますか？

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/100)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/101)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/261)
:end_tab:

:begin_tab:`jax`
[Discussions](https://discuss.d2l.ai/t/17987)
:end_tab:\n