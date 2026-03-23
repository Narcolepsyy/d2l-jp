{.python .input  n=1}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow', 'jax'])
```

# スクラッチからのソフトマックス回帰の実装
:label:`sec_softmax_scratch`

ソフトマックス回帰は非常に基本的な手法なので、
ぜひ自分で実装できるようになっておくべきだと考えます。
ここでは、モデルのソフトマックス固有の部分の定義に限定し、
線形回帰の節で使った他の構成要素、
たとえば学習ループなどは再利用します。

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
from flax import linen as nn
import jax
from jax import numpy as jnp
from functools import partial
```

## ソフトマックス

まず最も重要な部分から始めましょう。
それは、スカラーを確率へ写像することです。
復習として、 :numref:`subsec_lin-alg-reduction`
および :numref:`subsec_lin-alg-non-reduction`
で説明したように、テンソルの特定の次元に沿った和演算を思い出してください。
[**行列 `X` が与えられたとき、すべての要素を（デフォルトで）和にすることも、同じ軸上の要素だけを和にすることもできます。**]
`axis` 変数を使うと、行方向および列方向の和を計算できます。

```{.python .input}
%%tab all
X = d2l.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
d2l.reduce_sum(X, 0, keepdims=True), d2l.reduce_sum(X, 1, keepdims=True)
```

ソフトマックスの計算には3つの手順があります。
(i) 各項の指数関数を取る。
(ii) 各行について和を取り、各サンプルの正規化定数を計算する。
(iii) 各行をその正規化定数で割り、結果の和が1になるようにする。

[**
$$\mathrm{softmax}(\mathbf{X})_{ij} = \frac{\exp(\mathbf{X}_{ij})}{\sum_k \exp(\mathbf{X}_{ik})}.$$
**]

分母の（対数を取ったもの）は（対数）*分配関数*と呼ばれます。
これは熱力学的アンサンブルにおけるすべての可能な状態を総和するために、
[統計物理学](https://en.wikipedia.org/wiki/Partition_function_(statistical_mechanics))
で導入されました。
実装は簡単です。

```{.python .input}
%%tab all
def softmax(X):
    X_exp = d2l.exp(X)
    partition = d2l.reduce_sum(X_exp, 1, keepdims=True)
    return X_exp / partition  # ここではブロードキャスト機構が適用される
```

任意の入力 `X` に対して、[**各要素を非負の数に変換します。
各行の和は1になります。**]
これは確率として必要な性質です。注意: 上のコードは、非常に大きい値や非常に小さい値に対しては*頑健ではありません*。何が起きているかを示すには十分ですが、実用上の目的でこのコードをそのまま使うべきではありません。深層学習フレームワークにはこのような保護機構が組み込まれており、今後は組み込みのソフトマックスを使います。

```{.python .input}
%%tab mxnet
X = d2l.rand(2, 5)
X_prob = softmax(X)
X_prob, d2l.reduce_sum(X_prob, 1)
```

```{.python .input}
%%tab tensorflow, pytorch
X = d2l.rand((2, 5))
X_prob = softmax(X)
X_prob, d2l.reduce_sum(X_prob, 1)
```

```{.python .input}
%%tab jax
X = jax.random.uniform(jax.random.PRNGKey(d2l.get_seed()), (2, 5))
X_prob = softmax(X)
X_prob, d2l.reduce_sum(X_prob, 1)
```

## モデル

これで、[**ソフトマックス回帰モデルを実装するために必要なものはすべて揃いました。**]
線形回帰の例と同様に、各インスタンスは固定長ベクトルで表現されます。
ここでの生データは $28 \times 28$ ピクセルの画像からなるので、
[**各画像を平坦化し、長さ784のベクトルとして扱います。**]
後の章では、空間構造をより自然に活用できる
畳み込みニューラルネットワークを紹介します。


ソフトマックス回帰では、ネットワークの出力数はクラス数と等しくなければなりません。
[**データセットには10クラスあるので、ネットワークの出力次元は10です。**]
したがって、重みは $784 \times 10$ の行列と、
バイアス用の $1 \times 10$ の行ベクトルから構成されます。
線形回帰と同様に、重み `W` はガウスノイズで初期化します。
バイアスはゼロで初期化します。

```{.python .input}
%%tab mxnet
class SoftmaxRegressionScratch(d2l.Classifier):
    def __init__(self, num_inputs, num_outputs, lr, sigma=0.01):
        super().__init__()
        self.save_hyperparameters()
        self.W = np.random.normal(0, sigma, (num_inputs, num_outputs))
        self.b = np.zeros(num_outputs)
        self.W.attach_grad()
        self.b.attach_grad()

    def collect_params(self):
        return [self.W, self.b]
```

```{.python .input}
%%tab pytorch
class SoftmaxRegressionScratch(d2l.Classifier):
    def __init__(self, num_inputs, num_outputs, lr, sigma=0.01):
        super().__init__()
        self.save_hyperparameters()
        self.W = torch.normal(0, sigma, size=(num_inputs, num_outputs),
                              requires_grad=True)
        self.b = torch.zeros(num_outputs, requires_grad=True)

    def parameters(self):
        return [self.W, self.b]
```

```{.python .input}
%%tab tensorflow
class SoftmaxRegressionScratch(d2l.Classifier):
    def __init__(self, num_inputs, num_outputs, lr, sigma=0.01):
        super().__init__()
        self.save_hyperparameters()
        self.W = tf.random.normal((num_inputs, num_outputs), 0, sigma)
        self.b = tf.zeros(num_outputs)
        self.W = tf.Variable(self.W)
        self.b = tf.Variable(self.b)
```

```{.python .input}
%%tab jax
class SoftmaxRegressionScratch(d2l.Classifier):
    num_inputs: int
    num_outputs: int
    lr: float
    sigma: float = 0.01

    def setup(self):
        self.W = self.param('W', nn.initializers.normal(self.sigma),
                            (self.num_inputs, self.num_outputs))
        self.b = self.param('b', nn.initializers.zeros, self.num_outputs)
```

以下のコードでは、ネットワークが各入力をどのように出力へ写像するかを定義します。
バッチ内の各 $28 \times 28$ ピクセル画像は、モデルに通す前に `reshape` を使ってベクトルに平坦化していることに注意してください。

```{.python .input}
%%tab all
@d2l.add_to_class(SoftmaxRegressionScratch)
def forward(self, X):
    X = d2l.reshape(X, (-1, self.W.shape[0]))
    return softmax(d2l.matmul(X, self.W) + self.b)
```

## 交差エントロピー損失

次に、交差エントロピー損失関数を実装する必要があります
(:numref:`subsec_softmax-regression-loss-func` で導入しました)。
これは深層学習全体の中でも最も一般的な損失関数かもしれません。
現時点では、深層学習の応用のうち、
回帰問題よりも分類問題として自然に定式化できるものの方がはるかに多くあります。

交差エントロピーは、真のラベルに割り当てられた予測確率の負の対数尤度を取ることを思い出してください。
効率のため、Python の for ループは避け、代わりにインデックス参照を使います。
特に、$\mathbf{y}$ の one-hot エンコーディングにより、
$\hat{\mathbf{y}}$ の対応する項を選択できます。

これを確認するために、[**3クラスに対する予測確率を持つ2つのサンプル `y_hat` と、それに対応するラベル `y` を作成します。**]
正しいラベルはそれぞれ $0$ と $2$（すなわち第1クラスと第3クラス）です。
[**`y` を `y_hat` 内の確率のインデックスとして使うことで、**]
項を効率よく取り出せます。

```{.python .input}
%%tab mxnet, pytorch, jax
y = d2l.tensor([0, 2])
y_hat = d2l.tensor([[0.1, 0.3, 0.6], [0.3, 0.2, 0.5]])
y_hat[[0, 1], y]
```

```{.python .input}
%%tab tensorflow
y_hat = tf.constant([[0.1, 0.3, 0.6], [0.3, 0.2, 0.5]])
y = tf.constant([0, 2])
tf.boolean_mask(y_hat, tf.one_hot(y, depth=y_hat.shape[-1]))
```

:begin_tab:`pytorch, mxnet, tensorflow`
これで、選択した確率の対数を平均することで、[**交差エントロピー損失関数を実装できます。**]
:end_tab:

:begin_tab:`jax`
これで、選択した確率の対数を平均することで、[**交差エントロピー損失関数を実装できます。**]

`jax.jit` を使って JAX 実装を高速化し、`loss` が純粋関数であることを保証するために、
`cross_entropy` 関数は `loss` の内部で再定義されています。
これにより、`loss` 関数を不純にしてしまう可能性のあるグローバル変数や関数の使用を避けています。
`jax.jit` と純粋関数については、[JAX のドキュメント](https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html#pure-functions) を参照してください。
:end_tab:

```{.python .input}
%%tab mxnet, pytorch, jax
def cross_entropy(y_hat, y):
    return -d2l.reduce_mean(d2l.log(y_hat[list(range(len(y_hat))), y]))

cross_entropy(y_hat, y)
```

```{.python .input}
%%tab tensorflow
def cross_entropy(y_hat, y):
    return -tf.reduce_mean(tf.math.log(tf.boolean_mask(
        y_hat, tf.one_hot(y, depth=y_hat.shape[-1]))))

cross_entropy(y_hat, y)
```

```{.python .input}
%%tab pytorch, mxnet, tensorflow
@d2l.add_to_class(SoftmaxRegressionScratch)
def loss(self, y_hat, y):
    return cross_entropy(y_hat, y)
```

```{.python .input}
%%tab jax
@d2l.add_to_class(SoftmaxRegressionScratch)
@partial(jax.jit, static_argnums=(0))
def loss(self, params, X, y, state):
    def cross_entropy(y_hat, y):
        return -d2l.reduce_mean(d2l.log(y_hat[list(range(len(y_hat))), y]))
    y_hat = state.apply_fn({'params': params}, *X)
    # 返される空の辞書は補助データのプレースホルダであり、
    # 後で（たとえば batch norm のために）使われます
    return cross_entropy(y_hat, y), {}
```

## 学習

:numref:`sec_linear_scratch` で定義した `fit` メソッドを再利用して、[**10エポックでモデルを学習します。**]
エポック数（`max_epochs`）、
ミニバッチサイズ（`batch_size`）、
学習率（`lr`）は調整可能なハイパーパラメータであることに注意してください。
つまり、これらの値は主たる学習ループの中で
学習されるわけではありませんが、
学習性能と汎化性能の両方において、
モデルの性能に影響を与えます。
実際には、データの *検証* 分割に基づいてこれらの値を選び、
最終的には *テスト* 分割で最終モデルを評価したいでしょう。
:numref:`subsec_generalization-model-selection` で述べたように、
Fashion-MNIST のテストデータを検証セットとして扱い、
この分割上で検証損失と検証精度を報告します。

```{.python .input}
%%tab all
data = d2l.FashionMNIST(batch_size=256)
model = SoftmaxRegressionScratch(num_inputs=784, num_outputs=10, lr=0.1)
trainer = d2l.Trainer(max_epochs=10)
trainer.fit(model, data)
```

## 予測

学習が完了したので、
モデルは[**いくつかの画像を分類する準備ができました。**]

```{.python .input}
%%tab all
X, y = next(iter(data.val_dataloader()))
if tab.selected('pytorch', 'mxnet', 'tensorflow'):
    preds = d2l.argmax(model(X), axis=1)
if tab.selected('jax'):
    preds = d2l.argmax(model.apply({'params': trainer.state.params}, X), axis=1)
preds.shape
```

私たちがより関心を持つのは、*誤って*ラベル付けされた画像です。実際のラベル
（テキスト出力の1行目）と
モデルの予測
（テキスト出力の2行目）を比較することで、それらを可視化します。

```{.python .input}
%%tab all
wrong = d2l.astype(preds, y.dtype) != y
X, y, preds = X[wrong], y[wrong], preds[wrong]
labels = [a+'\n'+b for a, b in zip(
    data.text_labels(y), data.text_labels(preds))]
data.visualize([X, y], labels=labels)
```

## まとめ

ここまでで、線形回帰と分類問題を解く経験を少し積んできました。
これにより、統計モデリングの1960〜1970年代における
いわば最先端に到達したと言ってよいでしょう。
次の節では、このモデルを
深層学習フレームワークを活用して
はるかに効率よく実装する方法を示します。

## 演習

1. この節では、ソフトマックス演算の数学的定義に基づいてソフトマックス関数を直接実装しました。 :numref:`sec_softmax` で述べたように、これは数値的不安定性を引き起こす可能性があります。
    1. 入力に値 $100$ が含まれていても `softmax` が正しく動作するか確認せよ。
    1. すべての入力の最大値が $-100$ より小さい場合でも `softmax` が正しく動作するか確認せよ。
    1. 引数の最大要素に対する相対値を見ることで修正を実装せよ。
1. 交差エントロピー損失関数 $\sum_i y_i \log \hat{y}_i$ の定義に従う `cross_entropy` 関数を実装せよ。
    1. この節のコード例で試してみよ。
    1. なぜより遅く動作すると考えられるか。
    1. それを使うべきか。どのような場合に使うのが理にかなっているか。
    1. 何に注意する必要があるか。ヒント: 対数の定義域を考えよ。
1. 最もありそうなラベルを返すことは、常に良い考えだろうか。たとえば、医療診断でこれを行うだろうか。どのように対処しようとするか。
1. いくつかの特徴量に基づいて次の単語を予測するためにソフトマックス回帰を使いたいと仮定する。大きな語彙から生じる問題にはどのようなものがあるか。
1. この節のコードのハイパーパラメータをいろいろ試してみよ。特に:
    1. 学習率を変えたときに検証損失がどのように変化するかをプロットせよ。
    1. ミニバッチサイズを変えると検証損失と学習損失は変化するか。効果が見えるまでに、どれくらい大きくまたは小さくする必要があるか。


:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/50)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/51)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/225)
:end_tab:

:begin_tab:`jax`
[Discussions](https://discuss.d2l.ai/t/17982)
:end_tab:\n