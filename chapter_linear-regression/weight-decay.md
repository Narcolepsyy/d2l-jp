{.python .input}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow', 'jax'])
```

# 重み減衰
:label:`sec_weight_decay`

過学習の問題を特徴づけたので、最初の*正則化*手法を導入できます。
過学習は、より多くの訓練データを集めることで常に緩和できることを思い出してください。
しかし、それにはコストがかかり、時間もかかり、
あるいは完全に私たちの制御外であることもあり、
短期的には不可能です。
ひとまず、私たちはすでに
資源が許す限り十分に高品質なデータを持っていると仮定し、
データセットが与えられたものとして扱うときに利用できる手段に集中しましょう。

多項式回帰の例
(:numref:`subsec_polynomial-curve-fitting`)
では、当てはめる多項式の次数を調整することで
モデルの容量を制限できました。
実際、特徴量の数を制限することは
過学習を抑えるための一般的な手法です。
しかし、単に特徴量を切り捨てるだけでは
あまりに大雑把すぎることがあります。
多項式回帰の例に戻って、
高次元入力で何が起こりうるかを考えてみましょう。
多変量データへの多項式の自然な拡張は
*単項式*と呼ばれ、変数のべき乗の積にすぎません。
単項式の次数は、べきの和です。
たとえば、$x_1^2 x_2$ と $x_3 x_5^2$
はいずれも次数3の単項式です。

次数 $d$ の項の数は、$d$ が大きくなるにつれて
急速に爆発的に増えることに注意してください。
$k$ 個の変数があるとき、次数 $d$ の単項式の数は
${k - 1 + d} \choose {k - 1}$ です。
次数を $2$ から $3$ に変えるといった小さな変化でも、
モデルの複雑さは劇的に増加します。
したがって、関数の複雑さを調整するには、
よりきめ細かな手段がしばしば必要になります。

```{.python .input}
%%tab mxnet
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import autograd, gluon, init, np, npx
from mxnet.gluon import nn
npx.set_np()
```

```{.python .input}
%%tab pytorch
%matplotlib inline
from d2l import torch as d2l
import torch
from torch import nn
```

```{.python .input}
%%tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
import tensorflow as tf
```

```{.python .input}
%%tab jax
%matplotlib inline
from d2l import jax as d2l
import jax
from jax import numpy as jnp
import optax
```

## ノルムと重み減衰

(**パラメータの数を直接操作する代わりに、
*重み減衰*は、パラメータが取りうる値を制限することで機能します。**)
深層学習以外の分野では、ミニバッチ確率的勾配降下法で最適化するとき、
より一般には $\ell_2$ 正則化と呼ばれます。
重み減衰は、パラメトリック機械学習モデルを正則化するための
最も広く使われている手法の一つかもしれません。
この手法は、すべての関数 $f$ の中で、
関数 $f = 0$
（すべての入力に値 $0$ を割り当てるもの）が
ある意味で*最も単純*であり、
関数の複雑さはパラメータがゼロからどれだけ離れているかで測れる、
という基本的な直観に動機づけられています。
しかし、関数とゼロの間の距離を
正確にはどのように測ればよいのでしょうか？
唯一の正解があるわけではありません。
実際、関数解析の一部や
バナッハ空間の理論を含む数学の大きな分野全体が、
このような問題に取り組むために存在しています。

一つの単純な解釈としては、
線形関数 $f(\mathbf{x}) = \mathbf{w}^\top \mathbf{x}$
の複雑さを、その重みベクトルの何らかのノルム、たとえば $\| \mathbf{w} \|^2$ で測ることが考えられます。
: numref:`subsec_lin-algebra-norms` で、
より一般的な $\ell_p$ ノルムの特殊な場合である
$\ell_2$ ノルムと $\ell_1$ ノルムを導入したことを思い出してください。
小さな重みベクトルを確保する最も一般的な方法は、
そのノルムを損失最小化問題に罰則項として加えることです。
したがって、元の目的関数、
すなわち*訓練ラベルに対する予測損失を最小化すること*を、
新しい目的関数、
すなわち*予測損失と罰則項の和を最小化すること*に置き換えます。
こうすると、重みベクトルが大きくなりすぎた場合、
学習アルゴリズムは訓練誤差の最小化よりも
重みノルム $\| \mathbf{w} \|^2$ の最小化に
注力するかもしれません。
それこそが私たちの望むことです。
コードで示すために、
: numref:`sec_linear_regression` の線形回帰の例を再び取り上げます。
そこでは、損失は次のように与えられていました。

$$L(\mathbf{w}, b) = \frac{1}{n}\sum_{i=1}^n \frac{1}{2}\left(\mathbf{w}^\top \mathbf{x}^{(i)} + b - y^{(i)}\right)^2.$$

$\mathbf{x}^{(i)}$ は特徴量、
$y^{(i)}$ はデータ例 $i$ のラベルであり、$(\mathbf{w}, b)$
はそれぞれ重みパラメータとバイアスパラメータです。
重みベクトルの大きさに罰則を与えるには、
何らかの形で損失関数に $\| \mathbf{w} \|^2$ を加える必要がありますが、
この新しい加法的な罰則に対して、モデルは標準的な損失を
どのようにトレードオフすべきでしょうか？
実際には、このトレードオフを
*正則化定数* $\lambda$ によって特徴づけます。
これは非負のハイパーパラメータであり、
検証データを用いて調整します。

$$L(\mathbf{w}, b) + \frac{\lambda}{2} \|\mathbf{w}\|^2.$$


$\lambda = 0$ なら、元の損失関数に戻ります。
$\lambda > 0$ なら、$\| \mathbf{w} \|$ の大きさを制限します。
$2$ で割るのは慣習です。
二次関数の微分を取るとき、
$2$ と $1/2$ が打ち消し合い、更新式が
見た目にもきれいで簡潔になります。
鋭い読者は、なぜ標準ノルム（すなわちユークリッド距離）ではなく、
二乗したノルムを使うのかと疑問に思うかもしれません。
これは計算上の都合です。
$\ell_2$ ノルムを二乗することで平方根が消え、
重みベクトルの各成分の二乗和だけが残ります。
これにより、罰則項の微分を簡単に計算できます。
すなわち、和の微分は微分の和に等しいのです。


さらに、そもそもなぜ $\ell_1$ ノルムではなく
$\ell_2$ ノルムを使うのか、と疑問に思うかもしれません。
実際、他の選択肢も有効であり、
統計学では広く使われています。
$\ell_2$ 正則化された線形モデルは古典的な
*リッジ回帰*アルゴリズムを構成しますが、
$\ell_1$ 正則化された線形回帰は
同様に基本的な統計手法であり、
一般に *ラッソ回帰* として知られています。
$\ell_2$ ノルムを使う一つの理由は、
重みベクトルの大きな成分に対して
特に大きな罰則を課すことです。
これにより学習アルゴリズムは、
より多くの特徴量に重みを均等に分配するモデルへと
偏ります。
実際には、これは単一変数の測定誤差に対して
より頑健にするかもしれません。
対照的に、$\ell_1$ 罰則は、
他の重みをゼロにしてしまうことで、
少数の特徴量に重みを集中させるモデルを導きます。
これにより、*特徴選択*のための有効な手法が得られ、
別の理由から望ましいことがあります。
たとえば、モデルが少数の特徴量にしか依存しないなら、
他の（捨てられた）特徴量についてデータを収集、保存、送信する必要が
なくなるかもしれません。 

:eqref:`eq_linreg_batch_update` と同じ記法を用いると、
ミニバッチ確率的勾配降下法による
$\ell_2$ 正則化回帰の更新は次のようになります。

$$\begin{aligned}
\mathbf{w} & \leftarrow \left(1- \eta\lambda \right) \mathbf{w} - \frac{\eta}{|\mathcal{B}|} \sum_{i \in \mathcal{B}} \mathbf{x}^{(i)} \left(\mathbf{w}^\top \mathbf{x}^{(i)} + b - y^{(i)}\right).
\end{aligned}$$

これまでと同様に、推定値が観測値からどれだけずれているかに基づいて
$\mathbf{w}$ を更新します。
しかし同時に、$\mathbf{w}$ の大きさをゼロへ向かって縮小します。
そのため、この手法はしばしば「重み減衰」と呼ばれます。
罰則項だけを考えると、
最適化アルゴリズムは学習の各ステップで重みを*減衰*させます。
特徴選択とは対照的に、重み減衰は関数の複雑さを
連続的に調整する仕組みを与えてくれます。
$\lambda$ が小さいほど $\mathbf{w}$ への制約は弱くなり、
一方で $\lambda$ が大きいほど
$\mathbf{w}$ はより強く制約されます。
対応するバイアス罰則 $b^2$ を含めるかどうかは
実装によって異なり、
ニューラルネットワークの層ごとに異なる場合もあります。
多くの場合、バイアス項は正則化しません。
さらに、
$\ell_2$ 正則化が他の最適化アルゴリズムでは重み減衰と等価でない場合があるとしても、
重みの大きさを縮小することによる正則化という考え方自体は
依然として成り立ちます。

## 高次元線形回帰

重み減衰の利点は、
単純な合成例を通して示すことができます。

まず、以前と同様に [**データを生成します**]：

(**$$y = 0.05 + \sum_{i = 1}^d 0.01 x_i + \epsilon \textrm{ where }
\epsilon \sim \mathcal{N}(0, 0.01^2).$$**)

この合成データセットでは、ラベルは入力の背後にある線形関数によって与えられ、
平均0、標準偏差0.01のガウスノイズによって
汚されています。
説明のために、
問題の次元を $d = 200$ に増やし、
20例しかない小さな訓練セットで学習することで、
過学習の影響を顕著にできます。

```{.python .input}
%%tab all
class Data(d2l.DataModule):
    def __init__(self, num_train, num_val, num_inputs, batch_size):
        self.save_hyperparameters()                
        n = num_train + num_val 
        if tab.selected('mxnet') or tab.selected('pytorch'):
            self.X = d2l.randn(n, num_inputs)
            noise = d2l.randn(n, 1) * 0.01
        if tab.selected('tensorflow'):
            self.X = d2l.normal((n, num_inputs))
            noise = d2l.normal((n, 1)) * 0.01
        if tab.selected('jax'):
            self.X = jax.random.normal(jax.random.PRNGKey(0), (n, num_inputs))
            noise = jax.random.normal(jax.random.PRNGKey(0), (n, 1)) * 0.01
        w, b = d2l.ones((num_inputs, 1)) * 0.01, 0.05
        self.y = d2l.matmul(self.X, w) + b + noise

    def get_dataloader(self, train):
        i = slice(0, self.num_train) if train else slice(self.num_train, None)
        return self.get_tensorloader([self.X, self.y], train, i)
```

## ゼロからの実装

では、重み減衰をゼロから実装してみましょう。
ミニバッチ確率的勾配降下法が最適化手法なので、
元の損失関数に二乗した $\ell_2$ 罰則を加えるだけで十分です。

### (**$\ell_2$ ノルム罰則の定義**)

この罰則を実装する最も便利な方法は、
すべての項をその場で二乗してから和を取ることかもしれません。

```{.python .input}
%%tab all
def l2_penalty(w):
    return d2l.reduce_sum(w**2) / 2
```

### モデルの定義

最終的なモデルでは、
線形回帰と二乗損失は :numref:`sec_linear_scratch` から変わっていないので、
`d2l.LinearRegressionScratch` のサブクラスを定義するだけでよいでしょう。
ここでの唯一の変更点は、損失に罰則項が含まれることです。

```{.python .input}
%%tab pytorch, mxnet, tensorflow
class WeightDecayScratch(d2l.LinearRegressionScratch):
    def __init__(self, num_inputs, lambd, lr, sigma=0.01):
        super().__init__(num_inputs, lr, sigma)
        self.save_hyperparameters()
        
    def loss(self, y_hat, y):
        return (super().loss(y_hat, y) +
                self.lambd * l2_penalty(self.w))
```

```{.python .input}
%%tab jax
class WeightDecayScratch(d2l.LinearRegressionScratch):
    lambd: int = 0
        
    def loss(self, params, X, y, state):
        return (super().loss(params, X, y, state) +
                self.lambd * l2_penalty(params['w']))
```

次のコードは、20例の訓練セットでモデルを学習し、100例の検証セットで評価します。

```{.python .input}
%%tab all
data = Data(num_train=20, num_val=100, num_inputs=200, batch_size=5)
trainer = d2l.Trainer(max_epochs=10)

def train_scratch(lambd):    
    model = WeightDecayScratch(num_inputs=200, lambd=lambd, lr=0.01)
    model.board.yscale='log'
    trainer.fit(model, data)
    if tab.selected('pytorch', 'mxnet', 'tensorflow'):
        print('L2 norm of w:', float(l2_penalty(model.w)))
    if tab.selected('jax'):
        print('L2 norm of w:',
              float(l2_penalty(trainer.state.params['w'])))
```

### [**正則化なしでの学習**]

ここでは `lambd = 0` としてこのコードを実行し、
重み減衰を無効にします。
訓練誤差は下がる一方で検証誤差は下がらず、
ひどく過学習していることに注意してください。
これは過学習の典型例です。

```{.python .input}
%%tab all
train_scratch(0)
```

### [**重み減衰の使用**]

以下では、かなり強い重み減衰をかけて実行します。
訓練誤差は増加しますが、
検証誤差は減少することに注意してください。
これはまさに正則化から期待される効果です。

```{.python .input}
%%tab all
train_scratch(3)
```

## [**簡潔な実装**]

重み減衰はニューラルネットワーク最適化で
至るところに使われているため、深層学習フレームワークでは特に便利に扱えます。
最適化アルゴリズム自体に重み減衰を統合し、
任意の損失関数と組み合わせて簡単に使えるようにしています。
さらに、この統合は計算上の利点もあり、
追加の計算オーバーヘッドなしに、
実装上の工夫でアルゴリズムに重み減衰を加えられます。
更新の重み減衰部分は
各パラメータの現在値のみに依存するため、
最適化器はどうせ各パラメータに一度は触れる必要があります。

:begin_tab:`mxnet`
以下では、`Trainer` をインスタンス化するときに
`wd` を通じて重み減衰ハイパーパラメータを直接指定します。
デフォルトでは、Gluon は
重みとバイアスの両方を同時に減衰させます。
ハイパーパラメータ `wd`
はモデルパラメータ更新時に `wd_mult`
と掛け合わされることに注意してください。
したがって、`wd_mult` をゼロに設定すると、
バイアスパラメータ $b$ は減衰しません。
:end_tab:

:begin_tab:`pytorch`
以下では、最適化器をインスタンス化するときに
`weight_decay` を通じて重み減衰ハイパーパラメータを直接指定します。
デフォルトでは、PyTorch は
重みとバイアスの両方を同時に減衰させますが、
最適化器を異なるパラメータに対して異なる方針で扱うように設定できます。
ここでは、重みに対してのみ（`net.weight` パラメータに対してのみ）
`weight_decay` を設定しているため、
バイアス（`net.bias` パラメータ）は減衰しません。
:end_tab:

:begin_tab:`tensorflow`
以下では、重み減衰ハイパーパラメータ `wd` を持つ $\ell_2$ 正則化器を作成し、
`kernel_regularizer` 引数を通じて層の重みに適用します。
:end_tab:

```{.python .input}
%%tab mxnet
class WeightDecay(d2l.LinearRegression):
    def __init__(self, wd, lr):
        super().__init__(lr)
        self.save_hyperparameters()
        self.wd = wd
        
    def configure_optimizers(self):
        self.collect_params('.*bias').setattr('wd_mult', 0)
        return gluon.Trainer(self.collect_params(),
                             'sgd', 
                             {'learning_rate': self.lr, 'wd': self.wd})
```

```{.python .input}
%%tab pytorch
class WeightDecay(d2l.LinearRegression):
    def __init__(self, wd, lr):
        super().__init__(lr)
        self.save_hyperparameters()
        self.wd = wd

    def configure_optimizers(self):
        return torch.optim.SGD([
            {'params': self.net.weight, 'weight_decay': self.wd},
            {'params': self.net.bias}], lr=self.lr)
```

```{.python .input}
%%tab tensorflow
class WeightDecay(d2l.LinearRegression):
    def __init__(self, wd, lr):
        super().__init__(lr)
        self.save_hyperparameters()
        self.net = tf.keras.layers.Dense(
            1, kernel_regularizer=tf.keras.regularizers.l2(wd),
            kernel_initializer=tf.keras.initializers.RandomNormal(0, 0.01)
        )
        
    def loss(self, y_hat, y):
        return super().loss(y_hat, y) + self.net.losses
```

```{.python .input}
%%tab jax
class WeightDecay(d2l.LinearRegression):
    wd: int = 0
    
    def configure_optimizers(self):
        # Weight Decay is not available directly within optax.sgd, but
        # optax allows chaining several transformations together
        return optax.chain(optax.additive_weight_decay(self.wd),
                           optax.sgd(self.lr))
```

[**プロットは、ゼロから重み減衰を実装したときと似ています**]。
しかし、この版のほうが高速に動作し、
実装も容易です。
問題が大きくなり、作業がより日常的になるにつれて、
これらの利点はさらに顕著になります。

```{.python .input}
%%tab all
model = WeightDecay(wd=3, lr=0.01)
model.board.yscale='log'
trainer.fit(model, data)

if tab.selected('jax'):
    print('L2 norm of w:', float(l2_penalty(model.get_w_b(trainer.state)[0])))
if tab.selected('pytorch', 'mxnet', 'tensorflow'):
    print('L2 norm of w:', float(l2_penalty(model.get_w_b()[0])))
```

ここまでで、単純な線形関数を構成するものについて
一つの考え方に触れました。
しかし、単純な非線形関数であっても、状況ははるかに複雑になりえます。これを見るために、[再生核ヒルベルト空間（RKHS）](https://en.wikipedia.org/wiki/Reproducing_kernel_Hilbert_space) の概念は、
線形関数のために導入された道具を
非線形の文脈に適用することを可能にします。
残念ながら、RKHS ベースのアルゴリズムは
大規模で高次元のデータに対しては
スケーリングがうまくいかない傾向があります。
この本では、しばしば
重み減衰を深層ネットワークのすべての層に適用する
という一般的なヒューリスティックを採用します。

## まとめ

正則化は過学習に対処するための一般的な方法です。古典的な正則化手法では、学習時に損失関数へ罰則項を加えることで、学習されたモデルの複雑さを抑えます。
モデルを単純に保つための特定の選択肢の一つが、$\ell_2$ 罰則を使うことです。これにより、ミニバッチ確率的勾配降下法の更新ステップに重み減衰が現れます。
実際には、重み減衰の機能は深層学習フレームワークの最適化器に備わっています。
同じ訓練ループの中でも、異なるパラメータ集合に対して異なる更新挙動を持たせることができます。



## 演習

1. この節の推定問題で $\lambda$ の値を変えて実験しなさい。訓練精度と検証精度を $\lambda$ の関数としてプロットしなさい。何が観察できますか？
1. 検証セットを用いて $\lambda$ の最適値を見つけなさい。本当に最適値でしょうか？ それは重要でしょうか？
1. 罰則として $\|\mathbf{w}\|^2$ の代わりに $\sum_i |w_i|$ を用いた場合（$\ell_1$ 正則化）、更新方程式はどのようになりますか？
1. $\|\mathbf{w}\|^2 = \mathbf{w}^\top \mathbf{w}$ であることは分かっています。行列に対しても同様の式を見つけられますか（:numref:`subsec_lin-algebra-norms` のフロベニウスノルムを参照）？
1. 訓練誤差と汎化誤差の関係を復習しなさい。重み減衰に加えて、訓練の増加や適切な複雑さを持つモデルの使用以外に、過学習に対処するのに役立つ方法は何でしょうか？
1. ベイズ統計では、事前分布と尤度の積を用いて $P(w \mid x) \propto P(x \mid w) P(w)$ により事後分布を得ます。$P(w)$ を正則化とどのように対応づけられますか？

:begin_tab:`mxnet`
[議論](https://discuss.d2l.ai/t/98)
:end_tab:

:begin_tab:`pytorch`
[議論](https://discuss.d2l.ai/t/99)
:end_tab:

:begin_tab:`tensorflow`
[議論](https://discuss.d2l.ai/t/236)
:end_tab:

:begin_tab:`jax`
[議論](https://discuss.d2l.ai/t/17979)
:end_tab:\n