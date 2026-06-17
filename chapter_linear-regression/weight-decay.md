{.python .input}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow', 'jax'])
```

# 重み減衰
:label:`sec_weight_decay`

**重み減衰（weight decay）** は、機械学習モデルの過学習を防ぐために、損失関数へパラメータ（重み）の大きさに応じたペナルティ（典型的には $\ell_2$ ノルム）を加え、モデルの複雑さを抑える代表的な正則化手法である。

過学習の問題を見てきたので、最初の*正則化*手法を導入できる。
過学習は、訓練データをさらに集めれば常にある程度は緩和できることを思い出そう。
しかし、それには費用も時間もかかり、
そもそも制御できない場合もあるため、
短期的には実行不可能なことが多い。
ここでは、すでに
利用可能な範囲で十分に高品質なデータを持っていると仮定し、
与えられたデータセットのもとで使える手法に焦点を当てる。

多項式回帰の例
(:numref:`subsec_polynomial-curve-fitting`)
では、当てはめる多項式の次数を調整することで
モデル容量を制御できた。
実際、特徴量の数を制限することは
過学習を抑える一般的な方法である。
しかし、単に特徴量を削るだけでは
粗すぎることがある。
多項式回帰の例に戻り、
高次元入力で何が起こるかを考えよう。
多変量データに対する多項式の自然な拡張は
*単項式*であり、変数のべきの積で表される。
単項式の次数は、それぞれのべきの和である。
たとえば、$x_1^2 x_2$ と $x_3 x_5^2$
はいずれも次数3の単項式である。

次数 $d$ の項の数は、$d$ が大きくなるにつれて
急速に増大することに注意されたい。
$k$ 個の変数があるとき、次数 $d$ の単項式の数は
${k - 1 + d} \choose {k - 1}$ である。
次数を $2$ から $3$ に上げるといった小さな変更でも、
モデルの複雑さは劇的に増える。
したがって、関数の複雑さを調整するには、
よりきめ細かな手段がしばしば必要になる。

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

(**パラメータ数を直接制限する代わりに、
*重み減衰*はパラメータが取りうる値を制約することで機能する。**)
深層学習以外の文脈では、ミニバッチ確率的勾配降下法で最適化する場合、
より一般に $\ell_2$ 正則化と呼ばれる。
重み減衰は、パラメトリックな機械学習モデルを正則化するための
最も広く使われている手法の一つである。
この手法の背後にある基本的な直観は、
すべての関数 $f$ の中で、
関数 $f = 0$
（あらゆる入力に対して値 $0$ を返す関数）が
ある意味で*最も単純*であり、
関数の複雑さはパラメータがゼロからどれだけ離れているかで測れる、
というものである。
では、関数とゼロとの距離を
どのように測ればよいのだろうか。
唯一の正解があるわけではない。
実際、関数解析や
バナッハ空間の理論を含む数学の大きな分野が、
この種の問題を扱っている。

一つの単純な考え方は、
線形関数 $f(\mathbf{x}) = \mathbf{w}^\top \mathbf{x}$
の複雑さを、その重みベクトルの何らかのノルム、たとえば $\| \mathbf{w} \|^2$ で測ることである。
:numref:`subsec_lin-algebra-norms` で、
より一般的な $\ell_p$ ノルムの特殊な場合として
$\ell_2$ ノルムと $\ell_1$ ノルムを導入したことを思い出そう。
重みベクトルを小さく保つ最も一般的な方法は、
そのノルムを損失最小化問題に罰則項として加えることである。
したがって、元の目的関数、
すなわち*訓練ラベルに対する予測損失を最小化すること*を、
新しい目的関数、
すなわち*予測損失と罰則項の和を最小化すること*に置き換える。
こうすると、重みベクトルが大きくなりすぎたとき、
学習アルゴリズムは訓練誤差の最小化よりも
重みノルム $\| \mathbf{w} \|^2$ の最小化を
より重視するようになる。
それこそが狙いである。
これをコードで示すために、
:numref:`sec_linear_regression` の線形回帰の例を再び取り上げる。
そこでは、損失は次のように与えられていた。

$$L(\mathbf{w}, b) = \frac{1}{n}\sum_{i=1}^n \frac{1}{2}\left(\mathbf{w}^\top \mathbf{x}^{(i)} + b - y^{(i)}\right)^2.$$

$\mathbf{x}^{(i)}$ は特徴量、
$y^{(i)}$ はデータ例 $i$ のラベルであり、$(\mathbf{w}, b)$
はそれぞれ重みパラメータとバイアスパラメータである。
重みベクトルの大きさに罰則を課すには、
損失関数に何らかの形で $\| \mathbf{w} \|^2$ を加える必要がある。
では、この新しい加法的な罰則に対して、モデルは標準的な損失と
どのように折り合いをつけるべきだろうか。
実際には、このトレードオフを
*正則化定数* $\lambda$ で表す。
これは非負のハイパーパラメータであり、
検証データを用いて調整する。

$$L(\mathbf{w}, b) + \frac{\lambda}{2} \|\mathbf{w}\|^2.$$


$\lambda = 0$ なら、元の損失関数に戻る。
$\lambda > 0$ なら、$\| \mathbf{w} \|$ の大きさを制限する。
$2$ で割るのは慣習である。
二次関数を微分するとき、
$2$ と $1/2$ が打ち消し合うため、更新式が
見通しよく簡潔になる。
鋭い読者は、なぜ標準ノルム（すなわちユークリッド距離）そのものではなく、
その二乗を使うのかと疑問に思うかもしれない。
これは計算上の都合による。
$\ell_2$ ノルムを二乗すると平方根が消え、
重みベクトルの各成分の二乗和だけが残る。
その結果、罰則項の微分を容易に計算できる。
すなわち、和の微分は各項の微分の和に等しい。

さらに、そもそもなぜ $\ell_1$ ノルムではなく
$\ell_2$ ノルムを使うのか、とも考えられる。
実際、他の選択肢も有効であり、
統計学では広く用いられている。
$\ell_2$ 正則化を施した線形モデルは古典的な
*リッジ回帰*を与える一方で、
$\ell_1$ 正則化を施した線形回帰も
同様に基本的な統計手法であり、
一般に *ラッソ回帰* と呼ばれる。
$\ell_2$ ノルムを使う理由の一つは、
重みベクトルの大きな成分に対して
特に強い罰則を課す点にある。
その結果、学習アルゴリズムは、
より多くの特徴量に重みを分散させるモデルへと
偏る傾向がある。
実際、これは単一の変数の測定誤差に対して
より頑健になる可能性がある。
これに対して、$\ell_1$ 罰則は、
他の重みをゼロにすることで、
少数の特徴量に重みを集中させるモデルを導く。
これにより、*特徴選択*のための有効な手法が得られ、
別の理由から望ましい場合がある。
たとえば、モデルが少数の特徴量にしか依存しないなら、
他の（捨てられた）特徴量についてデータを収集、保存、送信する必要が
なくなるかもしれない。

:eqref:`eq_linreg_batch_update` と同じ記法を用いると、
ミニバッチ確率的勾配降下法による
$\ell_2$ 正則化回帰の更新は次のようになる。

$$\begin{aligned}
\mathbf{w} & \leftarrow \left(1- \eta\lambda \right) \mathbf{w} - \frac{\eta}{|\mathcal{B}|} \sum_{i \in \mathcal{B}} \mathbf{x}^{(i)} \left(\mathbf{w}^\top \mathbf{x}^{(i)} + b - y^{(i)}\right).
\end{aligned}$$

これまでと同様に、推定値が観測値からどれだけずれているかに基づいて
$\mathbf{w}$ を更新する。
しかし同時に、$\mathbf{w}$ の大きさをゼロへ向かって縮小する。
そのため、この手法はしばしば「重み減衰」と呼ばれる。
罰則項だけを考えれば、
最適化アルゴリズムは各学習ステップで重みを*減衰*させる。
特徴選択とは対照的に、重み減衰は関数の複雑さを
連続的に調整する仕組みを与える。
$\lambda$ が小さいほど $\mathbf{w}$ への制約は弱くなり、
一方で $\lambda$ が大きいほど
$\mathbf{w}$ はより強く制約される。
対応するバイアス罰則 $b^2$ を含めるかどうかは
実装によって異なり、
ニューラルネットワークでは層ごとに異なる場合もある。
多くの場合、バイアス項は正則化しない。
さらに、
$\ell_2$ 正則化が他の最適化アルゴリズムでは重み減衰と等価でない場合があるとしても、
重みの大きさを縮小することで正則化するという考え方自体は
依然として有効である。

## 高次元線形回帰

重み減衰の利点は、
単純な合成例で示せる。

まず、以前と同様に [**データを生成する**]：

[**$$y = 0.05 + \sum_{i = 1}^d 0.01 x_i + \epsilon \textrm{ where }
\epsilon \sim \mathcal{N}(0, 0.01^2).$$**]

この合成データセットでは、ラベルは入力の背後にある線形関数によって与えられ、
平均0、標準偏差0.01のガウスノイズで
汚されている。
説明のために、
問題の次元を $d = 200$ に増やし、
20例しかない小さな訓練集合で学習することで、
過学習の影響を顕著にする。

```{.python .input}
%%tab pytorch, mxnet
class Data(d2l.DataModule):
    def __init__(self, num_train, num_val, num_inputs, batch_size):
        self.save_hyperparameters()                
        n = num_train + num_val 
        self.X = d2l.randn(n, num_inputs)
        noise = d2l.randn(n, 1) * 0.01
        w, b = d2l.ones((num_inputs, 1)) * 0.01, 0.05
        self.y = d2l.matmul(self.X, w) + b + noise

    def get_dataloader(self, train):
        i = slice(0, self.num_train) if train else slice(self.num_train, None)
        return self.get_tensorloader([self.X, self.y], train, i)
```

```{.python .input}
%%tab jax
class Data(d2l.DataModule):
    def __init__(self, num_train, num_val, num_inputs, batch_size):
        self.save_hyperparameters()                
        n = num_train + num_val 
        self.X = jax.random.normal(jax.random.PRNGKey(0), (n, num_inputs))
        noise = jax.random.normal(jax.random.PRNGKey(0), (n, 1)) * 0.01
        w, b = d2l.ones((num_inputs, 1)) * 0.01, 0.05
        self.y = d2l.matmul(self.X, w) + b + noise

    def get_dataloader(self, train):
        i = slice(0, self.num_train) if train else slice(self.num_train, None)
        return self.get_tensorloader([self.X, self.y], train, i)
```

```{.python .input}
%%tab tensorflow
class Data(d2l.DataModule):
    def __init__(self, num_train, num_val, num_inputs, batch_size):
        self.save_hyperparameters()                
        n = num_train + num_val 
        self.X = d2l.normal((n, num_inputs))
        noise = d2l.normal((n, 1)) * 0.01
        w, b = d2l.ones((num_inputs, 1)) * 0.01, 0.05
        self.y = d2l.matmul(self.X, w) + b + noise

    def get_dataloader(self, train):
        i = slice(0, self.num_train) if train else slice(self.num_train, None)
        return self.get_tensorloader([self.X, self.y], train, i)
```

## ゼロからの実装

では、重み減衰をゼロから実装しよう。
最適化にはミニバッチ確率的勾配降下法を用いるので、
元の損失関数に二乗した $\ell_2$ 罰則を加えるだけでよい。

### [**$\ell_2$ ノルム罰則の定義**]

この罰則を実装する最も簡単な方法は、
各項をその場で二乗してから和を取ることである。

```{.python .input}
%%tab all
def l2_penalty(w):
    return d2l.reduce_sum(w**2) / 2
```

### モデルの定義

最終的なモデルでは、
線形回帰と二乗損失は :numref:`sec_linear_scratch` と同じなので、
`d2l.LinearRegressionScratch` のサブクラスを定義するだけでよい。
ここでの唯一の変更点は、損失に罰則項を含めることである。

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

次のコードは、20例の訓練集合でモデルを学習し、100例の検証集合で評価する。

```{.python .input}
%%tab pytorch, mxnet, tensorflow
data = Data(num_train=20, num_val=100, num_inputs=200, batch_size=5)
trainer = d2l.Trainer(max_epochs=10)

def train_scratch(lambd):    
    model = WeightDecayScratch(num_inputs=200, lambd=lambd, lr=0.01)
    model.board.yscale='log'
    trainer.fit(model, data)
    print('L2 norm of w:', float(l2_penalty(model.w)))
```

```{.python .input}
%%tab jax
data = Data(num_train=20, num_val=100, num_inputs=200, batch_size=5)
trainer = d2l.Trainer(max_epochs=10)

def train_scratch(lambd):    
    model = WeightDecayScratch(num_inputs=200, lambd=lambd, lr=0.01)
    model.board.yscale='log'
    trainer.fit(model, data)
    print('L2 norm of w:',
          float(l2_penalty(trainer.state.params['w'])))
```

### [**正則化なしでの学習**]

ここでは `lambd = 0` としてこのコードを実行し、
重み減衰を無効にする。
訓練誤差は下がる一方で検証誤差は下がらず、
深刻な過学習が起きていることに注意されたい。
過学習の典型例である。

```{.python .input}
%%tab all
train_scratch(0)
```

### [**重み減衰の使用**]

以下では、かなり強い重み減衰をかけて実行する。
訓練誤差は増加するが、
検証誤差は減少することに注意されたい。
これはまさに正則化に期待される効果である。

```{.python .input}
%%tab all
train_scratch(3)
```

## [**簡潔な実装**]

重み減衰はニューラルネットワークの最適化で
広く使われているため、深層学習フレームワークでは特に扱いやすい。
最適化アルゴリズム自体に重み減衰を組み込み、
任意の損失関数と容易に組み合わせられるようにしている。
さらに、この統合には計算上の利点もあり、
追加の計算オーバーヘッドなしに、
実装上の工夫によってアルゴリズムへ重み減衰を組み込める。
更新のうち重み減衰に対応する部分は
各パラメータの現在値のみに依存するため、
最適化器はどうせ各パラメータに一度はアクセスする必要があるからである。

:begin_tab:`mxnet`
以下では、`Trainer` をインスタンス化するときに
`wd` を通じて重み減衰ハイパーパラメータを直接指定する。
デフォルトでは、Gluon は
重みとバイアスの両方を同時に減衰させる。
ハイパーパラメータ `wd`
はモデルパラメータ更新時に `wd_mult`
と掛け合わされることに注意されたい。
したがって、`wd_mult` をゼロに設定すると、
バイアスパラメータ $b$ は減衰しない。
:end_tab:

:begin_tab:`pytorch`
以下では、最適化器をインスタンス化するときに
`weight_decay` を通じて重み減衰ハイパーパラメータを直接指定する。
デフォルトでは、PyTorch は
重みとバイアスの両方を同時に減衰させるが、
最適化器を異なるパラメータ群に対して異なる方針で動作するよう設定できる。
ここでは、重みに対してのみ（`net.weight` パラメータに対してのみ）
`weight_decay` を設定しているため、
バイアス（`net.bias` パラメータ）は減衰しない。
:end_tab:

:begin_tab:`tensorflow`
以下では、重み減衰ハイパーパラメータ `wd` を持つ $\ell_2$ 正則化器を作成し、
`kernel_regularizer` 引数を通じて層の重みに適用する。
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
        # Weight Decayはoptax.sgd内では直接利用できないが
        # optaxは複数の変換を連鎖させることができる
        return optax.chain(optax.additive_weight_decay(self.wd),
                           optax.sgd(self.lr))
```

[**プロットは、ゼロから重み減衰を実装した場合とよく似ている**]。
しかし、この実装のほうが高速であり、
記述も容易である。
問題が大きくなり、作業がより日常的になるほど、
これらの利点はさらに大きくなる。

```{.python .input}
%%tab pytorch, mxnet, tensorflow
model = WeightDecay(wd=3, lr=0.01)
model.board.yscale='log'
trainer.fit(model, data)

print('L2 norm of w:', float(l2_penalty(model.get_w_b()[0])))
```

```{.python .input}
%%tab jax
model = WeightDecay(wd=3, lr=0.01)
model.board.yscale='log'
trainer.fit(model, data)

print('L2 norm of w:', float(l2_penalty(model.get_w_b(trainer.state)[0])))
```

ここまでで、単純な線形関数を何によって単純とみなすかについて
一つの考え方を見た。
しかし、単純な非線形関数であっても、状況ははるかに複雑になりうる。これを理解するうえで有用なのが[再生核ヒルベルト空間（RKHS）](https://en.wikipedia.org/wiki/Reproducing_kernel_Hilbert_space)の概念であり、これを用いることで、
線形関数のために導入した道具を
非線形の文脈へ拡張できる。
残念ながら、RKHS に基づくアルゴリズムは
大規模かつ高次元のデータに対しては
スケーラビリティに難があることが多い。
この本では、しばしば
重み減衰を深層ネットワークのすべての層に適用する
という一般的なヒューリスティックを採用する。

## まとめ

正則化は過学習に対処するための一般的な方法である。古典的な正則化手法では、学習時に損失関数へ罰則項を加えることで、学習されたモデルの複雑さを抑える。
モデルを単純に保つための代表的な選択肢の一つが、$\ell_2$ 罰則を用いることである。これにより、ミニバッチ確率的勾配降下法の更新ステップに重み減衰が現れる。
実際には、重み減衰の機能は深層学習フレームワークの最適化器に組み込まれている。
同じ訓練ループの中でも、異なるパラメータ集合に対して異なる更新挙動を与えられる。

## 演習

1. この節の推定問題で $\lambda$ の値を変えて実験しなさい。訓練精度と検証精度を $\lambda$ の関数としてプロットしなさい。何が観察できるか。
1. 検証集合を用いて $\lambda$ の最適値を見つけなさい。それは本当に最適値だろうか。重要だろうか。
1. 罰則として $\|\mathbf{w}\|^2$ の代わりに $\sum_i |w_i|$ を用いた場合（$\ell_1$ 正則化）、更新方程式はどのようになるか。
1. $\|\mathbf{w}\|^2 = \mathbf{w}^\top \mathbf{w}$ であることは分かっている。行列に対しても同様の式を見つけられるか（:numref:`subsec_lin-algebra-norms` のフロベニウスノルムを参照）。
1. 訓練誤差と汎化誤差の関係を復習しなさい。重み減衰に加えて、訓練データを増やすことや適切な複雑さを持つモデルを使うこと以外に、過学習に対処するのに役立つ方法は何だろうか。
1. ベイズ統計では、事前分布と尤度の積を用いて $P(w \mid x) \propto P(x \mid w) P(w)$ により事後分布を得る。$P(w)$ を正則化とどのように対応づけられるか。