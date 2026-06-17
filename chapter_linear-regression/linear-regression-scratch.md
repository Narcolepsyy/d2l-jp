{.python .input  n=1}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow', 'jax'])
```

# 線形回帰のスクラッチ実装
:label:`sec_linear_scratch`

ここで、線形回帰を完全に動作する形で実装する準備が整った。  
この節では、[**(i) モデル、(ii) 損失関数、(iii) ミニバッチ確率的勾配降下法オプティマイザ、(iv) これらを結び付ける学習関数**] を含め、全体をスクラッチから実装する。  
最後に、:numref:`sec_synthetic-regression-data` の合成データ生成器を実行し、得られたデータセットにモデルを適用する。  
現代の深層学習フレームワークはこの作業の大半を自動化するが、スクラッチから実装することは、何を行っているかを本当に理解するための最良の方法である。  
さらに、モデルをカスタマイズしたり、自前の層や損失関数を定義したりする際には、内部で何が起きているかの理解が大いに役立つ。  
この節では、テンソルと自動微分だけを用いる。  
後の節では、同じ構造を保ちながら、深層学習フレームワークの便利な機能を活用した、より簡潔な実装を紹介する。

```{.python .input  n=2}
%%tab mxnet
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import autograd, np, npx
npx.set_np()
```

```{.python .input  n=3}
%%tab pytorch
%matplotlib inline
from d2l import torch as d2l
import torch
```

```{.python .input  n=4}
%%tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
import tensorflow as tf
```

```{.python .input  n=5}
%%tab jax
%matplotlib inline
from d2l import jax as d2l
from flax import linen as nn
import jax
from jax import numpy as jnp
import optax
```

## モデルの定義

ミニバッチSGDで[**モデルのパラメータを最適化し始める前に**]、[**まずパラメータそのものを用意しなければならない。**]  
以下では、平均0、標準偏差0.01の正規分布から乱数をサンプリングして重みを初期化する。  
この 0.01 という値は実務上しばしばうまく機能するが、引数 `sigma` を通じて別の値を指定してもよい。  
一方、バイアスは 0 に設定する。  
オブジェクト指向設計では、このコードを `d2l.Module` のサブクラスの `__init__` メソッドに追加する（:numref:`subsec_oo-design-models` で導入）。

```{.python .input  n=6}
%%tab pytorch, mxnet, tensorflow
class LinearRegressionScratch(d2l.Module):  #@save
    """The linear regression model implemented from scratch."""
    def __init__(self, num_inputs, lr, sigma=0.01):
        super().__init__()
        self.save_hyperparameters()
        if tab.selected('mxnet'):
            self.w = d2l.normal(0, sigma, (num_inputs, 1))
            self.b = d2l.zeros(1)
            self.w.attach_grad()
            self.b.attach_grad()
        if tab.selected('pytorch'):
            self.w = d2l.normal(0, sigma, (num_inputs, 1), requires_grad=True)
            self.b = d2l.zeros(1, requires_grad=True)
        if tab.selected('tensorflow'):
            w = tf.random.normal((num_inputs, 1), mean=0, stddev=0.01)
            b = tf.zeros(1)
            self.w = tf.Variable(w, trainable=True)
            self.b = tf.Variable(b, trainable=True)
```

```{.python .input  n=7}
%%tab jax
class LinearRegressionScratch(d2l.Module):  #@save
    """The linear regression model implemented from scratch."""
    num_inputs: int
    lr: float
    sigma: float = 0.01

    def setup(self):
        self.w = self.param('w', nn.initializers.normal(self.sigma),
                            (self.num_inputs, 1))
        self.b = self.param('b', nn.initializers.zeros, (1))
```

次に、[**入力とパラメータを出力へ対応付けるモデルを定義する必要がある。**]  
:eqref:`eq_linreg-y-vec` と同じ記法を用いると、線形モデルは入力特徴量 $\mathbf{X}$ とモデルの重み $\mathbf{w}$ の行列ベクトル積を計算し、各例にオフセット $b$ を加えるだけである。  
積 $\mathbf{Xw}$ はベクトルであり、$b$ はスカラーである。  
ブロードキャスト機構（:numref:`subsec_broadcasting` を参照）により、ベクトルとスカラーを加えると、スカラーはベクトルの各成分に加えられる。  
以下の `forward` メソッドは、`add_to_class`（:numref:`oo-design-utilities` で導入）を通じて `LinearRegressionScratch` クラスに登録される。

```{.python .input  n=8}
%%tab all
@d2l.add_to_class(LinearRegressionScratch)  #@save
def forward(self, X):
    return d2l.matmul(X, self.w) + self.b
```

## 損失関数の定義

モデルを更新するには損失関数の勾配が必要なので、[**まず損失関数を定義しなければならない。**]  
ここでは :eqref:`eq_mse` の二乗損失関数を用いる。  
実装では、真の値 `y` を予測値 `y_hat` の形状に合わせて変形する必要がある。  
以下のメソッドが返す結果も `y_hat` と同じ形状になる。  
さらに、ミニバッチ内の全例にわたる平均損失も返す。

```{.python .input  n=9}
%%tab pytorch, mxnet, tensorflow
@d2l.add_to_class(LinearRegressionScratch)  #@save
def loss(self, y_hat, y):
    l = (y_hat - y) ** 2 / 2
    return d2l.reduce_mean(l)
```

```{.python .input  n=10}
%%tab jax
@d2l.add_to_class(LinearRegressionScratch)  #@save
def loss(self, params, X, y, state):
    y_hat = state.apply_fn({'params': params}, *X)  # タプルから展開されたX
    l = (y_hat - d2l.reshape(y, y_hat.shape)) ** 2 / 2
    return d2l.reduce_mean(l)
```

## 最適化アルゴリズムの定義

:numref:`sec_linear_regression` で議論したように、線形回帰には閉形式解が存在する。  
しかし、ここでの目的は、より一般的なニューラルネットワークをどのように学習するかを示すことである。したがって、ミニバッチSGDの使い方を学ぶ必要がある。  
そこで、この機会にSGDの最初の実装例を示す。  
各ステップでは、データセットからランダムに取り出したミニバッチを用いて、パラメータに関する損失の勾配を推定する。  
次に、損失を減少させる方向へパラメータを更新する。

以下のコードは、パラメータ集合と学習率 `lr` が与えられたときに更新を適用する。  
損失はミニバッチ上の平均として計算されるため、バッチサイズに応じて学習率を調整する必要はない。  
後の章では、分散大規模学習で現れるような非常に大きいミニバッチに対して、学習率をどのように調整すべきかを調べる。  
今のところ、この依存関係は無視してよい。

:begin_tab:`mxnet`
`d2l.HyperParameters`（:numref:`oo-design-utilities` で導入）のサブクラスとして `SGD` クラスを定義し、組み込みのSGDオプティマイザと同様のAPIを持たせる。  
パラメータは `step` メソッドで更新する。  
`batch_size` 引数を受け取るが、ここでは無視してよい。
:end_tab:

:begin_tab:`pytorch`
`d2l.HyperParameters`（:numref:`oo-design-utilities` で導入）のサブクラスとして `SGD` クラスを定義し、組み込みのSGDオプティマイザと同様のAPIを持たせる。  
パラメータは `step` メソッドで更新する。  
`zero_grad` メソッドはすべての勾配を 0 に設定し、逆伝播の前に実行しなければならない。
:end_tab:

:begin_tab:`tensorflow`
`d2l.HyperParameters`（:numref:`oo-design-utilities` で導入）のサブクラスとして `SGD` クラスを定義し、組み込みのSGDオプティマイザと同様のAPIを持たせる。  
パラメータは `apply_gradients` メソッドで更新する。  
このメソッドは、パラメータと勾配の組のリストを受け取る。
:end_tab:

```{.python .input  n=11}
%%tab mxnet, pytorch
class SGD(d2l.HyperParameters):  #@save
    """Minibatch stochastic gradient descent."""
    def __init__(self, params, lr):
        self.save_hyperparameters()

    if tab.selected('mxnet'):
        def step(self, _):
            for param in self.params:
                param -= self.lr * param.grad

    if tab.selected('pytorch'):
        def step(self):
            for param in self.params:
                param -= self.lr * param.grad

        def zero_grad(self):
            for param in self.params:
                if param.grad is not None:
                    param.grad.zero_()
```

```{.python .input  n=12}
%%tab tensorflow
class SGD(d2l.HyperParameters):  #@save
    """Minibatch stochastic gradient descent."""
    def __init__(self, lr):
        self.save_hyperparameters()

    def apply_gradients(self, grads_and_vars):
        for grad, param in grads_and_vars:
            param.assign_sub(self.lr * grad)
```

```{.python .input  n=13}
%%tab jax
class SGD(d2l.HyperParameters):  #@save
    """Minibatch stochastic gradient descent."""
    # Optaxの主要な変換はGradientTransformationである
    # initとupdateという2つのメソッドで定義される。
    # initは状態を初期化し、updateは勾配を変換する。
    # https://github.com/deepmind/optax/blob/master/optax/_src/transform.py
    def __init__(self, lr):
        self.save_hyperparameters()

    def init(self, params):
        # 未使用のパラメータを削除する
        del params
        return optax.EmptyState

    def update(self, updates, state, params=None):
        del params
        # state.apply_gradientsメソッドが呼び出されてFlaxのを更新する際に
        # train_stateオブジェクト。内部でoptax.apply_updatesメソッドを呼び出す
        # 以下で定義する更新式にパラメータを加える。
        updates = jax.tree_util.tree_map(lambda g: -self.lr * g, updates)
        return updates, state

    def __call__():
        return optax.GradientTransformation(self.init, self.update)
```

次に、`SGD` クラスのインスタンスを返す `configure_optimizers` メソッドを定義する。

```{.python .input  n=14}
%%tab all
@d2l.add_to_class(LinearRegressionScratch)  #@save
def configure_optimizers(self):
    if tab.selected('mxnet') or tab.selected('pytorch'):
        return SGD([self.w, self.b], self.lr)
    if tab.selected('tensorflow', 'jax'):
        return SGD(self.lr)
```

## 学習

これで、すべての部品（パラメータ、損失関数、モデル、オプティマイザ）が揃い、[**主要な学習ループを実装する準備が整った。**]  
このコードを完全に理解することはきわめて重要である。というのも、この本で扱う他のすべての深層学習モデルでも、同様の学習ループを用いるからである。  
各 *epoch* では、訓練データセット全体を一巡し、すべての例を1回ずつ処理する（例の数がバッチサイズで割り切れると仮定する）。  
各 *iteration* では、訓練例のミニバッチを取り出し、モデルの `training_step` メソッドを通して損失を計算する。  
次に、各パラメータに関する勾配を計算する。  
最後に、最適化アルゴリズムを呼び出してモデルパラメータを更新する。  
要するに、以下のループを実行する。

* パラメータ $(\mathbf{w}, b)$ を初期化する
* 完了するまで繰り返す
    * 勾配 $\mathbf{g} \leftarrow \partial_{(\mathbf{w},b)} \frac{1}{|\mathcal{B}|} \sum_{i \in \mathcal{B}} l(\mathbf{x}^{(i)}, y^{(i)}, \mathbf{w}, b)$ を計算する
    * パラメータ $(\mathbf{w}, b) \leftarrow (\mathbf{w}, b) - \eta \mathbf{g}$ を更新する
 
:numref:`sec_synthetic-regression-data` で生成した合成回帰データセットには検証データセットが含まれていないことを思い出そう。  
しかし一般には、モデルの品質を測定するために検証データセットが必要になる。  
ここでは、各 epoch ごとに検証データローダーを1回通してモデル性能を測定する。  
オブジェクト指向設計に従い、`prepare_batch` と `fit_epoch` メソッドは `d2l.Trainer` クラスに登録される（:numref:`oo-design-training` で導入）。

```{.python .input  n=15}
%%tab all    
@d2l.add_to_class(d2l.Trainer)  #@save
def prepare_batch(self, batch):
    return batch
```

```{.python .input  n=16}
%%tab pytorch
@d2l.add_to_class(d2l.Trainer)  #@save
def fit_epoch(self):
    self.model.train()        
    for batch in self.train_dataloader:        
        loss = self.model.training_step(self.prepare_batch(batch))
        self.optim.zero_grad()
        with torch.no_grad():
            loss.backward()
            if self.gradient_clip_val > 0:  # 後で議論する
                self.clip_gradients(self.gradient_clip_val, self.model)
            self.optim.step()
        self.train_batch_idx += 1
    if self.val_dataloader is None:
        return
    self.model.eval()
    for batch in self.val_dataloader:
        with torch.no_grad():            
            self.model.validation_step(self.prepare_batch(batch))
        self.val_batch_idx += 1
```

```{.python .input  n=17}
%%tab mxnet
@d2l.add_to_class(d2l.Trainer)  #@save
def fit_epoch(self):
    for batch in self.train_dataloader:
        with autograd.record():
            loss = self.model.training_step(self.prepare_batch(batch))
        loss.backward()
        if self.gradient_clip_val > 0:
            self.clip_gradients(self.gradient_clip_val, self.model)
        self.optim.step(1)
        self.train_batch_idx += 1
    if self.val_dataloader is None:
        return
    for batch in self.val_dataloader:        
        self.model.validation_step(self.prepare_batch(batch))
        self.val_batch_idx += 1
```

```{.python .input  n=18}
%%tab tensorflow
@d2l.add_to_class(d2l.Trainer)  #@save
def fit_epoch(self):
    self.model.training = True
    for batch in self.train_dataloader:            
        with tf.GradientTape() as tape:
            loss = self.model.training_step(self.prepare_batch(batch))
        grads = tape.gradient(loss, self.model.trainable_variables)
        if self.gradient_clip_val > 0:
            grads = self.clip_gradients(self.gradient_clip_val, grads)
        self.optim.apply_gradients(zip(grads, self.model.trainable_variables))
        self.train_batch_idx += 1
    if self.val_dataloader is None:
        return
    self.model.training = False
    for batch in self.val_dataloader:        
        self.model.validation_step(self.prepare_batch(batch))
        self.val_batch_idx += 1
```

```{.python .input  n=19}
%%tab jax
@d2l.add_to_class(d2l.Trainer)  #@save
def fit_epoch(self):
    self.model.training = True
    if self.state.batch_stats:
        # 可変状態は後で使用される（例: バッチ正規化）
        for batch in self.train_dataloader:
            (_, mutated_vars), grads = self.model.training_step(self.state.params,
                                                           self.prepare_batch(batch),
                                                           self.state)
            self.state = self.state.apply_gradients(grads=grads)
            # Dropout層を持たないモデルでは無視できる
            self.state = self.state.replace(
                dropout_rng=jax.random.split(self.state.dropout_rng)[0])
            self.state = self.state.replace(batch_stats=mutated_vars['batch_stats'])
            self.train_batch_idx += 1
    else:
        for batch in self.train_dataloader:
            _, grads = self.model.training_step(self.state.params,
                                                self.prepare_batch(batch),
                                                self.state)
            self.state = self.state.apply_gradients(grads=grads)
            # Dropout層を持たないモデルでは無視できる
            self.state = self.state.replace(
                dropout_rng=jax.random.split(self.state.dropout_rng)[0])
            self.train_batch_idx += 1

    if self.val_dataloader is None:
        return
    self.model.training = False
    for batch in self.val_dataloader:
        self.model.validation_step(self.state.params,
                                   self.prepare_batch(batch),
                                   self.state)
        self.val_batch_idx += 1
```

モデルを学習する準備はほぼ整ったが、その前に学習データが必要である。  
ここでは `SyntheticRegressionData` クラスを用い、真のパラメータを与える。  
そして、学習率 `lr=0.03` でモデルを学習し、`max_epochs=3` に設定する。  
一般に、エポック数と学習率はいずれもハイパーパラメータであることに注意されたい。  
ハイパーパラメータの設定は一般に難しく、通常はデータセットを3つに分割したくなる。すなわち、1つは学習用、2つ目はハイパーパラメータ選択用の検証用、そして3つ目は最終評価用である。  
ここではこれらの詳細は省くが、後ほど改めて詳しく扱う。

```{.python .input  n=20}
%%tab all
model = LinearRegressionScratch(2, lr=0.03)
data = d2l.SyntheticRegressionData(w=d2l.tensor([2, -3.4]), b=4.2)
trainer = d2l.Trainer(max_epochs=3)
trainer.fit(model, data)
```

データセットを自分で合成したので、真のパラメータを正確に把握している。  
したがって、学習ループを通じて[**真のパラメータと学習されたパラメータを比較することで、学習が成功したかどうかを評価できる**]。  
実際、両者は非常に近い値になる。

```{.python .input  n=21}
%%tab pytorch
with torch.no_grad():
    print(f'error in estimating w: {data.w - d2l.reshape(model.w, data.w.shape)}')
    print(f'error in estimating b: {data.b - model.b}')
```

```{.python .input  n=22}
%%tab mxnet, tensorflow
print(f'error in estimating w: {data.w - d2l.reshape(model.w, data.w.shape)}')
print(f'error in estimating b: {data.b - model.b}')
```

```{.python .input  n=23}
%%tab jax
params = trainer.state.params
print(f"error in estimating w: {data.w - d2l.reshape(params['w'], data.w.shape)}")
print(f"error in estimating b: {data.b - params['b']}")
```

真のパラメータを正確に復元できることを当然視すべきではない。  
一般に、深いモデルではパラメータの一意な解は存在せず、線形モデルであっても、ある特徴量が他の特徴量に対して線形従属でない場合に限って、パラメータを正確に復元できる。  
しかし機械学習では、真の潜在パラメータを復元することよりも、高精度な予測をもたらすパラメータのほうが重要であることが多い :cite:`Vapnik.1992`。  
幸い、難しい最適化問題であっても、確率的勾配降下法はしばしば驚くほど良い解を見つける。深層ネットワークでは高精度な予測をもたらすパラメータ構成が多数存在することが、その一因である。


## まとめ

この節では、完全に機能するニューラルネットワークモデルと学習ループを実装し、深層学習システム設計に向けて大きく前進した。  
この過程で、データローダー、モデル、損失関数、最適化手続き、可視化・監視ツールを構築した。  
これは、モデル学習に関わるすべての要素を含む Python オブジェクトを組み立てることで実現した。  
まだ実運用向けの品質には達していないが、完全に機能しており、この種のコードは小規模な問題を素早く解くのにすでに有用である。  
今後の節では、これを *より簡潔に*（定型コードを減らして）かつ *より効率的に*（GPUを最大限活用して）行う方法を見ていく。



## 演習

1. 重みを0で初期化するとどうなるだろうか。アルゴリズムはそれでも動作するだろうか。では、パラメータを0.01ではなく分散1000で初期化するとどうだろうか。
1. [Georg Simon Ohm](https://en.wikipedia.org/wiki/Georg_Ohm) になったつもりで、電圧と電流を関係付ける抵抗のモデルを考えるとしよう。自動微分を使ってモデルのパラメータを学習できるか。
1. [プランクの法則](https://en.wikipedia.org/wiki/Planck%27s_law) を用いて、スペクトルエネルギー密度から物体の温度を求められるだろうか。参考までに、黒体から放射される放射のスペクトル密度 $B$ は $B(\lambda, T) = \frac{2 hc^2}{\lambda^5} \cdot \left(\exp \frac{h c}{\lambda k T} - 1\right)^{-1}$ である。ここで、$\lambda$ は波長、$T$ は温度、$c$ は光速、$h$ はプランク定数、$k$ はボルツマン定数である。異なる波長 $\lambda$ に対するエネルギーを測定したとすると、スペクトル密度曲線をプランクの法則にフィットさせる必要がある。
1. 損失の2階微分を計算したい場合、どのような問題に遭遇する可能性があるか。それらをどう解決するか。
1. `loss` 関数で `reshape` メソッドが必要なのはなぜであるか。
1. 異なる学習率を使って実験し、損失関数の値がどれほど速く下がるかを調べよ。学習エポック数を増やすことで誤差を減らせるだろうか。
1. 例の数がバッチサイズで割り切れない場合、epoch の終わりで `data_iter` はどうなるか。
1. 絶対値損失 `(y_hat - d2l.reshape(y, y_hat.shape)).abs().sum()` のような別の損失関数を実装してみよ。
    1. 通常のデータで何が起こるか確認せよ。
    1. $\mathbf{y}$ のいくつかの要素、たとえば $y_5 = 10000$ を意図的に摂動させた場合、挙動に違いがあるか確認せよ。
    1. 二乗損失と絶対値損失の長所を組み合わせる安価な解決策を考えられるだろうか。
       ヒント: 非常に大きな勾配値をどう避けるか。
1. なぜデータセットを再シャッフルする必要があるのだろうか。悪意を持って構成されたデータセットが、そうしないと最適化アルゴリズムを破綻させるような例を設計できるか。