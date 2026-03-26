{.python .input}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow', 'jax'])
```

# バッチ正規化
:label:`sec_batch_norm`

深層ニューラルネットワークの学習は難しいです。
妥当な時間内に収束させるのは厄介なことがあります。
この節では、深層ネットワークの収束を一貫して加速する、広く使われている効果的な手法である *バッチ正規化* を説明します :cite:`Ioffe.Szegedy.2015`。
後ほど :numref:`sec_resnet` で扱う残差ブロックと組み合わせることで、バッチ正規化により、実務者が100層を超えるネットワークを日常的に学習できるようになりました。
バッチ正規化には、副次的な（思いがけない）利点として、本質的な正則化効果もあります。

```{.python .input}
%%tab mxnet
from d2l import mxnet as d2l
from mxnet import autograd, np, npx, init
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
from jax import numpy as jnp
import jax
import optax
```

## 深層ネットワークの学習

データを扱うとき、学習の前に前処理を行うことがよくあります。
データ前処理に関する選択は、最終結果に非常に大きな違いをもたらすことがあります。
家の価格を予測するために MLP を適用した例（:numref:`sec_kaggle_house`）を思い出してください。
実データを扱う際の最初のステップは、入力特徴量を複数の観測にわたって平均0、分散1になるよう標準化することでした $\boldsymbol{\mu} = 0$ と $\boldsymbol{\Sigma} = \boldsymbol{1}$ :cite:`friedman1987exploratory`。しばしば後者は対角成分が1になるように再スケーリングされ、すなわち $\Sigma_{ii} = 1$ とします。
別の戦略として、ベクトルを単位長に再スケーリングし、場合によっては *各観測ごと* に平均0にする方法もあります。
これは、たとえば空間センサーデータではうまく機能することがあります。これらの前処理技術や他の多くの技術は、推定問題を適切に制御するうえで有益です。 
特徴選択と特徴抽出のレビューについては、たとえば :citet:`guyon2008feature` の論文を参照してください。
ベクトルを標準化することには、そこに作用する関数の複雑さを制約するという嬉しい副作用もあります。たとえば、サポートベクターマシンにおける有名な radius-margin bound :cite:`Vapnik95` やパーセプトロン収束定理 :cite:`Novikoff62` は、ノルムが有界な入力に依存しています。 

直感的には、この標準化は最適化器と相性がよいです。
というのも、パラメータを *a priori* に同程度のスケールに置くからです。
そうであれば、深層ネットワークの *内部* に対応する正規化のステップが有益でないかを問うのは自然です。
これがバッチ正規化 :cite:`Ioffe.Szegedy.2015` の発明に至った厳密な理由ではありませんが、バッチ正規化とその親戚である層正規化 :cite:`Ba.Kiros.Hinton.2016` を統一的な枠組みで理解するうえで有用な見方です。

第二に、典型的な MLP や CNN では、学習が進むにつれて、
中間層の変数
（たとえば MLP のアフィン変換の出力）は、
入力から出力へ向かう層方向、同じ層内のユニット間、そしてモデルパラメータの更新による時間変化のために、
大きさが大きく異なる値を取ることがあります。
バッチ正規化の発明者たちは、こうした変数の分布のドリフトがネットワークの収束を妨げる可能性があると、非公式に仮定しました。
直感的には、ある層の活性化が別の層の100倍の大きさで変動するなら、学習率に補償的な調整が必要になるかもしれません。
AdaGrad :cite:`Duchi.Hazan.Singer.2011`、Adam :cite:`Kingma.Ba.2014`、Yogi :cite:`Zaheer.Reddi.Sachan.ea.2018`、あるいは Distributed Shampoo :cite:`anil2020scalable` のような適応的ソルバは、二次法の要素を加えるなどして、最適化の観点からこの問題に対処しようとします。 
別の方法は、単に適応的な正規化によって問題の発生を防ぐことです。

第三に、より深いネットワークは複雑であり、過学習しやすい傾向があります。
これは、正則化がより重要になることを意味します。正則化の一般的な手法はノイズ
注入です。これは昔から知られており、たとえば入力へのノイズ注入 :cite:`Bishop.1995` に関しても研究されてきました。これは :numref:`sec_dropout` におけるドロップアウトの基礎にもなっています。結果として、かなり幸運なことに、バッチ正規化はこの3つの利点すべて、すなわち前処理、数値安定性、正則化をもたらします。

バッチ正規化は個々の層に適用され、あるいは必要に応じてすべての層に適用されます。
各学習反復で、
まず（バッチ正規化の）入力を
平均を引き、
標準偏差で割ることで正規化します。
この平均と標準偏差は、現在のミニバッチの統計量に基づいて推定されます。
次に、失われた自由度を回復するためにスケール係数とオフセットを適用します。まさに *バッチ* 統計量に基づく *正規化* であることから、
*バッチ正規化* という名前が付いています。

サイズ1のミニバッチでバッチ正規化を適用しようとしても、何も学習できないことに注意してください。
平均を引いた後、
各隠れユニットの値は0になるからです。
おそらく予想できるように、この節全体をバッチ正規化に割いているのは、十分に大きなミニバッチではこの手法が有効で安定だからです。
ここから得られる教訓の一つは、バッチ正規化を適用する際には、
バッチサイズの選択が
バッチ正規化なしの場合よりもさらに重要であり、少なくとも、
バッチサイズを調整するのと同様に適切な較正が必要だということです。

ミニバッチを $\mathcal{B}$ とし、$\mathbf{x} \in \mathcal{B}$ を
バッチ正規化（$\textrm{BN}$）への入力とします。このとき、バッチ正規化は次のように定義されます。

$$\textrm{BN}(\mathbf{x}) = \boldsymbol{\gamma} \odot \frac{\mathbf{x} - \hat{\boldsymbol{\mu}}_\mathcal{B}}{\hat{\boldsymbol{\sigma}}_\mathcal{B}} + \boldsymbol{\beta}.$$
:eqlabel:`eq_batchnorm`

:eqref:`eq_batchnorm` において、
$\hat{\boldsymbol{\mu}}_\mathcal{B}$ はミニバッチ $\mathcal{B}$ の標本平均であり、
$\hat{\boldsymbol{\sigma}}_\mathcal{B}$ は標本標準偏差です。
標準化を適用すると、
得られるミニバッチは
平均0、分散1になります。
分散1
（他の任意の値ではなく）を選ぶのは恣意的です。この自由度は、
$\mathbf{x}$ と同じ形状を持つ要素ごとの
*スケールパラメータ* $\boldsymbol{\gamma}$ と *シフトパラメータ* $\boldsymbol{\beta}$
を含めることで回復します。どちらもモデル学習の一部として
学習する必要があるパラメータです。

中間層の変数の大きさは、
バッチ正規化が $\hat{\boldsymbol{\mu}}_\mathcal{B}$ と ${\hat{\boldsymbol{\sigma}}_\mathcal{B}}$ を通じて、それらを所定の平均と大きさに積極的に中心化・再スケーリングするため、学習中に発散しません。
特徴量の再スケーリングについて述べたときに触れたように、実践的な経験は、バッチ正規化によってより大きな学習率を使えることを示しています。
:eqref:`eq_batchnorm` における $\hat{\boldsymbol{\mu}}_\mathcal{B}$ と ${\hat{\boldsymbol{\sigma}}_\mathcal{B}}$ は次のように計算します。

$$\hat{\boldsymbol{\mu}}_\mathcal{B} = \frac{1}{|\mathcal{B}|} \sum_{\mathbf{x} \in \mathcal{B}} \mathbf{x}
\textrm{ and }
\hat{\boldsymbol{\sigma}}_\mathcal{B}^2 = \frac{1}{|\mathcal{B}|} \sum_{\mathbf{x} \in \mathcal{B}} (\mathbf{x} - \hat{\boldsymbol{\mu}}_{\mathcal{B}})^2 + \epsilon.$$

分散推定値に小さな定数 $\epsilon > 0$
を加えることで、
たとえ経験的分散推定値が非常に小さい、あるいは消失する場合でも、
ゼロ除算を決して行わないようにしていることに注意してください。
$\hat{\boldsymbol{\mu}}_\mathcal{B}$ と ${\hat{\boldsymbol{\sigma}}_\mathcal{B}}$ の推定は、平均と分散のノイズを含む推定を用いることでスケーリングの問題に対処します。
このノイズは問題だと思うかもしれません。
しかし逆に、実際には有益です。

これは深層学習で繰り返し現れるテーマです。
理論的にはまだ十分に特徴づけられていない理由により、最適化におけるさまざまなノイズ源は、
しばしば学習の高速化と過学習の抑制につながります。
この変動は正則化の一形態として働いているように見えます。
:citet:`Teye.Azizpour.Smith.2018` と :citet:`Luo.Wang.Shao.ea.2018`
は、バッチ正規化の性質をそれぞれベイズ事前分布とペナルティに関連付けました。 
特にこれは、
なぜバッチ正規化が 50〜100 程度の中程度のミニバッチサイズで最もよく機能するのか、という謎に光を当てます。
この特定のミニバッチサイズは、$\hat{\boldsymbol{\sigma}}$ によるスケールの面でも、$\hat{\boldsymbol{\mu}}$ によるオフセットの面でも、各層にちょうど「適切な量」のノイズを注入しているように見えます。
より大きなミニバッチでは推定が安定するため正則化効果が弱まり、逆に小さすぎるミニバッチでは分散が大きすぎて有用な信号が失われます。さらにこの方向を探究し、別種の前処理やフィルタリングを考えることで、他の有効な正則化手法が見つかるかもしれません。

学習済みモデルを固定すると、平均と分散の推定にはデータセット全体を使う方が望ましいと思うかもしれません。
学習が終わった後で、同じ画像が属するバッチによって分類結果が変わるのはなぜでしょうか。
学習中は、すべてのデータ例に対する中間変数が
モデルを更新するたびに変化するため、このような厳密な計算は不可能です。
しかし、モデルが学習された後なら、
各層の変数の平均と分散をデータセット全体に基づいて計算できます。
実際、これはバッチ正規化を用いるモデルでは標準的な実践です。
したがって、バッチ正規化層は *学習モード*（ミニバッチ統計量で正規化）と *予測モード*（データセット統計量で正規化）で異なる振る舞いをします。
この形では、ノイズが学習中にのみ注入される :numref:`sec_dropout` のドロップアウト正則化の振る舞いによく似ています。


## バッチ正規化層

全結合層と畳み込み層のバッチ正規化の実装は少し異なります。
バッチ正規化と他の層との重要な違いの一つは、
前者が一度にミニバッチ全体を対象に動作するため、
他の層を導入したときのようにバッチ次元を無視できないことです。

### 全結合層

全結合層にバッチ正規化を適用する場合、
:citet:`Ioffe.Szegedy.2015` の元論文では、アフィン変換の後、
非線形活性化関数の *前* にバッチ正規化を挿入していました。後の応用では、
活性化関数の直後にバッチ正規化を挿入する実験も行われました。
全結合層への入力を $\mathbf{x}$、
アフィン変換を
$\mathbf{W}\mathbf{x} + \mathbf{b}$（重みパラメータを $\mathbf{W}$、バイアスパラメータを $\mathbf{b}$ とする）、
活性化関数を $\phi$ とすると、
バッチ正規化を用いた全結合層の出力 $\mathbf{h}$ は次のように表せます。

$$\mathbf{h} = \phi(\textrm{BN}(\mathbf{W}\mathbf{x} + \mathbf{b}) ).$$

平均と分散は、
変換を適用するのと *同じ* ミニバッチ上で計算されることを思い出してください。

### 畳み込み層

同様に、畳み込み層では、
畳み込みの後、
非線形活性化関数の前にバッチ正規化を適用できます。全結合層におけるバッチ正規化との重要な違いは、操作を各チャネルごとに
*すべての位置にわたって* 適用することです。これは、畳み込みを導いた位置不変性の仮定と整合的です。つまり、画像内のパターンの具体的な位置は、理解の目的にとって本質的ではないと仮定していました。

ミニバッチに $m$ 個の例が含まれ、
各チャネルについて、
畳み込みの出力の高さが $p$、幅が $q$ であるとします。
畳み込み層では、各バッチ正規化を
出力チャネルごとに $m \cdot p \cdot q$ 個の要素全体に対して同時に行います。
したがって、平均と分散を計算するときには
すべての空間位置の値を集め、
その結果、
あるチャネル内では同じ平均と分散を
各空間位置の値の正規化に適用します。
各チャネルには独自のスケールパラメータとシフトパラメータがあり、
どちらもスカラーです。

### 層正規化
:label:`subsec_layer-normalization-in-bn`

畳み込みの文脈では、サイズ1のミニバッチでもバッチ正規化は十分に定義されることに注意してください。結局のところ、画像全体の位置を平均に使えるからです。したがって、たとえ単一の観測内であっても、平均と分散は十分に定義されます。この考察から :citet:`Ba.Kiros.Hinton.2016` は *層正規化* の概念を導入しました。これはバッチ正規化と同様に動作しますが、1つの観測に対して一度に適用される点だけが異なります。したがって、オフセットとスケーリング係数の両方はスカラーです。$n$ 次元ベクトル $\mathbf{x}$ に対して、層正規化は次のように与えられます。 

$$\mathbf{x} \rightarrow \textrm{LN}(\mathbf{x}) =  \frac{\mathbf{x} - \hat{\mu}}{\hat\sigma},$$

ここでスケーリングとオフセットは係数ごとに適用され、
次のように与えられます。 

$$\hat{\mu} \stackrel{\textrm{def}}{=} \frac{1}{n} \sum_{i=1}^n x_i \textrm{ and }
\hat{\sigma}^2 \stackrel{\textrm{def}}{=} \frac{1}{n} \sum_{i=1}^n (x_i - \hat{\mu})^2 + \epsilon.$$

前と同様に、ゼロ除算を防ぐために小さなオフセット $\epsilon > 0$ を加えます。層正規化を使う大きな利点の一つは、発散を防ぐことです。結局のところ、$\epsilon$ を無視すれば、層正規化の出力はスケールに依存しません。すなわち、任意の $\alpha \neq 0$ に対して $\textrm{LN}(\mathbf{x}) \approx \textrm{LN}(\alpha \mathbf{x})$ が成り立ちます。これは $|\alpha| \to \infty$ で等式になります（近似等式であるのは、分散に対するオフセット $\epsilon$ のためです）。 

層正規化のもう一つの利点は、ミニバッチサイズに依存しないことです。また、学習時かテスト時かにも依存しません。言い換えると、これは単に、活性化を所定のスケールに標準化する決定論的な変換です。これは最適化における発散を防ぐうえで非常に有益です。ここではこれ以上の詳細は省き、興味のある読者には元論文を参照することを勧めます。

### 予測時のバッチ正規化

前に述べたように、バッチ正規化は通常、学習モードと予測モードで異なる振る舞いをします。
第一に、各ミニバッチで推定したときに生じる標本平均と標本分散のノイズは、モデルを学習し終えた後にはもはや望ましくありません。
第二に、バッチごとの正規化統計量を計算する余裕がない場合もあります。
たとえば、
モデルを1件ずつ予測に適用しなければならないことがあります。

通常、学習後には、変数統計量の安定した推定値を得るためにデータセット全体を使い、
予測時にはそれらを固定します。
したがって、バッチ正規化はテスト時と学習時で異なる振る舞いをします。
ドロップアウトにも同様の特徴があることを思い出してください。

## [**スクラッチからの実装**]

バッチ正規化が実際にどのように機能するかを見るために、以下でスクラッチから実装します。

```{.python .input}
%%tab mxnet
def batch_norm(X, gamma, beta, moving_mean, moving_var, eps, momentum):
    # Use autograd to determine whether we are in training mode
    if not autograd.is_training():
        # In prediction mode, use mean and variance obtained by moving average
        X_hat = (X - moving_mean) / np.sqrt(moving_var + eps)
    else:
        assert len(X.shape) in (2, 4)
        if len(X.shape) == 2:
            # When using a fully connected layer, calculate the mean and
            # variance on the feature dimension
            mean = X.mean(axis=0)
            var = ((X - mean) ** 2).mean(axis=0)
        else:
            # When using a two-dimensional convolutional layer, calculate the
            # mean and variance on the channel dimension (axis=1). Here we
            # need to maintain the shape of X, so that the broadcasting
            # operation can be carried out later
            mean = X.mean(axis=(0, 2, 3), keepdims=True)
            var = ((X - mean) ** 2).mean(axis=(0, 2, 3), keepdims=True)
        # In training mode, the current mean and variance are used 
        X_hat = (X - mean) / np.sqrt(var + eps)
        # Update the mean and variance using moving average
        moving_mean = (1.0 - momentum) * moving_mean + momentum * mean
        moving_var = (1.0 - momentum) * moving_var + momentum * var
    Y = gamma * X_hat + beta  # Scale and shift
    return Y, moving_mean, moving_var
```

```{.python .input}
%%tab pytorch
def batch_norm(X, gamma, beta, moving_mean, moving_var, eps, momentum):
    # Use is_grad_enabled to determine whether we are in training mode
    if not torch.is_grad_enabled():
        # In prediction mode, use mean and variance obtained by moving average
        X_hat = (X - moving_mean) / torch.sqrt(moving_var + eps)
    else:
        assert len(X.shape) in (2, 4)
        if len(X.shape) == 2:
            # When using a fully connected layer, calculate the mean and
            # variance on the feature dimension
            mean = X.mean(dim=0)
            var = ((X - mean) ** 2).mean(dim=0)
        else:
            # When using a two-dimensional convolutional layer, calculate the
            # mean and variance on the channel dimension (axis=1). Here we
            # need to maintain the shape of X, so that the broadcasting
            # operation can be carried out later
            mean = X.mean(dim=(0, 2, 3), keepdim=True)
            var = ((X - mean) ** 2).mean(dim=(0, 2, 3), keepdim=True)
        # In training mode, the current mean and variance are used 
        X_hat = (X - mean) / torch.sqrt(var + eps)
        # Update the mean and variance using moving average
        moving_mean = (1.0 - momentum) * moving_mean + momentum * mean
        moving_var = (1.0 - momentum) * moving_var + momentum * var
    Y = gamma * X_hat + beta  # Scale and shift
    return Y, moving_mean.data, moving_var.data
```

```{.python .input}
%%tab tensorflow
def batch_norm(X, gamma, beta, moving_mean, moving_var, eps):
    # Compute reciprocal of square root of the moving variance elementwise
    inv = tf.cast(tf.math.rsqrt(moving_var + eps), X.dtype)
    # Scale and shift
    inv *= gamma
    Y = X * inv + (beta - moving_mean * inv)
    return Y
```

```{.python .input}
%%tab jax
def batch_norm(X, deterministic, gamma, beta, moving_mean, moving_var, eps,
               momentum):
    # Use `deterministic` to determine whether the current mode is training
    # mode or prediction mode
    if deterministic:
        # In prediction mode, use mean and variance obtained by moving average
        # `linen.Module.variables` have a `value` attribute containing the array
        X_hat = (X - moving_mean.value) / jnp.sqrt(moving_var.value + eps)
    else:
        assert len(X.shape) in (2, 4)
        if len(X.shape) == 2:
            # When using a fully connected layer, calculate the mean and
            # variance on the feature dimension
            mean = X.mean(axis=0)
            var = ((X - mean) ** 2).mean(axis=0)
        else:
            # When using a two-dimensional convolutional layer, calculate the
            # mean and variance on the channel dimension (axis=1). Here we
            # need to maintain the shape of X, so that the broadcasting
            # operation can be carried out later
            mean = X.mean(axis=(0, 2, 3), keepdims=True)
            var = ((X - mean) ** 2).mean(axis=(0, 2, 3), keepdims=True)
        # In training mode, the current mean and variance are used
        X_hat = (X - mean) / jnp.sqrt(var + eps)
        # Update the mean and variance using moving average
        moving_mean.value = momentum * moving_mean.value + (1.0 - momentum) * mean
        moving_var.value = momentum * moving_var.value + (1.0 - momentum) * var
    Y = gamma * X_hat + beta  # Scale and shift
    return Y
```

これで [**適切な `BatchNorm` 層を作成**] できます。
この層は、スケール `gamma` とシフト `beta` のための適切なパラメータを保持し、
これらは学習の過程で更新されます。
さらに、この層は
モデル予測時に後で使うための平均と分散の移動平均も保持します。

アルゴリズムの詳細はさておき、実装の背後にある設計パターンに注目してください。
通常、数学的な処理は `batch_norm` のような別関数で定義します。
その後、この機能をカスタム層に統合し、
そのコードは主に、データを適切なデバイスコンテキストへ移すこと、
必要な変数を割り当てて初期化すること、
移動平均（ここでは平均と分散）を追跡することなどの管理作業を担当します。
このパターンにより、数学と定型コードをきれいに分離できます。
また、便宜上ここでは入力形状を自動推論することを考慮していないため、
特徴数を通して指定する必要があります。
現在では、すべての主要な深層学習フレームワークが、高レベルのバッチ正規化 API でサイズと形状を自動検出できます（実際にはこちらを使います）。

```{.python .input}
%%tab mxnet
class BatchNorm(nn.Block):
    # `num_features`: the number of outputs for a fully connected layer
    # or the number of output channels for a convolutional layer. `num_dims`:
    # 2 for a fully connected layer and 4 for a convolutional layer
    def __init__(self, num_features, num_dims, **kwargs):
        super().__init__(**kwargs)
        if num_dims == 2:
            shape = (1, num_features)
        else:
            shape = (1, num_features, 1, 1)
        # The scale parameter and the shift parameter (model parameters) are
        # initialized to 1 and 0, respectively
        self.gamma = self.params.get('gamma', shape=shape, init=init.One())
        self.beta = self.params.get('beta', shape=shape, init=init.Zero())
        # The variables that are not model parameters are initialized to 0 and
        # 1
        self.moving_mean = np.zeros(shape)
        self.moving_var = np.ones(shape)

    def forward(self, X):
        # If `X` is not on the main memory, copy `moving_mean` and
        # `moving_var` to the device where `X` is located
        if self.moving_mean.ctx != X.ctx:
            self.moving_mean = self.moving_mean.copyto(X.ctx)
            self.moving_var = self.moving_var.copyto(X.ctx)
        # Save the updated `moving_mean` and `moving_var`
        Y, self.moving_mean, self.moving_var = batch_norm(
            X, self.gamma.data(), self.beta.data(), self.moving_mean,
            self.moving_var, eps=1e-12, momentum=0.1)
        return Y
```

```{.python .input}
%%tab pytorch
class BatchNorm(nn.Module):
    # num_features: the number of outputs for a fully connected layer or the
    # number of output channels for a convolutional layer. num_dims: 2 for a
    # fully connected layer and 4 for a convolutional layer
    def __init__(self, num_features, num_dims):
        super().__init__()
        if num_dims == 2:
            shape = (1, num_features)
        else:
            shape = (1, num_features, 1, 1)
        # The scale parameter and the shift parameter (model parameters) are
        # initialized to 1 and 0, respectively
        self.gamma = nn.Parameter(torch.ones(shape))
        self.beta = nn.Parameter(torch.zeros(shape))
        # The variables that are not model parameters are initialized to 0 and
        # 1
        self.moving_mean = torch.zeros(shape)
        self.moving_var = torch.ones(shape)

    def forward(self, X):
        # If X is not on the main memory, copy moving_mean and moving_var to
        # the device where X is located
        if self.moving_mean.device != X.device:
            self.moving_mean = self.moving_mean.to(X.device)
            self.moving_var = self.moving_var.to(X.device)
        # Save the updated moving_mean and moving_var
        Y, self.moving_mean, self.moving_var = batch_norm(
            X, self.gamma, self.beta, self.moving_mean,
            self.moving_var, eps=1e-5, momentum=0.1)
        return Y
```

```{.python .input}
%%tab tensorflow
class BatchNorm(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(BatchNorm, self).__init__(**kwargs)

    def build(self, input_shape):
        weight_shape = [input_shape[-1], ]
        # The scale parameter and the shift parameter (model parameters) are
        # initialized to 1 and 0, respectively
        self.gamma = self.add_weight(name='gamma', shape=weight_shape,
            initializer=tf.initializers.ones, trainable=True)
        self.beta = self.add_weight(name='beta', shape=weight_shape,
            initializer=tf.initializers.zeros, trainable=True)
        # The variables that are not model parameters are initialized to 0
        self.moving_mean = self.add_weight(name='moving_mean',
            shape=weight_shape, initializer=tf.initializers.zeros,
            trainable=False)
        self.moving_variance = self.add_weight(name='moving_variance',
            shape=weight_shape, initializer=tf.initializers.ones,
            trainable=False)
        super(BatchNorm, self).build(input_shape)

    def assign_moving_average(self, variable, value):
        momentum = 0.1
        delta = (1.0 - momentum) * variable + momentum * value
        return variable.assign(delta)

    @tf.function
    def call(self, inputs, training):
        if training:
            axes = list(range(len(inputs.shape) - 1))
            batch_mean = tf.reduce_mean(inputs, axes, keepdims=True)
            batch_variance = tf.reduce_mean(tf.math.squared_difference(
                inputs, tf.stop_gradient(batch_mean)), axes, keepdims=True)
            batch_mean = tf.squeeze(batch_mean, axes)
            batch_variance = tf.squeeze(batch_variance, axes)
            mean_update = self.assign_moving_average(
                self.moving_mean, batch_mean)
            variance_update = self.assign_moving_average(
                self.moving_variance, batch_variance)
            self.add_update(mean_update)
            self.add_update(variance_update)
            mean, variance = batch_mean, batch_variance
        else:
            mean, variance = self.moving_mean, self.moving_variance
        output = batch_norm(inputs, moving_mean=mean, moving_var=variance,
            beta=self.beta, gamma=self.gamma, eps=1e-5)
        return output
```

```{.python .input}
%%tab jax
class BatchNorm(nn.Module):
    # `num_features`: the number of outputs for a fully connected layer
    # or the number of output channels for a convolutional layer.
    # `num_dims`: 2 for a fully connected layer and 4 for a convolutional layer
    # Use `deterministic` to determine whether the current mode is training
    # mode or prediction mode
    num_features: int
    num_dims: int
    deterministic: bool = False

    @nn.compact
    def __call__(self, X):
        if self.num_dims == 2:
            shape = (1, self.num_features)
        else:
            shape = (1, 1, 1, self.num_features)

        # The scale parameter and the shift parameter (model parameters) are
        # initialized to 1 and 0, respectively
        gamma = self.param('gamma', jax.nn.initializers.ones, shape)
        beta = self.param('beta', jax.nn.initializers.zeros, shape)

        # The variables that are not model parameters are initialized to 0 and
        # 1. Save them to the 'batch_stats' collection
        moving_mean = self.variable('batch_stats', 'moving_mean', jnp.zeros, shape)
        moving_var = self.variable('batch_stats', 'moving_var', jnp.ones, shape)
        Y = batch_norm(X, self.deterministic, gamma, beta,
                       moving_mean, moving_var, eps=1e-5, momentum=0.9)

        return Y
```

`momentum` を用いて、過去の平均と分散の推定値をどの程度集約するかを制御しました。これはやや紛らわしい名称で、最適化における *モメンタム* 項とはまったく関係がありません。それでも、この項に対して一般に採用されている名前であり、API の命名規則に敬意を払って、コードでも同じ変数名を使っています。

## [**バッチ正規化付き LeNet**]

`BatchNorm` を文脈の中でどう適用するかを見るために、
以下では従来の LeNet モデル（:numref:`sec_lenet`）に適用します。
バッチ正規化は
対応する活性化関数の前に、
畳み込み層または全結合層の後に適用されることを思い出してください。

```{.python .input}
%%tab pytorch, mxnet, tensorflow
class BNLeNetScratch(d2l.Classifier):
    def __init__(self, lr=0.1, num_classes=10):
        super().__init__()
        self.save_hyperparameters()
        if tab.selected('mxnet'):
            self.net = nn.Sequential()
            self.net.add(
                nn.Conv2D(6, kernel_size=5), BatchNorm(6, num_dims=4),
                nn.Activation('sigmoid'),
                nn.AvgPool2D(pool_size=2, strides=2),
                nn.Conv2D(16, kernel_size=5), BatchNorm(16, num_dims=4),
                nn.Activation('sigmoid'),
                nn.AvgPool2D(pool_size=2, strides=2), nn.Dense(120),
                BatchNorm(120, num_dims=2), nn.Activation('sigmoid'),
                nn.Dense(84), BatchNorm(84, num_dims=2),
                nn.Activation('sigmoid'), nn.Dense(num_classes))
            self.initialize()
        if tab.selected('pytorch'):
            self.net = nn.Sequential(
                nn.LazyConv2d(6, kernel_size=5), BatchNorm(6, num_dims=4),
                nn.Sigmoid(), nn.AvgPool2d(kernel_size=2, stride=2),
                nn.LazyConv2d(16, kernel_size=5), BatchNorm(16, num_dims=4),
                nn.Sigmoid(), nn.AvgPool2d(kernel_size=2, stride=2),
                nn.Flatten(), nn.LazyLinear(120),
                BatchNorm(120, num_dims=2), nn.Sigmoid(), nn.LazyLinear(84),
                BatchNorm(84, num_dims=2), nn.Sigmoid(),
                nn.LazyLinear(num_classes))
        if tab.selected('tensorflow'):
            self.net = tf.keras.models.Sequential([
                tf.keras.layers.Conv2D(filters=6, kernel_size=5,
                                       input_shape=(28, 28, 1)),
                BatchNorm(), tf.keras.layers.Activation('sigmoid'),
                tf.keras.layers.AvgPool2D(pool_size=2, strides=2),
                tf.keras.layers.Conv2D(filters=16, kernel_size=5),
                BatchNorm(), tf.keras.layers.Activation('sigmoid'),
                tf.keras.layers.AvgPool2D(pool_size=2, strides=2),
                tf.keras.layers.Flatten(), tf.keras.layers.Dense(120),
                BatchNorm(), tf.keras.layers.Activation('sigmoid'),
                tf.keras.layers.Dense(84), BatchNorm(),
                tf.keras.layers.Activation('sigmoid'),
                tf.keras.layers.Dense(num_classes)])
```

```{.python .input}
%%tab jax
class BNLeNetScratch(d2l.Classifier):
    lr: float = 0.1
    num_classes: int = 10
    training: bool = True

    def setup(self):
        self.net = nn.Sequential([
            nn.Conv(6, kernel_size=(5, 5)),
            BatchNorm(6, num_dims=4, deterministic=not self.training),
            nn.sigmoid,
            lambda x: nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2)),
            nn.Conv(16, kernel_size=(5, 5)),
            BatchNorm(16, num_dims=4, deterministic=not self.training),
            nn.sigmoid,
            lambda x: nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2)),
            lambda x: x.reshape((x.shape[0], -1)),
            nn.Dense(120),
            BatchNorm(120, num_dims=2, deterministic=not self.training),
            nn.sigmoid,
            nn.Dense(84),
            BatchNorm(84, num_dims=2, deterministic=not self.training),
            nn.sigmoid,
            nn.Dense(self.num_classes)])
```

:begin_tab:`jax`
`BatchNorm` 層はバッチ統計量（平均と分散）を計算する必要があるため、Flax は `batch_stats` 辞書を追跡し、各ミニバッチで更新します。`batch_stats` のようなコレクションは、`d2l.Trainer` クラスで定義された `TrainState` オブジェクト（:numref:`oo-design-training`）の属性として保存でき、モデルの順伝播では `mutable` 引数に渡す必要があります。そうすると、Flax は変更された変数を返します。
:end_tab:

```{.python .input}
%%tab jax
@d2l.add_to_class(d2l.Classifier)  #@save
@partial(jax.jit, static_argnums=(0, 5))
def loss(self, params, X, Y, state, averaged=True):
    Y_hat, updates = state.apply_fn({'params': params,
                                     'batch_stats': state.batch_stats},
                                    *X, mutable=['batch_stats'],
                                    rngs={'dropout': state.dropout_rng})
    Y_hat = d2l.reshape(Y_hat, (-1, Y_hat.shape[-1]))
    Y = d2l.reshape(Y, (-1,))
    fn = optax.softmax_cross_entropy_with_integer_labels
    return (fn(Y_hat, Y).mean(), updates) if averaged else (fn(Y_hat, Y), updates)
```

前と同様に、ネットワークを Fashion-MNIST データセットで [**学習**] します。
このコードは、LeNet を最初に学習したときとほぼ同じです。

```{.python .input}
%%tab mxnet, pytorch, jax
trainer = d2l.Trainer(max_epochs=10, num_gpus=1)
data = d2l.FashionMNIST(batch_size=128)
model = BNLeNetScratch(lr=0.1)
if tab.selected('pytorch'):
    model.apply_init([next(iter(data.get_dataloader(True)))[0]], d2l.init_cnn)
trainer.fit(model, data)
```

```{.python .input}
%%tab tensorflow
trainer = d2l.Trainer(max_epochs=10)
data = d2l.FashionMNIST(batch_size=128)
with d2l.try_gpu():
    model = BNLeNetScratch(lr=0.5)
    trainer.fit(model, data)
```

最初のバッチ正規化層から学習されたスケールパラメータ `gamma`
とシフトパラメータ `beta` を [**見てみましょう**]。

```{.python .input}
%%tab mxnet
model.net[1].gamma.data().reshape(-1,), model.net[1].beta.data().reshape(-1,)
```

```{.python .input}
%%tab pytorch
model.net[1].gamma.reshape((-1,)), model.net[1].beta.reshape((-1,))
```

```{.python .input}
%%tab tensorflow
tf.reshape(model.net.layers[1].gamma, (-1,)), tf.reshape(
    model.net.layers[1].beta, (-1,))
```

```{.python .input}
%%tab jax
trainer.state.params['net']['layers_1']['gamma'].reshape((-1,)), \
trainer.state.params['net']['layers_1']['beta'].reshape((-1,))
```

## [**簡潔な実装**]

先ほど自分で定義した `BatchNorm` クラスと比べると、
深層学習フレームワークの高レベル API で定義された `BatchNorm` クラスを直接使うことができます。
コードは上の実装とほぼ同じですが、次元を正しく扱うための追加引数を与える必要がない点が異なります。

```{.python .input}
%%tab pytorch, tensorflow, mxnet
class BNLeNet(d2l.Classifier):
    def __init__(self, lr=0.1, num_classes=10):
        super().__init__()
        self.save_hyperparameters()
        if tab.selected('mxnet'):
            self.net = nn.Sequential()
            self.net.add(
                nn.Conv2D(6, kernel_size=5), nn.BatchNorm(),
                nn.Activation('sigmoid'),
                nn.AvgPool2D(pool_size=2, strides=2),
                nn.Conv2D(16, kernel_size=5), nn.BatchNorm(),
                nn.Activation('sigmoid'),
                nn.AvgPool2D(pool_size=2, strides=2),
                nn.Dense(120), nn.BatchNorm(), nn.Activation('sigmoid'),
                nn.Dense(84), nn.BatchNorm(), nn.Activation('sigmoid'),
                nn.Dense(num_classes))
            self.initialize()
        if tab.selected('pytorch'):
            self.net = nn.Sequential(
                nn.LazyConv2d(6, kernel_size=5), nn.LazyBatchNorm2d(),
                nn.Sigmoid(), nn.AvgPool2d(kernel_size=2, stride=2),
                nn.LazyConv2d(16, kernel_size=5), nn.LazyBatchNorm2d(),
                nn.Sigmoid(), nn.AvgPool2d(kernel_size=2, stride=2),
                nn.Flatten(), nn.LazyLinear(120), nn.LazyBatchNorm1d(),
                nn.Sigmoid(), nn.LazyLinear(84), nn.LazyBatchNorm1d(),
                nn.Sigmoid(), nn.LazyLinear(num_classes))
        if tab.selected('tensorflow'):
            self.net = tf.keras.models.Sequential([
                tf.keras.layers.Conv2D(filters=6, kernel_size=5,
                                       input_shape=(28, 28, 1)),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Activation('sigmoid'),
                tf.keras.layers.AvgPool2D(pool_size=2, strides=2),
                tf.keras.layers.Conv2D(filters=16, kernel_size=5),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Activation('sigmoid'),
                tf.keras.layers.AvgPool2D(pool_size=2, strides=2),
                tf.keras.layers.Flatten(), tf.keras.layers.Dense(120),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Activation('sigmoid'),
                tf.keras.layers.Dense(84),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Activation('sigmoid'),
                tf.keras.layers.Dense(num_classes)])
```

```{.python .input}
%%tab jax
class BNLeNet(d2l.Classifier):
    lr: float = 0.1
    num_classes: int = 10
    training: bool = True

    def setup(self):
        self.net = nn.Sequential([
            nn.Conv(6, kernel_size=(5, 5)),
            nn.BatchNorm(not self.training),
            nn.sigmoid,
            lambda x: nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2)),
            nn.Conv(16, kernel_size=(5, 5)),
            nn.BatchNorm(not self.training),
            nn.sigmoid,
            lambda x: nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2)),
            lambda x: x.reshape((x.shape[0], -1)),
            nn.Dense(120),
            nn.BatchNorm(not self.training),
            nn.sigmoid,
            nn.Dense(84),
            nn.BatchNorm(not self.training),
            nn.sigmoid,
            nn.Dense(self.num_classes)])
```

以下では、[**同じハイパーパラメータを使ってモデルを学習**] します。
通常どおり、高レベル API 版の方がはるかに高速に動作することに注意してください。
これは、そのコードが C++ や CUDA にコンパイルされている一方で、カスタム実装は Python によって解釈される必要があるためです。

```{.python .input}
%%tab mxnet, pytorch, jax
trainer = d2l.Trainer(max_epochs=10, num_gpus=1)
data = d2l.FashionMNIST(batch_size=128)
model = BNLeNet(lr=0.1)
if tab.selected('pytorch'):
    model.apply_init([next(iter(data.get_dataloader(True)))[0]], d2l.init_cnn)
trainer.fit(model, data)
```

```{.python .input}
%%tab tensorflow
trainer = d2l.Trainer(max_epochs=10)
data = d2l.FashionMNIST(batch_size=128)
with d2l.try_gpu():
    model = BNLeNet(lr=0.5)
    trainer.fit(model, data)
```

## 議論

直感的には、バッチ正規化は最適化の地形をより滑らかにすると考えられています。
しかし、深層モデルの学習時に観測される現象については、推測的な直感と真の説明を区別しなければなりません。
より単純な深層ニューラルネットワーク（MLP や従来型 CNN）が、そもそもなぜうまく汎化するのかすら、私たちはまだ知りません。
ドロップアウトや重み減衰を加えても、それらは依然として非常に柔軟であり、未知データへの汎化能力には、より洗練された学習理論的な汎化保証が必要である可能性が高いです。

バッチ正規化を提案した元論文 :cite:`Ioffe.Szegedy.2015` は、強力で有用な道具を導入しただけでなく、それがなぜ機能するのかについての説明も与えました。
それは *internal covariate shift* を減らすからだ、という説明です。
おそらく *internal covariate shift* とは、上で述べた直感、すなわち学習の過程で変数値の分布が変化するという考えを指していたのでしょう。
しかし、この説明には2つの問題がありました。
i) このドリフトは *covariate shift* とは大きく異なり、その名称は不適切です。むしろ concept drift に近いものです。 
ii) この説明は十分に規定されていない直感を与えるだけで、*なぜこの手法が正確に機能するのか* という問いは、厳密な説明を求める未解決のまま残します。
本書全体を通して、実務者が深層ニューラルネットワークの開発を導くために用いる直感を伝えることを目指しています。
しかし、こうした指針となる直感と、確立された科学的事実を区別することが重要だと考えています。
やがてこの内容を習得し、自分で研究論文を書き始めるときには、
技術的主張と勘や推測を明確に区別したいと思うはずです。

バッチ正規化の成功以降、
*internal covariate shift* による説明は、技術文献や機械学習研究の提示方法をめぐるより広い議論の中で、繰り返し取り上げられてきました。
2017年の NeurIPS で Test of Time Award を受賞した際の印象的なスピーチで、Ali Rahimi は *internal covariate shift* を焦点として、現代の深層学習の実践を錬金術になぞらえる議論を展開しました。
その後、この例は、機械学習における懸念すべき傾向を概説したポジションペーパーで詳しく再検討されました :cite:`Lipton.Steinhardt.2018`。
他の著者たちは、バッチ正規化の成功について別の説明を提案しており、その中には :cite:`Santurkar.Tsipras.Ilyas.ea.2018`
のように、バッチ正規化の成功は、元論文で主張されたものとはある意味で逆の振る舞いを示しているにもかかわらず達成されている、と主張するものもあります。


*internal covariate shift* は、技術的な機械学習文献で毎年なされる何千もの同様に曖昧な主張と比べて、特に批判に値するわけではないことも指摘しておきます。
おそらく、この議論の焦点として強く響いたのは、対象読者にとって広く認識しやすかったからでしょう。
バッチ正規化は不可欠な手法であることが証明されており、実運用されている画像分類器のほぼすべてに適用されています。その結果、この手法を導入した論文は数万件もの引用を集めました。
とはいえ、ノイズ注入による正則化、再スケーリングによる高速化、そして最後に前処理という指針は、将来さらに新しい層や手法の発明につながるだろうと私たちは考えています。

より実践的な観点から、バッチ正規化について覚えておくべき点がいくつかあります。

* モデル学習中、バッチ正規化はミニバッチの平均と標準偏差を利用してネットワークの中間出力を継続的に調整し、ニューラルネットワーク全体の各層における中間出力の値をより安定させます。
* バッチ正規化は、全結合層と畳み込み層で少し異なります。実際、畳み込み層では、代わりに層正規化を使える場合があります。
* ドロップアウト層と同様に、バッチ正規化層は学習モードと予測モードで異なる振る舞いをします。
* バッチ正規化は、正則化と最適化の収束改善に有用です。対照的に、内部共変量シフトを減らすという元の動機は、妥当な説明ではないようです。
* 入力摂動に対してより頑健で、影響を受けにくいモデルを得たい場合は、バッチ正規化を取り除くことを検討してください :cite:`wang2022removing`。

## 演習

1. バッチ正規化の前に、全結合層または畳み込み層からバイアスパラメータを取り除くべきでしょうか。なぜですか。
1. バッチ正規化ありとなしの LeNet について、学習率を比較してください。
    1. 検証精度の向上をプロットしてください。
    1. 最適化が失敗する前に、両方の場合で学習率をどこまで大きくできますか。
1. すべての層にバッチ正規化が必要でしょうか。実験してみてください。
1. 平均だけを取り除く「lite」版のバッチ正規化、あるいは分散だけを取り除く版を実装してください。どのように振る舞いますか。
1. `beta` と `gamma` を固定してください。結果を観察し、分析してください。
1. ドロップアウトをバッチ正規化で置き換えられますか。振る舞いはどう変わりますか。
1. 研究アイデア: 適用できそうな他の正規化変換を考えてみてください。
    1. 確率積分変換を適用できますか。
    1. フルランクの共分散推定を使えますか。なぜおそらく使うべきではないのでしょうか。 
    1. 他のコンパクトな行列変種（ブロック対角、低変位ランク、Monarch など）を使えますか。
    1. スパース化圧縮は正則化として働きますか。
    1. 他の射影（たとえば凸錐、対称群固有の変換）を使えますか。\n
