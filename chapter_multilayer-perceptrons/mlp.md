{.python .input}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow', 'jax'])
```

# 多層パーセプトロン（MLP）
:label:`sec_mlp`

**多層パーセプトロン（MLP: Multi-Layer Perceptron）** とは、入力層と出力層の間に1つ以上の隠れ層をもつ順伝播型ニューラルネットワークであり、活性化関数を導入することで非線形な問題を扱える、深層学習における最も基本的なアーキテクチャの1つである。

:numref:`sec_softmax` ではソフトマックス回帰を導入し、
そのアルゴリズムをスクラッチから実装し（:numref:`sec_softmax_scratch`）、
さらに高水準 API を用いて実装した（:numref:`sec_softmax_concise`）。
その結果、低解像度画像から 10 種類の衣類を識別する分類器を学習できた。
この過程で、データの整形方法、
出力を妥当な確率分布へ変換する方法、
適切な損失関数の適用方法、
そしてその損失をモデルのパラメータに関して最小化する方法を学んだ。
単純な線形モデルの文脈でこれらの仕組みを理解したので、
次は深層ニューラルネットワークへ進む。
本書の中心となるのは、より高い表現力をもつこれらのモデルである。

```{.python .input}
%%tab mxnet
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import autograd, np, npx
npx.set_np()
```

```{.python .input}
%%tab pytorch
%matplotlib inline
from d2l import torch as d2l
import torch
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
from jax import grad, vmap
```

## 隠れ層

:numref:`subsec_linear_model` では、アフィン変換を
バイアスを加えた線形変換として説明した。
まず、:numref:`fig_softmaxreg` に示した
ソフトマックス回帰のモデル構造を思い出そう。
このモデルでは、単一のアフィン変換の後にソフトマックス演算を適用し、
入力を直接出力へ写像する。
もしラベルが本当に単純なアフィン変換によって
入力データと結び付いているなら、
この方法で十分である。
しかし、線形性（より正確にはアフィン性）は *強い* 仮定である。

### 線形モデルの限界

たとえば、線形性は *単調性* という、より *弱い* 仮定を含意する。
すなわち、ある特徴量が増加したとき、
モデルの出力は必ず増加するか（対応する重みが正の場合）、
あるいは必ず減少するか（対応する重みが負の場合）のいずれかでなければならない。
これは場合によっては妥当である。
たとえば、ある人がローンを返済するかどうかを予測したいとしよう。
他の条件が同じなら、収入が高い申請者のほうが
収入が低い申請者よりも返済する可能性が高い、
と仮定するのは自然かもしれない。
この関係は単調ではあるが、返済確率と線形に結び付いているとは
考えにくい。
収入が \$0 から \$50,000 に増えることは、
\$1 million から \$1.05 million に増えることよりも、
返済可能性をはるかに大きく高めるだろう。
この問題への1つの対処法は、ロジスティック写像（したがって確率の対数オッズ）を用いて、
出力を後処理し、線形性の仮定をより妥当なものにすることである。

単調性すら成り立たない例も容易に挙げられる。
たとえば、体温の関数として健康状態を予測したいとしよう。
37°C（98.6°F）を超える範囲では、
体温が高いほどリスクが高いと考えられる。
しかし、体温が 37°C を下回ると、
今度は体温が低いほどリスクが高くなる。
この場合も、37°C からの距離を特徴量として用いるなど、
工夫した前処理によって問題を解決できるかもしれない。

では、猫と犬の画像分類はどうだろうか。
位置 (13, 17) にある画素の強度を増やすと、
その画像が犬である確率は常に増える（あるいは常に減る）べきだろうか。
線形モデルに依存することは、
猫と犬を区別するために必要なのは各画素の明るさを個別に評価することだけだ、
という暗黙の仮定に対応する。
画像を反転してもカテゴリが変わらないような世界では、
この方法はうまく機能しない。

しかも、この場合の線形性は前の例よりいっそう不自然に見えるにもかかわらず、
単純な前処理の工夫だけで問題を解決できるかどうかは明らかでない。
なぜなら、任意の画素の意味は
その文脈、すなわち周囲の画素の値に複雑に依存するからである。
特徴量間の重要な相互作用を取り込んだ表現が存在し、
その表現の上では線形モデルが適切になる可能性はある。
しかし、そのような表現を手作業で設計する方法は分からない。
深層ニューラルネットワークでは、観測データから
隠れ層による表現と、その表現に作用する線形予測器の両方を
同時に学習する。

このような非線形性の問題は、少なくとも 1 世紀にわたって研究されてきた :cite:`Fisher.1928`。
たとえば、決定木は最も基本的な形では、
一連の二値決定によってクラス所属を判定する :cite:`quinlan2014c4`。
同様に、カーネル法も非線形な依存関係をモデル化するために
長年用いられてきた :cite:`Aronszajn.1950`。
非パラメトリックなスプラインモデル :cite:`Wahba.1990` や
カーネル法 :cite:`Scholkopf.Smola.2002` もこの流れに位置付けられる。
また、脳はこの問題を自然に解いているように見える。
結局のところ、ニューロンは他のニューロンへ入力を送り、
そのニューロンはさらに別のニューロンへ入力を送る :cite:`Cajal.Azoulay.1894`。
その結果、比較的単純な変換が連なった構造が生まれる。

### 隠れ層の導入

1 つ以上の隠れ層を導入することで、
線形モデルの限界を乗り越えられる。
最も単純な方法は、多数の全結合層を
順に積み重ねることである。
各層はその上の層へ入力を渡し、
最終的に出力を生成する。
最初の $L-1$ 層を表現として、
最後の層を線形予測器とみなせる。
この構造は一般に *多層パーセプトロン* と呼ばれ、
しばしば *MLP* と略される（:numref:`fig_mlp`）。

![5 個の隠れユニットを持つ隠れ層を備えた MLP。](../img/mlp.svg)
:label:`fig_mlp`

この MLP は 4 つの入力と 3 つの出力をもち、
隠れ層には 5 個の隠れユニットがある。
入力層には計算が含まれないため、
このネットワークで出力を生成するには
隠れ層と出力層の両方を実装する必要がある。
したがって、この MLP は 2 層ネットワークである。
なお、どちらの層も全結合である。
すべての入力は隠れ層のすべてのニューロンに影響し、
各隠れユニットはさらに出力層のすべてのニューロンに影響する。
ただし、これだけではまだ不十分である。

### 線形から非線形へ

これまでと同様に、$\mathbf{X} \in \mathbb{R}^{n \times d}$ を、
各データ例が $d$ 個の入力（特徴量）をもつ $n$ 個のデータ例からなるミニバッチを表す行列とする。
隠れ層が $h$ 個の隠れユニットをもつ 1 隠れ層 MLP について、
$\mathbf{H} \in \mathbb{R}^{n \times h}$ を隠れ層の出力、
すなわち *隠れ表現* とする。
隠れ層と出力層はいずれも全結合なので、
隠れ層の重み $\mathbf{W}^{(1)} \in \mathbb{R}^{d \times h}$ とバイアス $\mathbf{b}^{(1)} \in \mathbb{R}^{1 \times h}$、
出力層の重み $\mathbf{W}^{(2)} \in \mathbb{R}^{h \times q}$ とバイアス $\mathbf{b}^{(2)} \in \mathbb{R}^{1 \times q}$ をもつ。
このとき、1 隠れ層 MLP の出力 $\mathbf{O} \in \mathbb{R}^{n \times q}$ は次のように計算できる。

$$
\begin{aligned}
    \mathbf{H} & = \mathbf{X} \mathbf{W}^{(1)} + \mathbf{b}^{(1)}, \\
    \mathbf{O} & = \mathbf{H}\mathbf{W}^{(2)} + \mathbf{b}^{(2)}.
\end{aligned}
$$

隠れ層を追加すると、
モデルは新たに増えたパラメータ群を保持し更新しなければならない。
では、その見返りとして何が得られるのだろうか。
驚くべきことに、上で定義したモデルでは、*苦労のわりに何も得ていない*。
理由は明白である。
上の隠れユニットは入力のアフィン関数であり、
出力（ソフトマックス適用前）もまた隠れユニットのアフィン関数にすぎない。
アフィン関数のアフィン関数は、やはりアフィン関数である。
しかも、線形モデルはすでに任意のアフィン関数を表現できる。

これを形式的に示すには、上の定義から隠れ層を消去すればよい。
すると、パラメータ $\mathbf{W} = \mathbf{W}^{(1)}\mathbf{W}^{(2)}$ と $\mathbf{b} = \mathbf{b}^{(1)} \mathbf{W}^{(2)} + \mathbf{b}^{(2)}$ をもつ等価な単層モデルが得られる。

$$
\mathbf{O} = (\mathbf{X} \mathbf{W}^{(1)} + \mathbf{b}^{(1)})\mathbf{W}^{(2)} + \mathbf{b}^{(2)} = \mathbf{X} \mathbf{W}^{(1)}\mathbf{W}^{(2)} + \mathbf{b}^{(1)} \mathbf{W}^{(2)} + \mathbf{b}^{(2)} = \mathbf{X} \mathbf{W} + \mathbf{b}.
$$

多層構造の潜在能力を引き出すには、
もう 1 つ重要な要素が必要である。
それが、アフィン変換の後に各隠れユニットへ適用する
非線形 *活性化関数* $\sigma$ である。
たとえば、よく用いられる選択肢として ReLU（rectified linear unit）活性化関数 :cite:`Nair.Hinton.2010`
$\sigma(x) = \mathrm{max}(0, x)$ がある。
これは引数に要素ごとに作用する。
活性化関数 $\sigma(\cdot)$ の出力は *活性化* と呼ばれる。
一般に、活性化関数を導入すると、
MLP を線形モデルへ畳み込むことはもはやできない。

$$
\begin{aligned}
    \mathbf{H} & = \sigma(\mathbf{X} \mathbf{W}^{(1)} + \mathbf{b}^{(1)}), \\
    \mathbf{O} & = \mathbf{H}\mathbf{W}^{(2)} + \mathbf{b}^{(2)}.\\
\end{aligned}
$$

$\mathbf{X}$ の各行はミニバッチ中の 1 つのデータ例に対応するので、
記法をやや乱用して、非線形性 $\sigma$ は入力に行ごとに作用する、
すなわち 1 データ例ずつ適用されるものとする。
:numref:`subsec_softmax_vectorization` で行ごとの演算を表す際に
ソフトマックスでも同じ記法を用いたことに注意されたい。
実際には、ここで扱う活性化関数の多くは
行ごとであるだけでなく要素ごとにも作用する。
したがって、層の線形部分を計算した後は、
他の隠れユニットの値を参照せずに
各活性化を計算できる。

より一般的な MLP を構築するには、
このような隠れ層をさらに積み重ねればよい。
たとえば、$\mathbf{H}^{(1)} = \sigma_1(\mathbf{X} \mathbf{W}^{(1)} + \mathbf{b}^{(1)})$、
$\mathbf{H}^{(2)} = \sigma_2(\mathbf{H}^{(1)} \mathbf{W}^{(2)} + \mathbf{b}^{(2)})$
のように層を順に重ねることで、
より表現力の高いモデルが得られる。

### 万能近似器

脳がきわめて高度な統計的処理を行えることは明らかである。
したがって、深いネットワークがどれほど強力になり得るかを問う価値がある。
この問いには複数の答えが与えられており、たとえば MLP の文脈では :citet:`Cybenko.1989`、
再生核ヒルベルト空間の文脈では :citet:`micchelli1984interpolation` があり、
これは 1 つの隠れ層をもつ RBF（radial basis function）ネットワークとみなせる形である。
これらの結果（および関連する結果）は、
たとえ 1 隠れ層ネットワークであっても、
十分な数のノード（おそらく途方もなく多い）と
適切な重みがあれば、
任意の関数を近似できることを示唆している。
ただし、問題はその関数を実際に学習することが難しい点にある。
ニューラルネットワークは C 言語のようなものだと考えるとよい。
この言語は、他の現代的な言語と同様に、
計算可能なプログラムなら何でも表現できる。
しかし、仕様を満たすプログラムを実際に書くことは容易でない。

さらに、1 隠れ層ネットワークが
*任意の関数を学習できる* からといって、
すべての問題をそれだけで解くべきだという意味ではない。
実際、この場合にはカーネル法のほうがはるかに有効なこともある。
なぜなら、無限次元空間であっても問題を
*厳密に* 解けるからである :cite:`Kimeldorf.Wahba.1971,Scholkopf.Herbrich.Smola.2001`。
また、より深い（単により広いのではない）ネットワークを用いることで、
多くの関数をはるかにコンパクトに近似できる :cite:`Simonyan.Zisserman.2014`。
より厳密な議論は後続の章で扱う。

## 活性化関数
:label:`subsec_activation-functions`

活性化関数は、重み付き和を計算し、さらにバイアスを加えた後に、
ニューロンをどのように活性化するかを決める。
これは入力信号を出力へ変換する微分可能な演算子であり、
その多くは非線形性を導入する。
活性化関数は深層学習の基礎であるため、
[**代表的なものを簡単に見ていこう**]。

### ReLU 関数

実装が容易で、さまざまな予測タスクで良好な性能を示すことから、
最も広く使われている選択肢は *rectified linear unit*（*ReLU*）である :cite:`Nair.Hinton.2010`。
[**ReLU はきわめて単純な非線形変換を与える**]。
要素 $x$ に対して、この関数は $x$ と 0 の最大値として定義される。

$$\operatorname{ReLU}(x) = \max(x, 0).$$

直感的には、ReLU 関数は正の要素だけを残し、
対応する活性化を 0 にすることで負の要素をすべて捨てる。
直感を得るために、この関数をプロットしてみよう。
見て分かるように、活性化関数は区分線形である。

```{.python .input}
%%tab mxnet
x = np.arange(-8.0, 8.0, 0.1)
x.attach_grad()
with autograd.record():
    y = npx.relu(x)
d2l.plot(x, y, 'x', 'relu(x)', figsize=(5, 2.5))
```

```{.python .input}
%%tab pytorch
x = torch.arange(-8.0, 8.0, 0.1, requires_grad=True)
y = torch.relu(x)
d2l.plot(x.detach(), y.detach(), 'x', 'relu(x)', figsize=(5, 2.5))
```

```{.python .input}
%%tab tensorflow
x = tf.Variable(tf.range(-8.0, 8.0, 0.1), dtype=tf.float32)
y = tf.nn.relu(x)
d2l.plot(x.numpy(), y.numpy(), 'x', 'relu(x)', figsize=(5, 2.5))
```

```{.python .input}
%%tab jax
x = jnp.arange(-8.0, 8.0, 0.1)
y = jax.nn.relu(x)
d2l.plot(x, y, 'x', 'relu(x)', figsize=(5, 2.5))
```

入力が負のとき、
ReLU 関数の導関数は 0 であり、
入力が正のとき、
ReLU 関数の導関数は 1 である。
ReLU 関数は、入力がちょうど 0 のときには微分可能でないことに注意されたい。
この場合には左側導関数を採用し、
入力が 0 のときの導関数を 0 とする。
これは実用上ほとんど問題にならない。
実際、入力がちょうど 0 になることは通常ほとんどなく、
数学的には非微分可能な点は測度 0 の集合にすぎない。
微妙な境界条件が重要になるなら、
工学ではなく純粋数学をしているのだ、という古い格言がある。
ここでもその感覚は当てはまるかもしれない。
少なくとも、制約付き最適化を扱っていないことは関係しているだろう :cite:`Mangasarian.1965,Rockafellar.1970`。
以下に ReLU 関数の導関数をプロットする。

```{.python .input}
%%tab mxnet
y.backward()
d2l.plot(x, x.grad, 'x', 'grad of relu', figsize=(5, 2.5))
```

```{.python .input}
%%tab pytorch
y.backward(torch.ones_like(x), retain_graph=True)
d2l.plot(x.detach(), x.grad, 'x', 'grad of relu', figsize=(5, 2.5))
```

```{.python .input}
%%tab tensorflow
with tf.GradientTape() as t:
    y = tf.nn.relu(x)
d2l.plot(x.numpy(), t.gradient(y, x).numpy(), 'x', 'grad of relu',
         figsize=(5, 2.5))
```

```{.python .input}
%%tab jax
grad_relu = vmap(grad(jax.nn.relu))
d2l.plot(x, grad_relu(x), 'x', 'grad of relu', figsize=(5, 2.5))
```

ReLU が広く使われる理由は、
その導関数が非常に扱いやすいからである。
すなわち、0 になるか、そのまま通すかのどちらかである。
これにより最適化が容易になり、
以前のニューラルネットワークを悩ませていた
勾配消失問題を軽減した（これについては後で詳しく述べる）。

ReLU 関数には多くの変種があり、
たとえば *parametrized ReLU*（*pReLU*）関数 :cite:`He.Zhang.Ren.ea.2015` がある。
この変種では ReLU に線形項を加えるため、
引数が負でも一部の情報が通過する。

$$\operatorname{pReLU}(x) = \max(0, x) + \alpha \min(0, x).$$

### シグモイド関数

[**シグモイド関数は、値が $\mathbb{R}$ にある入力を**]
[**区間 (0, 1) 上の出力へ変換する。**]
そのため、シグモイドはしばしば *圧縮関数* と呼ばれる。
すなわち、$(-\infty, \infty)$ の任意の入力を
$(0, 1)$ の範囲の値へ押し込める。

$$\operatorname{sigmoid}(x) = \frac{1}{1 + \exp(-x)}.$$

初期のニューラルネットワークでは、研究者たちは
生物学的ニューロンが *発火する* か *発火しない* かをモデル化することに関心をもっていた。
そのため、この分野の先駆者たち、すなわち人工ニューロンの発明者である
McCulloch と Pitts にまでさかのぼる研究では、
しきい値ユニットに注目していた :cite:`McCulloch.Pitts.1943`。
しきい値型活性化は、入力があるしきい値を下回ると 0、
しきい値を超えると 1 をとる。

関心が勾配ベースの学習へ移ると、
シグモイド関数は自然な選択となった。
なぜなら、しきい値ユニットの滑らかで微分可能な近似だからである。
シグモイドは、二値分類問題で出力を確率として解釈したいときに、
出力ユニットの活性化関数として今でも広く用いられている。
また、シグモイドはソフトマックスの特殊な場合とみなせる。
しかし、隠れ層における多くの用途では、
シグモイドはより単純で学習しやすい ReLU にほぼ置き換えられた。
大きな理由は、
シグモイドが最適化を難しくするからである :cite:`LeCun.Bottou.Orr.ea.1998`。
大きな正の引数でも負の引数でも勾配が消えてしまうため、
抜け出しにくい平坦な領域が生じることがある。
それでもシグモイドは重要である。
後の章（たとえば :numref:`sec_lstm`）で扱う再帰ニューラルネットワークでは、
時間方向の情報の流れを制御するために
シグモイドユニットを利用する構造を説明する。

以下にシグモイド関数をプロットする。
入力が 0 に近いとき、
シグモイド関数は線形変換に近づくことに注意されたい。

```{.python .input}
%%tab mxnet
with autograd.record():
    y = npx.sigmoid(x)
d2l.plot(x, y, 'x', 'sigmoid(x)', figsize=(5, 2.5))
```

```{.python .input}
%%tab pytorch
y = torch.sigmoid(x)
d2l.plot(x.detach(), y.detach(), 'x', 'sigmoid(x)', figsize=(5, 2.5))
```

```{.python .input}
%%tab tensorflow
y = tf.nn.sigmoid(x)
d2l.plot(x.numpy(), y.numpy(), 'x', 'sigmoid(x)', figsize=(5, 2.5))
```

```{.python .input}
%%tab jax
y = jax.nn.sigmoid(x)
d2l.plot(x, y, 'x', 'sigmoid(x)', figsize=(5, 2.5))
```

シグモイド関数の導関数は次式で与えられる。

$$\frac{d}{dx} \operatorname{sigmoid}(x) = \frac{\exp(-x)}{(1 + \exp(-x))^2} = \operatorname{sigmoid}(x)\left(1-\operatorname{sigmoid}(x)\right).$$


シグモイド関数の導関数を以下に示す。
入力が 0 のとき、
シグモイド関数の導関数は最大値 0.25 に達する。
入力が 0 からどちらの方向へ離れても、
導関数は 0 に近づく。

```{.python .input}
%%tab mxnet
y.backward()
d2l.plot(x, x.grad, 'x', 'grad of sigmoid', figsize=(5, 2.5))
```

```{.python .input}
%%tab pytorch
# 前回の勾配をクリアする
x.grad.data.zero_()
y.backward(torch.ones_like(x),retain_graph=True)
d2l.plot(x.detach(), x.grad, 'x', 'grad of sigmoid', figsize=(5, 2.5))
```

```{.python .input}
%%tab tensorflow
with tf.GradientTape() as t:
    y = tf.nn.sigmoid(x)
d2l.plot(x.numpy(), t.gradient(y, x).numpy(), 'x', 'grad of sigmoid',
         figsize=(5, 2.5))
```

```{.python .input}
%%tab jax
grad_sigmoid = vmap(grad(jax.nn.sigmoid))
d2l.plot(x, grad_sigmoid(x), 'x', 'grad of sigmoid', figsize=(5, 2.5))
```

### tanh 関数
:label:`subsec_tanh`

シグモイド関数と同様に、[**tanh（双曲線正接）
関数も入力を圧縮し**]、
それを [**-1 と 1 の間**] の区間上の値へ変換する。

$$\operatorname{tanh}(x) = \frac{1 - \exp(-2x)}{1 + \exp(-2x)}.$$

以下に tanh 関数をプロットする。入力が 0 に近づくと、tanh 関数は線形変換に近づくことに注意されたい。関数の形はシグモイド関数に似ているが、tanh 関数は座標系の原点に関して点対称である :cite:`Kalman.Kwasny.1992`。

```{.python .input}
%%tab mxnet
with autograd.record():
    y = np.tanh(x)
d2l.plot(x, y, 'x', 'tanh(x)', figsize=(5, 2.5))
```

```{.python .input}
%%tab pytorch
y = torch.tanh(x)
d2l.plot(x.detach(), y.detach(), 'x', 'tanh(x)', figsize=(5, 2.5))
```

```{.python .input}
%%tab tensorflow
y = tf.nn.tanh(x)
d2l.plot(x.numpy(), y.numpy(), 'x', 'tanh(x)', figsize=(5, 2.5))
```

```{.python .input}
%%tab jax
y = jax.nn.tanh(x)
d2l.plot(x, y, 'x', 'tanh(x)', figsize=(5, 2.5))
```

tanh 関数の導関数は次のとおりである。

$$\frac{d}{dx} \operatorname{tanh}(x) = 1 - \operatorname{tanh}^2(x).$$

以下に示す。
入力が 0 に近づくと、
tanh 関数の導関数は最大値 1 に近づく。
そしてシグモイド関数で見たように、
入力が 0 からどちらの方向へ離れても、
tanh 関数の導関数は 0 に近づく。

```{.python .input}
%%tab mxnet
y.backward()
d2l.plot(x, x.grad, 'x', 'grad of tanh', figsize=(5, 2.5))
```

```{.python .input}
%%tab pytorch
# 前回の勾配をクリアする
x.grad.data.zero_()
y.backward(torch.ones_like(x),retain_graph=True)
d2l.plot(x.detach(), x.grad, 'x', 'grad of tanh', figsize=(5, 2.5))
```

```{.python .input}
%%tab tensorflow
with tf.GradientTape() as t:
    y = tf.nn.tanh(x)
d2l.plot(x.numpy(), t.gradient(y, x).numpy(), 'x', 'grad of tanh',
         figsize=(5, 2.5))
```

```{.python .input}
%%tab jax
grad_tanh = vmap(grad(jax.nn.tanh))
d2l.plot(x, grad_tanh(x), 'x', 'grad of tanh', figsize=(5, 2.5))
```

## 要約と考察

これで、非線形性を組み込むことで
表現力の高い多層ニューラルネットワークを構築する方法が分かった。
見方を変えれば、ここまでの知識だけでも
1990 年頃の実務家と同程度の道具立てを使えることになる。
ある意味では、当時の誰よりも有利である。
強力なオープンソースの深層学習フレームワークを利用して、
わずかなコードで素早くモデルを構築できるからである。
かつては、これらのネットワークを学習させるために、
研究者が C、Fortran、あるいは（LeNet の場合には）Lisp で
層や導関数を明示的に実装しなければならなかった。

副次的な利点として、ReLU はシグモイドや tanh 関数よりも
最適化にかなり適している。
これは、過去 10 年における深層学習の復活を支えた
重要な革新の 1 つであったと言える。
ただし、活性化関数の研究が止まったわけではない。
たとえば、
GELU（Gaussian error linear unit）
活性化関数 $x \Phi(x)$ は :citet:`Hendrycks.Gimpel.2016` によるもので（$\Phi(x)$
は標準ガウス累積分布関数）、
また
Swish 活性化関数
$\sigma(x) = x \operatorname{sigmoid}(\beta x)$ は :citet:`Ramachandran.Zoph.Le.2017` により提案され、
多くの場合により高い精度をもたらす。

## 演習

1. 非線形性 $\sigma$ をもたない、すなわち *線形* な深層ネットワークでは、層を追加してもネットワークの表現力が決して増えないことを示しなさい。さらに、表現力が実際に低下する例を挙げなさい。
1. pReLU 活性化関数の導関数を求めなさい。
1. Swish 活性化関数 $x \operatorname{sigmoid}(\beta x)$ の導関数を求めなさい。
1. ReLU（または pReLU）のみを用いる MLP が、連続な区分線形関数を構成することを示しなさい。
1. シグモイドと tanh は非常によく似ている。
    1. $\operatorname{tanh}(x) + 1 = 2 \operatorname{sigmoid}(2x)$ を示しなさい。
    1. 両方の非線形性でパラメータ化される関数クラスが同一であることを証明しなさい。ヒント: アフィン層にもバイアス項がある。
1. バッチ正規化 :cite:`Ioffe.Szegedy.2015` のように、1 つのミニバッチごとに作用する非線形性があると仮定する。どのような問題が生じると予想されるか。
1. シグモイド活性化関数で勾配が消失する例を示しなさい。