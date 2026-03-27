{.python .input}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow', 'jax'])
```

# 多層パーセプトロン
:label:`sec_mlp`

:numref:`sec_softmax` では、ソフトマックス回帰を導入し、
アルゴリズムをスクラッチから実装し（:numref:`sec_softmax_scratch`）、
高水準 API を用いて実装しました（:numref:`sec_softmax_concise`）。
これにより、低解像度画像から 10 種類の衣類を認識できる分類器を学習できた。
その過程で、データを整形する方法、
出力を妥当な確率分布に変換する方法、
適切な損失関数を適用する方法、
そしてモデルのパラメータに関してそれを最小化する方法を学んだ。
単純な線形モデルの文脈でこれらの仕組みを習得した今、
いよいよ深層ニューラルネットワークの探究を始める可能である。
本書が主として扱うのは、より豊かなクラスに属するこれらのモデルである。

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
まず、 :numref:`fig_softmaxreg` に示した
ソフトマックス回帰の例に対応するモデル構造を思い出ししよう。
このモデルは、単一のアフィン変換の後にソフトマックス演算を適用することで、
入力を直接出力へ写像する。
もしラベルが本当に単純なアフィン変換によって
入力データと結び付いているのであれば、
この方法で十分である。
しかし、線形性（アフィン変換における）は *強い* 仮定である。

### 線形モデルの限界

たとえば、線形性は *単調性* という、より *弱い* 仮定を含意する。
すなわち、特徴量が増加したとき、
モデルの出力は必ず増加するか（対応する重みが正の場合）、
あるいは必ず減少するか（対応する重みが負の場合）のどちらかでなければなりない。
これは場合によっては理にかなっている。
たとえば、ある個人がローンを返済するかどうかを予測したいとししよう。
他の条件が同じなら、収入が高い申請者のほうが
低い申請者よりも返済する可能性が常に高い、
と仮定するのは妥当かもしれない。
この関係は単調ではありますが、返済確率と線形に結び付いているとは
おそらく言えない。
収入が \$0 から \$50,000 に増えることは、
\$1 million から \$1.05 million に増えることよりも、
返済可能性を大きく高めるだろう。
この問題への一つの対処法は、ロジスティック写像（したがって結果の確率の対数）を用いて、
線形性がよりもっともらしくなるように出力を後処理することである。

単調性すら破る例も簡単に思いつくる。
たとえば、体温の関数として健康状態を予測したいとししよう。
37°C（98.6°F）を超える正常体温の個人では、
体温が高いほどリスクが高いことを示する。
しかし、体温が 37°C を下回ると、
今度は体温が低いほどリスクが高いのです！
ここでも、37°C からの距離を特徴量として使うなど、
何らかの巧妙な前処理で問題を解決できるかもしれない。

では、猫と犬の画像分類はどうでしょうか？
位置 (13, 17) にある画素の強度を増やすと、
その画像が犬である確率は常に増える（あるいは常に減る）べきでしょうか？
線形モデルに依存することは、
猫と犬を区別するために必要なのは個々の画素の明るさを評価することだけだ、
という暗黙の仮定に対応する。
画像を反転してもカテゴリが保存されるような世界では、
この方法は失敗する運命にある。

それでも、ここでの線形性は前の例と比べて一見ばかげているにもかかわらず、
単純な前処理の修正で問題を解決できるかはそれほど明らかではありない。
というのも、任意の画素の意味は
その文脈（周囲の画素の値）に複雑に依存するからである。
特徴量間の関連する相互作用を考慮した表現が存在し、
その上で線形モデルが適切になる可能性はありますが、
それを手作業で計算する方法は分かりない。
深層ニューラルネットワークでは、観測データを用いて、
隠れ層による表現と、その表現に作用する線形予測器の両方を
同時に学習した。

この非線形性の問題は、少なくとも 1 世紀にわたって研究されてきました :cite:`Fisher.1928`。
たとえば、決定木は最も基本的な形では、
一連の二値決定を用いてクラス所属を判定します :cite:`quinlan2014c4`。
同様に、カーネル法は非線形依存をモデル化するために
何十年も使われてきました :cite:`Aronszajn.1950`。
これは非パラメトリックなスプラインモデル :cite:`Wahba.1990` や
カーネル法 :cite:`Scholkopf.Smola.2002` にもつながっている。
また、脳はこれをごく自然に解いている。
結局のところ、ニューロンは他のニューロンへ入力し、
そのニューロンはさらに別のニューロンへ入力します :cite:`Cajal.Azoulay.1894`。
その結果、比較的単純な変換が連なった構造になる。

### 隠れ層の導入

1 つ以上の隠れ層を組み込むことで、
線形モデルの限界を克服できる。
最も簡単な方法は、多数の全結合層を
順に積み重ねることである。
各層はその上の層へ入力を渡し、
最終的に出力を生成する。
最初の $L-1$ 層を表現として、
最後の層を線形予測器として考える可能である。
この構造は一般に *多層パーセプトロン* と呼ばれ、
しばしば *MLP* と略されます（:numref:`fig_mlp`）。

![5 個の隠れユニットを持つ隠れ層を備えた MLP。](../img/mlp.svg)
:label:`fig_mlp`

この MLP は 4 つの入力、3 つの出力を持ち、
隠れ層には 5 個の隠れユニットがある。
入力層には計算が含まれないため、
このネットワークで出力を生成するには
隠れ層と出力層の両方の計算を実装する必要がある。
したがって、この MLP の層数は 2 である。
なお、どちらの層も全結合である。
すべての入力は隠れ層のすべてのニューロンに影響し、
それぞれのニューロンはさらに出力層のすべてのニューロンに影響する。
とはいえ、まだ終わりではありない。

### 線形から非線形へ

これまでと同様に、$\mathbf{X} \in \mathbb{R}^{n \times d}$ を、
各データ例が $d$ 個の入力（特徴量）を持つ $n$ 個のデータ例からなるミニバッチを表す行列とする。
隠れ層が $h$ 個の隠れユニットを持つ 1 隠れ層 MLP について、
$\mathbf{H} \in \mathbb{R}^{n \times h}$ を隠れ層の出力、
すなわち *隠れ表現* とする。
隠れ層と出力層はいずれも全結合なので、
隠れ層の重み $\mathbf{W}^{(1)} \in \mathbb{R}^{d \times h}$ とバイアス $\mathbf{b}^{(1)} \in \mathbb{R}^{1 \times h}$、
出力層の重み $\mathbf{W}^{(2)} \in \mathbb{R}^{h \times q}$ とバイアス $\mathbf{b}^{(2)} \in \mathbb{R}^{1 \times q}$ を持つ。
これにより、1 隠れ層 MLP の出力 $\mathbf{O} \in \mathbb{R}^{n \times q}$ を次のように計算できる。

$$
\begin{aligned}
    \mathbf{H} & = \mathbf{X} \mathbf{W}^{(1)} + \mathbf{b}^{(1)}, \\
    \mathbf{O} & = \mathbf{H}\mathbf{W}^{(2)} + \mathbf{b}^{(2)}.
\end{aligned}
$$

隠れ層を追加すると、
モデルは新たに追加されたパラメータ群を追跡し更新する必要がある。
では、その代わりに何を得たのでしょうか？
驚くかもしれませんが、上で定義したモデルでは、*苦労のわりに何も得ていません*！
理由は明白である。
上の隠れユニットは入力のアフィン関数で与えられ、
出力（ソフトマックス適用前）は隠れユニットのアフィン関数にすぎない。
アフィン関数のアフィン関数は、やはりアフィン関数である。
さらに、線形モデルはすでに任意のアフィン関数を表現できた。

これを形式的に見るには、上の定義から隠れ層を消去すればよく、
パラメータ $\mathbf{W} = \mathbf{W}^{(1)}\mathbf{W}^{(2)}$ と $\mathbf{b} = \mathbf{b}^{(1)} \mathbf{W}^{(2)} + \mathbf{b}^{(2)}$ を持つ等価な単層モデルが得られる。

$$
\mathbf{O} = (\mathbf{X} \mathbf{W}^{(1)} + \mathbf{b}^{(1)})\mathbf{W}^{(2)} + \mathbf{b}^{(2)} = \mathbf{X} \mathbf{W}^{(1)}\mathbf{W}^{(2)} + \mathbf{b}^{(1)} \mathbf{W}^{(2)} + \mathbf{b}^{(2)} = \mathbf{X} \mathbf{W} + \mathbf{b}.
$$

多層構造の潜在能力を引き出すには、
もう 1 つ重要な要素が必要である。
それは、アフィン変換の後に各隠れユニットへ適用する
非線形 *活性化関数* $\sigma$ である。
たとえば、よく使われる選択肢は ReLU（rectified linear unit）活性化関数 :cite:`Nair.Hinton.2010`
$\sigma(x) = \mathrm{max}(0, x)$ で、引数に要素ごとに作用する。
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
記法を少し乱用して、非線形性 $\sigma$ は入力に行ごとに作用する、
すなわち 1 データ例ずつ適用されると定義する。
:numref:`subsec_softmax_vectorization` で行ごとの演算を表す際に
ソフトマックスでも同じ記法を使ったことに注意されたい。
私たちが使う活性化関数は、行ごとだけでなく
要素ごとに作用することが非常に多いである。
つまり、層の線形部分を計算した後は、
他の隠れユニットの値を見なくても
各活性化を計算できる。

より一般的な MLP を構築するには、
このような隠れ層をさらに積み重ねればよいである。
たとえば、$\mathbf{H}^{(1)} = \sigma_1(\mathbf{X} \mathbf{W}^{(1)} + \mathbf{b}^{(1)})$、
$\mathbf{H}^{(2)} = \sigma_2(\mathbf{H}^{(1)} \mathbf{W}^{(2)} + \mathbf{b}^{(2)})$
のように、層を次々と重ねることで、
ますます表現力の高いモデルが得られる。

### 万能近似器

脳が非常に高度な統計解析を行えることは分かっている。
したがって、深いネットワークがどれほど強力になり得るのかを問う価値がある。
この問いには複数回答えが与えられており、たとえば MLP の文脈では :citet:`Cybenko.1989`、
再生核ヒルベルト空間の文脈では :citet:`micchelli1984interpolation` があり、
これは 1 つの隠れ層を持つ RBF（radial basis function）ネットワークと見なせる形である。
これらの結果（および関連する結果）は、
たとえ 1 隠れ層ネットワークであっても、
十分な数のノード（おそらく途方もなく多く）と
適切な重みがあれば、
任意の関数をモデル化できることを示唆している。
ただし、その関数を実際に学習することが難しいのである。
ニューラルネットワークは、C 言語のようなものだと考えるとよいだろう。
この言語は、他の現代的な言語と同様に、
計算可能なプログラムなら何でも表現できる。
しかし、仕様を満たすプログラムを実際に作ることが難しいのである。

さらに、1 隠れ層ネットワークが
*任意の関数を学習できる* からといって、
すべての問題をそれ 1 つで解くべきだという意味ではありない。
実際、この場合はカーネル法のほうがはるかに有効である。
なぜなら、無限次元空間であっても問題を
*厳密に* 解けるからです :cite:`Kimeldorf.Wahba.1971,Scholkopf.Herbrich.Smola.2001`。
実際、より深い（より広いのではなく）ネットワークを使うことで、
多くの関数をはるかにコンパクトに近似できます :cite:`Simonyan.Zisserman.2014`。
より厳密な議論は後続の章で扱いる。

## 活性化関数
:label:`subsec_activation-functions`

活性化関数は、重み付き和を計算し、さらにバイアスを加えることで、
ニューロンを活性化するかどうかを決定する。
それらは入力信号を出力へ変換する微分可能な演算子であり、
その多くは非線形性を付加する。
活性化関数は深層学習の基礎なので、
[**代表的なものを簡単に見ておきましょう**]。

### ReLU 関数

実装が簡単で、さまざまな予測タスクで良好な性能を示すことから、
最も人気のある選択肢は *rectified linear unit*（*ReLU*）です :cite:`Nair.Hinton.2010`。
[**ReLU は非常に単純な非線形変換を提供します**]。
要素 $x$ に対して、この関数はその要素と 0 の最大値として定義される。

$$\operatorname{ReLU}(x) = \max(x, 0).$$

直感的には、ReLU 関数は正の要素だけを残し、
対応する活性化を 0 にすることで負の要素をすべて捨てる。
直感を得るために、この関数をプロットしてみしよう。
見てのとおり、活性化関数は区分線形である。

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
この場合は左側微分を採用し、
入力が 0 のときの導関数は 0 とする。
これは、入力が実際には 0 にならないかもしれないので問題ありない
（数学者は、これは測度 0 の集合上で非微分可能だと言うだろう）。
微妙な境界条件が重要なら、
私たちはおそらく工学ではなく（本当の）数学をしているのだ、という古い格言がある。
ここでもその常識が当てはまるかもしれませんし、
少なくとも制約付き最適化を行っていないという事実が関係しているのかもしれません :cite:`Mangasarian.1965,Rockafellar.1970`。
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

ReLU を使う理由は、
その導関数が非常に扱いやすいからである。
つまり、0 になるか、引数をそのまま通すかのどちらかである。
これにより最適化が扱いやすくなり、
以前のニューラルネットワークを悩ませていた
勾配消失問題を軽減した（これについては後で詳しく述べる）。

ReLU 関数には多くの変種があり、
たとえば *parametrized ReLU*（*pReLU*）関数 :cite:`He.Zhang.Ren.ea.2015` がある。
この変種は ReLU に線形項を加えるため、
引数が負でも一部の情報が通過する。

$$\operatorname{pReLU}(x) = \max(0, x) + \alpha \min(0, x).$$

### シグモイド関数

[**シグモイド関数は、値が $\mathbb{R}$ の範囲にある入力を**]
[**区間 (0, 1) 上の出力へ変換する。**]
そのため、シグモイドはしばしば *圧縮関数* と呼ばれる。
つまり、(-inf, inf) の範囲にある任意の入力を
(0, 1) の範囲のある値へ押し込めるのである。

$$\operatorname{sigmoid}(x) = \frac{1}{1 + \exp(-x)}.$$

初期のニューラルネットワークでは、科学者たちは
生物学的ニューロンが *発火する* か *発火しない* かをモデル化することに関心を持っていた。
そのため、この分野の先駆者たち、人工ニューロンの発明者である
McCulloch と Pitts にまでさかのぼる研究では、
しきい値ユニットに注目していました :cite:`McCulloch.Pitts.1943`。
しきい値型活性化は、入力があるしきい値を下回ると 0、
しきい値を超えると 1 を取る。

勾配ベース学習へ関心が移ると、
シグモイド関数は自然な選択であった。
なぜなら、それはしきい値ユニットの滑らかで微分可能な近似だからである。
シグモイドは、二値分類問題で出力を確率として解釈したいときに、
出力ユニットの活性化関数として今でも広く使われている。
シグモイドはソフトマックスの特殊な場合と考える可能である。
しかし、隠れ層での多くの用途では、
シグモイドはより単純で学習しやすい ReLU にほぼ置き換えられた。
その大きな理由は、
シグモイドが最適化において難しさをもたらすからです :cite:`LeCun.Bottou.Orr.ea.1998`。
大きな正の引数でも負の引数でも勾配が消えてしまうためである。
その結果、抜け出しにくい平坦領域が生じることがある。
それでもシグモイドは重要である。
後の章（たとえば :numref:`sec_lstm`）で扱う再帰ニューラルネットワークでは、
時間をまたいだ情報の流れを制御するために
シグモイドユニットを活用する構造を説明する。

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

シグモイド関数の導関数は次の式で与えられる。

$$\frac{d}{dx} \operatorname{sigmoid}(x) = \frac{\exp(-x)}{(1 + \exp(-x))^2} = \operatorname{sigmoid}(x)\left(1-\operatorname{sigmoid}(x)\right).$$


シグモイド関数の導関数を以下に示する。
入力が 0 のとき、
シグモイド関数の導関数は最大値 0.25 に達する。
入力が 0 からどちらの方向へ離れても、
導関数は 0 に近づくる。

```{.python .input}
%%tab mxnet
y.backward()
d2l.plot(x, x.grad, 'x', 'grad of sigmoid', figsize=(5, 2.5))
```

```{.python .input}
%%tab pytorch
# Clear out previous gradients
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
それらを [**-1 と 1 の間**] の区間上の要素へ変換する。

$$\operatorname{tanh}(x) = \frac{1 - \exp(-2x)}{1 + \exp(-2x)}.$$

以下に tanh 関数をプロットする。入力が 0 に近づくと、tanh 関数は線形変換に近づくことに注意されたい。関数の形はシグモイド関数に似ていますが、tanh 関数は座標系の原点に関して点対称です :cite:`Kalman.Kwasny.1992`。

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

以下に示する。
入力が 0 に近づくと、
tanh 関数の導関数は最大値 1 に近づくる。
そしてシグモイド関数で見たように、
入力が 0 からどちらの方向へ離れても、
tanh 関数の導関数は 0 に近づくる。

```{.python .input}
%%tab mxnet
y.backward()
d2l.plot(x, x.grad, 'x', 'grad of tanh', figsize=(5, 2.5))
```

```{.python .input}
%%tab pytorch
# Clear out previous gradients
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

これで、非線形性を組み込んで
表現力の高い多層ニューラルネットワーク構造を構築する方法が分かった。
補足すると、ここまでの知識だけでも
1990 年頃の実務家と同程度の道具立てを使いこなせることになる。
ある意味では、当時の誰よりも有利である。
強力なオープンソースの深層学習フレームワークを活用して、
わずかなコードで素早くモデルを構築できるからである。
以前は、これらのネットワークを学習させるには、
研究者が C、Fortran、あるいは（LeNet の場合は）Lisp で
層や導関数を明示的に実装する必要があった。

副次的な利点として、ReLU はシグモイドや tanh 関数よりも
最適化にかなり適している。
これは、過去 10 年の深層学習の復活を支えた
重要な革新の 1 つだったと言えるだろう。
ただし、活性化関数の研究が止まったわけではありない。
たとえば、
GELU（Gaussian error linear unit）
活性化関数 $x \Phi(x)$ は :citet:`Hendrycks.Gimpel.2016` によるもので（$\Phi(x)$
は標準ガウス累積分布関数）、
また
Swish 活性化関数
$\sigma(x) = x \operatorname{sigmoid}(\beta x)$ は :citet:`Ramachandran.Zoph.Le.2017` により提案され、
多くの場合でより高い精度をもたらする。

## 演習

1. 非線形性 $\sigma$ を持たない、すなわち *線形* な深層ネットワークに層を追加しても、ネットワークの表現力は決して増えないことを示しなさい。逆に、それを実際に低下させる例を挙げなさい。
1. pReLU 活性化関数の導関数を求めなさい。
1. Swish 活性化関数 $x \operatorname{sigmoid}(\beta x)$ の導関数を求めなさい。
1. ReLU（または pReLU）だけを用いる MLP が、連続な区分線形関数を構成することを示しなさい。
1. シグモイドと tanh は非常によく似ている。
    1. $\operatorname{tanh}(x) + 1 = 2 \operatorname{sigmoid}(2x)$ を示しなさい。
    1. 両方の非線形性でパラメータ化される関数クラスが同一であることを証明しなさい。ヒント: アフィン層にもバイアス項がある。
1. バッチ正規化 :cite:`Ioffe.Szegedy.2015` のように、1 つのミニバッチごとに作用する非線形性があると仮定する。どのような問題が生じると予想されますか？
1. シグモイド活性化関数で勾配が消失する例を示しなさい。

