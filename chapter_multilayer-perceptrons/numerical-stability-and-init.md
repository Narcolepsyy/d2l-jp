{.python .input}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow', 'jax'])
```

# 数値安定性と初期化
:label:`sec_numerical_stability`


これまでに実装してきたすべてのモデルでは、
あらかじめ指定された分布に従って
パラメータを初期化する必要がありました。
これまでは初期化手法を当然のものとして扱い、
その選択がどのように行われるかという詳細には触れてきませんでした。
そのため、こうした選択はそれほど重要ではない
という印象を持ったかもしれません。
しかし実際には、初期化手法の選択は
ニューラルネットワークの学習において重要な役割を果たし、
数値安定性を保つうえで決定的になることもあります。
さらに、これらの選択は
非線形活性化関数の選択と興味深い形で結びついています。
どの関数を選ぶか、そしてパラメータをどう初期化するかによって、
最適化アルゴリズムがどれだけ速く収束するかが決まります。
ここでの選択を誤ると、学習中に
勾配爆発や勾配消失に遭遇することがあります。
この節では、これらの話題をより詳しく掘り下げ、
深層学習のキャリアを通じて役立つ
いくつかの有用な経験則を紹介します。

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

## 勾配消失と勾配爆発

入力 $\mathbf{x}$ と出力 $\mathbf{o}$ を持つ、
$L$ 層の深いネットワークを考えます。
各層 $l$ は重み $\mathbf{W}^{(l)}$ によってパラメータ化された変換 $f_l$
で定義され、その隠れ層出力を $\mathbf{h}^{(l)}$ とします（$\mathbf{h}^{(0)} = \mathbf{x}$ とする）。
このとき、ネットワークは次のように表せます。

$$\mathbf{h}^{(l)} = f_l (\mathbf{h}^{(l-1)}) \textrm{ and thus } \mathbf{o} = f_L \circ \cdots \circ f_1(\mathbf{x}).$$

すべての隠れ層出力と入力がベクトルであるとき、
$\mathbf{o}$ の任意のパラメータ集合 $\mathbf{W}^{(l)}$ に関する勾配は次のように書けます。

$$\partial_{\mathbf{W}^{(l)}} \mathbf{o} = \underbrace{\partial_{\mathbf{h}^{(L-1)}} \mathbf{h}^{(L)}}_{ \mathbf{M}^{(L)} \stackrel{\textrm{def}}{=}} \cdots \underbrace{\partial_{\mathbf{h}^{(l)}} \mathbf{h}^{(l+1)}}_{ \mathbf{M}^{(l+1)} \stackrel{\textrm{def}}{=}} \underbrace{\partial_{\mathbf{W}^{(l)}} \mathbf{h}^{(l)}}_{ \mathbf{v}^{(l)} \stackrel{\textrm{def}}{=}}.$$

言い換えると、この勾配は
$L-l$ 個の行列
$\mathbf{M}^{(L)} \cdots \mathbf{M}^{(l+1)}$
と勾配ベクトル $\mathbf{v}^{(l)}$ の積です。
したがって、これは
あまりにも多くの確率を掛け合わせたときにしばしば現れる
数値アンダーフローの問題と同じような影響を受けます。
確率を扱うときの一般的な工夫は、
対数空間に移ること、すなわち
数値表現の仮数部から指数部へと
負荷を移すことです。
残念ながら、上の問題はそれより深刻です。
初期状態では、行列 $\mathbf{M}^{(l)}$ はさまざまな固有値を持ちえます。
それらは小さいことも大きいこともあり、
その積は *非常に大きく* も *非常に小さく* もなりえます。

不安定な勾配がもたらす危険は、
数値表現の問題にとどまりません。
予測不能な大きさの勾配は、
最適化アルゴリズムの安定性も脅かします。
パラメータ更新が
(i) 過度に大きくなってモデルを壊してしまう
（*勾配爆発* 問題）か、
あるいは (ii) 過度に小さくなって
（*勾配消失* 問題）、
更新ごとにパラメータがほとんど動かず
学習が不可能になるかもしれません。


### (**勾配消失**)

勾配消失問題のよくある原因の一つは、
各層の線形演算の後に付加される活性化関数 $\sigma$
の選択です。
歴史的には、シグモイド関数
$1/(1 + \exp(-x))$（:numref:`sec_mlp` で導入）
は、しきい値関数に似ているため人気がありました。
初期の人工ニューラルネットワークは
生物学的ニューラルネットワークに着想を得ていたため、
生体ニューロンのように *完全に* 発火するか *まったく* 発火しないかの
どちらかであるニューロンの考え方は魅力的に見えました。
シグモイドを詳しく見て、
なぜ勾配消失を引き起こしうるのかを確認しましょう。

```{.python .input}
%%tab mxnet
x = np.arange(-8.0, 8.0, 0.1)
x.attach_grad()
with autograd.record():
    y = npx.sigmoid(x)
y.backward()

d2l.plot(x, [y, x.grad], legend=['sigmoid', 'gradient'], figsize=(4.5, 2.5))
```

```{.python .input}
%%tab pytorch
x = torch.arange(-8.0, 8.0, 0.1, requires_grad=True)
y = torch.sigmoid(x)
y.backward(torch.ones_like(x))

d2l.plot(x.detach().numpy(), [y.detach().numpy(), x.grad.numpy()],
         legend=['sigmoid', 'gradient'], figsize=(4.5, 2.5))
```

```{.python .input}
%%tab tensorflow
x = tf.Variable(tf.range(-8.0, 8.0, 0.1))
with tf.GradientTape() as t:
    y = tf.nn.sigmoid(x)
d2l.plot(x.numpy(), [y.numpy(), t.gradient(y, x).numpy()],
         legend=['sigmoid', 'gradient'], figsize=(4.5, 2.5))
```

```{.python .input}
%%tab jax
x = jnp.arange(-8.0, 8.0, 0.1)
y = jax.nn.sigmoid(x)
grad_sigmoid = vmap(grad(jax.nn.sigmoid))
d2l.plot(x, [y, grad_sigmoid(x)],
         legend=['sigmoid', 'gradient'], figsize=(4.5, 2.5))
```

ご覧のとおり、(**シグモイドの勾配は、入力が大きいときも小さいときも消失します**）。
さらに、多くの層を逆伝播するとき、
多くのシグモイドへの入力がゼロに近い
ちょうどよい範囲にない限り、
全体の積の勾配は消失してしまう可能性があります。
ネットワークが多層になるほど、
注意しないと、どこかの層で勾配が
途切れてしまうでしょう。
実際、この問題はかつて深層ネットワークの学習を悩ませていました。
その結果、より安定している
（ただし生物学的にはあまりもっともらしくない）
ReLU が、実務家にとっての標準的な選択肢として
広く使われるようになりました。


### [**勾配爆発**]

逆の問題である勾配爆発も、
同様に厄介です。
これを少し分かりやすく示すために、
100 個のガウス乱数行列を生成し、
ある初期行列と掛け合わせてみます。
私たちが選んだスケール
（分散 $\sigma^2=1$ の選択）では、
行列積は爆発します。
これが深いネットワークの初期化によって起こると、
勾配降下法の最適化器を収束させる見込みはありません。

```{.python .input}
%%tab mxnet
M = np.random.normal(size=(4, 4))
print('a single matrix', M)
for i in range(100):
    M = np.dot(M, np.random.normal(size=(4, 4)))
print('after multiplying 100 matrices', M)
```

```{.python .input}
%%tab pytorch
M = torch.normal(0, 1, size=(4, 4))
print('a single matrix \n',M)
for i in range(100):
    M = M @ torch.normal(0, 1, size=(4, 4))
print('after multiplying 100 matrices\n', M)
```

```{.python .input}
%%tab tensorflow
M = tf.random.normal((4, 4))
print('a single matrix \n', M)
for i in range(100):
    M = tf.matmul(M, tf.random.normal((4, 4)))
print('after multiplying 100 matrices\n', M.numpy())
```

```{.python .input}
%%tab jax
get_key = lambda: jax.random.PRNGKey(d2l.get_seed())  # Generate PRNG keys
M = jax.random.normal(get_key(), (4, 4))
print('a single matrix \n', M)
for i in range(100):
    M = jnp.matmul(M, jax.random.normal(get_key(), (4, 4)))
print('after multiplying 100 matrices\n', M)
```

### 対称性を破る

ニューラルネットワーク設計における
もう一つの問題は、パラメータ化に内在する対称性です。
1 つの隠れ層と 2 つのユニットを持つ
単純な MLP を考えます。
この場合、最初の層の重み $\mathbf{W}^{(1)}$
を入れ替え、同様に出力層の重みも入れ替えることで、
同じ関数を得ることができます。
第 1 隠れユニットと第 2 隠れユニットを
区別する特別なものはありません。
言い換えると、各層の隠れユニットの間には
置換対称性があります。

これは単なる理論上の厄介ごとではありません。
先ほどの 1 隠れ層・2 ユニットの MLP を考えましょう。
説明のために、
出力層が 2 つの隠れユニットを 1 つの出力ユニットに変換するとします。
もし隠れ層のすべてのパラメータを
$\mathbf{W}^{(1)} = c$（ある定数 $c$）として初期化したら、
何が起こるでしょうか。
この場合、順伝播では
どちらの隠れユニットも同じ入力とパラメータを受け取り、
同じ活性化を生成し、
それが出力ユニットに渡されます。
逆伝播では、
出力ユニットをパラメータ $\mathbf{W}^{(1)}$ で微分すると、
すべての要素が同じ値を取る勾配が得られます。
したがって、勾配ベースの反復（たとえばミニバッチ確率的勾配降下法）を行っても、
$\mathbf{W}^{(1)}$ のすべての要素は
依然として同じ値のままです。
このような反復だけでは
自力で対称性を *破る* ことは決してできず、
ネットワークの表現力を
実現できないままになるかもしれません。
隠れ層は、あたかも
1 つのユニットしか持たないかのように振る舞うでしょう。
ミニバッチ確率的勾配降下法ではこの対称性は破れませんが、
（後で導入する）ドロップアウト正則化なら破ることができる点に注意してください。


## パラメータ初期化

上で述べた問題に対処する、あるいは少なくとも軽減する
一つの方法は、慎重な初期化です。
後で見るように、
最適化時にさらに注意を払い、
適切な正則化を行うことで、安定性をさらに高められます。


### デフォルトの初期化

前の節、たとえば :numref:`sec_linear_concise` では、
重みの値を初期化するために
正規分布を用いました。
初期化方法を指定しない場合、フレームワークは
デフォルトのランダム初期化方法を使います。これは
中程度の問題規模では実際によく機能することが多いです。



### Xavier 初期化
:label:`subsec_xavier`

非線形性 *なし* のある全結合層について、
出力 $o_{i}$ のスケール分布を見てみましょう。
この層に $n_\textrm{in}$ 個の入力 $x_j$
と、それに対応する重み $w_{ij}$ があるとすると、
出力は次のように与えられます。

$$o_{i} = \sum_{j=1}^{n_\textrm{in}} w_{ij} x_j.$$

重み $w_{ij}$ はすべて
同じ分布から独立にサンプルされるとします。
さらに、この分布の平均が 0、分散が $\sigma^2$ であると仮定します。
ここで、分布がガウス分布である必要はなく、
平均と分散が存在すればよいことに注意してください。
今のところ、この層への入力 $x_j$ も
平均 0、分散 $\gamma^2$ を持ち、
$w_{ij}$ と互いに独立で、かつ入力同士も独立であると仮定しましょう。
この場合、$o_i$ の平均は次のように計算できます。

$$
\begin{aligned}
    E[o_i] & = \sum_{j=1}^{n_\textrm{in}} E[w_{ij} x_j] \\&= \sum_{j=1}^{n_\textrm{in}} E[w_{ij}] E[x_j] \\&= 0, \end{aligned}$$

また分散は次のようになります。

$$
\begin{aligned}
    \textrm{Var}[o_i] & = E[o_i^2] - (E[o_i])^2 \\
        & = \sum_{j=1}^{n_\textrm{in}} E[w^2_{ij} x^2_j] - 0 \\
        & = \sum_{j=1}^{n_\textrm{in}} E[w^2_{ij}] E[x^2_j] \\
        & = n_\textrm{in} \sigma^2 \gamma^2.
\end{aligned}
$$

分散を一定に保つ一つの方法は、
$n_\textrm{in} \sigma^2 = 1$ とすることです。
次に逆伝播を考えます。
ここでも同様の問題に直面しますが、
今度は勾配が出力に近い層から伝播してきます。
順伝播の場合と同じ考え方を使うと、
$n_\textrm{out} \sigma^2 = 1$
でない限り、勾配の分散は爆発しうることが分かります。
ここで $n_\textrm{out}$ はこの層の出力数です。
すると、私たちはジレンマに陥ります。
この 2 つの条件を同時に満たすことはできません。
そこで、次を満たすことを目指します。

$$
\begin{aligned}
\frac{1}{2} (n_\textrm{in} + n_\textrm{out}) \sigma^2 = 1 \textrm{ or equivalently }
\sigma = \sqrt{\frac{2}{n_\textrm{in} + n_\textrm{out}}}.
\end{aligned}
$$

これが、現在では標準的で実用上も有益な
*Xavier 初期化* の理論的根拠です。
この手法は、その考案者の第一著者にちなんで名付けられました :cite:`Glorot.Bengio.2010`。
通常、Xavier 初期化では
平均 0、分散
$\sigma^2 = \frac{2}{n_\textrm{in} + n_\textrm{out}}$
のガウス分布から重みをサンプルします。
また、重みを一様分布からサンプルするときの
分散の選び方にも適用できます。
一様分布 $U(-a, a)$ の分散は $\frac{a^2}{3}$ であることに注意してください。
$\frac{a^2}{3}$ を $\sigma^2$ に関する条件へ代入すると、
次のように初期化すればよいことが分かります。

$$U\left(-\sqrt{\frac{6}{n_\textrm{in} + n_\textrm{out}}}, \sqrt{\frac{6}{n_\textrm{in} + n_\textrm{out}}}\right).$$

上の数学的な議論では
非線形性がないことを仮定していますが、
この仮定はニューラルネットワークでは簡単に破られます。
それでも、Xavier 初期化法は
実際にはうまく機能することが分かっています。


### さらに先へ

上の議論は、現代的なパラメータ初期化手法の
ほんの入口にすぎません。
深層学習フレームワークには、しばしば十数種類もの
異なる経験則が実装されています。
さらに、パラメータ初期化は
深層学習における基礎研究の
非常に活発な分野であり続けています。
そこには、共有パラメータ、超解像、系列モデル、
その他の状況に特化した経験則も含まれます。
たとえば、
:citet:`Xiao.Bahri.Sohl-Dickstein.ea.2018` は、慎重に設計された初期化法を用いることで、
アーキテクチャ上の工夫なしに
10,000 層のニューラルネットワークを学習できる可能性を示しました。

この話題に興味があるなら、
このモジュールで扱う各手法を深く掘り下げ、
それぞれの経験則を提案・解析した論文を読み、
さらにこの分野の最新論文を追ってみることを勧めます。
もしかすると、あなたは巧妙なアイデアを見つけたり、
あるいは発明したりして、
深層学習フレームワークに実装を貢献するかもしれません。


## まとめ

勾配消失と勾配爆発は、深いネットワークでよく見られる問題です。勾配とパラメータが適切に制御された状態を保つためには、パラメータ初期化に細心の注意が必要です。
初期勾配が大きすぎず小さすぎもしないようにするために、初期化の経験則が必要です。
ランダム初期化は、最適化の前に対称性が破られることを保証するうえで重要です。
Xavier 初期化は、各層について、任意の出力の分散が入力数の影響を受けず、任意の勾配の分散が出力数の影響を受けないことを示唆します。
ReLU 活性化関数は勾配消失問題を緩和します。これにより収束を加速できます。

## 演習

1. MLP の各層における置換対称性以外に、対称性を破る必要があるニューラルネットワークのケースを他に設計できますか？
1. 線形回帰や softmax 回帰では、すべての重みパラメータを同じ値に初期化してもよいでしょうか？
1. 2 つの行列の積の固有値に関する解析的な上界を調べてください。これは、勾配がよく条件付けられるようにすることについて何を示唆していますか？
1. ある項が発散すると分かっている場合、後から修正できるでしょうか？ layerwise adaptive rate scaling に関する論文を参考にしてください :cite:`You.Gitman.Ginsburg.2017`。


:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/103)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/104)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/235)
:end_tab:

:begin_tab:`jax`
[Discussions](https://discuss.d2l.ai/t/17986)
:end_tab:\n