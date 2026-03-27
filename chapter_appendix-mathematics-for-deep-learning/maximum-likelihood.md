# 最尤推定
:label:`sec_maximum_likelihood`

機械学習で最もよく出会う考え方の一つが、最尤の観点である。これは、未知のパラメータをもつ確率モデルを扱うとき、データに最も高い確率を与えるパラメータが最もありそうだ、という考え方である。

## 最尤原理

これはベイズ的な解釈を持っており、考えるうえで役に立ちる。パラメータ $\boldsymbol{\theta}$ をもつモデルと、データ例の集合 $X$ があるとしよう。具体的には、$\boldsymbol{\theta}$ を、コインを投げたときに表が出る確率を表す単一の値、$X$ を独立なコイン投げの列だと考えることができる。この例は後で詳しく見る。

モデルのパラメータとして最もありそうな値を見つけたいなら、次を求めたいことになる。

$$
\mathop{\mathrm{argmax}} P(\boldsymbol{\theta}\mid X).
$$
:eqlabel:`eq_max_like`

ベイズの定理によれば、これは次と同じである。

$$
\mathop{\mathrm{argmax}} \frac{P(X \mid \boldsymbol{\theta})P(\boldsymbol{\theta})}{P(X)}.
$$

式 $P(X)$ は、データを生成する確率をパラメータに依存せずに表したものなので、$\boldsymbol{\theta}$ にはまったく依存しない。したがって、$\boldsymbol{\theta}$ の最良の選択を変えずにこの項は無視できる。同様に、どのパラメータ集合が他より優れているという事前仮定はないと考えてよいので、$P(\boldsymbol{\theta})$ も $\theta$ に依存しないと宣言してよいだろう！ たとえば、コイン投げの例では、表が出る確率は $[0,1]$ の任意の値を取りうるので、それが公平かどうかについて事前の信念はない（これはしばしば *無情報事前分布* と呼ばれる）。したがって、ベイズの定理の適用により、$\boldsymbol{\theta}$ の最良の選択は $\boldsymbol{\theta}$ の最尤推定量であることがわかる。

$$
\hat{\boldsymbol{\theta}} = \mathop{\mathrm{argmax}} _ {\boldsymbol{\theta}} P(X \mid \boldsymbol{\theta}).
$$

一般的な用語として、パラメータが与えられたときのデータの確率（$P(X \mid \boldsymbol{\theta})$）は *尤度* と呼ばれる。

### 具体例

これが具体的にどう働くかを見てみよう。コイン投げで表が出る確率を表す単一のパラメータ $\theta$ があるとする。このとき、裏が出る確率は $1-\theta$ である。観測データ $X$ が $n_H$ 回の表と $n_T$ 回の裏からなる列であれば、独立な確率は掛け算できるので、

$$
P(X \mid \theta) = \theta^{n_H}(1-\theta)^{n_T}.
$$

13枚のコインを投げて "HHHTHTTHHHHHT" という列が得られたとする。これは $n_H = 9$、$n_T = 4$ だろうから、

$$
P(X \mid \theta) = \theta^9(1-\theta)^4.
$$

この例のよい点の一つは、答えをあらかじめ知っていることである。実際、口頭で「13回コインを投げて、9回表が出た。では、そのコインが表を出す確率の最良の推定値は？」と言えば、誰もが正しく $9/13$ と答えるだろう。この最尤法が与えてくれるのは、その数を第一原理から求める方法であり、しかもそれははるかに複雑な状況へ一般化できる。

この例に対する $P(X \mid \theta)$ のグラフは次のとおりである。

```{.python .input}
#@tab mxnet
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import autograd, np, npx
npx.set_np()

theta = np.arange(0, 1, 0.001)
p = theta**9 * (1 - theta)**4.

d2l.plot(theta, p, 'theta', 'likelihood')
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
import torch

theta = torch.arange(0, 1, 0.001)
p = theta**9 * (1 - theta)**4.

d2l.plot(theta, p, 'theta', 'likelihood')
```

```{.python .input}
#@tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
import tensorflow as tf

theta = tf.range(0, 1, 0.001)
p = theta**9 * (1 - theta)**4.

d2l.plot(theta, p, 'theta', 'likelihood')
```

これは、期待される $9/13 \approx 0.7\ldots$ の近くで最大値をとりる。これが正確にそこなのかを見るには、微積分を使える。最大値では関数の傾きが 0 になることに注意しよう。したがって、導関数が 0 となる $\theta$ の値を見つけ、その中で最も高い確率を与えるものを選べば、最尤推定 :eqref:`eq_max_like` を求められる。計算すると、

$$
\begin{aligned}
0 & = \frac{d}{d\theta} P(X \mid \theta) \\
& = \frac{d}{d\theta} \theta^9(1-\theta)^4 \\
& = 9\theta^8(1-\theta)^4 - 4\theta^9(1-\theta)^3 \\
& = \theta^8(1-\theta)^3(9-13\theta).
\end{aligned}
$$

これには 3 つの解、$0$、$1$、$9/13$ がある。最初の 2 つは、列に確率 0 を割り当てるので、明らかに最大値ではなく最小値である。最後の値は列に 0 の確率を割り当てないので、最尤推定量 $\hat \theta = 9/13$ でなければならない。

## 数値最適化と負の対数尤度

前の例はよいものであるが、もし何十億ものパラメータやデータ例があったらどうだろうか？

まず、すべてのデータ例が独立であると仮定すると、尤度は多数の確率の積になるため、実用上そのまま扱うことはできない。実際、各確率は $[0,1]$ にあり、典型的にはおよそ $1/2$ 程度の値だろうから、$(1/2)^{1000000000}$ の積は機械精度をはるかに下回りる。これを直接扱うことはできない。

しかし、対数をとると積が和に変わることを思い出する。このとき

$$
\log((1/2)^{1000000000}) = 1000000000\cdot\log(1/2) \approx -301029995.6\ldots
$$

となる。この数は、単精度の 32 ビット浮動小数点数にすら十分収まりる。したがって、*対数尤度* を考えるべきである。これは

$$
\log(P(X \mid \boldsymbol{\theta})).
$$

である。関数 $x \mapsto \log(x)$ は単調増加なので、尤度を最大化することは対数尤度を最大化することと同じである。実際、 :numref:`sec_naive_bayes` では、ナイーブベイズ分類器という具体例でこの考え方が使われるのを見る。

私たちはしばしば損失関数を扱い、その損失を最小化したいと考える。最大尤度は、$-\log(P(X \mid \boldsymbol{\theta}))$ をとることで損失の最小化に変えられる。これが *負の対数尤度* である。

これを示すために、先ほどのコイン投げ問題を考え、閉形式解を知らないふりをしよう。すると、

$$
-\log(P(X \mid \boldsymbol{\theta})) = -\log(\theta^{n_H}(1-\theta)^{n_T}) = -(n_H\log(\theta) + n_T\log(1-\theta)).
$$

と計算できる。これはコードに書けますし、何十億回のコイン投げに対しても自由に最適化できる。

```{.python .input}
#@tab mxnet
# Set up our data
n_H = 8675309
n_T = 256245

# Initialize our paramteres
theta = np.array(0.5)
theta.attach_grad()

# Perform gradient descent
lr = 1e-9
for iter in range(100):
    with autograd.record():
        loss = -(n_H * np.log(theta) + n_T * np.log(1 - theta))
    loss.backward()
    theta -= lr * theta.grad

# Check output
theta, n_H / (n_H + n_T)
```

```{.python .input}
#@tab pytorch
# Set up our data
n_H = 8675309
n_T = 256245

# Initialize our paramteres
theta = torch.tensor(0.5, requires_grad=True)

# Perform gradient descent
lr = 1e-9
for iter in range(100):
    loss = -(n_H * torch.log(theta) + n_T * torch.log(1 - theta))
    loss.backward()
    with torch.no_grad():
        theta -= lr * theta.grad
    theta.grad.zero_()

# Check output
theta, n_H / (n_H + n_T)
```

```{.python .input}
#@tab tensorflow
# Set up our data
n_H = 8675309
n_T = 256245

# Initialize our paramteres
theta = tf.Variable(tf.constant(0.5))

# Perform gradient descent
lr = 1e-9
for iter in range(100):
    with tf.GradientTape() as t:
        loss = -(n_H * tf.math.log(theta) + n_T * tf.math.log(1 - theta))
    theta.assign_sub(lr * t.gradient(loss, theta))

# Check output
theta, n_H / (n_H + n_T)
```

数値計算上の都合だけが、負の対数尤度を使う理由ではない。ほかにも、これを好む理由がいくつかある。

対数尤度を考える第 2 の理由は、微積分の規則を簡単に適用できることである。上で述べたように、独立性の仮定により、機械学習で出会う確率の多くは個々の確率の積になる。

$$
P(X\mid\boldsymbol{\theta}) = p(x_1\mid\boldsymbol{\theta})\cdot p(x_2\mid\boldsymbol{\theta})\cdots p(x_n\mid\boldsymbol{\theta}).
$$

これは、導関数を求めるために直接積の微分法則を適用すると、

$$
\begin{aligned}
\frac{\partial}{\partial \boldsymbol{\theta}} P(X\mid\boldsymbol{\theta}) & = \left(\frac{\partial}{\partial \boldsymbol{\theta}}P(x_1\mid\boldsymbol{\theta})\right)\cdot P(x_2\mid\boldsymbol{\theta})\cdots P(x_n\mid\boldsymbol{\theta}) \\
& \quad + P(x_1\mid\boldsymbol{\theta})\cdot \left(\frac{\partial}{\partial \boldsymbol{\theta}}P(x_2\mid\boldsymbol{\theta})\right)\cdots P(x_n\mid\boldsymbol{\theta}) \\
& \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \vdots \\
& \quad + P(x_1\mid\boldsymbol{\theta})\cdot P(x_2\mid\boldsymbol{\theta}) \cdots \left(\frac{\partial}{\partial \boldsymbol{\theta}}P(x_n\mid\boldsymbol{\theta})\right).
\end{aligned}
$$

となる。これには $n(n-1)$ 回の乗算と $(n-1)$ 回の加算が必要なので、入力に対して二次時間に比例する！ 項のまとめ方をうまく工夫すれば線形時間に減らせるが、少し考える必要がある。これに対して負の対数尤度では、

$$
-\log\left(P(X\mid\boldsymbol{\theta})\right) = -\log(P(x_1\mid\boldsymbol{\theta})) - \log(P(x_2\mid\boldsymbol{\theta})) \cdots - \log(P(x_n\mid\boldsymbol{\theta})),
$$

となり、したがって

$$
- \frac{\partial}{\partial \boldsymbol{\theta}} \log\left(P(X\mid\boldsymbol{\theta})\right) = \frac{1}{P(x_1\mid\boldsymbol{\theta})}\left(\frac{\partial}{\partial \boldsymbol{\theta}}P(x_1\mid\boldsymbol{\theta})\right) + \cdots + \frac{1}{P(x_n\mid\boldsymbol{\theta})}\left(\frac{\partial}{\partial \boldsymbol{\theta}}P(x_n\mid\boldsymbol{\theta})\right).
$$

これには $n$ 回の除算と $n-1$ 回の和しか必要なく、入力に対して線形時間である。

負の対数尤度を考える第 3 の、そして最後の理由は、情報理論との関係である。これについては :numref:`sec_information_theory` で詳しく議論する。これは、確率変数における情報量やランダムさの程度を測る方法を与える厳密な数学理論である。この分野で中心となる対象はエントロピーであり、

$$
H(p) = -\sum_{i} p_i \log_2(p_i),
$$

である。これはソースのランダムさを測りる。これが単に平均の $-\log$ 確率にほかならないことに注意する。したがって、負の対数尤度をデータ例の数で割れば、クロスエントロピーとして知られるエントロピーの親戚が得られる。この理論的解釈だけでも、データセット全体にわたる平均負の対数尤度をモデル性能の指標として報告する十分な動機になる。

## 連続変数に対する最尤推定

これまで行ってきたことはすべて離散確率変数を扱うことを前提としていたが、連続変数を扱いたい場合はどうだろうか？

短く言えば、確率をすべて確率密度に置き換える以外、何も変わりない。密度は小文字の $p$ で書くことを思い出すと、たとえば今は次のように書く。

$$
-\log\left(p(X\mid\boldsymbol{\theta})\right) = -\log(p(x_1\mid\boldsymbol{\theta})) - \log(p(x_2\mid\boldsymbol{\theta})) \cdots - \log(p(x_n\mid\boldsymbol{\theta})) = -\sum_i \log(p(x_i \mid \theta)).
$$

問題は、「なぜこれでよいのか？」である。そもそも、密度を導入した理由は、特定の結果が起こる確率そのものが 0 だからでした。では、どのパラメータ集合に対してもデータを生成する確率は 0 ではないのだろうか？

実際そのとおりである。そして、なぜ密度に切り替えられるのかを理解するには、$\epsilon$ に何が起こるかを追跡する練習になる。

まず、目的を再定義しよう。連続確率変数については、ちょうど正しい値を得る確率を計算するのではなく、ある範囲 $\epsilon$ 以内で一致することを考えるとする。簡単のため、データは同一分布の確率変数 $X_1, \ldots, X_N$ の繰り返し観測 $x_1, \ldots, x_N$ だと仮定する。前に見たように、これは次のように書ける。

$$
\begin{aligned}
&P(X_1 \in [x_1, x_1+\epsilon], X_2 \in [x_2, x_2+\epsilon], \ldots, X_N \in [x_N, x_N+\epsilon]\mid\boldsymbol{\theta}) \\
\approx &\epsilon^Np(x_1\mid\boldsymbol{\theta})\cdot p(x_2\mid\boldsymbol{\theta}) \cdots p(x_n\mid\boldsymbol{\theta}).
\end{aligned}
$$

したがって、これの負の対数をとると、

$$
\begin{aligned}
&-\log(P(X_1 \in [x_1, x_1+\epsilon], X_2 \in [x_2, x_2+\epsilon], \ldots, X_N \in [x_N, x_N+\epsilon]\mid\boldsymbol{\theta})) \\
\approx & -N\log(\epsilon) - \sum_{i} \log(p(x_i\mid\boldsymbol{\theta})).
\end{aligned}
$$

となる。この式を見ると、$\epsilon$ が現れるのは加法定数 $-N\log(\epsilon)$ の部分だけである。これは $\boldsymbol{\theta}$ にまったく依存しないので、$\boldsymbol{\theta}$ の最適な選択は $\epsilon$ の選び方に依存しない！ 4 桁の精度を要求しても 400 桁を要求しても、$\boldsymbol{\theta}$ の最良の選択は同じである。したがって、$\epsilon$ を自由に捨てて、最適化したいものが

$$
- \sum_{i} \log(p(x_i\mid\boldsymbol{\theta})).
$$

であることがわかる。

このように、最大尤度の観点は、確率を確率密度に置き換えるだけで、離散確率変数と同じくらい容易に連続確率変数にも適用できることがわかる。

## まとめ
* 最尤原理は、与えられたデータセットに最もよく適合するモデルとは、そのデータを最も高い確率で生成するモデルだと述べる。
* 実際には、数値安定性、積を和に変換できること（およびそれに伴う勾配計算の簡略化）、情報理論との理論的なつながりなど、さまざまな理由から負の対数尤度がよく使われる。
* 離散設定で最も説明しやすいであるが、データ点に割り当てられた確率密度を最大化することで、連続設定にも自由に一般化できる。

## 演習
1. 非負の確率変数が、ある $\alpha>0$ に対して密度 $\alpha e^{-\alpha x}$ をもつことがわかっているとする。この確率変数から 1 つの観測値として数 $3$ を得た。$\alpha$ の最尤推定値は何だろうか？
2. 平均は未知だが分散は $1$ のガウス分布から、サンプル $\{x_i\}_{i=1}^N$ からなるデータセットが与えられているとする。平均の最尤推定値は何だろうか？
