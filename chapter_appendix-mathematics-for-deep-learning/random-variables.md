# 確率変数
:label:`sec_random_variables`

:numref:`sec_prob` では、離散確率変数の扱い方の基礎を見た。ここでいう離散確率変数とは、有限個の取りうる値、または整数値をとる確率変数のことでした。この節では、任意の実数値をとりうる *連続確率変数* の理論を展開する。

## 連続確率変数

連続確率変数は、離散確率変数よりもかなり繊細な話題である。技術的な飛躍としては、数のリストを足し合わせることから関数を積分することへの飛躍に匹敵すると考えるとよいだろう。そのため、理論を構築するのに少し時間をかける必要がある。

### 離散から連続へ

連続確率変数を扱う際に生じる追加の技術的課題を理解するために、思考実験をしてみよう。ダーツをダーツボードに投げて、ボードの中心からちょうど $2 \textrm{cm}$ の位置に当たる確率を知りたいとする。

まず、1桁の精度で測ることを考える。つまり、$0 \textrm{cm}$、$1 \textrm{cm}$、$2 \textrm{cm}$ などのビンに分けるのである。たとえばダーツを $100$ 本投げて、そのうち $20$ 本が $2\textrm{cm}$ のビンに入ったなら、投げたダーツの $20\%$ が中心から $2 \textrm{cm}$ の位置に当たったと結論づける。

しかし、よく見ると、これは私たちの問いと一致していない。私たちが知りたかったのは「ちょうど」当たる確率であって、これらのビンはたとえば $1.5\textrm{cm}$ から $2.5\textrm{cm}$ の間に入ったものをすべて含んでいるからである。

それでもあきらめず、さらに進む。今度は $1.9\textrm{cm}$、$2.0\textrm{cm}$、$2.1\textrm{cm}$ のように、もっと精密に測りる。すると、$100$ 本のうち $3$ 本が $2.0\textrm{cm}$ のビンに入ったとしよう。そこで確率は $3\%$ だと結論づける。

しかし、これでも何も解決していない。問題を1桁先送りしただけである。少し抽象化してみよう。最初の $k$ 桁が $2.00000\ldots$ と一致する確率がわかっていて、最初の $k+1$ 桁まで一致する確率を知りたいとする。${k+1}^{\textrm{th}}$ 桁は、集合 $\{0, 1, 2, \ldots, 9\}$ からほぼランダムに選ばれると考えるのはかなり自然である。少なくとも、中心から何マイクロメートル離れているかの末尾が $7$ になるか $3$ になるかを強制するような、物理的に意味のある過程は思いつかない。

つまり、本質的には、精度を1桁上げるごとに一致する確率は $10$ 分の $1$ に減るはずである。言い換えると、次のように期待できる。

$$
P(\textrm{distance is}\; 2.00\ldots, \;\textrm{to}\; k \;\textrm{digits} ) \approx p\cdot10^{-k}.
$$

ここでの値 $p$ は、最初の数桁で何が起こるかを実質的に表し、$10^{-k}$ は残りを扱う。

小数点以下 $k=4$ 桁まで正確に位置がわかるということは、値がたとえば $[1.99995,2.00005]$ のような区間に入ることを意味する。この区間の長さは $2.00005-1.99995 = 10^{-4}$ である。したがって、この区間の長さを $\epsilon$ と呼ぶと、

$$
P(\textrm{distance is in an}\; \epsilon\textrm{-sized interval around}\; 2 ) \approx \epsilon \cdot p.
$$

これをさらに一歩進めよう。ここまでずっと点 $2$ について考えてきたが、他の点については考えていなかった。そこでも本質的には何も違いませんが、値 $p$ はおそらく異なる。少なくとも、ダーツ投げでは $20\textrm{cm}$ より $2\textrm{cm}$ のように中心に近い点に当たりやすいと期待したいところである。したがって、値 $p$ は固定ではなく、点 $x$ に依存すべきである。すると、次を期待すべきだとわかる。

$$P(\textrm{distance is in an}\; \epsilon \textrm{-sized interval around}\; x ) \approx \epsilon \cdot p(x).$$
:eqlabel:`eq_pdf_deriv`

実際、:eqref:`eq_pdf_deriv` は *確率密度関数* を正確に定義している。これは、ある点の近くに当たる相対的な確率を表す関数 $p(x)$ である。このような関数がどのような形をしているかを可視化してみよう。

```{.python .input}
#@tab mxnet
%matplotlib inline
from d2l import mxnet as d2l
from IPython import display
from mxnet import np, npx
npx.set_np()

# Plot the probability density function for some random variable
x = np.arange(-5, 5, 0.01)
p = 0.2*np.exp(-(x - 3)**2 / 2)/np.sqrt(2 * np.pi) + \
    0.8*np.exp(-(x + 1)**2 / 2)/np.sqrt(2 * np.pi)

d2l.plot(x, p, 'x', 'Density')
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
from IPython import display
import torch
torch.pi = torch.acos(torch.zeros(1)).item() * 2  # Define pi in torch

# Plot the probability density function for some random variable
x = torch.arange(-5, 5, 0.01)
p = 0.2*torch.exp(-(x - 3)**2 / 2)/torch.sqrt(2 * torch.tensor(torch.pi)) + \
    0.8*torch.exp(-(x + 1)**2 / 2)/torch.sqrt(2 * torch.tensor(torch.pi))

d2l.plot(x, p, 'x', 'Density')
```

```{.python .input}
#@tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
from IPython import display
import tensorflow as tf
tf.pi = tf.acos(tf.zeros(1)).numpy() * 2  # Define pi in TensorFlow

# Plot the probability density function for some random variable
x = tf.range(-5, 5, 0.01)
p = 0.2*tf.exp(-(x - 3)**2 / 2)/tf.sqrt(2 * tf.constant(tf.pi)) + \
    0.8*tf.exp(-(x + 1)**2 / 2)/tf.sqrt(2 * tf.constant(tf.pi))

d2l.plot(x, p, 'x', 'Density')
```

関数値が大きい場所は、確率変数が見つかりやすい領域を示している。値が低い部分は、確率変数が見つかりにくい領域である。

### 確率密度関数

では、これをさらに詳しく調べよう。確率変数 $X$ に対する確率密度関数とは何かをすでに直感的に見た。すなわち、密度関数とは

$$P(X \; \textrm{is in an}\; \epsilon \textrm{-sized interval around}\; x ) \approx \epsilon \cdot p(x).$$
:eqlabel:`eq_pdf_def`

を満たす関数 $p(x)$ のことである。

では、これは $p(x)$ の性質について何を意味するのだろうか。

まず、確率は負にならないので、$p(x) \ge 0$ であると期待される。

次に、$\mathbb{R}$ を幅 $\epsilon$ の無限個の区間、たとえば $(\epsilon\cdot i, \epsilon \cdot (i+1)]$ に分割することを考える。それぞれについて、:eqref:`eq_pdf_def` より確率はおおよそ

$$
P(X \; \textrm{is in an}\; \epsilon\textrm{-sized interval around}\; x ) \approx \epsilon \cdot p(\epsilon \cdot i),
$$

だろうから、すべてを足し合わせると

$$
P(X\in\mathbb{R}) \approx \sum_i \epsilon \cdot p(\epsilon\cdot i).
$$

これは :numref:`sec_integral_calculus` で議論した積分の近似にほかならない。したがって、

$$
P(X\in\mathbb{R}) = \int_{-\infty}^{\infty} p(x) \; dx.
$$

確率変数は *何らかの* 値をとらなければならないので、$P(X\in\mathbb{R}) = 1$ である。したがって、任意の密度について

$$
\int_{-\infty}^{\infty} p(x) \; dx = 1.
$$

実際、さらに掘り下げると、任意の $a$ と $b$ について

$$
P(X\in(a, b]) = \int _ {a}^{b} p(x) \; dx.
$$

が成り立ちる。

これをコードで近似するには、先ほどと同じ離散近似法を使える。この場合、青い領域に入る確率を近似できる。

```{.python .input}
#@tab mxnet
# Approximate probability using numerical integration
epsilon = 0.01
x = np.arange(-5, 5, 0.01)
p = 0.2*np.exp(-(x - 3)**2 / 2) / np.sqrt(2 * np.pi) + \
    0.8*np.exp(-(x + 1)**2 / 2) / np.sqrt(2 * np.pi)

d2l.set_figsize()
d2l.plt.plot(x, p, color='black')
d2l.plt.fill_between(x.tolist()[300:800], p.tolist()[300:800])
d2l.plt.show()

f'approximate Probability: {np.sum(epsilon*p[300:800])}'
```

```{.python .input}
#@tab pytorch
# Approximate probability using numerical integration
epsilon = 0.01
x = torch.arange(-5, 5, 0.01)
p = 0.2*torch.exp(-(x - 3)**2 / 2) / torch.sqrt(2 * torch.tensor(torch.pi)) +\
    0.8*torch.exp(-(x + 1)**2 / 2) / torch.sqrt(2 * torch.tensor(torch.pi))

d2l.set_figsize()
d2l.plt.plot(x, p, color='black')
d2l.plt.fill_between(x.tolist()[300:800], p.tolist()[300:800])
d2l.plt.show()

f'approximate Probability: {torch.sum(epsilon*p[300:800])}'
```

```{.python .input}
#@tab tensorflow
# Approximate probability using numerical integration
epsilon = 0.01
x = tf.range(-5, 5, 0.01)
p = 0.2*tf.exp(-(x - 3)**2 / 2) / tf.sqrt(2 * tf.constant(tf.pi)) +\
    0.8*tf.exp(-(x + 1)**2 / 2) / tf.sqrt(2 * tf.constant(tf.pi))

d2l.set_figsize()
d2l.plt.plot(x, p, color='black')
d2l.plt.fill_between(x.numpy().tolist()[300:800], p.numpy().tolist()[300:800])
d2l.plt.show()

f'approximate Probability: {tf.reduce_sum(epsilon*p[300:800])}'
```

これら2つの性質は、可能な確率密度関数の空間をちょうど記述している（よく使われる略称として *p.d.f.* とも呼ばれます）。すなわち、非負関数 $p(x) \ge 0$ であって

$$\int_{-\infty}^{\infty} p(x) \; dx = 1.$$
:eqlabel:`eq_pdf_int_one`

を満たすものである。

この関数は、確率変数が特定の区間に入る確率を積分によって求めることで解釈する。

$$P(X\in(a, b]) = \int _ {a}^{b} p(x) \; dx.$$
:eqlabel:`eq_pdf_int_int`

:numref:`sec_distributions` では、いくつかの代表的な分布を見ますが、ここでは抽象的な議論を続けましょう。

### 累積分布関数

前節では p.d.f. の概念を見た。実際には、これは連続確率変数を扱う際によく使われる方法であるが、1つ大きな落とし穴がある。それは、p.d.f. の値そのものは確率ではなく、確率を得るために積分しなければならない関数だということである。密度が $10$ より大きくても、それが長さ $1/10$ を超える区間にわたって大きくなければ問題ない。これは直感に反することがあるため、人々はしばしば *累積分布関数*、すなわち c.d.f. でも考える。こちらは *確率* である。

特に、:eqref:`eq_pdf_int_int` を用いて、密度 $p(x)$ をもつ確率変数 $X$ の c.d.f. を

$$
F(x) = \int _ {-\infty}^{x} p(x) \; dx = P(X \le x).
$$

と定義する。

いくつかの性質を見てみよう。

* $x\rightarrow -\infty$ のとき $F(x) \rightarrow 0$。
* $x\rightarrow \infty$ のとき $F(x) \rightarrow 1$。
* $F(x)$ は単調非減少である（$y > x \implies F(y) \ge F(x)$）。
* $X$ が連続確率変数なら、$F(x)$ は連続である（飛びがない）。

4つ目について、$X$ が離散的、たとえば $0$ と $1$ をそれぞれ確率 $1/2$ でとる場合には、これは成り立たないことに注意する。その場合、

$$
F(x) = \begin{cases}
0 & x < 0, \\
\frac{1}{2} & x < 1, \\
1 & x \ge 1.
\end{cases}
$$

この例から、c.d.f. を使う利点の1つがわかる。連続確率変数と離散確率変数を同じ枠組みで扱えること、さらには両者の混合（コインを投げて、表ならサイコロの目を返し、裏ならダーツボードの中心からの距離を返す）も扱えることである。

### 平均

確率変数 $X$ を扱っているとしよう。分布そのものは解釈が難しいことがある。確率変数の振る舞いを簡潔に要約できると便利なことがよくある。確率変数の振る舞いを捉えるのに役立つ数を *要約統計量* と呼ぶ。最もよく出てくるものは *平均*、*分散*、*標準偏差* である。

*平均* は確率変数の平均値を表する。離散確率変数 $X$ が値 $x_i$ を確率 $p_i$ でとるなら、平均は重み付き平均で与えられる。すなわち、値にその値をとる確率を掛けて足し合わせる。

$$\mu_X = E[X] = \sum_i x_i p_i.$$
:eqlabel:`eq_exp_def`

平均の解釈は、注意は必要であるが、本質的には確率変数がどのあたりに位置しがちかを示すものだと考えればよいだろう。

この節を通して扱う最小限の例として、$X$ を、確率 $p$ で $a-2$、確率 $p$ で $a+2$、確率 $1-2p$ で $a$ をとる確率変数とする。:eqref:`eq_exp_def` を使うと、$a$ と $p$ のどのような選び方に対しても平均は

$$
\mu_X = E[X] = \sum_i x_i p_i = (a-2)p + a(1-2p) + (a+2)p = a.
$$

となる。したがって、平均は $a$ である。これは、確率変数を中心化した位置が $a$ であるという直感と一致している。

役に立つので、いくつかの性質をまとめておきよう。

* 任意の確率変数 $X$ と数 $a, b$ に対して、$\mu_{aX+b} = a\mu_X + b$ である。
* 2つの確率変数 $X$ と $Y$ に対して、$\mu_{X+Y} = \mu_X+\mu_Y$ である。

平均は確率変数の平均的な振る舞いを理解するのに有用であるが、平均だけでは十分な直感的理解には至りない。1回の販売あたり $\$10 \pm \$1$ の利益を得るのと、$\$10 \pm \$15$ の利益を得るのとでは、平均値が同じでもまったく違いる。後者のほうが変動がはるかに大きく、したがってリスクもはるかに大きいのである。したがって、確率変数の振る舞いを理解するには、少なくとももう1つ、確率変数がどれだけ広く変動するかを測る尺度が必要である。

### 分散

そこで、確率変数の *分散* を考える。これは、確率変数が平均からどれだけずれるかを定量的に測るものである。式 $X - \mu_X$ を考えよう。これは確率変数の平均からの偏差である。この値は正にも負にもなりうるので、偏差の大きさを測るために正にする工夫が必要である。

試してみる自然な方法は $\left|X-\mu_X\right|$ を見ることである。実際、これは *平均絶対偏差* と呼ばれる有用な量につながりるが、数学や統計の他分野とのつながりのため、しばしば別の方法が使われる。

具体的には、$(X-\mu_X)^2$ を見る。この量の典型的な大きさを平均で見れば、分散に到達する。

$$\sigma_X^2 = \textrm{Var}(X) = E\left[(X-\mu_X)^2\right] = E[X^2] - \mu_X^2.$$
:eqlabel:`eq_var_def`

:eqref:`eq_var_def` の最後の等式は、中央の定義を展開し、期待値の性質を適用することで得られる。

例として、$X$ が確率 $p$ で $a-2$、確率 $p$ で $a+2$、確率 $1-2p$ で $a$ をとる確率変数を考えよう。この場合 $\mu_X = a$ なので、計算すべきなのは $E\left[X^2\right]$ だけである。これはすぐに求められる。

$$
E\left[X^2\right] = (a-2)^2p + a^2(1-2p) + (a+2)^2p = a^2 + 8p.
$$

したがって、:eqref:`eq_var_def` より分散は

$$
\sigma_X^2 = \textrm{Var}(X) = E[X^2] - \mu_X^2 = a^2 + 8p - a^2 = 8p.
$$

この結果もまた納得できる。$p$ の最大値は $1/2$ であり、これは $a-2$ か $a+2$ をコイン投げで選ぶ場合に対応する。このとき分散が $4$ になるのは、$a-2$ と $a+2$ のどちらも平均から $2$ だけ離れており、$2^2 = 4$ だからである。逆に、$p=0$ ならこの確率変数は常に $0$ をとるので、分散はまったくない。

分散の性質をいくつか挙げておきる。

* 任意の確率変数 $X$ に対して、$\textrm{Var}(X) \ge 0$ であり、$\textrm{Var}(X) = 0$ となるのは $X$ が定数であるとき、かつそのときに限る。
* 任意の確率変数 $X$ と数 $a, b$ に対して、$\textrm{Var}(aX+b) = a^2\textrm{Var}(X)$ である。
* 2つの *独立* な確率変数 $X$ と $Y$ に対して、$\textrm{Var}(X+Y) = \textrm{Var}(X) + \textrm{Var}(Y)$ である。

これらの値を解釈する際には、少し引っかかる点がある。特に、単位を追跡しながらこの計算がどうなるかを考えてみよう。ウェブページ上で製品に付けられた星評価を扱っているとする。このとき $a$、$a-2$、$a+2$ はすべて星の単位で測られる。同様に、平均 $\mu_X$ も（重み付き平均なので）星の単位である。しかし分散に進むと、すぐに問題にぶつかりる。$(X-\mu_X)^2$ を見る必要があり、これは *二乗された星* の単位になるからである。つまり、分散そのものは元の測定値と比較しにくいのである。解釈可能にするには、元の単位に戻る必要がある。

### 標準偏差

この要約統計量は、平方根を取ることで分散から常に導ける。そこで *標準偏差* を

$$
\sigma_X = \sqrt{\textrm{Var}(X)}.
$$

と定義する。

この例では、標準偏差は $\sigma_X = 2\sqrt{2p}$ になる。レビューの例で単位が星なら、$\sigma_X$ も再び星の単位である。

分散について述べた性質は、標準偏差について次のように言い換えられる。

* 任意の確率変数 $X$ に対して、$\sigma_{X} \ge 0$。
* 任意の確率変数 $X$ と数 $a, b$ に対して、$\sigma_{aX+b} = |a|\sigma_{X}$。
* 2つの *独立* な確率変数 $X$ と $Y$ に対して、$\sigma_{X+Y} = \sqrt{\sigma_{X}^2 + \sigma_{Y}^2}$。

ここで、「標準偏差は元の確率変数と同じ単位を持つのなら、その確率変数について何か描けるものを表しているのだろうか？」と問うのは自然である。答えは大いにイエスである。平均が確率変数の典型的な位置を示したのと同様に、標準偏差はその確率変数の典型的な変動範囲を与える。これはチェビシェフの不等式として知られるものによって厳密にできる。

$$P\left(X \not\in [\mu_X - \alpha\sigma_X, \mu_X + \alpha\sigma_X]\right) \le \frac{1}{\alpha^2}.$$
:eqlabel:`eq_chebyshev`

言葉で言えば、$\alpha=10$ の場合、どんな確率変数でも標本の $99\%$ は平均から $10$ 標準偏差以内に入る。これにより、標準的な要約統計量の解釈がすぐに得られる。

この主張がかなり繊細であることを見るために、先ほどの例、すなわち $X$ が確率 $p$ で $a-2$、確率 $p$ で $a+2$、確率 $1-2p$ で $a$ をとる確率変数をもう一度見てみよう。平均は $a$、標準偏差は $2\sqrt{2p}$ でした。したがって、チェビシェフの不等式 :eqref:`eq_chebyshev` を $\alpha = 2$ で使うと、

$$
P\left(X \not\in [a - 4\sqrt{2p}, a + 4\sqrt{2p}]\right) \le \frac{1}{4}.
$$

となる。つまり、どんな $p$ に対しても、この確率変数は $75\%$ の確率でこの区間に入る。ここで、$p \rightarrow 0$ とすると、この区間は単一点 $a$ に収束する。しかし、確率変数は $a-2, a, a+2$ の値しかとらないので、最終的には $a-2$ と $a+2$ が区間の外に出ることは確実である。問題は、それがどの $p$ で起こるかである。そこで、$a+4\sqrt{2p} = a+2$ となる $p$ を解くと、$p=1/8$ であり、これは分布からの標本の $1/4$ 以下しか区間の外に出ないという主張に反しないまま、それが起こりうる最初の $p$ です（左に $1/8$、右に $1/8$）。

これを可視化してみよう。3つの値が出る確率を、それぞれ確率に比例した高さの3本の縦棒で示する。区間は中央に水平線として描く。最初の図は、$p > 1/8$ のとき、区間がすべての点を安全に含む様子を示する。

```{.python .input}
#@tab mxnet
# Define a helper to plot these figures
def plot_chebyshev(a, p):
    d2l.set_figsize()
    d2l.plt.stem([a-2, a, a+2], [p, 1-2*p, p], use_line_collection=True)
    d2l.plt.xlim([-4, 4])
    d2l.plt.xlabel('x')
    d2l.plt.ylabel('p.m.f.')

    d2l.plt.hlines(0.5, a - 4 * np.sqrt(2 * p),
                   a + 4 * np.sqrt(2 * p), 'black', lw=4)
    d2l.plt.vlines(a - 4 * np.sqrt(2 * p), 0.53, 0.47, 'black', lw=1)
    d2l.plt.vlines(a + 4 * np.sqrt(2 * p), 0.53, 0.47, 'black', lw=1)
    d2l.plt.title(f'p = {p:.3f}')

    d2l.plt.show()

# Plot interval when p > 1/8
plot_chebyshev(0.0, 0.2)
```

```{.python .input}
#@tab pytorch
# Define a helper to plot these figures
def plot_chebyshev(a, p):
    d2l.set_figsize()
    d2l.plt.stem([a-2, a, a+2], [p, 1-2*p, p], use_line_collection=True)
    d2l.plt.xlim([-4, 4])
    d2l.plt.xlabel('x')
    d2l.plt.ylabel('p.m.f.')

    d2l.plt.hlines(0.5, a - 4 * torch.sqrt(2 * p),
                   a + 4 * torch.sqrt(2 * p), 'black', lw=4)
    d2l.plt.vlines(a - 4 * torch.sqrt(2 * p), 0.53, 0.47, 'black', lw=1)
    d2l.plt.vlines(a + 4 * torch.sqrt(2 * p), 0.53, 0.47, 'black', lw=1)
    d2l.plt.title(f'p = {p:.3f}')

    d2l.plt.show()

# Plot interval when p > 1/8
plot_chebyshev(0.0, torch.tensor(0.2))
```

```{.python .input}
#@tab tensorflow
# Define a helper to plot these figures
def plot_chebyshev(a, p):
    d2l.set_figsize()
    d2l.plt.stem([a-2, a, a+2], [p, 1-2*p, p], use_line_collection=True)
    d2l.plt.xlim([-4, 4])
    d2l.plt.xlabel('x')
    d2l.plt.ylabel('p.m.f.')

    d2l.plt.hlines(0.5, a - 4 * tf.sqrt(2 * p),
                   a + 4 * tf.sqrt(2 * p), 'black', lw=4)
    d2l.plt.vlines(a - 4 * tf.sqrt(2 * p), 0.53, 0.47, 'black', lw=1)
    d2l.plt.vlines(a + 4 * tf.sqrt(2 * p), 0.53, 0.47, 'black', lw=1)
    d2l.plt.title(f'p = {p:.3f}')

    d2l.plt.show()

# Plot interval when p > 1/8
plot_chebyshev(0.0, tf.constant(0.2))
```

2つ目は、$p = 1/8$ のとき、区間がちょうど2点に接することを示する。これは、不等式を真に保ちながらこれ以上小さい区間を取れないので、不等式が *鋭い* ことを示している。

```{.python .input}
#@tab mxnet
# Plot interval when p = 1/8
plot_chebyshev(0.0, 0.125)
```

```{.python .input}
#@tab pytorch
# Plot interval when p = 1/8
plot_chebyshev(0.0, torch.tensor(0.125))
```

```{.python .input}
#@tab tensorflow
# Plot interval when p = 1/8
plot_chebyshev(0.0, tf.constant(0.125))
```

3つ目は、$p < 1/8$ のとき、区間が中心だけを含むことを示する。これは不等式を破るものではない。なぜなら、区間の外に出る確率が $1/4$ を超えないことだけを保証すればよいからである。したがって、$p < 1/8$ になれば、$a-2$ と $a+2$ の2点は捨ててよいのである。

```{.python .input}
#@tab mxnet
# Plot interval when p < 1/8
plot_chebyshev(0.0, 0.05)
```

```{.python .input}
#@tab pytorch
# Plot interval when p < 1/8
plot_chebyshev(0.0, torch.tensor(0.05))
```

```{.python .input}
#@tab tensorflow
# Plot interval when p < 1/8
plot_chebyshev(0.0, tf.constant(0.05))
```

### 連続体における平均と分散

ここまでは離散確率変数について述べてきたが、連続確率変数の場合も同様である。これを直感的に理解するために、実数直線を長さ $\epsilon$ の区間 $(\epsilon i, \epsilon (i+1)]$ に分割することを考えよう。そうすると、連続確率変数は離散化され、:eqref:`eq_exp_def` を使って

$$
\begin{aligned}
\mu_X & \approx \sum_{i} (\epsilon i)P(X \in (\epsilon i, \epsilon (i+1)]) \\
& \approx \sum_{i} (\epsilon i)p_X(\epsilon i)\epsilon, \\
\end{aligned}
$$

と書ける。ここで $p_X$ は $X$ の密度である。これは $xp_X(x)$ の積分の近似なので、

$$
\mu_X = \int_{-\infty}^\infty xp_X(x) \; dx.
$$

と結論できる。

同様に、:eqref:`eq_var_def` を使うと分散は

$$
\sigma^2_X = E[X^2] - \mu_X^2 = \int_{-\infty}^\infty x^2p_X(x) \; dx - \left(\int_{-\infty}^\infty xp_X(x) \; dx\right)^2.
$$

と書ける。

平均、分散、標準偏差について上で述べたことは、この場合にもすべて当てはまりる。たとえば、密度

$$
p(x) = \begin{cases}
1 & x \in [0,1], \\
0 & \textrm{otherwise}.
\end{cases}
$$

をもつ確率変数を考えると、

$$
\mu_X = \int_{-\infty}^\infty xp(x) \; dx = \int_0^1 x \; dx = \frac{1}{2}.
$$

また

$$
\sigma_X^2 = \int_{-\infty}^\infty x^2p(x) \; dx - \left(\frac{1}{2}\right)^2 = \frac{1}{3} - \frac{1}{4} = \frac{1}{12}.
$$

警告として、もう1つの例、*コーシー分布* を見てみよう。これは p.d.f. が

$$
p(x) = \frac{1}{1+x^2}.
$$

で与えられる分布である。

```{.python .input}
#@tab mxnet
# Plot the Cauchy distribution p.d.f.
x = np.arange(-5, 5, 0.01)
p = 1 / (1 + x**2)

d2l.plot(x, p, 'x', 'p.d.f.')
```

```{.python .input}
#@tab pytorch
# Plot the Cauchy distribution p.d.f.
x = torch.arange(-5, 5, 0.01)
p = 1 / (1 + x**2)

d2l.plot(x, p, 'x', 'p.d.f.')
```

```{.python .input}
#@tab tensorflow
# Plot the Cauchy distribution p.d.f.
x = tf.range(-5, 5, 0.01)
p = 1 / (1 + x**2)

d2l.plot(x, p, 'x', 'p.d.f.')
```

この関数は無害に見え、実際、積分表を参照すると下の面積が1であることがわかるので、連続確率変数を定義している。

何がまずいのかを見るために、これの分散を計算してみよう。これは :eqref:`eq_var_def` を使って

$$
\int_{-\infty}^\infty \frac{x^2}{1+x^2}\; dx
$$

を計算することに相当する。

中の関数は次のようになる。

```{.python .input}
#@tab mxnet
# Plot the integrand needed to compute the variance
x = np.arange(-20, 20, 0.01)
p = x**2 / (1 + x**2)

d2l.plot(x, p, 'x', 'integrand')
```

```{.python .input}
#@tab pytorch
# Plot the integrand needed to compute the variance
x = torch.arange(-20, 20, 0.01)
p = x**2 / (1 + x**2)

d2l.plot(x, p, 'x', 'integrand')
```

```{.python .input}
#@tab tensorflow
# Plot the integrand needed to compute the variance
x = tf.range(-20, 20, 0.01)
p = x**2 / (1 + x**2)

d2l.plot(x, p, 'x', 'integrand')
```

この関数は、ほぼ定数1であり、0付近に小さなくぼみがあるだけなので、明らかに下の面積は無限大である。実際、

$$
\int_{-\infty}^\infty \frac{x^2}{1+x^2}\; dx = \infty.
$$

と示せる。つまり、有限で定義された分散を持たない。

しかし、さらに深く見ると、もっと気がかりな結果が出る。:eqref:`eq_exp_def` を使って平均を計算してみよう。変数変換の公式を使うと、

$$
\mu_X = \int_{-\infty}^{\infty} \frac{x}{1+x^2} \; dx = \frac{1}{2}\int_1^\infty \frac{1}{u} \; du.
$$

となる。右辺の積分は対数の定義なので、これは本質的には $\log(\infty) = \infty$ であり、平均値も定義できない。

機械学習の研究者は、通常はこうした問題を扱わなくて済むようにモデルを設計し、ほとんどの場合、平均と分散がきちんと定義された確率変数を扱う。しかし、時折、*heavy tails*（大きな値をとる確率が十分大きく、平均や分散のような量が定義できなくなる確率変数）をもつ確率変数が物理系のモデリングに役立つことがある。そのようなものが存在することは知っておく価値がある。

### 同時密度関数

ここまでの議論は、1つの実数値確率変数を扱う場合を仮定していた。しかし、強く相関した2つ以上の確率変数を扱う場合はどうだろうか。この状況は機械学習では普通である。たとえば、画像の $(i, j)$ 座標にある画素の赤成分を表す $R_{i, j}$ や、時刻 $t$ における株価を表す確率変数 $P_t$ を考えてみよ。近くの画素は似た色を持ち、近い時刻の価格は似た値になりがちである。これらを別々の確率変数として扱い、その上で成功するモデルを作れるとは期待できない（:numref:`sec_naive_bayes` では、そのような仮定のために性能が劣るモデルを見ることになる）。相関した連続確率変数を扱うための数学的言語を整備する必要がある。

幸い、 :numref:`sec_integral_calculus` の多重積分を使えば、そのような言語を構築できる。簡単のため、相関しうる2つの確率変数 $X, Y$ があるとしよう。このとき、1変数の場合と同様に、次の問いを考えられる。

$$
P(X \;\textrm{is in an}\; \epsilon \textrm{-sized interval around}\; x \; \textrm{and} \;Y \;\textrm{is in an}\; \epsilon \textrm{-sized interval around}\; y ).
$$

1変数の場合と同じ推論により、これはおおよそ

$$
P(X \;\textrm{is in an}\; \epsilon \textrm{-sized interval around}\; x \; \textrm{and} \;Y \;\textrm{is in an}\; \epsilon \textrm{-sized interval around}\; y ) \approx \epsilon^{2}p(x, y),
$$

となるはずである。ここで $p(x, y)$ はある関数である。これは $X$ と $Y$ の同時密度と呼ばれる。1変数の場合と同様に、次の性質が成り立ちる。

* $p(x, y) \ge 0$;
* $\int _ {\mathbb{R}^2} p(x, y) \;dx \;dy = 1$;
* $P((X, Y) \in \mathcal{D}) = \int _ {\mathcal{D}} p(x, y) \;dx \;dy$。

このようにして、複数の、しかも相関しうる確率変数を扱える。2つより多くの確率変数を扱いたい場合は、$p(\mathbf{x}) = p(x_1, \ldots, x_n)$ と考えることで、多変量密度を任意の次元に拡張できる。非負であり、全積分が1であるという性質は同様に成り立ちる。

### 周辺分布
複数の変数を扱うとき、関係を無視して「この1つの変数はどのように分布しているのか」と知りたいことがよくある。このような分布を *周辺分布* と呼ぶ。

具体的に、同時密度が $p _ {X, Y}(x, y)$ で与えられる2つの確率変数 $X, Y$ があるとしよう。ここでは、密度がどの確率変数に対するものかを示すために添字を使う。周辺分布を求めるとは、この関数から $p _ X(x)$ を求めることである。

多くのことと同様に、何が成り立つべきかを考えるには直感的な図に戻るのが最善である。密度とは

$$
P(X \in [x, x+\epsilon]) \approx \epsilon \cdot p _ X(x).
$$

を満たす関数 $p _ X$ だったことを思い出する。

ここには $Y$ の記述がないが、与えられているのが $p _{X, Y}$ だけなら、何らかの形で $Y$ を含める必要がある。まず、これは次と同じだとわかる。

$$
P(X \in [x, x+\epsilon] \textrm{, and } Y \in \mathbb{R}) \approx \epsilon \cdot p _ X(x).
$$

この場合については密度から直接はわからないので、$y$ 方向にも小区間に分割する必要がある。すると、次のように書ける。

$$
\begin{aligned}
\epsilon \cdot p _ X(x) & \approx \sum _ {i} P(X \in [x, x+\epsilon] \textrm{, and } Y \in [\epsilon \cdot i, \epsilon \cdot (i+1)]) \\
& \approx \sum _ {i} \epsilon^{2} p _ {X, Y}(x, \epsilon\cdot i).
\end{aligned}
$$

![確率の配列の列に沿って和をとることで、$\mathit{x}$ 軸に対応する確率変数だけの周辺分布を得ることができる。](../img/marginal.svg)
:label:`fig_marginal`

これは、 :numref:`fig_marginal` に示すように、密度の値を一列に並んだ正方形に沿って足し合わせることを意味する。実際、両辺から $\epsilon$ の因子を1つ打ち消し、右辺の和が $y$ に関する積分であることを認識すると、

$$
\begin{aligned}
 p _ X(x) &  \approx \sum _ {i} \epsilon p _ {X, Y}(x, \epsilon\cdot i) \\
 & \approx \int_{-\infty}^\infty p_{X, Y}(x, y) \; dy.
\end{aligned}
$$

と結論できる。

したがって、

$$
p _ X(x) = \int_{-\infty}^\infty p_{X, Y}(x, y) \; dy.
$$

となる。つまり、周辺分布を得るには、関心のない変数について積分すればよいのである。この過程は、しばしば不要な変数を *積分消去する*、あるいは *周辺化する* と呼ばれる。

### 共分散

複数の確率変数を扱うとき、知っておくと役立つもう1つの要約統計量が *共分散* である。これは、2つの確率変数がどの程度一緒に変動するかを測りる。

2つの確率変数 $X$ と $Y$ があるとする。まず、それらが離散的で、値 $(x_i, y_j)$ を確率 $p_{ij}$ でとるとしよう。この場合、共分散は

$$\sigma_{XY} = \textrm{Cov}(X, Y) = \sum_{i, j} (x_i - \mu_X) (y_j-\mu_Y) p_{ij}. = E[XY] - E[X]E[Y].$$
:eqlabel:`eq_cov_def`

と定義される。

直感的に考えるために、次の確率変数の組を見てみよう。$X$ は値 $1$ と $3$ をとり、$Y$ は値 $-1$ と $3$ をとるとする。次の確率を持つとしよう。

$$
\begin{aligned}
P(X = 1 \; \textrm{and} \; Y = -1) & = \frac{p}{2}, \\
P(X = 1 \; \textrm{and} \; Y = 3) & = \frac{1-p}{2}, \\
P(X = 3 \; \textrm{and} \; Y = -1) & = \frac{1-p}{2}, \\
P(X = 3 \; \textrm{and} \; Y = 3) & = \frac{p}{2},
\end{aligned}
$$

ここで $p$ は $[0,1]$ の範囲で選べるパラメータである。$p=1$ なら、両者は常に同時に最小値または最大値をとり、$p=0$ なら、同時に反転した値をとることが保証される（片方が大きいとき他方は小さく、その逆も同様です）。$p=1/2$ なら4つの可能性はすべて同確率で、互いに関係がないはずである。共分散を計算してみよう。まず、$\mu_X = 2$、$\mu_Y = 1$ なので、:eqref:`eq_cov_def` を使って

$$
\begin{aligned}
\textrm{Cov}(X, Y) & = \sum_{i, j} (x_i - \mu_X) (y_j-\mu_Y) p_{ij} \\
& = (1-2)(-1-1)\frac{p}{2} + (1-2)(3-1)\frac{1-p}{2} + (3-2)(-1-1)\frac{1-p}{2} + (3-2)(3-1)\frac{p}{2} \\
& = 4p-2.
\end{aligned}
$$

$p=1$（両者が同時に最大限正または負になる場合）では共分散は $2$ である。$p=0$（反転している場合）では共分散は $-2$ である。最後に、$p=1/2$（互いに無関係な場合）では共分散は $0$ である。したがって、共分散はこれら2つの確率変数がどのように関係しているかを測ることがわかる。

共分散についての簡単な注意として、これは線形関係しか測らないという点がある。$Y$ が $\{-2, -1, 0, 1, 2\}$ から等確率で選ばれ、$X = Y^2$ のようなより複雑な関係は見逃されることがある。実際、簡単な計算により、一方が他方の決定論的関数であるにもかかわらず、これらの確率変数の共分散は0になる。

連続確率変数についても、話はほぼ同じである。ここまでで離散と連続の移行にはかなり慣れてきたので、導出なしで :eqref:`eq_cov_def` の連続版を示す。

$$
\sigma_{XY} = \int_{\mathbb{R}^2} (x-\mu_X)(y-\mu_Y)p(x, y) \;dx \;dy.
$$

可視化のために、共分散を調整できる確率変数の集まりを見てみよう。

```{.python .input}
#@tab mxnet
# Plot a few random variables adjustable covariance
covs = [-0.9, 0.0, 1.2]
d2l.plt.figure(figsize=(12, 3))
for i in range(3):
    X = np.random.normal(0, 1, 500)
    Y = covs[i]*X + np.random.normal(0, 1, (500))

    d2l.plt.subplot(1, 4, i+1)
    d2l.plt.scatter(X.asnumpy(), Y.asnumpy())
    d2l.plt.xlabel('X')
    d2l.plt.ylabel('Y')
    d2l.plt.title(f'cov = {covs[i]}')
d2l.plt.show()
```

```{.python .input}
#@tab pytorch
# Plot a few random variables adjustable covariance
covs = [-0.9, 0.0, 1.2]
d2l.plt.figure(figsize=(12, 3))
for i in range(3):
    X = torch.randn(500)
    Y = covs[i]*X + torch.randn(500)

    d2l.plt.subplot(1, 4, i+1)
    d2l.plt.scatter(X.numpy(), Y.numpy())
    d2l.plt.xlabel('X')
    d2l.plt.ylabel('Y')
    d2l.plt.title(f'cov = {covs[i]}')
d2l.plt.show()
```

```{.python .input}
#@tab tensorflow
# Plot a few random variables adjustable covariance
covs = [-0.9, 0.0, 1.2]
d2l.plt.figure(figsize=(12, 3))
for i in range(3):
    X = tf.random.normal((500, ))
    Y = covs[i]*X + tf.random.normal((500, ))

    d2l.plt.subplot(1, 4, i+1)
    d2l.plt.scatter(X.numpy(), Y.numpy())
    d2l.plt.xlabel('X')
    d2l.plt.ylabel('Y')
    d2l.plt.title(f'cov = {covs[i]}')
d2l.plt.show()
```

共分散の性質をいくつか見てみよう。

* 任意の確率変数 $X$ に対して、$\textrm{Cov}(X, X) = \textrm{Var}(X)$。
* 任意の確率変数 $X, Y$ と数 $a, b$ に対して、$\textrm{Cov}(aX+b, Y) = \textrm{Cov}(X, aY+b) = a\textrm{Cov}(X, Y)$。
* $X$ と $Y$ が独立なら、$\textrm{Cov}(X, Y) = 0$。

さらに、共分散を使うと、先ほど見た関係を拡張できる。$X$ と $Y$ が2つの独立な確率変数なら、

$$
\textrm{Var}(X+Y) = \textrm{Var}(X) + \textrm{Var}(Y).
$$

でした。共分散の知識を使うと、この関係を一般化できる。実際、少し代数計算をすると、一般には

$$
\textrm{Var}(X+Y) = \textrm{Var}(X) + \textrm{Var}(Y) + 2\textrm{Cov}(X, Y).
$$

となる。これにより、相関した確率変数に対する分散の加法則を一般化できる。

### 相関

平均と分散の場合と同様に、単位について考えてみよう。$X$ がある単位（たとえばインチ）で測られ、$Y$ が別の単位（たとえばドル）で測られるなら、共分散の単位はこれら2つの単位の積 $\textrm{inches} \times \textrm{dollars}$ になる。この単位は解釈しにくいことがある。この場合にしばしば欲しいのは、単位に依存しない関係の尺度である。実際、厳密な定量的相関よりも、相関が同じ方向かどうか、そして関係がどれだけ強いかを知りたいことが多いのである。

何が自然かを見るために、思考実験をしてみよう。インチとドルで測っていた確率変数を、インチとセントで測るように変換するとする。この場合、確率変数 $Y$ は $100$ 倍される。定義をたどると、$\textrm{Cov}(X, Y)$ も $100$ 倍されることがわかる。つまり、単位を変えると共分散は $100$ 倍変わる。したがって、単位不変な相関の尺度を得るには、同じく $100$ 倍にスケールされる別の量で割る必要がある。そこで有力な候補が標準偏差である。実際、*相関係数* を

$$\rho(X, Y) = \frac{\textrm{Cov}(X, Y)}{\sigma_{X}\sigma_{Y}},$$
:eqlabel:`eq_cor_def`

と定義すると、これは単位のない値になる。少し数学を使えば、この数は $-1$ から $1$ の間にあり、$1$ は最大の正の相関、$-1$ は最大の負の相関を意味することが示せる。

先ほどの具体的な離散例に戻ると、$\sigma_X = 1$、$\sigma_Y = 2$ なので、:eqref:`eq_cor_def` を使って2つの確率変数の相関を計算すると

$$
\rho(X, Y) = \frac{4p-2}{1\cdot 2} = 2p-1.
$$

となる。これは $-1$ から $1$ の範囲をとり、$1$ が最も相関が強く、$-1$ が最も相関が弱いという期待通りの振る舞いをする。

別の例として、$X$ を任意の確率変数、$Y=aX+b$ を $X$ の任意の線形決定論的関数とする。このとき、

$$\sigma_{Y} = \sigma_{aX+b} = |a|\sigma_{X},$$

$$\textrm{Cov}(X, Y) = \textrm{Cov}(X, aX+b) = a\textrm{Cov}(X, X) = a\textrm{Var}(X),$$

であり、したがって :eqref:`eq_cor_def` より

$$
\rho(X, Y) = \frac{a\textrm{Var}(X)}{|a|\sigma_{X}^2} = \frac{a}{|a|} = \textrm{sign}(a).
$$

となる。したがって、$a > 0$ なら相関は $+1$、$a < 0$ なら $-1$ である。これは、相関が2つの確率変数の関係の程度と方向を測るのであって、変動のスケールを測るのではないことを示している。

再び、調整可能な相関をもつ確率変数の集まりを描いてみよう。

```{.python .input}
#@tab mxnet
# Plot a few random variables adjustable correlations
cors = [-0.9, 0.0, 1.0]
d2l.plt.figure(figsize=(12, 3))
for i in range(3):
    X = np.random.normal(0, 1, 500)
    Y = cors[i] * X + np.sqrt(1 - cors[i]**2) * np.random.normal(0, 1, 500)

    d2l.plt.subplot(1, 4, i + 1)
    d2l.plt.scatter(X.asnumpy(), Y.asnumpy())
    d2l.plt.xlabel('X')
    d2l.plt.ylabel('Y')
    d2l.plt.title(f'cor = {cors[i]}')
d2l.plt.show()
```

```{.python .input}
#@tab pytorch
# Plot a few random variables adjustable correlations
cors = [-0.9, 0.0, 1.0]
d2l.plt.figure(figsize=(12, 3))
for i in range(3):
    X = torch.randn(500)
    Y = cors[i] * X + torch.sqrt(torch.tensor(1) -
                                 cors[i]**2) * torch.randn(500)

    d2l.plt.subplot(1, 4, i + 1)
    d2l.plt.scatter(X.numpy(), Y.numpy())
    d2l.plt.xlabel('X')
    d2l.plt.ylabel('Y')
    d2l.plt.title(f'cor = {cors[i]}')
d2l.plt.show()
```

```{.python .input}
#@tab tensorflow
# Plot a few random variables adjustable correlations
cors = [-0.9, 0.0, 1.0]
d2l.plt.figure(figsize=(12, 3))
for i in range(3):
    X = tf.random.normal((500, ))
    Y = cors[i] * X + tf.sqrt(tf.constant(1.) -
                                 cors[i]**2) * tf.random.normal((500, ))

    d2l.plt.subplot(1, 4, i + 1)
    d2l.plt.scatter(X.numpy(), Y.numpy())
    d2l.plt.xlabel('X')
    d2l.plt.ylabel('Y')
    d2l.plt.title(f'cor = {cors[i]}')
d2l.plt.show()
```

相関の性質をいくつか挙げておきる。

* 任意の確率変数 $X$ に対して、$\rho(X, X) = 1$。
* 任意の確率変数 $X, Y$ と数 $a, b$ に対して、$\rho(aX+b, Y) = \rho(X, aY+b) = \rho(X, Y)$。
* $X$ と $Y$ が独立で、かつ分散が0でないなら、$\rho(X, Y) = 0$。

最後に、いくつかの式が見覚えがあるように感じるかもしれない。実際、$\mu_X = \mu_Y = 0$ と仮定してすべて展開すると、これは

$$
\rho(X, Y) = \frac{\sum_{i, j} x_iy_ip_{ij}}{\sqrt{\sum_{i, j}x_i^2 p_{ij}}\sqrt{\sum_{i, j}y_j^2 p_{ij}}}.
$$

となる。これは、項の積の和を、項の和の平方根で割った形である。これは、確率 $p_{ij}$ で各座標が重み付けされた2つのベクトル $\mathbf{v}, \mathbf{w}$ のなす角の余弦の公式そのものである。

$$
\cos(\theta) = \frac{\mathbf{v}\cdot \mathbf{w}}{\|\mathbf{v}\|\|\mathbf{w}\|} = \frac{\sum_{i} v_iw_i}{\sqrt{\sum_{i}v_i^2}\sqrt{\sum_{i}w_i^2}}.
$$

実際、ノルムを標準偏差に、相関を角度の余弦に対応するものと考えれば、幾何学から得られる直感の多くを確率変数の理解に応用できる。

## まとめ
* 連続確率変数は、連続体の値をとりうる確率変数である。離散確率変数に比べて扱いが難しい技術的問題がある。
* 確率密度関数を使うと、ある区間で曲線の下の面積がその区間に標本点が入る確率を与える関数として、連続確率変数を扱える。
* 累積分布関数は、確率変数があるしきい値以下である確率である。離散変数と連続変数を統一的に扱える有用な別視点を与える。
* 平均は確率変数の平均値である。
* 分散は、確率変数とその平均の差の二乗の期待値である。
* 標準偏差は分散の平方根である。確率変数がとりうる値の範囲を測るものと考えられる。
* チェビシェフの不等式により、確率変数が大抵の場合に含まれる明示的な区間を与えることで、この直感を厳密にできる。
* 同時密度を使うと、相関した確率変数を扱える。不要な確率変数について積分することで同時密度を周辺化し、目的の確率変数の分布を得られる。
* 共分散と相関係数は、2つの相関した確率変数の線形関係を測る方法を与える。

## 演習
1. 密度が $p(x) = \frac{1}{x^2}$ for $x \ge 1$ であり、それ以外では $p(x) = 0$ である確率変数を考える。$P(X > 2)$ はいくらか。
2. ラプラス分布は、密度が $p(x = \frac{1}{2}e^{-|x|}$ で与えられる確率変数である。この関数の平均と標準偏差はいくらか。ヒントとして、$\int_0^\infty xe^{-x} \; dx = 1$ と $\int_0^\infty x^2e^{-x} \; dx = 2$ を用いてよい。
3. 私が街であなたに近づいてきて、「平均が $1$、標準偏差が $2$ の確率変数を持っていて、標本の $25\%$ が $9$ より大きい値をとるのを観測した」と言ったとする。あなたは私の話を信じるだろうか。なぜそう思い、なぜそう思わないのだろうか。
4. 2つの確率変数 $X, Y$ があり、同時密度が $p_{XY}(x, y) = 4xy$ for $x, y \in [0,1]$、それ以外では $p_{XY}(x, y) = 0$ で与えられているとする。$X$ と $Y$ の共分散はいくらか。
