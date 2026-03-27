# 分布
:label:`sec_distributions`

これまで、離散的な場合と連続的な場合の両方で確率を扱う方法を学んできた。ここでは、よく出会う代表的な分布について見ていきましょう。機械学習の分野によっては、これらよりはるかに多くの分布に精通している必要があるかもしれませんし、深層学習のある分野ではまったく不要なこともある。しかし、基本的な一覧として知っておくのは有益である。まず、いくつかの共通ライブラリをインポートしよう。

```{.python .input}
#@tab mxnet
%matplotlib inline
from d2l import mxnet as d2l
from IPython import display
from math import erf, factorial
import numpy as np
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
from IPython import display
from math import erf, factorial
import torch

torch.pi = torch.acos(torch.zeros(1)) * 2  # Define pi in torch
```

```{.python .input}
#@tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
from IPython import display
from math import erf, factorial
import tensorflow as tf
import tensorflow_probability as tfp

tf.pi = tf.acos(tf.zeros(1)) * 2  # Define pi in TensorFlow
```

## ベルヌーイ分布

これは通常最初に出会う最も単純な確率変数である。この確率変数は、コイン投げを表し、$1$ が確率 $p$、$0$ が確率 $1-p$ で出ることを表する。この分布に従う確率変数 $X$ は

$$
X \sim \textrm{Bernoulli}(p).
$$

と書く。

累積分布関数は

$$F(x) = \begin{cases} 0 & x < 0, \\ 1-p & 0 \le x < 1, \\ 1 & x >= 1 . \end{cases}$$
:eqlabel:`eq_bernoulli_cdf`

である。

確率質量関数を以下に描く。

```{.python .input}
#@tab all
p = 0.3

d2l.set_figsize()
d2l.plt.stem([0, 1], [1 - p, p], use_line_collection=True)
d2l.plt.xlabel('x')
d2l.plt.ylabel('p.m.f.')
d2l.plt.show()
```

では、累積分布関数 :eqref:`eq_bernoulli_cdf` を描いてみよう。

```{.python .input}
#@tab mxnet
x = np.arange(-1, 2, 0.01)

def F(x):
    return 0 if x < 0 else 1 if x > 1 else 1 - p

d2l.plot(x, np.array([F(y) for y in x]), 'x', 'c.d.f.')
```

```{.python .input}
#@tab pytorch
x = torch.arange(-1, 2, 0.01)

def F(x):
    return 0 if x < 0 else 1 if x > 1 else 1 - p

d2l.plot(x, torch.tensor([F(y) for y in x]), 'x', 'c.d.f.')
```

```{.python .input}
#@tab tensorflow
x = tf.range(-1, 2, 0.01)

def F(x):
    return 0 if x < 0 else 1 if x > 1 else 1 - p

d2l.plot(x, tf.constant([F(y) for y in x]), 'x', 'c.d.f.')
```

$X \sim \textrm{Bernoulli}(p)$ ならば、

* $\mu_X = p$、
* $\sigma_X^2 = p(1-p)$。

ベルヌーイ確率変数から任意の形状の配列をサンプルするには、次のようにする。

```{.python .input}
#@tab mxnet
1*(np.random.rand(10, 10) < p)
```

```{.python .input}
#@tab pytorch
1*(torch.rand(10, 10) < p)
```

```{.python .input}
#@tab tensorflow
tf.cast(tf.random.uniform((10, 10)) < p, dtype=tf.float32)
```

## 離散一様分布

次によく現れる確率変数は離散一様分布である。ここでは、$\{1, 2, \ldots, n\}$ の整数上に台を持つものを考えますが、他の任意の値集合を自由に選んでも構いない。この文脈で *一様* というのは、取りうるすべての値が等確率であることを意味する。各値 $i \in \{1, 2, 3, \ldots, n\}$ の確率は $p_i = \frac{1}{n}$ である。この分布に従う確率変数 $X$ は

$$
X \sim U(n).
$$

と表する。

累積分布関数は

$$F(x) = \begin{cases} 0 & x < 1, \\ \frac{k}{n} & k \le x < k+1 \textrm{ with } 1 \le k < n, \\ 1 & x >= n . \end{cases}$$
:eqlabel:`eq_discrete_uniform_cdf`

である。

まず確率質量関数を描きましょう。

```{.python .input}
#@tab all
n = 5

d2l.plt.stem([i+1 for i in range(n)], n*[1 / n], use_line_collection=True)
d2l.plt.xlabel('x')
d2l.plt.ylabel('p.m.f.')
d2l.plt.show()
```

では、累積分布関数 :eqref:`eq_discrete_uniform_cdf` を描いてみよう。

```{.python .input}
#@tab mxnet
x = np.arange(-1, 6, 0.01)

def F(x):
    return 0 if x < 1 else 1 if x > n else np.floor(x) / n

d2l.plot(x, np.array([F(y) for y in x]), 'x', 'c.d.f.')
```

```{.python .input}
#@tab pytorch
x = torch.arange(-1, 6, 0.01)

def F(x):
    return 0 if x < 1 else 1 if x > n else torch.floor(x) / n

d2l.plot(x, torch.tensor([F(y) for y in x]), 'x', 'c.d.f.')
```

```{.python .input}
#@tab tensorflow
x = tf.range(-1, 6, 0.01)

def F(x):
    return 0 if x < 1 else 1 if x > n else tf.floor(x) / n

d2l.plot(x, [F(y) for y in x], 'x', 'c.d.f.')
```

$X \sim U(n)$ ならば、

* $\mu_X = \frac{1+n}{2}$、
* $\sigma_X^2 = \frac{n^2-1}{12}$。

離散一様確率変数から任意の形状の配列をサンプルするには、次のようにする。

```{.python .input}
#@tab mxnet
np.random.randint(1, n, size=(10, 10))
```

```{.python .input}
#@tab pytorch
torch.randint(1, n, size=(10, 10))
```

```{.python .input}
#@tab tensorflow
tf.random.uniform((10, 10), 1, n, dtype=tf.int32)
```

## 連続一様分布

次に、連続一様分布について説明しよう。この確率変数の考え方は、離散一様分布の $n$ を増やし、それを区間 $[a, b]$ に収まるようにスケーリングすると、$[a, b]$ の中の任意の値をすべて等確率で選ぶ連続確率変数に近づく、というものである。この分布は

$$
X \sim U(a, b).
$$

と表する。

確率密度関数は

$$p(x) = \begin{cases} \frac{1}{b-a} & x \in [a, b], \\ 0 & x \not\in [a, b].\end{cases}$$
:eqlabel:`eq_cont_uniform_pdf`

である。

累積分布関数は

$$F(x) = \begin{cases} 0 & x < a, \\ \frac{x-a}{b-a} & x \in [a, b], \\ 1 & x >= b . \end{cases}$$
:eqlabel:`eq_cont_uniform_cdf`

である。

まず確率密度関数 :eqref:`eq_cont_uniform_pdf` を描きましょう。

```{.python .input}
#@tab mxnet
a, b = 1, 3

x = np.arange(0, 4, 0.01)
p = (x > a)*(x < b)/(b - a)

d2l.plot(x, p, 'x', 'p.d.f.')
```

```{.python .input}
#@tab pytorch
a, b = 1, 3

x = torch.arange(0, 4, 0.01)
p = (x > a).type(torch.float32)*(x < b).type(torch.float32)/(b-a)
d2l.plot(x, p, 'x', 'p.d.f.')
```

```{.python .input}
#@tab tensorflow
a, b = 1, 3

x = tf.range(0, 4, 0.01)
p = tf.cast(x > a, tf.float32) * tf.cast(x < b, tf.float32) / (b - a)
d2l.plot(x, p, 'x', 'p.d.f.')
```

では、累積分布関数 :eqref:`eq_cont_uniform_cdf` を描いてみよう。

```{.python .input}
#@tab mxnet
def F(x):
    return 0 if x < a else 1 if x > b else (x - a) / (b - a)

d2l.plot(x, np.array([F(y) for y in x]), 'x', 'c.d.f.')
```

```{.python .input}
#@tab pytorch
def F(x):
    return 0 if x < a else 1 if x > b else (x - a) / (b - a)

d2l.plot(x, torch.tensor([F(y) for y in x]), 'x', 'c.d.f.')
```

```{.python .input}
#@tab tensorflow
def F(x):
    return 0 if x < a else 1 if x > b else (x - a) / (b - a)

d2l.plot(x, [F(y) for y in x], 'x', 'c.d.f.')
```

$X \sim U(a, b)$ ならば、

* $\mu_X = \frac{a+b}{2}$、
* $\sigma_X^2 = \frac{(b-a)^2}{12}$。

一様確率変数から任意の形状の配列をサンプルするには、次のようにする。デフォルトでは $U(0,1)$ からサンプルされるので、別の範囲が欲しい場合はスケーリングする必要がある。

```{.python .input}
#@tab mxnet
(b - a) * np.random.rand(10, 10) + a
```

```{.python .input}
#@tab pytorch
(b - a) * torch.rand(10, 10) + a
```

```{.python .input}
#@tab tensorflow
(b - a) * tf.random.uniform((10, 10)) + a
```

## 二項分布

少し複雑にして、*二項* 確率変数を見てみよう。この確率変数は、成功確率 $p$ の独立な試行を $n$ 回行い、成功が何回起こるかを数えることから生まれる。

これを数学的に表しよう。各試行は独立な確率変数 $X_i$ であり、成功を $1$、失敗を $0$ で表する。それぞれが成功確率 $p$ の独立なコイン投げなので、$X_i \sim \textrm{Bernoulli}(p)$ と書ける。すると、二項確率変数は

$$
X = \sum_{i=1}^n X_i.
$$

である。

このとき、

$$
X \sim \textrm{Binomial}(n, p).
$$

と書く。

累積分布関数を得るには、ちょうど $k$ 回成功する場合が $\binom{n}{k} = \frac{n!}{k!(n-k)!}$ 通りあり、それぞれが確率 $p^k(1-p)^{n-k}$ で起こることに注意する必要がある。したがって、累積分布関数は

$$F(x) = \begin{cases} 0 & x < 0, \\ \sum_{m \le k} \binom{n}{m} p^m(1-p)^{n-m}  & k \le x < k+1 \textrm{ with } 0 \le k < n, \\ 1 & x >= n . \end{cases}$$
:eqlabel:`eq_binomial_cdf`

である。

まず確率質量関数を描きましょう。

```{.python .input}
#@tab mxnet
n, p = 10, 0.2

# Compute binomial coefficient
def binom(n, k):
    comb = 1
    for i in range(min(k, n - k)):
        comb = comb * (n - i) // (i + 1)
    return comb

pmf = np.array([p**i * (1-p)**(n - i) * binom(n, i) for i in range(n + 1)])

d2l.plt.stem([i for i in range(n + 1)], pmf, use_line_collection=True)
d2l.plt.xlabel('x')
d2l.plt.ylabel('p.m.f.')
d2l.plt.show()
```

```{.python .input}
#@tab pytorch
n, p = 10, 0.2

# Compute binomial coefficient
def binom(n, k):
    comb = 1
    for i in range(min(k, n - k)):
        comb = comb * (n - i) // (i + 1)
    return comb

pmf = d2l.tensor([p**i * (1-p)**(n - i) * binom(n, i) for i in range(n + 1)])

d2l.plt.stem([i for i in range(n + 1)], pmf, use_line_collection=True)
d2l.plt.xlabel('x')
d2l.plt.ylabel('p.m.f.')
d2l.plt.show()
```

```{.python .input}
#@tab tensorflow
n, p = 10, 0.2

# Compute binomial coefficient
def binom(n, k):
    comb = 1
    for i in range(min(k, n - k)):
        comb = comb * (n - i) // (i + 1)
    return comb

pmf = tf.constant([p**i * (1-p)**(n - i) * binom(n, i) for i in range(n + 1)])

d2l.plt.stem([i for i in range(n + 1)], pmf, use_line_collection=True)
d2l.plt.xlabel('x')
d2l.plt.ylabel('p.m.f.')
d2l.plt.show()
```

では、累積分布関数 :eqref:`eq_binomial_cdf` を描いてみよう。

```{.python .input}
#@tab mxnet
x = np.arange(-1, 11, 0.01)
cmf = np.cumsum(pmf)

def F(x):
    return 0 if x < 0 else 1 if x > n else cmf[int(x)]

d2l.plot(x, np.array([F(y) for y in x.tolist()]), 'x', 'c.d.f.')
```

```{.python .input}
#@tab pytorch
x = torch.arange(-1, 11, 0.01)
cmf = torch.cumsum(pmf, dim=0)

def F(x):
    return 0 if x < 0 else 1 if x > n else cmf[int(x)]

d2l.plot(x, torch.tensor([F(y) for y in x.tolist()]), 'x', 'c.d.f.')
```

```{.python .input}
#@tab tensorflow
x = tf.range(-1, 11, 0.01)
cmf = tf.cumsum(pmf)

def F(x):
    return 0 if x < 0 else 1 if x > n else cmf[int(x)]

d2l.plot(x, [F(y) for y in x.numpy().tolist()], 'x', 'c.d.f.')
```

$X \sim \textrm{Binomial}(n, p)$ ならば、

* $\mu_X = np$、
* $\sigma_X^2 = np(1-p)$。

これは、$n$ 個のベルヌーイ確率変数の和に対する期待値の線形性と、独立な確率変数の和の分散が分散の和になることから従う。これは次のようにサンプルできる。

```{.python .input}
#@tab mxnet
np.random.binomial(n, p, size=(10, 10))
```

```{.python .input}
#@tab pytorch
m = torch.distributions.binomial.Binomial(n, p)
m.sample(sample_shape=(10, 10))
```

```{.python .input}
#@tab tensorflow
m = tfp.distributions.Binomial(n, p)
m.sample(sample_shape=(10, 10))
```

## ポアソン分布
では、思考実験をしてみよう。私たちはバス停に立っていて、次の1分間に何台のバスが到着するかを知りたいとする。まず、$X^{(1)} \sim \textrm{Bernoulli}(p)$ を考える。これは、1分間の窓の中でバスが到着する確率そのものである。都市中心部から離れたバス停では、これはかなり良い近似かもしれない。1分間に2台以上のバスを見ることはまずないでしょう。

しかし、混雑した地域にいるなら、2台のバスが到着することもありえますし、むしろ起こりやすいかもしれない。これを、最初の30秒と次の30秒に分けて確率変数を分割することでモデル化できる。この場合、

$$
X^{(2)} \sim X^{(2)}_1 + X^{(2)}_2,
$$

と書ける。ここで $X^{(2)}$ は合計であり、$X^{(2)}_i \sim \textrm{Bernoulli}(p/2)$ である。すると全体の分布は $X^{(2)} \sim \textrm{Binomial}(2, p/2)$ になる。

ここで止める必要はあるだろうか。さらにその1分を $n$ 個に分割してみよう。上と同じ推論により、

$$X^{(n)} \sim \textrm{Binomial}(n, p/n).$$
:eqlabel:`eq_eq_poisson_approx`

が得られる。

これらの確率変数を考える。前節より、:eqref:`eq_eq_poisson_approx` の平均は $\mu_{X^{(n)}} = n(p/n) = p$、分散は $\sigma_{X^{(n)}}^2 = n(p/n)(1-(p/n)) = p(1-p/n)$ である。$n \rightarrow \infty$ とすると、これらの値は $\mu_{X^{(\infty)}} = p$、分散 $\sigma_{X^{(\infty)}}^2 = p$ に安定することがわかる。これは、この無限分割の極限で定義できる確率変数が *存在するかもしれない* ことを示している。

現実世界ではバスの到着回数を数えればよいだけなので、これはそれほど驚くことではありませんが、数学的モデルがきちんと定義されていることを見るのは良いことである。この議論は *稀な事象の法則* として形式化できる。

この推論を丁寧にたどると、次のモデルに到達できる。$X \sim \textrm{Poisson}(\lambda)$ とは、$\{0,1,2, \ldots\}$ の値を確率

$$p_k = \frac{\lambda^ke^{-\lambda}}{k!}.$$
:eqlabel:`eq_poisson_mass`

でとる確率変数のことである。

$\lambda > 0$ は *率*（あるいは *形状* パラメータ）と呼ばれ、1単位時間あたりに期待される到着回数を表する。

この確率質量関数を和をとって累積分布関数を得ることができる。

$$F(x) = \begin{cases} 0 & x < 0, \\ e^{-\lambda}\sum_{m = 0}^k \frac{\lambda^m}{m!} & k \le x < k+1 \textrm{ with } 0 \le k. \end{cases}$$
:eqlabel:`eq_poisson_cdf`

まず確率質量関数 :eqref:`eq_poisson_mass` を描きましょう。

```{.python .input}
#@tab mxnet
lam = 5.0

xs = [i for i in range(20)]
pmf = np.array([np.exp(-lam) * lam**k / factorial(k) for k in xs])

d2l.plt.stem(xs, pmf, use_line_collection=True)
d2l.plt.xlabel('x')
d2l.plt.ylabel('p.m.f.')
d2l.plt.show()
```

```{.python .input}
#@tab pytorch
lam = 5.0

xs = [i for i in range(20)]
pmf = torch.tensor([torch.exp(torch.tensor(-lam)) * lam**k
                    / factorial(k) for k in xs])

d2l.plt.stem(xs, pmf, use_line_collection=True)
d2l.plt.xlabel('x')
d2l.plt.ylabel('p.m.f.')
d2l.plt.show()
```

```{.python .input}
#@tab tensorflow
lam = 5.0

xs = [i for i in range(20)]
pmf = tf.constant([tf.exp(tf.constant(-lam)).numpy() * lam**k
                    / factorial(k) for k in xs])

d2l.plt.stem(xs, pmf, use_line_collection=True)
d2l.plt.xlabel('x')
d2l.plt.ylabel('p.m.f.')
d2l.plt.show()
```

では、累積分布関数 :eqref:`eq_poisson_cdf` を描いてみよう。

```{.python .input}
#@tab mxnet
x = np.arange(-1, 21, 0.01)
cmf = np.cumsum(pmf)
def F(x):
    return 0 if x < 0 else 1 if x > n else cmf[int(x)]

d2l.plot(x, np.array([F(y) for y in x.tolist()]), 'x', 'c.d.f.')
```

```{.python .input}
#@tab pytorch
x = torch.arange(-1, 21, 0.01)
cmf = torch.cumsum(pmf, dim=0)
def F(x):
    return 0 if x < 0 else 1 if x > n else cmf[int(x)]

d2l.plot(x, torch.tensor([F(y) for y in x.tolist()]), 'x', 'c.d.f.')
```

```{.python .input}
#@tab tensorflow
x = tf.range(-1, 21, 0.01)
cmf = tf.cumsum(pmf)
def F(x):
    return 0 if x < 0 else 1 if x > n else cmf[int(x)]

d2l.plot(x, [F(y) for y in x.numpy().tolist()], 'x', 'c.d.f.')
```

上で見たように、平均と分散は特に簡潔である。$X \sim \textrm{Poisson}(\lambda)$ ならば、

* $\mu_X = \lambda$、
* $\sigma_X^2 = \lambda$。

これは次のようにサンプルできる。

```{.python .input}
#@tab mxnet
np.random.poisson(lam, size=(10, 10))
```

```{.python .input}
#@tab pytorch
m = torch.distributions.poisson.Poisson(lam)
m.sample((10, 10))
```

```{.python .input}
#@tab tensorflow
m = tfp.distributions.Poisson(lam)
m.sample((10, 10))
```

## ガウス分布
では、別の、しかし関連した実験をしてみよう。ここでも、$n$ 個の独立な $\textrm{Bernoulli}(p)$ 測定 $X_i$ を行うとする。これらの和の分布は $X^{(n)} \sim \textrm{Binomial}(n, p)$ である。$n$ を増やし $p$ を減らす極限をとる代わりに、$p$ を固定して $n \rightarrow \infty$ としてみよう。この場合、$\mu_{X^{(n)}} = np \rightarrow \infty$ かつ $\sigma_{X^{(n)}}^2 = np(1-p) \rightarrow \infty$ となるので、この極限がうまく定義されるとは考えにくいである。

しかし、希望はまだある。平均と分散がうまく振る舞うように、次を定義しよう。

$$
Y^{(n)} = \frac{X^{(n)} - \mu_{X^{(n)}}}{\sigma_{X^{(n)}}}.
$$

これは平均0、分散1を持つことがわかるので、何らかの極限分布に収束すると考えるのはもっともらしいである。これらの分布がどのように見えるかを描いてみると、さらにそれがうまくいきそうだと確信できるでしょう。

```{.python .input}
#@tab mxnet
p = 0.2
ns = [1, 10, 100, 1000]
d2l.plt.figure(figsize=(10, 3))
for i in range(4):
    n = ns[i]
    pmf = np.array([p**i * (1-p)**(n-i) * binom(n, i) for i in range(n + 1)])
    d2l.plt.subplot(1, 4, i + 1)
    d2l.plt.stem([(i - n*p)/np.sqrt(n*p*(1 - p)) for i in range(n + 1)], pmf,
                 use_line_collection=True)
    d2l.plt.xlim([-4, 4])
    d2l.plt.xlabel('x')
    d2l.plt.ylabel('p.m.f.')
    d2l.plt.title("n = {}".format(n))
d2l.plt.show()
```

```{.python .input}
#@tab pytorch
p = 0.2
ns = [1, 10, 100, 1000]
d2l.plt.figure(figsize=(10, 3))
for i in range(4):
    n = ns[i]
    pmf = torch.tensor([p**i * (1-p)**(n-i) * binom(n, i)
                        for i in range(n + 1)])
    d2l.plt.subplot(1, 4, i + 1)
    d2l.plt.stem([(i - n*p)/torch.sqrt(torch.tensor(n*p*(1 - p)))
                  for i in range(n + 1)], pmf,
                 use_line_collection=True)
    d2l.plt.xlim([-4, 4])
    d2l.plt.xlabel('x')
    d2l.plt.ylabel('p.m.f.')
    d2l.plt.title("n = {}".format(n))
d2l.plt.show()
```

```{.python .input}
#@tab tensorflow
p = 0.2
ns = [1, 10, 100, 1000]
d2l.plt.figure(figsize=(10, 3))
for i in range(4):
    n = ns[i]
    pmf = tf.constant([p**i * (1-p)**(n-i) * binom(n, i)
                        for i in range(n + 1)])
    d2l.plt.subplot(1, 4, i + 1)
    d2l.plt.stem([(i - n*p)/tf.sqrt(tf.constant(n*p*(1 - p)))
                  for i in range(n + 1)], pmf,
                 use_line_collection=True)
    d2l.plt.xlim([-4, 4])
    d2l.plt.xlabel('x')
    d2l.plt.ylabel('p.m.f.')
    d2l.plt.title("n = {}".format(n))
d2l.plt.show()
```

注意すべき点が1つある。ポアソンの場合と比べると、ここでは標準偏差で割っているため、取りうる結果をより狭い領域に押し込めている。これは、この極限がもはや離散的ではなく、連続的になることを示している。

何が起こるかの導出はこの文書の範囲を超えますが、*中心極限定理* によれば、$n \rightarrow \infty$ のとき、これはガウス分布（あるいは正規分布）を与える。より明示的には、任意の $a, b$ に対して

$$
\lim_{n \rightarrow \infty} P(Y^{(n)} \in [a, b]) = P(\mathcal{N}(0,1) \in [a, b]),
$$

が成り立ちる。ここで、確率変数が平均 $\mu$、分散 $\sigma^2$ を持つ正規分布に従うとき、$X \sim \mathcal{N}(\mu, \sigma^2)$ と書き、その密度は

$$p_X(x) = \frac{1}{\sqrt{2\pi\sigma^2}}e^{-\frac{(x-\mu)^2}{2\sigma^2}}.$$
:eqlabel:`eq_gaussian_pdf`

である。

まず確率密度関数 :eqref:`eq_gaussian_pdf` を描きましょう。

```{.python .input}
#@tab mxnet
mu, sigma = 0, 1

x = np.arange(-3, 3, 0.01)
p = 1 / np.sqrt(2 * np.pi * sigma**2) * np.exp(-(x - mu)**2 / (2 * sigma**2))

d2l.plot(x, p, 'x', 'p.d.f.')
```

```{.python .input}
#@tab pytorch
mu, sigma = 0, 1

x = torch.arange(-3, 3, 0.01)
p = 1 / torch.sqrt(2 * torch.pi * sigma**2) * torch.exp(
    -(x - mu)**2 / (2 * sigma**2))

d2l.plot(x, p, 'x', 'p.d.f.')
```

```{.python .input}
#@tab tensorflow
mu, sigma = 0, 1

x = tf.range(-3, 3, 0.01)
p = 1 / tf.sqrt(2 * tf.pi * sigma**2) * tf.exp(
    -(x - mu)**2 / (2 * sigma**2))

d2l.plot(x, p, 'x', 'p.d.f.')
```

では、累積分布関数を描きましょう。ここでは詳しく述べませんが、ガウス分布の c.d.f. は、より初等的な関数を用いた閉じた形の式を持たない。そこで、`erf` を使ってこの積分を数値的に計算する。

```{.python .input}
#@tab mxnet
def phi(x):
    return (1.0 + erf((x - mu) / (sigma * np.sqrt(2)))) / 2.0

d2l.plot(x, np.array([phi(y) for y in x.tolist()]), 'x', 'c.d.f.')
```

```{.python .input}
#@tab pytorch
def phi(x):
    return (1.0 + erf((x - mu) / (sigma * torch.sqrt(d2l.tensor(2.))))) / 2.0

d2l.plot(x, torch.tensor([phi(y) for y in x.tolist()]), 'x', 'c.d.f.')
```

```{.python .input}
#@tab tensorflow
def phi(x):
    return (1.0 + erf((x - mu) / (sigma * tf.sqrt(tf.constant(2.))))) / 2.0

d2l.plot(x, [phi(y) for y in x.numpy().tolist()], 'x', 'c.d.f.')
```

注意深い読者なら、これらの項のいくつかに見覚えがあるでしょう。実際、この積分は :numref:`sec_integral_calculus` で扱いた。まさにその計算によって、この $p_X(x)$ の全体の面積が1であり、したがって有効な密度であることがわかる。

コイン投げを用いたのは計算を簡単にするためでしたが、その選択自体に本質的な意味はない。実際、独立同分布な確率変数 $X_i$ の任意の集まりを取り、

$$
X^{(N)} = \sum_{i=1}^N X_i.
$$

としたとき、

$$
\frac{X^{(N)} - \mu_{X^{(N)}}}{\sigma_{X^{(N)}}}
$$

はおおよそガウス分布に従う。うまく成り立つためには追加の条件が必要で、最も一般的には $E[X^4] < \infty$ であるが、考え方は明確である。

中心極限定理は、ガウス分布が確率、統計、機械学習において基本的である理由である。何か測定したものが多くの小さな独立な寄与の和だと言えるなら、その測定対象はガウス分布に近いと仮定できる。

ガウス分布にはさらに多くの興味深い性質があるが、ここではもう1つだけ述べましょう。ガウス分布は *最大エントロピー分布* として知られている。エントロピーについては :numref:`sec_information_theory` でより深く扱いますが、ここで知っておくべきことは、それがランダムさの尺度であるということだけである。厳密な数学的意味では、ガウス分布は平均と分散が固定された確率変数の中で *最も* ランダムな選択だと考えられる。したがって、確率変数の平均と分散がわかっているなら、ガウス分布はある意味で最も保守的な分布の選択である。

節を閉じる前に、$X \sim \mathcal{N}(\mu, \sigma^2)$ ならば、

* $\mu_X = \mu$、
* $\sigma_X^2 = \sigma^2$。

であることを思い出しよう。

ガウス分布（あるいは標準正規分布）からのサンプルは以下のように得られる。

```{.python .input}
#@tab mxnet
np.random.normal(mu, sigma, size=(10, 10))
```

```{.python .input}
#@tab pytorch
torch.normal(mu, sigma, size=(10, 10))
```

```{.python .input}
#@tab tensorflow
tf.random.normal((10, 10), mu, sigma)
```

## 指数型分布族
:label:`subsec_exponential_family`

上で挙げた分布に共通する性質の1つは、すべてが *指数型分布族* に属することである。指数型分布族は、密度を次の形で表せる分布の集合である。

$$p(\mathbf{x} \mid \boldsymbol{\eta}) = h(\mathbf{x}) \cdot \exp \left( \boldsymbol{\eta}^{\top} \cdot T(\mathbf{x}) - A(\boldsymbol{\eta}) \right)$$
:eqlabel:`eq_exp_pdf`

この定義は少し繊細なので、詳しく見ていきましょう。

まず、$h(\mathbf{x})$ は *基底測度* あるいは *ベース測度* と呼ばれる。これは、指数重みで修正する元の測度とみなせる。

次に、$\boldsymbol{\eta} = (\eta_1, \eta_2, ..., \eta_l) \in \mathbb{R}^l$ というベクトルがあり、これを *自然パラメータ* または *標準パラメータ* と呼ぶ。これらは、基底測度をどのように修正するかを定める。自然パラメータは、$\mathbf{x}= (x_1, x_2, ..., x_n) \in \mathbb{R}^n$ のある関数 $T(\cdot)$ に対してこれらのパラメータとの内積を取り、それを指数化することで新しい測度に入る。ベクトル $T(\mathbf{x})= (T_1(\mathbf{x}), T_2(\mathbf{x}), ..., T_l(\mathbf{x}))$ は、$\boldsymbol{\eta}$ に対する *十分統計量* と呼ばれる。この名前は、$T(\mathbf{x})$ に表される情報だけで確率密度を計算するのに十分であり、サンプル $\mathbf{x}$ の他の情報は不要だからである。

第三に、$A(\boldsymbol{\eta})$ がある。これは *累積関数* と呼ばれ、上の分布 :eqref:`eq_exp_pdf` が1に積分されること、すなわち

$$A(\boldsymbol{\eta})  = \log \left[\int h(\mathbf{x}) \cdot \exp
\left(\boldsymbol{\eta}^{\top} \cdot T(\mathbf{x}) \right) d\mathbf{x} \right].$$

を保証する。

具体的に、ガウス分布を考えてみよう。$\mathbf{x}$ が1変数であるとすると、その密度は

$$
\begin{aligned}
p(x \mid \mu, \sigma) &= \frac{1}{\sqrt{2 \pi \sigma^2}} \cdot \exp 
\left\{ \frac{-(x-\mu)^2}{2 \sigma^2} \right\} \\
&= \frac{1}{\sqrt{2 \pi}} \cdot \exp \left\{ \frac{\mu}{\sigma^2}x
-\frac{1}{2 \sigma^2} x^2 - \left( \frac{1}{2 \sigma^2} \mu^2
+\log(\sigma) \right) \right\}.
\end{aligned}
$$

これは指数型分布族の定義に次のように対応する。

* *基底測度*: $h(x) = \frac{1}{\sqrt{2 \pi}}$、
* *自然パラメータ*: $\boldsymbol{\eta} = \begin{bmatrix} \eta_1 \\ \eta_2
\end{bmatrix} = \begin{bmatrix} \frac{\mu}{\sigma^2} \\
\frac{1}{2 \sigma^2} \end{bmatrix}$、
* *十分統計量*: $T(x) = \begin{bmatrix}x\\-x^2\end{bmatrix}$、および
* *累積関数*: $A({\boldsymbol\eta}) = \frac{1}{2 \sigma^2} \mu^2 + \log(\sigma)
= \frac{\eta_1^2}{4 \eta_2} - \frac{1}{2}\log(2 \eta_2)$。

上の各項の正確な選び方は、ある程度任意であることに注意する価値がある。重要なのは、分布がこの形で表せることであって、その厳密な形そのものではない。

:numref:`subsec_softmax_and_derivatives` でほのめかしたように、広く使われる手法の1つは、最終出力 $\mathbf{y}$ が指数型分布族に従うと仮定することである。指数型分布族は、機械学習で頻繁に現れる、一般的で強力な分布族である。

## まとめ
* ベルヌーイ確率変数は、はい/いいえの結果を持つ事象をモデル化できる。
* 離散一様分布は、有限個の候補からの選択をモデル化する。
* 連続一様分布は、区間からの選択をモデル化する。
* 二項分布は、ベルヌーイ確率変数の系列をモデル化し、成功回数を数える。
* ポアソン確率変数は、稀な事象の到着をモデル化する。
* ガウス確率変数は、多数の独立な確率変数を足し合わせた結果をモデル化する。
* 上記の分布はすべて指数型分布族に属する。

## 演習

1. 独立な二項確率変数 $X, Y \sim \textrm{Binomial}(16, 1/2)$ の差 $X-Y$ である確率変数の標準偏差はいくつか。
2. ポアソン確率変数 $X \sim \textrm{Poisson}(\lambda)$ を取り、$\lambda \rightarrow \infty$ のときの $(X - \lambda)/\sqrt{\lambda}$ を考えると、これが近似的にガウス分布になることを示せる。なぜこれは自然なのか。
3. $n$ 個の要素を持つ2つの離散一様確率変数の和の確率質量関数は何か。
