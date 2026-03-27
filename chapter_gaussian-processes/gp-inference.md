{.python .input}
%load_ext d2lbook.tab
tab.interact_select(["pytorch"])
#required_libs("gpytorch")
```

# ガウス過程推論

この節では、前節で導入した GP 事前分布を用いて、事後推論を行い予測を作成する方法を示します。まずは回帰から始めます。回帰では、推論を _閉形式_ で行うことができます。これは、実際にガウス過程をすぐ使い始めるための「GP の要点」セクションです。まず、基本的な操作をすべてゼロから実装し、その後で [GPyTorch](https://gpytorch.ai/) を導入します。これにより、最先端のガウス過程を扱ったり、深層ニューラルネットワークと統合したりすることがずっと便利になります。これらのより高度な話題については、次の節で詳しく扱います。その節では、近似推論が必要となる設定――分類、点過程、あるいは非ガウス尤度全般――についても考えます。 

## 回帰における事後推論

_観測_ モデルは、学習したい関数 $f(x)$ と、観測値 $y(x)$ を結びつけます。どちらもある入力 $x$ によって添字付けられます。分類では、$x$ は画像の画素であり、$y$ は対応するクラスラベルです。回帰では、$y$ は通常、地表温度、海面水位、$CO_2$ 濃度などの連続値出力を表します。  

回帰では、出力は潜在的なノイズのない関数 $f(x)$ に、独立同分布なガウス雑音 $\epsilon(x)$ が加わったものだと仮定することがよくあります。 

$$y(x) = f(x) + \epsilon(x),$$
:eqlabel:`eq_gp-regression`

ここで $\epsilon(x) \sim \mathcal{N}(0,\sigma^2)$ です。$\mathbf{y} = y(X) = (y(x_1),\dots,y(x_n))^{\top}$ を訓練観測のベクトル、$\textbf{f} = (f(x_1),\dots,f(x_n))^{\top}$ を、訓練入力 $X = {x_1, \dots, x_n}$ で問い合わせた潜在的なノイズのない関数値のベクトルとします。

ここでは $f(x) \sim \mathcal{GP}(m,k)$ を仮定します。これは、任意の関数値の集まり $\textbf{f}$ が、平均ベクトル $\mu_i = m(x_i)$ と共分散行列 $K_{ij} = k(x_i,x_j)$ をもつ同時多変量ガウス分布に従うことを意味します。RBF カーネル $k(x_i,x_j) = a^2 \exp\left(-\frac{1}{2\ell^2}||x_i-x_j||^2\right)$ は、標準的な共分散関数の選択肢です。記法を簡単にするため、平均関数は $m(x)=0$ と仮定します。導出は後で容易に一般化できます。

入力の集合 $$X_* = x_{*1},x_{*2},\dots,x_{*m}.$$ で予測したいとします。すると、$x^2$ と $p(\mathbf{f}_* | \mathbf{y}, X)$ を求めたいことになります。回帰設定では、$\mathbf{f}_* = f(X_*)$ と $\mathbf{y}$ の同時分布を求めた後、ガウスの恒等式を使ってこの分布を便利に求めることができます。 

式 :eqref:`eq_gp-regression` を訓練入力 $X$ で評価すると、$\mathbf{y} = \mathbf{f} + \mathbf{\epsilon}$ です。ガウス過程の定義（前節参照）より、$\mathbf{f} \sim \mathcal{N}(0,K(X,X))$ です。ここで $K(X,X)$ は、あり得るすべての入力対 $x_i, x_j \in X$ に対して共分散関数（別名 _カーネル_）を評価して得られる $n \times n$ 行列です。$\mathbf{\epsilon}$ は単に $\mathcal{N}(0,\sigma^2)$ からの iid サンプルからなるベクトルなので、分布は $\mathcal{N}(0,\sigma^2I)$ です。したがって $\mathbf{y}$ は、2つの独立な多変量ガウス変数の和であり、分布は $\mathcal{N}(0, K(X,X) + \sigma^2I)$ となります。また、$\textrm{cov}(\mathbf{f}_*, \mathbf{y}) = \textrm{cov}(\mathbf{y},\mathbf{f}_*)^{\top} = K(X_*,X)$ であることも示せます。ここで $K(X_*,X)$ は、テスト入力と訓練入力のすべての組に対してカーネルを評価して得られる $m \times n$ 行列です。 

$$
\begin{bmatrix}
\mathbf{y} \\
\mathbf{f}_*
\end{bmatrix}
\sim
\mathcal{N}\left(0, 
\mathbf{A} = \begin{bmatrix}
K(X,X)+\sigma^2I & K(X,X_*) \\
K(X_*,X) & K(X_*,X_*)
\end{bmatrix}
\right)
$$

その後、標準的なガウスの恒等式を使って、同時分布から条件付き分布を求めることができます（例えば Bishop 第2章を参照）。  
$\mathbf{f}_* | \mathbf{y}, X, X_* \sim \mathcal{N}(m_*,S_*)$ であり、$m_* = K(X_*,X)[K(X,X)+\sigma^2I]^{-1}\textbf{y}$、$S = K(X_*,X_*) - K(X_*,X)[K(X,X)+\sigma^2I]^{-1}K(X,X_*)$ です。

通常、予測共分散行列 $S$ 全体を使う必要はなく、各予測の不確実性として $S$ の対角成分だけを使います。そのため、しばしばテスト点の集合ではなく、単一のテスト点 $x_*$ に対する予測分布を書くことが多いです。 

カーネル行列には、上で述べた RBF カーネルの振幅 $a$ や長さ尺度 $\ell$ のように、推定したいパラメータ $\theta$ もあります。これらの目的のために、_周辺尤度_ $p(\textbf{y} | \theta, X)$ を使います。これは、$\mathbf{y},\mathbf{f}_*$ の同時分布を求めるために周辺分布を導出する過程ですでに得たものです。後で見るように、周辺尤度はモデル適合度とモデル複雑性の項に分解され、ハイパーパラメータ学習におけるオッカムの剃刀の概念を自動的に組み込みます。詳しくは MacKay Ch. 28 :cite:`mackay2003information` および Rasmussen and Williams Ch. 5 :cite:`rasmussen2006gaussian` を参照してください。

```{.python .input}
from d2l import torch as d2l
import numpy as np
from scipy.spatial import distance_matrix
from scipy import optimize
import matplotlib.pyplot as plt
import math
import torch
import gpytorch
import os

d2l.set_figsize()
```

## GP 回帰における予測とカーネルハイパーパラメータ学習のための式

ここでは、ガウス過程回帰でハイパーパラメータを学習し予測を行う際に使う式をまとめます。繰り返しになりますが、入力 $X = \{x_1,\dots,x_n\}$ によって添字付けられた回帰目標 $\textbf{y}$ のベクトルがあり、テスト入力 $x_*$ で予測したいとします。分散 $\sigma^2$ をもつ独立同分布な加法ゼロ平均ガウス雑音を仮定します。潜在的なノイズのない関数には、平均関数 $m$ とカーネル関数 $k$ をもつガウス過程事前分布 $f(x) \sim \mathcal{GP}(m,k)$ を使います。カーネル自体には、学習したいパラメータ $\theta$ があります。例えば RBF カーネル $k(x_i,x_j) = a^2\exp\left(-\frac{1}{2\ell^2}||x-x'||^2\right)$ を使うなら、$\theta = \{a^2, \ell^2\}$ を学習したいことになります。$K(X,X)$ は、$n$ 個の訓練入力のあり得るすべての組に対してカーネルを評価して得られる $n \times n$ 行列を表します。$K(x_*,X)$ は、$k(x_*, x_i)$ を $i=1,\dots,n$ について評価して得られる $1 \times n$ ベクトルを表します。$\mu$ は、すべての訓練点 $x$ で平均関数 $m(x)$ を評価して得られる平均ベクトルです。

通常、ガウス過程を扱うときは、2段階の手順に従います。 
1. 周辺尤度をこれらのハイパーパラメータに関して最大化することで、カーネルのハイパーパラメータ $\hat{\theta}$ を学習する。
2. 予測平均を点予測器として使い、予測標準偏差の 2 倍を用いて 95\% の信用集合を作る。ここでは、学習済みハイパーパラメータ $\hat{\theta}$ に条件付ける。

対数周辺尤度は単なる対数ガウス密度であり、次の形をしています:
$$\log p(\textbf{y} | \theta, X) = -\frac{1}{2}\textbf{y}^{\top}[K_{\theta}(X,X) + \sigma^2I]^{-1}\textbf{y} - \frac{1}{2}\log|K_{\theta}(X,X)| + c$$

予測分布は次の形をとります:
$$p(y_* | x_*, \textbf{y}, \theta) = \mathcal{N}(a_*,v_*)$$
$$a_* = k_{\theta}(x_*,X)[K_{\theta}(X,X)+\sigma^2I]^{-1}(\textbf{y}-\mu) + \mu$$
$$v_* = k_{\theta}(x_*,x_*) - K_{\theta}(x_*,X)[K_{\theta}(X,X)+\sigma^2I]^{-1}k_{\theta}(X,x_*)$$

## 学習と予測の式の解釈

ガウス過程の予測分布について、いくつか重要な点があります。

* モデルクラスは柔軟ですが、GP 回帰では _閉形式_ で _厳密な_ ベイズ推論を行うことができます。カーネルのハイパーパラメータを学習することを除けば、_訓練_ はありません。予測に使う式を正確に書き下せます。この点でガウス過程はかなり例外的であり、その便利さ、多用途性、そして今なお高い人気に大きく貢献しています。 

* 予測平均 $a_*$ は、訓練目標 $\textbf{y}$ の線形結合であり、重みはカーネル $k_{\theta}(x_*,X)[K_{\theta}(X,X)+\sigma^2I]^{-1}$ です。後で見るように、カーネル（とそのハイパーパラメータ）はモデルの汎化特性において極めて重要な役割を果たします。

* 予測平均は目標値 $\textbf{y}$ に明示的に依存しますが、予測分散は依存しません。代わりに、予測不確実性は、カーネル関数によって決まるように、テスト入力 $x_*$ が目標位置 $X$ から離れるにつれて増大します。ただし、不確実性は、データから学習されるカーネルハイパーパラメータ $\theta$ を通じて、間接的に目標値 $\textbf{y}$ に依存します。

* 周辺尤度は、モデル適合度とモデル複雑性（行列式の対数）に分解されます。周辺尤度は、データと整合的でありつつ最も単純な適合を与えるハイパーパラメータを選びがちです。 

* 主な計算ボトルネックは、$n$ 個の訓練点に対する $n \times n$ の対称正定値行列 $K(X,X)$ について、線形方程式を解くことと対数行列式を計算することです。素朴に行うと、これらの操作はそれぞれ $\mathcal{O}(n^3)$ の計算量と、カーネル（共分散）行列の各要素に対して $\mathcal{O}(n^2)$ のメモリを要し、しばしばコレスキー分解から始めます。歴史的には、これらのボトルネックのために GP はおよそ 10,000 点未満の問題に限られてきており、GP は「遅い」という評判を長年持っていましたが、これは今ではほぼ 10 年近く不正確です。高度な話題では、GP を何百万点もの問題にスケールさせる方法を議論します。

* よく使われるカーネル関数では、$K(X,X)$ はしばしば特異に近く、コレスキー分解や線形方程式を解くための他の操作で数値的問題を引き起こすことがあります。幸い、回帰ではしばしば $K_{\theta}(X,X)+\sigma^2I$ を扱うため、ノイズ分散 $\sigma^2$ が $K(X,X)$ の対角に加わり、条件数が大幅に改善されます。ノイズ分散が小さい場合、あるいはノイズなし回帰を行う場合は、条件数を改善するために対角へ $10^{-6}$ 程度の小さな "jitter" を加えるのが一般的です。


## ゼロからの実例

回帰データを作成し、そのデータを GP で当てはめます。すべての手順をゼロから実装します。 
次の式からデータをサンプルします。  
$$y(x) = \sin(x) + \frac{1}{2}\sin(4x) + \epsilon,$$ ただし $\epsilon \sim \mathcal{N}(0,\sigma^2)$ です。求めたいノイズのない関数は $f(x) = \sin(x) + \frac{1}{2}\sin(4x)$ です。まずはノイズの標準偏差を $\sigma = 0.25$ とします。

```{.python .input}
def data_maker1(x, sig):
    return np.sin(x) + 0.5 * np.sin(4 * x) + np.random.randn(x.shape[0]) * sig

sig = 0.25
train_x, test_x = np.linspace(0, 5, 50), np.linspace(0, 5, 500)
train_y, test_y = data_maker1(train_x, sig=sig), data_maker1(test_x, sig=0.)

d2l.plt.scatter(train_x, train_y)
d2l.plt.plot(test_x, test_y)
d2l.plt.xlabel("x", fontsize=20)
d2l.plt.ylabel("Observations y", fontsize=20)
d2l.plt.show()
```

ここでは、ノイズのある観測が丸印で、求めたい青色のノイズのない関数が見えます。 

では、潜在的なノイズのない関数 $f(x)\sim \mathcal{GP}(m,k)$ に対して GP 事前分布を指定しましょう。平均関数 $m(x) = 0$ と、RBF 共分散関数（カーネル）
$$k(x_i,x_j) = a^2\exp\left(-\frac{1}{2\ell^2}||x-x'||^2\right).$$
を使います。

```{.python .input}
mean = np.zeros(test_x.shape[0])
cov = d2l.rbfkernel(test_x, test_x, ls=0.2)
```

長さ尺度は 0.2 から始めています。データを当てはめる前に、妥当な事前分布を指定できているかを考えることが重要です。この事前分布からのサンプル関数と、95\% の信用集合（真の関数がこの領域内にある確率が 95\% だと考える）を可視化してみましょう。

```{.python .input}
prior_samples = np.random.multivariate_normal(mean=mean, cov=cov, size=5)
d2l.plt.plot(test_x, prior_samples.T, color='black', alpha=0.5)
d2l.plt.plot(test_x, mean, linewidth=2.)
d2l.plt.fill_between(test_x, mean - 2 * np.diag(cov), mean + 2 * np.diag(cov), 
                 alpha=0.25)
d2l.plt.show()
```

これらのサンプルは妥当に見えるでしょうか。関数の高レベルな性質は、モデル化したいデータの種類と整合しているでしょうか。

では、任意のテスト点 $x_*$ における事後予測分布の平均と分散を求めます。

$$
\bar{f}_{*} = K(x, x_*)^T (K(x, x) + \sigma^2 I)^{-1}y
$$

$$
V(f_{*}) = K(x_*, x_*) - K(x, x_*)^T (K(x, x) + \sigma^2 I)^{-1}K(x, x_*)
$$

予測を行う前に、カーネルのハイパーパラメータ $\theta$ とノイズ分散 $\sigma^2$ を学習する必要があります。事前分布の関数が、当てはめるデータに比べて変動が速すぎるように見えたので、長さ尺度の初期値を 0.75 にします。また、ノイズの標準偏差 $\sigma$ も 0.75 と仮定します。 

これらのパラメータを学習するために、これらのパラメータに関して周辺尤度を最大化します。

$$
\log p(y | X) = \log \int p(y | f, X)p(f | X)df
$$
$$
\log p(y | X) = -\frac{1}{2}y^T(K(x, x) + \sigma^2 I)^{-1}y - \frac{1}{2}\log |K(x, x) + \sigma^2 I| - \frac{n}{2}\log 2\pi
$$


おそらく事前分布の関数は変動が速すぎました。長さ尺度を 0.4 と仮定してみましょう。ノイズの標準偏差も 0.75 と仮定します。これらは単なるハイパーパラメータの初期値であり、周辺尤度からこれらのパラメータを学習します。

```{.python .input}
ell_est = 0.4
post_sig_est = 0.5

def neg_MLL(pars):
    K = d2l.rbfkernel(train_x, train_x, ls=pars[0])
    kernel_term = -0.5 * train_y @ \
        np.linalg.inv(K + pars[1] ** 2 * np.eye(train_x.shape[0])) @ train_y
    logdet = -0.5 * np.log(np.linalg.det(K + pars[1] ** 2 * \
                                         np.eye(train_x.shape[0])))
    const = -train_x.shape[0] / 2. * np.log(2 * np.pi)
    
    return -(kernel_term + logdet + const)


learned_hypers = optimize.minimize(neg_MLL, x0=np.array([ell_est,post_sig_est]), 
                                   bounds=((0.01, 10.), (0.01, 10.)))
ell = learned_hypers.x[0]
post_sig_est = learned_hypers.x[1]
```

この例では、長さ尺度 0.299 とノイズの標準偏差 0.24 を学習しました。学習されたノイズが真のノイズに非常に近いことに注意してください。これは、この問題に対して GP が非常によく適合していることを示しています。 

一般に、カーネルの選択とハイパーパラメータの初期化には慎重な検討が不可欠です。周辺尤度の最適化は初期値に対して比較的頑健ですが、悪い初期化の影響を受けないわけではありません。上のスクリプトをさまざまな初期値で実行し、どのような結果になるか試してみてください。

では、これらの学習済みハイパーパラメータで予測を行いましょう。

```{.python .input}
K_x_xstar = d2l.rbfkernel(train_x, test_x, ls=ell)
K_x_x = d2l.rbfkernel(train_x, train_x, ls=ell)
K_xstar_xstar = d2l.rbfkernel(test_x, test_x, ls=ell)

post_mean = K_x_xstar.T @ np.linalg.inv((K_x_x + \
                post_sig_est ** 2 * np.eye(train_x.shape[0]))) @ train_y
post_cov = K_xstar_xstar - K_x_xstar.T @ np.linalg.inv((K_x_x + \
                post_sig_est ** 2 * np.eye(train_x.shape[0]))) @ K_x_xstar

lw_bd = post_mean - 2 * np.sqrt(np.diag(post_cov))
up_bd = post_mean + 2 * np.sqrt(np.diag(post_cov))

d2l.plt.scatter(train_x, train_y)
d2l.plt.plot(test_x, test_y, linewidth=2.)
d2l.plt.plot(test_x, post_mean, linewidth=2.)
d2l.plt.fill_between(test_x, lw_bd, up_bd, alpha=0.25)
d2l.plt.legend(['Observed Data', 'True Function', 'Predictive Mean', '95% Set on True Func'])
d2l.plt.show()
```

オレンジ色の事後平均は、真のノイズのない関数とほぼ完全に一致していることがわかります。ここで示している 95\% の信用集合は、データ点ではなく、潜在的な _ノイズのない_（真の）関数に対するものです。この信用集合は真の関数を完全に含んでおり、広すぎも狭すぎもしないように見えます。データ点を含むことは期待していませんし、含む必要もありません。観測に対する信用集合が欲しいなら、次を計算します。

```{.python .input}
lw_bd_observed = post_mean - 2 * np.sqrt(np.diag(post_cov) + post_sig_est ** 2)
up_bd_observed = post_mean + 2 * np.sqrt(np.diag(post_cov) + post_sig_est ** 2)
```

不確実性には2つの源があります。_認識論的不確実性_ は _削減可能_ な不確実性を表し、_偶然的不確実性_ または _不可約不確実性_ もあります。ここでの _認識論的不確実性_ は、ノイズのない真の関数値に関する不確実性です。データから離れるにつれてこの不確実性は増大すべきです。なぜなら、データから離れるほど、データと整合する関数値の候補が増えるからです。より多くのデータを観測するにつれて、真の関数に対する信念はより確信を増し、認識論的不確実性は消えていきます。この例における _偶然的不確実性_ は観測ノイズです。データはこのノイズ付きで与えられており、減らすことはできません。

データにおける _認識論的不確実性_ は、潜在的なノイズのない関数の分散 np.diag(post\_cov) によって捉えられます。_偶然的不確実性_ はノイズ分散 post_sig_est**2 によって捉えられます。 

残念ながら、不確実性の表し方について人々はしばしば不注意です。多くの論文では、エラーバーがまったく定義されていなかったり、認識論的不確実性と偶然的不確実性のどちらを可視化しているのか、あるいは両方なのかが不明瞭だったり、ノイズ分散とノイズ標準偏差、標準偏差と標準誤差、信頼区間と信用集合などを混同していたりします。不確実性が何を表しているのかを正確にしなければ、それは本質的に無意味です。 

不確実性が何を表しているのかに注意を払うという観点から、ここではノイズのない関数の分散推定値の _平方根_ に _2倍_ を掛けていることが重要です。予測分布はガウス分布なので、この量によって 95\% の信用集合を作ることができ、真の関数を 95\% の確率で含むと考えられる区間に対する信念を表します。ノイズの _分散_ はまったく別のスケールにあり、はるかに解釈しにくいです。 

最後に、20 個の事後サンプルを見てみましょう。これらのサンプルは、事後的にどのような関数がデータに適合しうると考えているかを示します。

```{.python .input}
post_samples = np.random.multivariate_normal(post_mean, post_cov, size=20)
d2l.plt.scatter(train_x, train_y)
d2l.plt.plot(test_x, test_y, linewidth=2.)
d2l.plt.plot(test_x, post_mean, linewidth=2.)
d2l.plt.plot(test_x, post_samples.T, color='gray', alpha=0.25)
d2l.plt.fill_between(test_x, lw_bd, up_bd, alpha=0.25)
plt.legend(['Observed Data', 'True Function', 'Predictive Mean', 'Posterior Samples'])
d2l.plt.show()
```

基本的な回帰アプリケーションでは、事後予測平均と標準偏差を、それぞれ点予測器と不確実性の指標として使うのが最も一般的です。モンテカルロ獲得関数を用いたベイズ最適化や、モデルベース RL のためのガウス過程のような、より高度な応用では、事後サンプルを取る必要があることがよくあります。しかし、基本的な応用で厳密には必要でなくても、これらのサンプルはデータへの当てはまりについての直感を与えてくれ、可視化に含めると有用なことが多いです。 

## GPyTorch で簡単にする

見てきたように、基本的なガウス過程回帰は実際にはゼロからでもかなり簡単に実装できます。しかし、さまざまなカーネルを試したくなったり、近似推論を考えたり（これは分類でも必要です）、GP とニューラルネットワークを組み合わせたり、あるいは 10,000 点程度を超えるデータセットを扱ったりすると、ゼロからの実装は扱いにくく煩雑になります。SKI（KISS-GP とも呼ばれる）のようなスケーラブルな GP 推論の有力な手法の中には、高度な数値線形代数ルーチンを実装する数百行のコードを必要とするものもあります。 

このような場合、_GPyTorch_ ライブラリは大いに役立ちます。GPyTorch については、ガウス過程の数値計算や高度な手法に関する今後のノートブックでさらに詳しく扱います。GPyTorch ライブラリには [多くの例](https://github.com/cornellius-gp/gpytorch/tree/master/examples) があります。パッケージの雰囲気をつかむために、[単純な回帰の例](https://github.com/cornellius-gp/gpytorch/blob/master/examples/01_Exact_GPs/Simple_GP_Regression.ipynb) を見て、上の結果を GPyTorch で再現するようにどう適応できるかを示します。これは、上の基本的な回帰を再現するだけにしてはコードが多く見えるかもしれませんし、ある意味ではその通りです。しかし、数千行の新しいコードを書く代わりに、下の数行を変えるだけで、さまざまなカーネル、スケーラブルな推論手法、近似推論をすぐに使えるようになります。

```{.python .input}
# First let's convert our data into tensors for use with PyTorch
train_x = torch.tensor(train_x)
train_y = torch.tensor(train_y)
test_y = torch.tensor(test_y)

# We are using exact GP inference with a zero mean and RBF kernel
class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ZeroMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel())
    
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
```

このコードブロックでは、データを GPyTorch で使える形式に変換し、厳密推論を使うこと、そして使いたい平均関数（ゼロ）とカーネル関数（RBF）を指定しています。例えば gpytorch.kernels.matern_kernel() や gpyotrch.kernels.spectral_mixture_kernel() を呼び出すだけで、他のカーネルも簡単に使えます。ここまでで扱ってきたのは厳密推論だけであり、近似を行わずに予測分布を推定できる場合です。ガウス過程では、ガウス尤度がある場合にのみ厳密推論が可能です。より具体的には、観測がガウス過程で表されるノイズのない関数にガウス雑音が加わって生成されると仮定する場合です。今後のノートブックでは、これらの仮定が成り立たない分類などの他の設定を扱います。

```{.python .input}
# Initialize Gaussian likelihood
likelihood = gpytorch.likelihoods.GaussianLikelihood()
model = ExactGPModel(train_x, train_y, likelihood)
training_iter = 50
# Find optimal model hyperparameters
model.train()
likelihood.train()
# Use the adam optimizer, includes GaussianLikelihood parameters
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)  
# Set our loss as the negative log GP marginal likelihood
mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
```

ここでは、使いたい尤度（ガウス）、カーネルのハイパーパラメータを学習するために使う目的関数（ここでは周辺尤度）、そしてその目的関数を最適化するために使う手順（この場合は Adam）を明示的に指定しています。Adam は「確率的」最適化手法ですが、この場合はフルバッチ Adam であることに注意してください。周辺尤度はデータごとに因数分解できないため、データの「ミニバッチ」に対する最適化器を使っても収束が保証されません。L-BFGS のような他の最適化器も GPyTorch でサポートされています。標準的な深層学習とは異なり、周辺尤度をうまく最適化できることは良い汎化と強く結びついているため、計算コストが高すぎない限り、L-BFGS のような強力な最適化器を使いたくなることが多いです。

```{.python .input}
for i in range(training_iter):
    # Zero gradients from previous iteration
    optimizer.zero_grad()
    # Output from model
    output = model(train_x)
    # Calc loss and backprop gradients
    loss = -mll(output, train_y)
    loss.backward()
    if i % 10 == 0:
        print(f'Iter {i+1:d}/{training_iter:d} - Loss: {loss.item():.3f} '
              f'squared lengthscale: '
              f'{model.covar_module.base_kernel.lengthscale.item():.3f} '
              f'noise variance: {model.likelihood.noise.item():.3f}')
    optimizer.step()
```

ここで実際に最適化手順を実行し、10 イテレーションごとに損失の値を出力しています。

```{.python .input}
# Get into evaluation (predictive posterior) mode
test_x = torch.tensor(test_x)
model.eval()
likelihood.eval()
observed_pred = likelihood(model(test_x)) 
```

上のコードブロックにより、テスト入力に対して予測を行えるようになります。

```{.python .input}
with torch.no_grad():
    # Initialize plot
    f, ax = d2l.plt.subplots(1, 1, figsize=(4, 3))
    # Get upper and lower bounds for 95\% credible set (in this case, in
    # observation space)
    lower, upper = observed_pred.confidence_region()
    ax.scatter(train_x.numpy(), train_y.numpy())
    ax.plot(test_x.numpy(), test_y.numpy(), linewidth=2.)
    ax.plot(test_x.numpy(), observed_pred.mean.numpy(), linewidth=2.)
    ax.fill_between(test_x.numpy(), lower.numpy(), upper.numpy(), alpha=0.25)
    ax.set_ylim([-1.5, 1.5])
    ax.legend(['True Function', 'Predictive Mean', 'Observed Data',
               '95% Credible Set'])
```

最後に、当てはまりを描画します。

当てはまりは事実上同一であることがわかります。いくつか注意点があります。GPyTorch は _二乗された_ 長さ尺度と観測ノイズを扱っています。例えば、ゼロから書いたコードで学習されたノイズ標準偏差は約 0.283 でした。GPyTorch が見つけたノイズ分散は $0.81 \approx 0.283^2$ です。GPyTorch の図では、潜在関数空間ではなく _観測空間_ における信用集合も示しており、実際に観測データ点を覆っていることを示しています。

## まとめ

ガウス過程事前分布とデータを組み合わせて事後分布を作り、それを予測に使うことができます。また、周辺尤度を作ることもでき、これはガウス過程の変動速度などの性質を制御するカーネルハイパーパラメータの自動学習に役立ちます。回帰における事後の構成とカーネルハイパーパラメータの学習の仕組みは単純で、コードはおよそ十数行で済みます。このノートブックは、ガウス過程をすぐに「使い始めたい」読者にとって良い参考資料です。また、GPyTorch ライブラリも紹介しました。基本的な回帰のための GPyTorch コードは比較的長いものの、他のカーネル関数や、今後のノートブックで扱うより高度な機能――スケーラブル推論や分類のための非ガウス尤度など――へは、数行変えるだけで簡単に拡張できます。


## 演習

1. カーネルハイパーパラメータを _学習する_ ことの重要性と、ハイパーパラメータやカーネルがガウス過程の汎化特性に与える影響を強調してきました。ハイパーパラメータを学習する手順を飛ばし、代わりにさまざまな長さ尺度とノイズ分散を仮定して、予測への影響を確認してみてください。長さ尺度が大きいとどうなりますか。小さいとどうなりますか。ノイズ分散が大きいとどうなりますか。小さいとどうなりますか。
2. 周辺尤度は凸最適化問題ではないが、長さ尺度やノイズ分散のようなハイパーパラメータは GP 回帰で信頼性高く推定できる、と述べました。これは一般に正しいです。実際、周辺尤度は、経験的自己相関関数（「コビアログラム」）を当てはめる空間統計学の従来手法よりも、長さ尺度ハイパーパラメータの学習に _はるかに_ 優れています。少なくとも最近のスケーラブル推論の研究以前において、機械学習がガウス過程研究に与えた最大の貢献は、ハイパーパラメータ学習のための周辺尤度の導入だったと言えるでしょう。 

*しかし*、これらのパラメータの組み合わせが異なるだけで、多くのデータセットに対して解釈可能でもっともらしい説明が異なり、目的関数に局所最適が生じます。長さ尺度が大きい場合、真の基礎関数はゆっくり変化すると仮定していることになります。観測データが _実際に_ 大きく変動しているなら、大きな長さ尺度を正当化できるのは、大きなノイズ分散がある場合だけです。逆に、長さ尺度が小さい場合、当てはまりはデータの変動に非常に敏感になり、ノイズ（偶然的不確実性）で変動を説明する余地がほとんどなくなります。 

これらの局所最適を見つけられるか試してみてください。大きな長さ尺度と大きなノイズ、そして小さな長さ尺度と小さなノイズで初期化してみましょう。異なる解に収束しますか？
  
3. ベイズ法の根本的な利点の1つは、_認識論的不確実性_ を自然に表現できることだと述べました。上の例では、認識論的不確実性の効果を完全には見ることができません。代わりに `test_x = np.linspace(0, 10, 1000)` で予測してみてください。予測がデータを超えて進むにつれて、95\% の信用集合はどうなりますか。その区間で真の関数を覆いますか。その領域で偶然的不確実性だけを可視化するとどうなりますか。 

4. 上の例を、訓練点数を 10,000、20,000、40,000 にして実行し、実行時間を測定してみてください。訓練時間はどのようにスケールしますか。あるいは、テスト点数に対して実行時間はどうスケールしますか。予測平均と予測分散で違いはありますか。理論的に訓練・テスト時間の計算量を求めることと、上のコードを異なる点数で実行することの両方で答えてください。

5. GPyTorch の例を、Matern カーネルなど異なる共分散関数で実行してみてください。結果はどう変わりますか。GPyTorch ライブラリにある spectral mixture カーネルはどうでしょうか。周辺尤度で学習しやすいものとそうでないものはありますか。長距離予測と短距離予測で有用性に違いはありますか。

6. GPyTorch の例では観測ノイズを含めた予測分布を描きましたが、ゼロからの例では認識論的不確実性だけを含めました。GPyTorch の例をやり直し、今度は認識論的不確実性だけを描画して、ゼロからの結果と比較してください。予測分布は同じように見えますか？（同じはずです。）
