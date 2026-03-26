# 統計学
:label:`sec_statistics`

言うまでもなく、最先端で高精度なモデルを訓練できることは、優れた深層学習実践者になるうえで極めて重要である。しかし、改善が統計的に有意なのか、それとも訓練過程における偶然の変動の結果にすぎないのかは、しばしば明確ではない。推定値における不確実性について議論できるようになるために、私たちは統計学を学ぶ必要がある。


*統計学* に関する最も古い記録は、9世紀のアラブの学者アル＝キンディーにさかのぼることができる。彼は、暗号化されたメッセージを解読するために統計と頻度分析をどのように用いるかを詳細に記述した。それから800年後、近代統計学は1700年代のドイツで生まれた。当時の研究者たちは、人口統計および経済データの収集と分析に注目していた。今日、統計学はデータの収集、処理、分析、解釈、可視化を扱う学問である。さらに、統計学の中核理論は、学術界、産業界、政府における研究で広く用いられている。


より具体的には、統計学は *記述統計* と *統計的推論* に分けられる。前者は、観測されたデータ集合の特徴を要約し、説明することに焦点を当てる。このデータ集合は *標本* と呼ばれる。標本は *母集団* から抽出される。母集団とは、私たちの実験対象に関係する類似した個体、項目、または事象の全体集合を指する。記述統計とは対照的に、*統計的推論* は、標本分布がある程度母集団分布を再現できるという仮定に基づいて、与えられた *標本* から母集団の特性をさらに推定する。


「機械学習と統計学の本質的な違いは何か」と疑問に思うかもしれない。根本的には、統計学は推論問題に焦点を当てる。この種の問題には、因果推論のような変数間の関係のモデル化や、A/Bテストのようなモデルパラメータの統計的有意性の検定が含まれる。これに対して、機械学習は、各パラメータの機能を明示的にプログラムしたり理解したりすることなく、正確な予測を行うことを重視する。


この節では、3種類の統計的推論手法、すなわち推定量の評価と比較、仮説検定の実施、信頼区間の構成を紹介する。これらの手法は、与えられた母集団、すなわち真のパラメータ $\theta$ の特性を推定するのに役立つ。簡潔さのため、ここでは与えられた母集団の真のパラメータ $\theta$ がスカラー値であると仮定する。$\theta$ がベクトルやテンソルである場合への拡張は容易なので、ここでは議論を省略する。



## 推定量の評価と比較

統計学において、*推定量* とは、真のパラメータ $\theta$ を推定するために与えられた標本の関数である。標本 $\{x_1, x_2, \ldots, x_n\}$ を観測した後の $\theta$ の推定値を $\hat{\theta}_n = \hat{f}(x_1, \ldots, x_n)$ と書く。

推定量の簡単な例は、 :numref:`sec_maximum_likelihood` 節ですでに見た。ベルヌーイ確率変数からいくつかの標本があるとき、その確率変数が1である確率の最尤推定値は、観測された1の個数を数え、それを標本数で割ることで得られる。同様に、ある演習では、ガウス分布の平均の最尤推定値が、すべての標本の平均値で与えられることを示すよう求められた。これらの推定量が真のパラメータ値を与えることはほとんどないが、理想的には、標本数が十分に大きければ推定値は真値に近づく。

例として、平均0、分散1のガウス確率変数の真の密度と、そのガウス分布から得られた標本を以下に示す。各点が見えるように、また元の密度との関係がより明確になるように、$y$ 座標を構成している。

```{.python .input}
#@tab mxnet
from d2l import mxnet as d2l
from mxnet import np, npx
import random
npx.set_np()

# Sample datapoints and create y coordinate
epsilon = 0.1
random.seed(8675309)
xs = np.random.normal(loc=0, scale=1, size=(300,))

ys = [np.sum(np.exp(-(xs[:i] - xs[i])**2 / (2 * epsilon**2))
             / np.sqrt(2*np.pi*epsilon**2)) / len(xs) for i in range(len(xs))]

# Compute true density
xd = np.arange(np.min(xs), np.max(xs), 0.01)
yd = np.exp(-xd**2/2) / np.sqrt(2 * np.pi)

# Plot the results
d2l.plot(xd, yd, 'x', 'density')
d2l.plt.scatter(xs, ys)
d2l.plt.axvline(x=0)
d2l.plt.axvline(x=np.mean(xs), linestyle='--', color='purple')
d2l.plt.title(f'sample mean: {float(np.mean(xs)):.2f}')
d2l.plt.show()
```

```{.python .input}
#@tab pytorch
from d2l import torch as d2l
import torch

torch.pi = torch.acos(torch.zeros(1)) * 2  #define pi in torch

# Sample datapoints and create y coordinate
epsilon = 0.1
torch.manual_seed(8675309)
xs = torch.randn(size=(300,))

ys = torch.tensor(
    [torch.sum(torch.exp(-(xs[:i] - xs[i])**2 / (2 * epsilon**2))\
               / torch.sqrt(2*torch.pi*epsilon**2)) / len(xs)\
     for i in range(len(xs))])

# Compute true density
xd = torch.arange(torch.min(xs), torch.max(xs), 0.01)
yd = torch.exp(-xd**2/2) / torch.sqrt(2 * torch.pi)

# Plot the results
d2l.plot(xd, yd, 'x', 'density')
d2l.plt.scatter(xs, ys)
d2l.plt.axvline(x=0)
d2l.plt.axvline(x=torch.mean(xs), linestyle='--', color='purple')
d2l.plt.title(f'sample mean: {float(torch.mean(xs).item()):.2f}')
d2l.plt.show()
```

```{.python .input}
#@tab tensorflow
from d2l import tensorflow as d2l
import tensorflow as tf

tf.pi = tf.acos(tf.zeros(1)) * 2  # define pi in TensorFlow

# Sample datapoints and create y coordinate
epsilon = 0.1
xs = tf.random.normal((300,))

ys = tf.constant(
    [(tf.reduce_sum(tf.exp(-(xs[:i] - xs[i])**2 / (2 * epsilon**2)) \
               / tf.sqrt(2*tf.pi*epsilon**2)) / tf.cast(
        tf.size(xs), dtype=tf.float32)).numpy() \
     for i in range(tf.size(xs))])

# Compute true density
xd = tf.range(tf.reduce_min(xs), tf.reduce_max(xs), 0.01)
yd = tf.exp(-xd**2/2) / tf.sqrt(2 * tf.pi)

# Plot the results
d2l.plot(xd, yd, 'x', 'density')
d2l.plt.scatter(xs, ys)
d2l.plt.axvline(x=0)
d2l.plt.axvline(x=tf.reduce_mean(xs), linestyle='--', color='purple')
d2l.plt.title(f'sample mean: {float(tf.reduce_mean(xs).numpy()):.2f}')
d2l.plt.show()
```

パラメータ $\hat{\theta}_n$ の推定量を計算する方法は多くある。この節では、推定量を評価・比較するための3つの一般的な方法、すなわち平均二乗誤差、標準偏差、統計的バイアスを紹介する。

### 平均二乗誤差

推定量を評価するために用いられる最も単純な指標は、おそらく *平均二乗誤差（MSE）*（または $l_2$ 損失）であり、次のように定義できる。

$$\textrm{MSE} (\hat{\theta}_n, \theta) = E[(\hat{\theta}_n - \theta)^2].$$
:eqlabel:`eq_mse_est`

これにより、真の値からの平均的な二乗偏差を定量化できる。MSE は常に非負である。 :numref:`sec_linear_regression` を読んでいれば、これが最も一般的に使われる回帰損失関数であることがわかるでしょう。推定量を評価する尺度としては、その値が0に近いほど、推定量は真のパラメータ $\theta$ に近いことを意味する。


### 統計的バイアス

MSE は自然な指標であるが、それを大きくする要因は複数考えられる。根本的に重要なのは、データセット内のランダム性による推定量の変動と、推定手続きに起因する推定量の系統誤差の2つである。


まず、系統誤差を測定しよう。推定量 $\hat{\theta}_n$ に対して、*統計的バイアス* は数学的には次のように定義できる。

$$\textrm{bias}(\hat{\theta}_n) = E(\hat{\theta}_n - \theta) = E(\hat{\theta}_n) - \theta.$$
:eqlabel:`eq_bias`

$\textrm{bias}(\hat{\theta}_n) = 0$ のとき、推定量 $\hat{\theta}_n$ の期待値はパラメータの真値に等しいことに注意する。この場合、$\hat{\theta}_n$ は不偏推定量であると言う。一般に、不偏推定量は、その期待値が真のパラメータと同じであるため、偏りのある推定量よりも優れている。


ただし、実務では偏りのある推定量が頻繁に使われることに注意する価値がある。追加の仮定なしには不偏推定量が存在しない場合や、計算が困難な場合があるからである。これは推定量の重大な欠点のように見えるかもしれないが、実際に遭遇する推定量の大半は、少なくとも漸近的には不偏である。すなわち、利用可能な標本数が無限大に近づくにつれてバイアスが0に近づく：$\lim_{n \rightarrow \infty} \textrm{bias}(\hat{\theta}_n) = 0$。


### 分散と標準偏差

次に、推定量のランダム性を測定しよう。 :numref:`sec_random_variables` で見たように、*標準偏差*（または *標準誤差*）は分散の平方根として定義される。推定量の標準偏差または分散を測ることで、その推定量の変動の程度を評価できる。

$$\sigma_{\hat{\theta}_n} = \sqrt{\textrm{Var} (\hat{\theta}_n )} = \sqrt{E[(\hat{\theta}_n - E(\hat{\theta}_n))^2]}.$$
:eqlabel:`eq_var_est`

:eqref:`eq_var_est` と :eqref:`eq_mse_est` を比較することが重要である。この式では真の母集団値 $\theta$ と比較しているのではなく、代わりに $E(\hat{\theta}_n)$、すなわち期待される標本平均と比較している。したがって、推定量が真の値からどれだけ離れがちかを測っているのではなく、推定量そのものの変動を測っているのである。


### バイアス・分散トレードオフ

これら2つの主要な要素が平均二乗誤差に寄与することは直感的に明らかである。やや驚くべきことに、平均二乗誤差は、これら2つの寄与に第3の項を加えたものへと実際に *分解* できることを示せる。つまり、平均二乗誤差は、バイアスの二乗、分散、そして不可避誤差の和として書ける。

$$
\begin{aligned}
\textrm{MSE} (\hat{\theta}_n, \theta) &= E[(\hat{\theta}_n - \theta)^2] \\
 &= E[(\hat{\theta}_n)^2] + E[\theta^2] - 2E[\hat{\theta}_n\theta] \\
 &= \textrm{Var} [\hat{\theta}_n] + E[\hat{\theta}_n]^2 + \textrm{Var} [\theta] + E[\theta]^2 - 2E[\hat{\theta}_n]E[\theta] \\
 &= (E[\hat{\theta}_n] - E[\theta])^2 + \textrm{Var} [\hat{\theta}_n] + \textrm{Var} [\theta] \\
 &= (E[\hat{\theta}_n - \theta])^2 + \textrm{Var} [\hat{\theta}_n] + \textrm{Var} [\theta] \\
 &= (\textrm{bias} [\hat{\theta}_n])^2 + \textrm{Var} (\hat{\theta}_n) + \textrm{Var} [\theta].\\
\end{aligned}
$$

上の式を *バイアス・分散トレードオフ* と呼ぶ。平均二乗誤差は、3つの誤差源、すなわち高バイアスによる誤差、高分散による誤差、そして不可避誤差に分けられる。バイアス誤差は、特徴量と出力の間の高次元の関係を抽出できない単純なモデル（たとえば線形回帰モデル）でよく見られる。モデルが高バイアス誤差に悩まされるとき、 :numref:`sec_generalization_basics` で述べたように、そのモデルは *アンダーフィッティング* している、あるいは *柔軟性* が不足していると言う。高分散は通常、訓練データに過剰適合する複雑すぎるモデルから生じる。その結果、*過学習* したモデルはデータの小さな変動に敏感になる。モデルが高分散に悩まされるとき、 :numref:`sec_generalization_basics` で述べたように、そのモデルは *過学習* しており、*汎化* が不足していると言う。不可避誤差は、$\theta$ 自体に含まれるノイズに起因する。


### コードによる推定量の評価

推定量の標準偏差は、テンソル `a` に対して単に `a.std()` を呼び出すことで実装できるため、ここでは省略し、統計的バイアスと平均二乗誤差を実装する。

```{.python .input}
#@tab mxnet
# Statistical bias
def stat_bias(true_theta, est_theta):
    return(np.mean(est_theta) - true_theta)

# Mean squared error
def mse(data, true_theta):
    return(np.mean(np.square(data - true_theta)))
```

```{.python .input}
#@tab pytorch
# Statistical bias
def stat_bias(true_theta, est_theta):
    return(torch.mean(est_theta) - true_theta)

# Mean squared error
def mse(data, true_theta):
    return(torch.mean(torch.square(data - true_theta)))
```

```{.python .input}
#@tab tensorflow
# Statistical bias
def stat_bias(true_theta, est_theta):
    return(tf.reduce_mean(est_theta) - true_theta)

# Mean squared error
def mse(data, true_theta):
    return(tf.reduce_mean(tf.square(data - true_theta)))
```

バイアス・分散トレードオフの式を示すために、正規分布 $\mathcal{N}(\theta, \sigma^2)$ を $10,000$ 個の標本でシミュレーションしてみよう。ここでは $\theta = 1$、$\sigma = 4$ を用いる。推定量は与えられた標本の関数なので、ここではこの正規分布 $\mathcal{N}(\theta, \sigma^2)$ における真の $\theta$ の推定量として標本平均を用いる。

```{.python .input}
#@tab mxnet
theta_true = 1
sigma = 4
sample_len = 10000
samples = np.random.normal(theta_true, sigma, sample_len)
theta_est = np.mean(samples)
theta_est
```

```{.python .input}
#@tab pytorch
theta_true = 1
sigma = 4
sample_len = 10000
samples = torch.normal(theta_true, sigma, size=(sample_len, 1))
theta_est = torch.mean(samples)
theta_est
```

```{.python .input}
#@tab tensorflow
theta_true = 1
sigma = 4
sample_len = 10000
samples = tf.random.normal((sample_len, 1), theta_true, sigma)
theta_est = tf.reduce_mean(samples)
theta_est
```

推定量のバイアスの二乗と分散の和を計算して、トレードオフの式を検証しよう。まず、推定量のMSEを計算する。

```{.python .input}
#@tab all
mse(samples, theta_true)
```

次に、以下のように $\textrm{Var} (\hat{\theta}_n) + [\textrm{bias} (\hat{\theta}_n)]^2$ を計算する。見てわかるように、2つの値は数値精度の範囲で一致する。

```{.python .input}
#@tab mxnet
bias = stat_bias(theta_true, theta_est)
np.square(samples.std()) + np.square(bias)
```

```{.python .input}
#@tab pytorch
bias = stat_bias(theta_true, theta_est)
torch.square(samples.std(unbiased=False)) + torch.square(bias)
```

```{.python .input}
#@tab tensorflow
bias = stat_bias(theta_true, theta_est)
tf.square(tf.math.reduce_std(samples)) + tf.square(bias)
```

## 仮説検定の実施


統計的推論で最もよく遭遇する話題は仮説検定である。仮説検定は20世紀初頭に広まりましたが、その最初の使用は1700年代のジョン・アーブスノットにさかのぼることができる。ジョンはロンドンの80年間の出生記録を追跡し、毎年、女性よりも男性のほうが多く生まれていると結論づけた。その後、現代的な有意性検定は、$p$ 値とピアソンのカイ二乗検定を考案したカール・ピアソン、スチューデントの $t$ 分布の父であるウィリアム・ゴセット、そして帰無仮説と有意性検定を導入したロナルド・フィッシャーによって発展した。

*仮説検定* とは、母集団に関する既定の主張に対して、何らかの証拠を評価する方法である。この既定の主張を *帰無仮説* $H_0$ と呼び、観測データを用いてこれを棄却しようとする。ここでは、$H_0$ を統計的有意性検定の出発点として用いる。*対立仮説* $H_A$（または $H_1$）は、帰無仮説に反する主張である。帰無仮説は、変数間の関係を仮定する宣言的な形で述べられることが多いである。それはできるだけ明確に要件を反映し、統計理論によって検証可能でなければならない。

あなたが化学者だと想像する。研究室で何千時間も費やした後、数学を理解する能力を劇的に向上させる新薬を開発したとする。その魔法のような効力を示すには、それを検証する必要がある。当然ながら、何人かのボランティアにその薬を飲んでもらい、数学の学習が本当に良くなるかを確かめる必要があるでしょう。では、どう始めればよいだろうか。

まず、数学的理解能力が何らかの指標で測ったときに差がないよう、慎重にランダムに選ばれた2つのボランティア群が必要である。この2つの群は一般に、テスト群と対照群と呼ばれる。*テスト群*（または *処置群*）は薬を投与される個体群であり、*対照群* は基準として取っておく群、すなわちこの薬を飲まないこと以外は同一の環境設定に置かれた群を表する。このようにすることで、処置における独立変数の影響を除き、他のすべての変数の影響を最小化できる。

次に、一定期間薬を飲んだ後、同じ指標で2つの群の数学的理解度を測定する必要がある。たとえば、新しい数学公式を学んだ後に、ボランティアに同じテストを受けてもらうといった方法である。その後、成績を集めて結果を比較できる。この場合、帰無仮説は2つの群に差がないことであり、対立仮説は差があることになる。

しかし、これでもまだ完全に形式化されているわけではない。慎重に考えるべき詳細はたくさんある。たとえば、数学的理解能力を検定するのに適した指標は何だろうか。薬の有効性を主張できると確信するには、何人のボランティアが必要だろうか。検定はどれくらいの期間実施すべきだろうか。2つの群に差があるかどうかをどう判断すればよいだろうか。平均的な成績だけを気にするのか、それとも得点の変動範囲も気にするのか。などである。

このように、仮説検定は、実験計画と観測結果の確実性について推論するための枠組みを与える。もし帰無仮説が真である可能性が非常に低いことを示せれば、私たちはそれを自信をもって棄却できる。

仮説検定の扱い方を理解するために、ここで追加の用語を導入し、上で述べた概念のいくつかを形式化する必要がある。


### 統計的有意性

*統計的有意性* は、棄却すべきでない帰無仮説 $H_0$ を誤って棄却してしまう確率を表する。すなわち、

$$ \textrm{statistical significance }= 1 - \alpha = 1 - P(\textrm{reject } H_0 \mid H_0 \textrm{ is true} ).$$

これは *第I種の誤り* または *偽陽性* とも呼ばれる。$\alpha$ は *有意水準* と呼ばれ、一般的な値は $5\%$、すなわち $1-\alpha = 95\%$ である。有意水準は、真の帰無仮説を棄却するときに私たちが受け入れるリスクの大きさとして説明できる。

:numref:`fig_statistical_significance` は、2標本仮説検定における観測値と、与えられた正規分布の確率を示している。観測データの例が $95\%$ の閾値の外側にある場合、それは帰無仮説の下では非常に起こりにくい観測である。したがって、帰無仮説に何か問題がある可能性があり、それを棄却することになる。

![Statistical significance.](../img/statistical-significance.svg)
:label:`fig_statistical_significance`


### 検出力

*統計的検出力*（または *感度*）は、棄却すべき帰無仮説 $H_0$ を正しく棄却できる確率を表する。すなわち、

$$ \textrm{statistical power }= 1 - \beta = 1 - P(\textrm{ fail to reject } H_0  \mid H_0 \textrm{ is false} ).$$

*第I種の誤り* は、真である帰無仮説を棄却してしまうことによる誤りであり、*第II種の誤り* は、偽である帰無仮説を棄却できないことによる誤りである。第II種の誤りは通常 $\beta$ で表されるため、対応する統計的検出力は $1-\beta$ である。


直感的には、統計的検出力は、所望の有意水準のもとで、ある最小の大きさを持つ真の差異をどれだけ検出しやすいかを表していると解釈できる。$80\%$ はよく使われる検出力の閾値である。検出力が高いほど、真の差を検出できる可能性が高くなる。

統計的検出力の最も一般的な用途の1つは、必要な標本数を決めることである。帰無仮説が偽であるときにそれを棄却できる確率は、それがどの程度偽であるか（*効果量* と呼ばれます）と、持っている標本数に依存する。予想されるとおり、効果量が小さい場合、それを高い確率で検出するには非常に多くの標本が必要になる。詳細な導出はこの簡潔な付録の範囲を超えますが、例として、標本が平均0・分散1のガウス分布から来たという帰無仮説を棄却したいとし、実際には標本平均が1に近いと考えている場合、許容可能な誤差率でそれを行うのに必要な標本サイズはわずか $8$ である。しかし、母集団の真の平均が $0.01$ に近いと考えるなら、その差を検出するには約 $80000$ の標本サイズが必要になる。

検出力は水フィルターにたとえることができる。この比喩では、高検出力の仮説検定は、高品質な浄水システムのように、水中の有害物質をできるだけ多く除去する。一方、差異が小さい場合は低品質な水フィルターのようなもので、比較的小さな物質が隙間から容易に漏れ出してしまいる。同様に、統計的検出力が十分に高くないと、検定は小さな差異を見逃してしまうかもしれない。


### 検定統計量

*検定統計量* $T(x)$ は、標本データのある特徴を要約するスカラーである。このような統計量を定義する目的は、異なる分布を区別し、仮説検定を実施できるようにすることである。化学者の例に戻ると、ある母集団が別の母集団より優れていることを示したいなら、平均を検定統計量として取るのは妥当でしょう。検定統計量の選び方によって、統計検定の検出力は大きく変わり得る。

しばしば、$T(X)$（帰無仮説の下での検定統計量の分布）は、少なくとも近似的には、帰無仮説のもとで正規分布のような一般的な確率分布に従う。このような分布を明示的に導出でき、さらにデータセット上で検定統計量を測定できれば、その統計量が期待される範囲から大きく外れているときに、帰無仮説を安全に棄却できる。これを定量化したものが $p$ 値の概念である。


### $p$ 値

$p$ 値（または *確率値*）は、帰無仮説が *真* であると仮定したときに、$T(X)$ が観測された検定統計量 $T(x)$ と少なくとも同程度に極端である確率である。すなわち、

$$ p\textrm{-value} = P_{H_0}(T(X) \geq T(x)).$$

$p$ 値が、あらかじめ定めた固定の統計的有意水準 $\alpha$ 以下であれば、帰無仮説を棄却できる。そうでなければ、帰無仮説を棄却する証拠が不足していると結論づける。与えられた母集団分布に対して、*棄却域* は、$p$ 値が統計的有意水準 $\alpha$ より小さいすべての点を含む区間になる。


### 片側検定と両側検定

通常、有意性検定には2種類ある。片側検定と両側検定である。*片側検定*（または *片側仮説検定*）は、帰無仮説と対立仮説が一方向にしかない場合に適用される。たとえば、帰無仮説が真のパラメータ $\theta$ が値 $c$ 以下であると述べることがある。対立仮説は $\theta$ が $c$ より大きいことになる。つまり、棄却域は標本分布の片側にのみある。片側検定とは対照的に、*両側検定*（または *両側仮説検定*）は、棄却域が標本分布の両側にある場合に適用される。この場合の例として、帰無仮説が真のパラメータ $\theta$ が値 $c$ に等しいと述べることがある。対立仮説は $\theta$ が $c$ に等しくないことになる。


### 仮説検定の一般的な手順

上の概念に慣れたところで、仮説検定の一般的な手順を見ていきましょう。

1. 問題を定義し、帰無仮説 $H_0$ を立てる。
2. 統計的有意水準 $\alpha$ と統計的検出力 ($1 - \beta$) を設定する。
3. 実験を通じて標本を得る。必要な標本数は、統計的検出力と期待される効果量に依存する。
4. 検定統計量と $p$ 値を計算する。
5. $p$ 値と統計的有意水準 $\alpha$ に基づいて、帰無仮説を維持するか棄却するかを決定する。

仮説検定を行うには、まず帰無仮説と、私たちが受け入れるリスクの水準を定義する。次に、標本の検定統計量を計算し、その極端な値を帰無仮説に対する証拠として扱う。検定統計量が棄却域に入れば、対立仮説を支持して帰無仮説を棄却できる。

仮説検定は、臨床試験や A/B テストなど、さまざまな場面に適用できる。


## 信頼区間の構成


パラメータ $\theta$ の値を推定するとき、$\hat \theta$ のような点推定量は、不確実性の概念を含まないため、限られた有用性しかない。むしろ、真のパラメータ $\theta$ を高い確率で含む区間を作れたほうがはるかに望ましいでしょう。もし1世紀前にそのような考えに関心があったなら、1937年に信頼区間の概念を初めて導入した Jerzy Neyman の "Outline of a Theory of Statistical Estimation Based on the Classical Theory of Probability" :cite:`Neyman.1937` を読んで興奮したことでしょう。

有用であるためには、信頼区間は、与えられた確実性の程度に対してできるだけ小さくあるべきである。どのように導出するかを見てみよう。


### 定義

数学的には、真のパラメータ $\theta$ に対する *信頼区間* とは、標本データから計算される区間 $C_n$ であり、次を満たすものである。

$$P_{\theta} (C_n \ni \theta) \geq 1 - \alpha, \forall \theta.$$
:eqlabel:`eq_confidence`

ここで $\alpha \in (0, 1)$ であり、$1 - \alpha$ は区間の *信頼水準* または *被覆率* と呼ばれる。これは、上で述べた有意水準と同じ $\alpha$ である。

:eqref:`eq_confidence` は固定された $\theta$ についてではなく、変数 $C_n$ について述べていることに注意する。これを強調するために、$P_{\theta} (C_n \ni \theta)$ と書き、$P_{\theta} (\theta \in C_n)$ とは書かない。

### 解釈

$95\%$ 信頼区間を「真のパラメータが $95\%$ の確率で入っている区間」と解釈したくなりますが、残念ながらこれは正しくない。真のパラメータは固定されており、ランダムなのは区間のほうである。したがって、より適切な解釈は、この手続きで多数の信頼区間を生成したなら、そのうち $95\%$ の区間が真のパラメータを含む、というものである。

これは些細な区別に見えるかもしれないが、結果の解釈に実際の影響を及ぼし得る。特に、めったにそうならない限り、真の値を含まないことが *ほぼ確実* な区間を構成することで :eqref:`eq_confidence` を満たすこともできてしまう。この節の最後に、魅力的に見えるが誤りである3つの主張を挙げる。これらの点についての詳細な議論は :citet:`Morey.Hoekstra.Rouder.ea.2016` を参照する。

* **誤解 1**. 信頼区間が狭いほど、パラメータを精密に推定できる。
* **誤解 2**. 信頼区間の内側の値は、外側の値よりも真の値である可能性が高い。
* **誤解 3**. ある特定の観測された $95\%$ 信頼区間が真の値を含む確率は $95\%$ である。

言うまでもなく、信頼区間は繊細な対象である。しかし、解釈を明確に保てば、強力な道具になる。

### ガウス分布の例

最も古典的な例として、平均と分散が未知のガウス分布の平均に対する信頼区間を考えましょう。ガウス分布 $\mathcal{N}(\mu, \sigma^2)$ から $n$ 個の標本 $\{x_i\}_{i=1}^n$ を集めたとする。平均と分散の推定量は次のように計算できる。

$$\hat\mu_n = \frac{1}{n}\sum_{i=1}^n x_i \;\textrm{and}\; \hat\sigma^2_n = \frac{1}{n-1}\sum_{i=1}^n (x_i - \hat\mu)^2.$$

ここで確率変数

$$
T = \frac{\hat\mu_n - \mu}{\hat\sigma_n/\sqrt{n}},
$$

を考えると、*自由度* $n-1$ の *スチューデントの $t$ 分布* と呼ばれるよく知られた分布に従う確率変数が得られる。

この分布は非常によく研究されており、たとえば $n\rightarrow \infty$ のとき、近似的に標準ガウス分布になることが知られている。したがって、ガウス分布の c.d.f. の値を表で調べれば、$T$ の値が少なくとも $95\%$ の確率で区間 $[-1.96, 1.96]$ に入ると結論できる。有限の $n$ では、この区間はやや広くする必要があるが、値はよく知られており、表にあらかじめ計算されている。

したがって、大きな $n$ に対しては、

$$
P\left(\frac{\hat\mu_n - \mu}{\hat\sigma_n/\sqrt{n}} \in [-1.96, 1.96]\right) \ge 0.95.
$$

これを両辺に $\hat\sigma_n/\sqrt{n}$ を掛け、その後 $\hat\mu_n$ を加えて整理すると、

$$
P\left(\mu \in \left[\hat\mu_n - 1.96\frac{\hat\sigma_n}{\sqrt{n}}, \hat\mu_n + 1.96\frac{\hat\sigma_n}{\sqrt{n}}\right]\right) \ge 0.95.
$$

したがって、$95\%$ 信頼区間は次のように得られる。
$$\left[\hat\mu_n - 1.96\frac{\hat\sigma_n}{\sqrt{n}}, \hat\mu_n + 1.96\frac{\hat\sigma_n}{\sqrt{n}}\right].$$
:eqlabel:`eq_gauss_confidence`

:eqref:`eq_gauss_confidence` は統計学で最もよく使われる式の1つと言ってよいでしょう。では、これを実装して統計学の議論を締めくくりましょう。簡単のため、ここでは漸近的な状況にあると仮定する。小さい $N$ の場合には、プログラムで、あるいは $t$ 表から得た正しい `t_star` の値を使うべきである。

```{.python .input}
#@tab mxnet
# Number of samples
N = 1000

# Sample dataset
samples = np.random.normal(loc=0, scale=1, size=(N,))

# Lookup Students's t-distribution c.d.f.
t_star = 1.96

# Construct interval
mu_hat = np.mean(samples)
sigma_hat = samples.std(ddof=1)
(mu_hat - t_star*sigma_hat/np.sqrt(N), mu_hat + t_star*sigma_hat/np.sqrt(N))
```

```{.python .input}
#@tab pytorch
# PyTorch uses Bessel's correction by default, which means the use of ddof=1
# instead of default ddof=0 in numpy. We can use unbiased=False to imitate
# ddof=0.

# Number of samples
N = 1000

# Sample dataset
samples = torch.normal(0, 1, size=(N,))

# Lookup Students's t-distribution c.d.f.
t_star = 1.96

# Construct interval
mu_hat = torch.mean(samples)
sigma_hat = samples.std(unbiased=True)
(mu_hat - t_star*sigma_hat/torch.sqrt(torch.tensor(N, dtype=torch.float32)),\
 mu_hat + t_star*sigma_hat/torch.sqrt(torch.tensor(N, dtype=torch.float32)))
```

```{.python .input}
#@tab tensorflow
# Number of samples
N = 1000

# Sample dataset
samples = tf.random.normal((N,), 0, 1)

# Lookup Students's t-distribution c.d.f.
t_star = 1.96

# Construct interval
mu_hat = tf.reduce_mean(samples)
sigma_hat = tf.math.reduce_std(samples)
(mu_hat - t_star*sigma_hat/tf.sqrt(tf.constant(N, dtype=tf.float32)), \
 mu_hat + t_star*sigma_hat/tf.sqrt(tf.constant(N, dtype=tf.float32)))
```

## まとめ

* 統計学は推論問題に焦点を当てるのに対し、深層学習は各パラメータを明示的にプログラムしたり理解したりすることなく、正確な予測を行うことを重視する。
* 統計的推論の一般的な手法は3つある。推定量の評価と比較、仮説検定の実施、信頼区間の構成である。
* 最も一般的な3つの指標は、統計的バイアス、標準偏差、平均二乗誤差である。
* 信頼区間とは、与えられた標本から構成できる、真の母集団パラメータの推定範囲である。
* 仮説検定は、母集団に関する既定の主張に対して証拠を評価する方法である。


## 演習

1. $X_1, X_2, \ldots, X_n \overset{\textrm{iid}}{\sim} \textrm{Unif}(0, \theta)$ とする。ここで "iid" は *独立同分布* を意味する。$\theta$ の次の推定量を考える。
$$\hat{\theta} = \max \{X_1, X_2, \ldots, X_n \};$$
$$\tilde{\theta} = 2 \bar{X_n} = \frac{2}{n} \sum_{i=1}^n X_i.$$
    * $\hat{\theta}$ の統計的バイアス、標準偏差、平均二乗誤差を求めよ。
    * $\tilde{\theta}$ の統計的バイアス、標準偏差、平均二乗誤差を求めよ。
    * どちらの推定量がより良いか。
1. 導入部の化学者の例について、両側仮説検定を行うための5つの手順を導出できるか。有意水準 $\alpha = 0.05$ と統計的検出力 $1 - \beta = 0.8$ を用いよ。
1. $N=2$ と $\alpha = 0.5$ で信頼区間のコードを、独立に生成した100個のデータセットに対して実行し、得られた区間をプロットせよ（この場合 `t_star = 1.0`）。真の平均 $0$ を含むにはあまりにも遠い、非常に短い区間がいくつか見えるはずである。これは信頼区間の解釈と矛盾するか。短い区間を高精度な推定の指標として使うことに安心感を持てるか。\n
