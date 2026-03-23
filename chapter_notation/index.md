# 記法
:label:`chap_notation`

本書全体を通して、以下の記法規則に従います。  
これらの記号の一部はプレースホルダであり、他のものは特定の対象を指します。  
一般的な目安として、不定冠詞の "a" はしばしば、その記号がプレースホルダであり、同じ形式の記号が同種の別の対象を表しうることを示します。  
たとえば、"$x$: a scalar" は、小文字の文字が一般にスカラー値を表すことを意味しますが、"$\mathbb{Z}$: the set of integers" は記号 $\mathbb{Z}$ を特定して指しています。



## 数値オブジェクト

* $x$: スカラー
* $\mathbf{x}$: ベクトル
* $\mathbf{X}$: 行列
* $\mathsf{X}$: 一般のテンソル
* $\mathbf{I}$: 単位行列（ある与えられた次元のもの）、すなわち対角成分がすべて $1$、非対角成分がすべて $0$ の正方行列
* $x_i$, $[\mathbf{x}]_i$: ベクトル $\mathbf{x}$ の $i^\textrm{th}$ 要素
* $x_{ij}$, $x_{i,j}$,$[\mathbf{X}]_{ij}$, $[\mathbf{X}]_{i,j}$: 行列 $\mathbf{X}$ の第 $i$ 行第 $j$ 列の要素



## 集合論


* $\mathcal{X}$: 集合
* $\mathbb{Z}$: 整数全体の集合
* $\mathbb{Z}^+$: 正の整数全体の集合
* $\mathbb{R}$: 実数全体の集合
* $\mathbb{R}^n$: 実数からなる $n$ 次元ベクトル全体の集合
* $\mathbb{R}^{a\times b}$: $a$ 行 $b$ 列の実数行列全体の集合
* $|\mathcal{X}|$: 集合 $\mathcal{X}$ の濃度（要素数）
* $\mathcal{A}\cup\mathcal{B}$: 集合 $\mathcal{A}$ と $\mathcal{B}$ の和集合
* $\mathcal{A}\cap\mathcal{B}$: 集合 $\mathcal{A}$ と $\mathcal{B}$ の共通部分
* $\mathcal{A}\setminus\mathcal{B}$: $\mathcal{A}$ から $\mathcal{B}$ を引いた差集合（$\mathcal{A}$ のうち $\mathcal{B}$ に属さない要素のみを含む）



## 関数と演算子


* $f(\cdot)$: 関数
* $\log(\cdot)$: 自然対数（底 $e$）
* $\log_2(\cdot)$: 底 $2$ の対数
* $\exp(\cdot)$: 指数関数
* $\mathbf{1}(\cdot)$: 指示関数。ブール値の引数が真なら $1$、それ以外なら $0$ を返す
* $\mathbf{1}_{\mathcal{X}}(z)$: 集合所属指示関数。要素 $z$ が集合 $\mathcal{X}$ に属すれば $1$、それ以外なら $0$ を返す
* $\mathbf{(\cdot)}^\top$: ベクトルまたは行列の転置
* $\mathbf{X}^{-1}$: 行列 $\mathbf{X}$ の逆行列
* $\odot$: ハダマード積（要素ごとの積）
* $[\cdot, \cdot]$: 連結
* $\|\cdot\|_p$: $\ell_p$ ノルム
* $\|\cdot\|$: $\ell_2$ ノルム
* $\langle \mathbf{x}, \mathbf{y} \rangle$: ベクトル $\mathbf{x}$ と $\mathbf{y}$ の内積（ドット積）
* $\sum$: 要素の集合に対する総和
* $\prod$: 要素の集合に対する総乗
* $\stackrel{\textrm{def}}{=}$: 左辺の記号の定義として主張される等式



## 微積分

* $\frac{dy}{dx}$: $x$ に関する $y$ の導関数
* $\frac{\partial y}{\partial x}$: $x$ に関する $y$ の偏導関数
* $\nabla_{\mathbf{x}} y$: $\mathbf{x}$ に関する $y$ の勾配
* $\int_a^b f(x) \;dx$: $x$ に関して $a$ から $b$ までの $f$ の定積分
* $\int f(x) \;dx$: $x$ に関する $f$ の不定積分



## 確率と情報理論

* $X$: 確率変数
* $P$: 確率分布
* $X \sim P$: 確率変数 $X$ は分布 $P$ に従う
* $P(X=x)$: 確率変数 $X$ が値 $x$ をとる事象に割り当てられる確率
* $P(X \mid Y)$: $Y$ が与えられたときの $X$ の条件付き確率分布
* $p(\cdot)$: 分布 $P$ に対応する確率密度関数（PDF）
* ${E}[X]$: 確率変数 $X$ の期待値
* $X \perp Y$: 確率変数 $X$ と $Y$ は独立
* $X \perp Y \mid Z$: 確率変数 $X$ と $Y$ は、$Z$ が与えられたとき条件付き独立
* $\sigma_X$: 確率変数 $X$ の標準偏差
* $\textrm{Var}(X)$: 確率変数 $X$ の分散。$\sigma^2_X$ に等しい
* $\textrm{Cov}(X, Y)$: 確率変数 $X$ と $Y$ の共分散
* $\rho(X, Y)$: $X$ と $Y$ のピアソン相関係数。$\frac{\textrm{Cov}(X, Y)}{\sigma_X \sigma_Y}$ に等しい
* $H(X)$: 確率変数 $X$ のエントロピー
* $D_{\textrm{KL}}(P\|Q)$: 分布 $Q$ から分布 $P$ への KL ダイバージェンス（または相対エントロピー）



[Discussions](https://discuss.d2l.ai/t/25)\n