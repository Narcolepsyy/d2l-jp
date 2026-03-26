# 近似学習
:label:`sec_approx_train`

:numref:`sec_word2vec` での議論を思い出してください。  
skip-gram モデルの主な考え方は、softmax 演算を用いて、与えられた中心語 $w_c$ に基づいて文脈語 $w_o$ を生成する条件付き確率を :eqref:`eq_skip-gram-softmax` で計算することです。その対応する対数損失は :eqref:`eq_skip-gram-log` の符号を反転したものとして与えられます。



softmax 演算の性質上、文脈語は辞書 $\mathcal{V}$ のどれでもあり得るため、:eqref:`eq_skip-gram-log` の符号を反転した式には、語彙全体の大きさに等しい数の項の総和が含まれます。したがって、:eqref:`eq_skip-gram-grad` における skip-gram モデルの勾配計算と、:eqref:`eq_cbow-gradient` における continuous bag-of-words モデルの勾配計算の両方に、この総和が含まれます。残念ながら、このように大きな辞書（しばしば数十万語から数百万語）にわたって和を取る勾配の計算コストは非常に大きいのです！

前述の計算複雑性を削減するために、この節では 2 つの近似学習法、*negative sampling* と *hierarchical softmax* を紹介します。skip-gram モデルと continuous bag of words モデルの類似性のため、ここでは skip-gram モデルを例として、これら 2 つの近似学習法を説明します。

## Negative Sampling
:label:`subsec_negative-sampling`


Negative sampling は元の目的関数を修正します。中心語 $w_c$ の文脈ウィンドウが与えられたとき、任意の（文脈）語 $w_o$ がこの文脈ウィンドウから来るという事実を、次のようにモデル化される確率をもつ事象として考えます。


$$P(D=1\mid w_c, w_o) = \sigma(\mathbf{u}_o^\top \mathbf{v}_c),$$

ここで $\sigma$ はシグモイド活性化関数を次のように定義したものです。

$$\sigma(x) = \frac{1}{1+\exp(-x)}.$$
:eqlabel:`eq_sigma-f`

まず、単語埋め込みを学習するために、テキスト系列におけるこのような事象すべての同時確率を最大化することから始めましょう。具体的には、長さ $T$ のテキスト系列が与えられたとき、時刻 $t$ における単語を $w^{(t)}$ と表し、文脈ウィンドウサイズを $m$ とすると、次の同時確率を最大化することを考えます。


$$ \prod_{t=1}^{T} \prod_{-m \leq j \leq m,\ j \neq 0} P(D=1\mid w^{(t)}, w^{(t+j)}).$$
:eqlabel:`eq-negative-sample-pos`


しかし、:eqref:`eq-negative-sample-pos` は正例に関わる事象しか考慮していません。その結果、:eqref:`eq-negative-sample-pos` の同時確率は、すべての単語ベクトルが無限大に等しい場合にのみ 1 に最大化されます。もちろん、このような結果には意味がありません。目的関数をより意味のあるものにするために、*negative sampling* はあらかじめ定めた分布からサンプリングした負例を追加します。

$S$ を、文脈語 $w_o$ が中心語 $w_c$ の文脈ウィンドウから来るという事象とします。この $w_o$ を含む事象に対して、あらかじめ定めた分布 $P(w)$ から、この文脈ウィンドウに含まれない $K$ 個の *noise words* をサンプルします。$N_k$ を、ノイズ語 $w_k$ ($k=1, \ldots, K$) が $w_c$ の文脈ウィンドウから来ないという事象とします。正例と負例の両方に関わるこれらの事象 $S, N_1, \ldots, N_K$ は互いに独立であると仮定します。Negative sampling は、:eqref:`eq-negative-sample-pos` における（正例のみを含む）同時確率を次のように書き換えます。

$$ \prod_{t=1}^{T} \prod_{-m \leq j \leq m,\ j \neq 0} P(w^{(t+j)} \mid w^{(t)}),$$

ここで条件付き確率は、事象 $S, N_1, \ldots, N_K$ を通じて近似されます。

$$ P(w^{(t+j)} \mid w^{(t)}) =P(D=1\mid w^{(t)}, w^{(t+j)})\prod_{k=1,\ w_k \sim P(w)}^K P(D=0\mid w^{(t)}, w_k).$$
:eqlabel:`eq-negative-sample-conditional-prob`

テキスト系列の時刻 $t$ における単語 $w^{(t)}$ のインデックスを $i_t$、ノイズ語 $w_k$ のインデックスを $h_k$ とします。:eqref:`eq-negative-sample-conditional-prob` における条件付き確率に関する対数損失は次のようになります。

$$
\begin{aligned}
-\log P(w^{(t+j)} \mid w^{(t)})
=& -\log P(D=1\mid w^{(t)}, w^{(t+j)}) - \sum_{k=1,\ w_k \sim P(w)}^K \log P(D=0\mid w^{(t)}, w_k)\\
=&-  \log\, \sigma\left(\mathbf{u}_{i_{t+j}}^\top \mathbf{v}_{i_t}\right) - \sum_{k=1,\ w_k \sim P(w)}^K \log\left(1-\sigma\left(\mathbf{u}_{h_k}^\top \mathbf{v}_{i_t}\right)\right)\\
=&-  \log\, \sigma\left(\mathbf{u}_{i_{t+j}}^\top \mathbf{v}_{i_t}\right) - \sum_{k=1,\ w_k \sim P(w)}^K \log\sigma\left(-\mathbf{u}_{h_k}^\top \mathbf{v}_{i_t}\right).
\end{aligned}
$$


これにより、各学習ステップでの勾配計算コストは辞書サイズに依存せず、$K$ に線形に依存することがわかります。ハイパーパラメータ $K$ を小さく設定すれば、negative sampling を用いた各学習ステップでの勾配計算コストはより小さくなります。




## Hierarchical Softmax

別の近似学習法として、*hierarchical softmax* は二分木を用います。これは :numref:`fig_hi_softmax` に示すデータ構造で、木の各葉ノードが辞書 $\mathcal{V}$ の単語を表します。

![近似学習のための hierarchical softmax。各葉ノードは辞書中の単語を表す。](../img/hi-softmax.svg)
:label:`fig_hi_softmax`

$L(w)$ を、二分木において根ノードから単語 $w$ を表す葉ノードまでの経路上にあるノード数（両端を含む）とします。$n(w,j)$ をこの経路上の $j^\textrm{th}$ ノードとし、その文脈語ベクトルを $\mathbf{u}_{n(w, j)}$ とします。たとえば、 :numref:`fig_hi_softmax` では $L(w_3) = 4$ です。Hierarchical softmax は、:eqref:`eq_skip-gram-softmax` における条件付き確率を次のように近似します。


$$P(w_o \mid w_c) = \prod_{j=1}^{L(w_o)-1} \sigma\left( [\![  n(w_o, j+1) = \textrm{leftChild}(n(w_o, j)) ]\!] \cdot \mathbf{u}_{n(w_o, j)}^\top \mathbf{v}_c\right),$$

ここで関数 $\sigma$ は :eqref:`eq_sigma-f` で定義され、$\textrm{leftChild}(n)$ はノード $n$ の左の子ノードです。$x$ が真なら $[\![x]\!] = 1$、そうでなければ $[\![x]\!] = -1$ です。

例として、 :numref:`fig_hi_softmax` において、単語 $w_c$ が与えられたときに単語 $w_3$ を生成する条件付き確率を計算してみましょう。これには、$w_c$ の単語ベクトル $\mathbf{v}_c$ と、根から $w_3$ までの経路上にある非葉ノードのベクトル（:numref:`fig_hi_softmax` で太字で示された経路）との内積が必要です。この経路は左、右、左の順にたどられます。


$$P(w_3 \mid w_c) = \sigma(\mathbf{u}_{n(w_3, 1)}^\top \mathbf{v}_c) \cdot \sigma(-\mathbf{u}_{n(w_3, 2)}^\top \mathbf{v}_c) \cdot \sigma(\mathbf{u}_{n(w_3, 3)}^\top \mathbf{v}_c).$$

$\sigma(x)+\sigma(-x) = 1$ なので、任意の単語 $w_c$ に基づいて辞書 $\mathcal{V}$ のすべての単語を生成する条件付き確率の総和は 1 になります。

$$\sum_{w \in \mathcal{V}} P(w \mid w_c) = 1.$$
:eqlabel:`eq_hi-softmax-sum-one`

幸いなことに、二分木構造のため $L(w_o)-1$ は $\mathcal{O}(\textrm{log}_2|\mathcal{V}|)$ のオーダーです。したがって、辞書サイズ $\mathcal{V}$ が非常に大きい場合、hierarchical softmax を用いた各学習ステップの計算コストは、近似学習を用いない場合と比べて大幅に削減されます。

## まとめ

* Negative sampling は、正例と負例の両方に関わる互いに独立な事象を考慮して損失関数を構成します。学習時の計算コストは、各ステップでのノイズ語の数に線形に依存します。
* Hierarchical softmax は、二分木における根ノードから葉ノードまでの経路を用いて損失関数を構成します。学習時の計算コストは、各ステップでの辞書サイズの対数に依存します。

## 演習

1. Negative sampling では、どのようにノイズ語をサンプリングできますか？
1. :eqref:`eq_hi-softmax-sum-one` が成り立つことを確認しなさい。
1. continuous bag of words モデルを、それぞれ negative sampling と hierarchical softmax を用いてどのように学習しますか？
