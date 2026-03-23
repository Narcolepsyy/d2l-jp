# 時間を通した逆伝播
:label:`sec_bptt`

:numref:`sec_rnn-scratch` の演習を終えていれば、
勾配クリッピングが、まれに発生する巨大な勾配が
学習を不安定にするのを防ぐうえで不可欠であることを
見てきたはずです。
爆発する勾配は、長い系列にわたって逆伝播することに
起因すると示唆しました。
現代的なRNNアーキテクチャを多数紹介する前に、
*逆伝播* が系列モデルで数学的にどのように
機能するのかを、もう少し詳しく見てみましょう。
この議論によって、*消失* 勾配と *爆発* 勾配という概念に
いくらか精密さが加わることを期待しています。
:numref:`sec_backprop` でMLPを導入したときに、
計算グラフを通した順伝播と逆伝播についての議論を
思い出していただければ、RNNにおける順伝播は
比較的わかりやすいはずです。
RNNに逆伝播を適用することを
*時間を通した逆伝播* と呼びます :cite:`Werbos.1990`。
この手続きでは、RNNの計算グラフを
1タイムステップずつ展開（またはアンロール）する必要があります。
アンロールされたRNNは本質的にフィードフォワード型の
ニューラルネットワークであり、
同じパラメータがアンロールされたネットワーク全体で
繰り返し現れ、各タイムステップに登場するという
特別な性質を持っています。
その後は、通常のフィードフォワード型ニューラルネットワークと同様に、
連鎖律を適用して、アンロールされたネットワークを通して
勾配を逆伝播できます。
各パラメータに関する勾配は、
アンロールされたネットワーク内でそのパラメータが現れる
すべての箇所にわたって和を取らなければなりません。
このような重み共有の扱いは、
畳み込みニューラルネットワークの章で
すでに見てきたはずです。


系列はかなり長くなりうるため、問題が生じます。
1000個を超えるトークンからなるテキスト系列を扱うことは
珍しくありません。
これは、計算量の観点（メモリが多すぎる）と
最適化の観点（数値的不安定性）の両方で
問題を引き起こします。
最初のステップからの入力は、出力に到達するまでに
1000回以上の行列積を通過し、
勾配を計算するためにもさらに1000回の行列積が必要です。
ここでは、何がうまくいかなくなるのか、
そして実際にはどう対処するのかを分析します。


## RNNにおける勾配の解析
:label:`subsec_bptt_analysis`

まず、RNNがどのように動作するかについての
単純化したモデルから始めます。
このモデルでは、隠れ状態の具体的な詳細や
その更新方法は無視します。
ここでの数学記法では、
スカラー、ベクトル、行列を明示的に区別しません。
あくまで直感を養うことが目的です。
この単純化したモデルでは、
時刻 $t$ における隠れ状態を $h_t$、
入力を $x_t$、出力を $o_t$ と表します。
:numref:`subsec_rnn_w_hidden_states` で議論したように、
入力と隠れ状態は、隠れ層の1つの重み変数で
掛け算される前に連結できます。
したがって、隠れ層と出力層の重みをそれぞれ
$w_\textrm{h}$ と $w_\textrm{o}$ で表します。
その結果、各タイムステップにおける隠れ状態と出力は

$$\begin{aligned}h_t &= f(x_t, h_{t-1}, w_\textrm{h}),\\o_t &= g(h_t, w_\textrm{o}),\end{aligned}$$
:eqlabel:`eq_bptt_ht_ot`

となります。ここで $f$ と $g$ はそれぞれ
隠れ層と出力層の変換です。
したがって、相互に依存する
$\{\ldots, (x_{t-1}, h_{t-1}, o_{t-1}), (x_{t}, h_{t}, o_t), \ldots\}$
という値の連鎖が、再帰的な計算を通じて得られます。
順伝播はかなり単純です。
必要なのは、$(x_t, h_t, o_t)$ の組を
1タイムステップずつ順にループすることだけです。
その後、出力 $o_t$ と望ましい目標 $y_t$ の差は、
全 $T$ タイムステップにわたる目的関数で評価され、

$$L(x_1, \ldots, x_T, y_1, \ldots, y_T, w_\textrm{h}, w_\textrm{o}) = \frac{1}{T}\sum_{t=1}^T l(y_t, o_t).$$



逆伝播では、特に目的関数 $L$ のパラメータ $w_\textrm{h}$ に関する
勾配を計算するときに、少し厄介になります。
具体的には、連鎖律により、

$$\begin{aligned}\frac{\partial L}{\partial w_\textrm{h}}  & = \frac{1}{T}\sum_{t=1}^T \frac{\partial l(y_t, o_t)}{\partial w_\textrm{h}}  \\& = \frac{1}{T}\sum_{t=1}^T \frac{\partial l(y_t, o_t)}{\partial o_t} \frac{\partial g(h_t, w_\textrm{o})}{\partial h_t}  \frac{\partial h_t}{\partial w_\textrm{h}}.\end{aligned}$$
:eqlabel:`eq_bptt_partial_L_wh`

:eqref:`eq_bptt_partial_L_wh` における積の
最初と2番目の因子は簡単に計算できます。
3番目の因子 $\partial h_t/\partial w_\textrm{h}$ が難所であり、
パラメータ $w_\textrm{h}$ が $h_t$ に与える影響を
再帰的に計算する必要があります。
:eqref:`eq_bptt_ht_ot` の再帰的計算によれば、
$h_t$ は $h_{t-1}$ と $w_\textrm{h}$ の両方に依存し、
$h_{t-1}$ の計算もまた $w_\textrm{h}$ に依存します。
したがって、連鎖律を用いて $h_t$ の $w_\textrm{h}$ に関する
全微分を評価すると、

$$\frac{\partial h_t}{\partial w_\textrm{h}}= \frac{\partial f(x_{t},h_{t-1},w_\textrm{h})}{\partial w_\textrm{h}} +\frac{\partial f(x_{t},h_{t-1},w_\textrm{h})}{\partial h_{t-1}} \frac{\partial h_{t-1}}{\partial w_\textrm{h}}.$$
:eqlabel:`eq_bptt_partial_ht_wh_recur`


上の勾配を導くために、3つの系列 $\{a_{t}\},\{b_{t}\},\{c_{t}\}$ が
$a_{0}=0$ かつ $t=1, 2,\ldots$ に対して
$a_{t}=b_{t}+c_{t}a_{t-1}$ を満たすと仮定します。
すると $t\geq 1$ について、次が容易に示せます。

$$a_{t}=b_{t}+\sum_{i=1}^{t-1}\left(\prod_{j=i+1}^{t}c_{j}\right)b_{i}.$$
:eqlabel:`eq_bptt_at`

$a_t$、$b_t$、$c_t$ をそれぞれ

$$\begin{aligned}a_t &= \frac{\partial h_t}{\partial w_\textrm{h}},\\
b_t &= \frac{\partial f(x_{t},h_{t-1},w_\textrm{h})}{\partial w_\textrm{h}}, \\
c_t &= \frac{\partial f(x_{t},h_{t-1},w_\textrm{h})}{\partial h_{t-1}},\end{aligned}$$

に従って置き換えると、:eqref:`eq_bptt_partial_ht_wh_recur` の勾配計算は
$a_{t}=b_{t}+c_{t}a_{t-1}$ を満たします。
したがって、:eqref:`eq_bptt_at` により、
:eqref:`eq_bptt_partial_ht_wh_recur` における再帰計算を

$$\frac{\partial h_t}{\partial w_\textrm{h}}=\frac{\partial f(x_{t},h_{t-1},w_\textrm{h})}{\partial w_\textrm{h}}+\sum_{i=1}^{t-1}\left(\prod_{j=i+1}^{t} \frac{\partial f(x_{j},h_{j-1},w_\textrm{h})}{\partial h_{j-1}} \right) \frac{\partial f(x_{i},h_{i-1},w_\textrm{h})}{\partial w_\textrm{h}}.$$
:eqlabel:`eq_bptt_partial_ht_wh_gen`

で取り除くことができます。

連鎖律を使って $\partial h_t/\partial w_\textrm{h}$ を再帰的に計算できますが、
$t$ が大きいとこの連鎖は非常に長くなります。
この問題に対処するいくつかの戦略を議論しましょう。

### 全計算 ### 

1つの考え方は、:eqref:`eq_bptt_partial_ht_wh_gen` の
完全な和を計算することかもしれません。
しかし、これは非常に遅く、勾配が爆発する可能性があります。
というのも、初期条件のわずかな変化が
結果に大きく影響しうるからです。
つまり、初期条件の小さな変化が
結果に不釣り合いな変化をもたらす、
いわゆるバタフライ効果に似た現象が
起こりえます。
これは一般に望ましくありません。
結局のところ、私たちが求めているのは、
よく一般化する頑健な推定量です。
したがって、この戦略が実際に使われることは
ほとんどありません。

### タイムステップの切り詰め ###

別の方法として、
:eqref:`eq_bptt_partial_ht_wh_gen` の和を
$\tau$ ステップ後で打ち切ることができます。
これが、ここまで議論してきた方法です。
これは、和を $\partial h_{t-\tau}/\partial w_\textrm{h}$ で
単純に終了させることで、真の勾配の *近似* を与えます。
実際には、これはかなりうまく機能します。
これは一般に、切り詰めた
時間を通した逆伝播と呼ばれます :cite:`Jaeger.2002`。
この結果の1つとして、モデルは
長期的な結果よりも短期的な影響に
主として注目するようになります。
これは実際には *望ましい* ことであり、
より単純で安定したモデルへと推定を
偏らせるからです。


### ランダム化された切り詰め ### 

最後に、$\partial h_t/\partial w_\textrm{h}$ を、
期待値では正しいが系列を切り詰める
確率変数で置き換えることができます。
これは、あらかじめ定めた $0 \leq \pi_t \leq 1$ を持つ
$\xi_t$ の系列を用いて実現され、
$P(\xi_t = 0) = 1-\pi_t$ かつ
$P(\xi_t = \pi_t^{-1}) = \pi_t$ となるため、
$E[\xi_t] = 1$ です。
これを用いて、:eqref:`eq_bptt_partial_ht_wh_recur` における
勾配 $\partial h_t/\partial w_\textrm{h}$ を

$$z_t= \frac{\partial f(x_{t},h_{t-1},w_\textrm{h})}{\partial w_\textrm{h}} +\xi_t \frac{\partial f(x_{t},h_{t-1},w_\textrm{h})}{\partial h_{t-1}} \frac{\partial h_{t-1}}{\partial w_\textrm{h}}.$$

で置き換えます。


$\xi_t$ の定義から、$E[z_t] = \partial h_t/\partial w_\textrm{h}$ が
成り立ちます。
$\xi_t = 0$ のときはいつでも、再帰計算は
そのタイムステップ $t$ で終了します。
これにより、長さの異なる系列の重み付き和が得られ、
長い系列はまれですが適切に重み付けされます。
この考え方は
:citet:`Tallec.Ollivier.2017` によって提案されました。

### 戦略の比較

![RNNにおける勾配計算の戦略の比較。上から順に、ランダム化された切り詰め、通常の切り詰め、全計算。](../img/truncated-bptt.svg)
:label:`fig_truncated_bptt`


:numref:`fig_truncated_bptt` は、RNNに対して時間を通した逆伝播を用いて
*The Time Machine* の最初の数文字を解析するときの
3つの戦略を示しています。

* 1行目は、長さの異なる区間にテキストを分割する
  ランダム化された切り詰めです。
* 2行目は、同じ長さの部分系列にテキストを分割する
  通常の切り詰めです。これは、RNNの実験で
  これまで行ってきた方法です。
* 3行目は、計算上実行不可能な式につながる
  全時間逆伝播です。


残念ながら、理論的には魅力的であるものの、
ランダム化された切り詰めは通常の切り詰めよりも
大きく優れているわけではありません。
おそらく、いくつかの要因によるものです。
第1に、過去へいくつかの逆伝播ステップを経た後の
観測の影響は、実際には依存関係を捉えるのに
十分です。
第2に、分散の増加は、より多くのステップで
勾配がより正確になるという事実を打ち消します。
第3に、私たちは実際には、相互作用の範囲が
短いモデルを *望んで* います。
したがって、通常の切り詰めを用いた時間を通した逆伝播には、
望ましい正則化効果がわずかにあります。

## 時間を通した逆伝播の詳細

一般原理を議論したので、
次に時間を通した逆伝播を詳しく見ていきましょう。
:numref:`subsec_bptt_analysis` の解析とは対照的に、
以下では、分解されたすべてのモデルパラメータに関する
目的関数の勾配をどのように計算するかを示します。
簡単のため、バイアスパラメータを持たず、
隠れ層の活性化関数として恒等写像
（$\phi(x)=x$）を用いるRNNを考えます。
タイムステップ $t$ において、単一の例の入力と目標を
それぞれ $\mathbf{x}_t \in \mathbb{R}^d$ と $y_t$ とします。
隠れ状態 $\mathbf{h}_t \in \mathbb{R}^h$ と
出力 $\mathbf{o}_t \in \mathbb{R}^q$ は次のように計算されます。

$$\begin{aligned}\mathbf{h}_t &= \mathbf{W}_\textrm{hx} \mathbf{x}_t + \mathbf{W}_\textrm{hh} \mathbf{h}_{t-1},\\
\mathbf{o}_t &= \mathbf{W}_\textrm{qh} \mathbf{h}_{t},\end{aligned}$$

ここで $\mathbf{W}_\textrm{hx} \in \mathbb{R}^{h \times d}$、
$\mathbf{W}_\textrm{hh} \in \mathbb{R}^{h \times h}$、
$\mathbf{W}_\textrm{qh} \in \mathbb{R}^{q \times h}$ は
重みパラメータです。
時刻 $t$ における損失を $l(\mathbf{o}_t, y_t)$ と表します。
したがって、私たちの目的関数、すなわち系列の先頭から
$T$ タイムステップにわたる損失は

$$L = \frac{1}{T} \sum_{t=1}^T l(\mathbf{o}_t, y_t).$$


RNNの計算中にモデル変数とパラメータの間の依存関係を
可視化するために、:numref:`fig_rnn_bptt` に示すような
モデルの計算グラフを描くことができます。
たとえば、3番目のタイムステップの隠れ状態
$\mathbf{h}_3$ の計算は、モデルパラメータ
$\mathbf{W}_\textrm{hx}$ と $\mathbf{W}_\textrm{hh}$、
前のタイムステップの隠れ状態 $\mathbf{h}_2$、
および現在のタイムステップの入力 $\mathbf{x}_3$ に依存します。

![3タイムステップのRNNモデルにおける依存関係を示す計算グラフ。箱は変数（塗りつぶしなし）またはパラメータ（塗りつぶしあり）を表し、円は演算子を表す。](../img/rnn-bptt.svg)
:label:`fig_rnn_bptt`

先ほど述べたように、:numref:`fig_rnn_bptt` のモデルパラメータは
$\mathbf{W}_\textrm{hx}$、$\mathbf{W}_\textrm{hh}$、$\mathbf{W}_\textrm{qh}$ です。
一般に、このモデルの学習には、これらのパラメータに関する
勾配計算
$\partial L/\partial \mathbf{W}_\textrm{hx}$、
$\partial L/\partial \mathbf{W}_\textrm{hh}$、
$\partial L/\partial \mathbf{W}_\textrm{qh}$ が必要です。
:numref:`fig_rnn_bptt` の依存関係に従って、
矢印と逆向きにたどることで、順に勾配を計算して保存できます。
連鎖律において、異なる形状を持つ
行列、ベクトル、スカラーの積を柔軟に表現するために、
:numref:`sec_backprop` で説明したように
$\textrm{prod}$ 演算子を引き続き用います。


まず、任意のタイムステップ $t$ における
モデル出力に関して目的関数を微分するのは
かなり単純です。

$$\frac{\partial L}{\partial \mathbf{o}_t} =  \frac{\partial l (\mathbf{o}_t, y_t)}{T \cdot \partial \mathbf{o}_t} \in \mathbb{R}^q.$$
:eqlabel:`eq_bptt_partial_L_ot`

これで、出力層のパラメータ $\mathbf{W}_\textrm{qh}$ に関する
目的関数の勾配
$\partial L/\partial \mathbf{W}_\textrm{qh} \in \mathbb{R}^{q \times h}$ を計算できます。
:numref:`fig_rnn_bptt` に基づけば、
目的関数 $L$ は $\mathbf{o}_1, \ldots, \mathbf{o}_T$ を通じて
$\mathbf{W}_\textrm{qh}$ に依存します。
連鎖律を用いると、

$$
\frac{\partial L}{\partial \mathbf{W}_\textrm{qh}}
= \sum_{t=1}^T \textrm{prod}\left(\frac{\partial L}{\partial \mathbf{o}_t}, \frac{\partial \mathbf{o}_t}{\partial \mathbf{W}_\textrm{qh}}\right)
= \sum_{t=1}^T \frac{\partial L}{\partial \mathbf{o}_t} \mathbf{h}_t^\top,
$$

ここで $\partial L/\partial \mathbf{o}_t$ は
:eqref:`eq_bptt_partial_L_ot` で与えられます。

次に、:numref:`fig_rnn_bptt` に示すように、
最終タイムステップ $T$ では、目的関数 $L$ は
出力 $\mathbf{o}_T$ を通じてのみ隠れ状態 $\mathbf{h}_T$ に依存します。
したがって、連鎖律を用いて
隠れ状態に関する勾配
$\partial L/\partial \mathbf{h}_T \in \mathbb{R}^h$ を容易に求められます。

$$\frac{\partial L}{\partial \mathbf{h}_T} = \textrm{prod}\left(\frac{\partial L}{\partial \mathbf{o}_T}, \frac{\partial \mathbf{o}_T}{\partial \mathbf{h}_T} \right) = \mathbf{W}_\textrm{qh}^\top \frac{\partial L}{\partial \mathbf{o}_T}.$$
:eqlabel:`eq_bptt_partial_L_hT_final_step`

任意のタイムステップ $t < T$ では事情が少し複雑になります。
このとき、目的関数 $L$ は $\mathbf{h}_{t+1}$ と $\mathbf{o}_t$ を通じて
$\mathbf{h}_t$ に依存します。
連鎖律に従うと、任意のタイムステップ $t < T$ における
隠れ状態の勾配
$\partial L/\partial \mathbf{h}_t \in \mathbb{R}^h$ は
再帰的に次のように計算できます。


$$\frac{\partial L}{\partial \mathbf{h}_t} = \textrm{prod}\left(\frac{\partial L}{\partial \mathbf{h}_{t+1}}, \frac{\partial \mathbf{h}_{t+1}}{\partial \mathbf{h}_t} \right) + \textrm{prod}\left(\frac{\partial L}{\partial \mathbf{o}_t}, \frac{\partial \mathbf{o}_t}{\partial \mathbf{h}_t} \right) = \mathbf{W}_\textrm{hh}^\top \frac{\partial L}{\partial \mathbf{h}_{t+1}} + \mathbf{W}_\textrm{qh}^\top \frac{\partial L}{\partial \mathbf{o}_t}.$$
:eqlabel:`eq_bptt_partial_L_ht_recur`

解析のために、任意のタイムステップ $1 \leq t \leq T$ について
再帰計算を展開すると、

$$\frac{\partial L}{\partial \mathbf{h}_t}= \sum_{i=t}^T {\left(\mathbf{W}_\textrm{hh}^\top\right)}^{T-i} \mathbf{W}_\textrm{qh}^\top \frac{\partial L}{\partial \mathbf{o}_{T+t-i}}.$$
:eqlabel:`eq_bptt_partial_L_ht`

:eqref:`eq_bptt_partial_L_ht` からわかるように、
この単純な線形例でさえ、長い系列モデルの重要な問題のいくつかを
すでに示しています。
そこには、$\mathbf{W}_\textrm{hh}^\top$ の非常に大きな冪が
含まれうるのです。
その固有値が1より小さいと消失し、
1より大きいと発散します。
これは数値的に不安定であり、
消失勾配や爆発勾配として現れます。
これに対処する1つの方法は、
:numref:`subsec_bptt_analysis` で議論したように、
タイムステップを計算上扱いやすい大きさで
切り詰めることです。
実際には、この切り詰めは、
一定数のタイムステップの後で勾配を切り離すことでも
実現できます。
後ほど、長短期記憶（LSTM）のような
より洗練された系列モデルが、これをさらに緩和できることを見ます。 

最後に、:numref:`fig_rnn_bptt` は、
目的関数 $L$ が、隠れ状態
$\mathbf{h}_1, \ldots, \mathbf{h}_T$ を通じて、
隠れ層のモデルパラメータ
$\mathbf{W}_\textrm{hx}$ と $\mathbf{W}_\textrm{hh}$ に依存することを示しています。
このようなパラメータに関する勾配
$\partial L / \partial \mathbf{W}_\textrm{hx} \in \mathbb{R}^{h \times d}$ と
$\partial L / \partial \mathbf{W}_\textrm{hh} \in \mathbb{R}^{h \times h}$ を計算するには、
連鎖律を適用して

$$
\begin{aligned}
\frac{\partial L}{\partial \mathbf{W}_\textrm{hx}}
&= \sum_{t=1}^T \textrm{prod}\left(\frac{\partial L}{\partial \mathbf{h}_t}, \frac{\partial \mathbf{h}_t}{\partial \mathbf{W}_\textrm{hx}}\right)
= \sum_{t=1}^T \frac{\partial L}{\partial \mathbf{h}_t} \mathbf{x}_t^\top,\\
\frac{\partial L}{\partial \mathbf{W}_\textrm{hh}}
&= \sum_{t=1}^T \textrm{prod}\left(\frac{\partial L}{\partial \mathbf{h}_t}, \frac{\partial \mathbf{h}_t}{\partial \mathbf{W}_\textrm{hh}}\right)
= \sum_{t=1}^T \frac{\partial L}{\partial \mathbf{h}_t} \mathbf{h}_{t-1}^\top,
\end{aligned}
$$

ここで、:eqref:`eq_bptt_partial_L_hT_final_step` と
:eqref:`eq_bptt_partial_L_ht_recur` によって再帰的に計算される
$\partial L/\partial \mathbf{h}_t$ が、
数値安定性に影響する重要な量です。



時間を通した逆伝播は、RNNにおける逆伝播の適用そのものであるため、
:numref:`sec_backprop` で説明したように、
RNNの学習では順伝播と時間を通した逆伝播を交互に行います。
さらに、時間を通した逆伝播では、
上記の勾配を順に計算して保存します。
具体的には、保存された中間値を再利用して
重複計算を避けます。たとえば、
$\partial L/\partial \mathbf{h}_t$ を保存しておき、
$\partial L / \partial \mathbf{W}_\textrm{hx}$ と
$\partial L / \partial \mathbf{W}_\textrm{hh}$ の両方の計算に
使います。


## まとめ

時間を通した逆伝播は、隠れ状態を持つ系列モデルに
逆伝播を適用したものにすぎません。
通常の切り詰めやランダム化された切り詰めのような
切り詰めは、計算上の都合と数値安定性のために必要です。
行列の高次冪は、発散または消失する固有値を引き起こしえます。
これは、爆発勾配や消失勾配として現れます。
効率的に計算するために、時間を通した逆伝播の間、
中間値はキャッシュされます。



## 演習

1. 対称行列 $\mathbf{M} \in \mathbb{R}^{n \times n}$ があり、その固有値を $\lambda_i$、対応する固有ベクトルを $\mathbf{v}_i$（$i = 1, \ldots, n$）とします。一般性を失わずに、$|\lambda_i| \geq |\lambda_{i+1}|$ の順に並んでいると仮定します。 
   1. $\mathbf{M}^k$ の固有値が $\lambda_i^k$ であることを示しなさい。
   1. ランダムなベクトル $\mathbf{x} \in \mathbb{R}^n$ に対して、高い確率で $\mathbf{M}^k \mathbf{x}$ が $\mathbf{M}$ の固有ベクトル $\mathbf{v}_1$ に非常によく整列することを証明しなさい。この主張を形式化しなさい。
   1. 上の結果はRNNの勾配にとって何を意味するか。
1. 勾配クリッピング以外に、再帰型ニューラルネットワークにおける勾配爆発に対処する他の方法を思いつきますか？

[Discussions](https://discuss.d2l.ai/t/334)\n