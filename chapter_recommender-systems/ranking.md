# レコメンダシステムのための個人化ランキング

前の節では、明示的フィードバックのみを考慮し、モデルは観測された評価値で学習・テストされていました。このような手法には2つの欠点があります。第1に、現実世界のシナリオではフィードバックの大半は明示的ではなく暗黙的であり、明示的フィードバックの収集にはより高いコストがかかることです。第2に、ユーザーの興味を予測するうえで有用である可能性のある、未観測のユーザー・アイテム対が完全に無視されてしまうため、評価値がランダムに欠損しているのではなく、ユーザーの嗜好によって欠損している場合には、これらの手法は適していません。未観測のユーザー・アイテム対は、実際の負のフィードバック（ユーザーがそのアイテムに興味を持っていない）と欠損値（将来そのアイテムと相互作用するかもしれない）の混合です。行列分解と AutoRec では、未観測の対を単純に無視していました。明らかに、これらのモデルは観測済み対と未観測対を区別できず、通常、個人化ランキングタスクには適していません。

このため、暗黙的フィードバックから順位付けされた推薦リストを生成することを目的とした推薦モデル群が人気を集めています。一般に、個人化ランキングモデルは pointwise、pairwise、listwise のいずれかのアプローチで最適化できます。Pointwise アプローチは一度に1つの相互作用を考慮し、分類器または回帰器を学習して個々の嗜好を予測します。行列分解と AutoRec は pointwise の目的関数で最適化されます。Pairwise アプローチは各ユーザーについてアイテムのペアを考慮し、そのペアに対する最適な順序を近似することを目指します。通常、pairwise アプローチは、相対的な順序を予測することがランキングの本質に近いため、ランキングタスクにより適しています。Listwise アプローチは、アイテム全体のリストの順序を近似します。たとえば、Normalized Discounted Cumulative Gain ([NDCG](https://en.wikipedia.org/wiki/Discounted_cumulative_gain)) のようなランキング指標を直接最適化します。ただし、listwise アプローチは pointwise や pairwise よりも複雑で計算コストが高くなります。この節では、2つの pairwise 目的関数／損失である Bayesian Personalized Ranking loss と Hinge loss、およびそれぞれの実装を紹介します。

## Bayesian Personalized Ranking Loss とその実装

Bayesian personalized ranking (BPR) :cite:`Rendle.Freudenthaler.Gantner.ea.2009` は、最大事後推定量から導かれる pairwise の個人化ランキング損失です。これは既存の多くの推薦モデルで広く用いられています。BPR の学習データは、正例と負例（欠損値）の両方のペアから構成されます。これは、ユーザーが正例アイテムを他のすべての未観測アイテムより好むと仮定します。

形式的には、学習データは $(u, i, j)$ の形のタプルで構成され、これはユーザー $u$ がアイテム $i$ をアイテム $j$ より好むことを表します。事後確率を最大化することを目的とする BPR のベイズ的定式化は次のように与えられます。

$$
p(\Theta \mid >_u )  \propto  p(>_u \mid \Theta) p(\Theta)
$$

ここで $\Theta$ は任意の推薦モデルのパラメータを表し、$>_u$ はユーザー $u$ に対する全アイテムの望ましい個人化された全順序を表します。最大事後推定量を定式化することで、個人化ランキングタスクの一般的な最適化基準を導けます。

$$
\begin{aligned}
\textrm{BPR-OPT} : &= \ln p(\Theta \mid >_u) \\
         & \propto \ln p(>_u \mid \Theta) p(\Theta) \\
         &= \ln \prod_{(u, i, j \in D)} \sigma(\hat{y}_{ui} - \hat{y}_{uj}) p(\Theta) \\
         &= \sum_{(u, i, j \in D)} \ln \sigma(\hat{y}_{ui} - \hat{y}_{uj}) + \ln p(\Theta) \\
         &= \sum_{(u, i, j \in D)} \ln \sigma(\hat{y}_{ui} - \hat{y}_{uj}) - \lambda_\Theta \|\Theta \|^2
\end{aligned}
$$


ここで $D \stackrel{\textrm{def}}{=} \{(u, i, j) \mid i \in I^+_u \wedge j \in I \backslash I^+_u \}$ は学習集合であり、$I^+_u$ はユーザー $u$ が好んだアイテム、$I$ は全アイテム、$I \backslash I^+_u$ はユーザーが好んだアイテムを除くその他すべてのアイテムを表します。$\hat{y}_{ui}$ と $\hat{y}_{uj}$ は、それぞれユーザー $u$ に対するアイテム $i$ と $j$ の予測スコアです。事前分布 $p(\Theta)$ は、平均0、分散共分散行列 $\Sigma_\Theta$ をもつ正規分布です。ここでは $\Sigma_\Theta = \lambda_\Theta I$ とします。

![Illustration of Bayesian Personalized Ranking](../img/rec-ranking.svg)
基底クラス `mxnet.gluon.loss.Loss` を実装し、`forward` メソッドをオーバーライドして Bayesian personalized ranking loss を構成します。まず、Loss クラスと np モジュールをインポートします。

```{.python .input  n=5}
#@tab mxnet
from mxnet import gluon, np, npx
npx.set_np()
```

BPR loss の実装は次のとおりです。

```{.python .input  n=2}
#@tab mxnet
#@save
class BPRLoss(gluon.loss.Loss):
    def __init__(self, weight=None, batch_axis=0, **kwargs):
        super(BPRLoss, self).__init__(weight=None, batch_axis=0, **kwargs)

    def forward(self, positive, negative):
        distances = positive - negative
        loss = - np.sum(np.log(npx.sigmoid(distances)), 0, keepdims=True)
        return loss
```

## Hinge Loss とその実装

ランキングのための Hinge loss は、SVM などの分類器でよく使われる gluon ライブラリ提供の [hinge loss](https://mxnet.incubator.apache.org/api/python/gluon/loss.html#mxnet.gluon.loss.HingeLoss) とは異なる形をしています。レコメンダシステムにおけるランキング用の損失は、次の形をとります。

$$
 \sum_{(u, i, j \in D)} \max( m - \hat{y}_{ui} + \hat{y}_{uj}, 0)
$$

ここで $m$ は安全マージンの大きさです。これは、負のアイテムを正のアイテムから遠ざけることを目的としています。BPR と同様に、絶対的な出力ではなく正例と負例の相対的な距離を最適化しようとするため、レコメンダシステムに非常に適しています。

```{.python .input  n=3}
#@tab mxnet
#@save
class HingeLossbRec(gluon.loss.Loss):
    def __init__(self, weight=None, batch_axis=0, **kwargs):
        super(HingeLossbRec, self).__init__(weight=None, batch_axis=0,
                                            **kwargs)

    def forward(self, positive, negative, margin=1):
        distances = positive - negative
        loss = np.sum(np.maximum(- distances + margin, 0))
        return loss
```

これら2つの損失は、推薦における個人化ランキングで互換的に使用できます。

## まとめ

- レコメンダシステムにおける個人化ランキングタスクには、pointwise、pairwise、listwise の3種類のランキング損失があります。
- 2つの pairwise 損失である Bayesian personalized ranking loss と hinge loss は、互換的に使用できます。

## 演習

- BPR と hinge loss には、利用可能な変種はありますか？
- BPR または hinge loss を使用する推薦モデルを見つけられますか？

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/402)
:end_tab:\n