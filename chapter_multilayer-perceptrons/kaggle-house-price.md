{.python .input  n=1}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow', 'jax'])
```

# Kaggleで住宅価格を予測する
:label:`sec_kaggle_house`

ここまでで、深層ネットワークを構築・学習するための基本的な道具を導入し、
さらに重み減衰やドロップアウトといった手法でそれらを正則化する方法も学びました。
いよいよ、Kaggleコンペティションに参加して、
これまでの知識を実践に移す準備が整いました。
住宅価格予測コンペティションは、始めるのに最適な題材です。
データは比較的一般的で、音声や動画のように特殊なモデルを必要とするような
特異な構造は含んでいません。
このデータセットは :citet:`De-Cock.2011` によって収集されたもので、
2006--2010年のアイオワ州エイムズの住宅価格を扱っています。
これは、Harrison と Rubinfeld (1978) による有名な [Boston housing dataset](https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.names)
よりもかなり大きく、例の数も特徴量の数も多くなっています。


この節では、データ前処理、モデル設計、ハイパーパラメータ選択の詳細を順に説明します。
実践的なアプローチを通じて、
データサイエンティストとしてのキャリアに役立つ直感を
身につけてもらえればと思います。

```{.python .input}
%%tab mxnet
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import gluon, autograd, init, np, npx
from mxnet.gluon import nn
import pandas as pd

npx.set_np()
```

```{.python .input}
%%tab pytorch
%matplotlib inline
from d2l import torch as d2l
import torch
from torch import nn
import pandas as pd
```

```{.python .input}
%%tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
import tensorflow as tf
import pandas as pd
```

```{.python .input}
%%tab jax
%matplotlib inline
from d2l import jax as d2l
import jax
from jax import numpy as jnp
import numpy as np
import pandas as pd
```

## データのダウンロード

本書を通して、さまざまなダウンロード済みデータセットを用いてモデルを学習・評価します。
ここでは、zip または tar ファイルをダウンロードして展開するための
(**2つのユーティリティ関数を実装**)します。
ここでも、そのようなユーティリティ関数の実装詳細は省略します。

```{.python .input  n=2}
%%tab all
def download(url, folder, sha1_hash=None):
    """Download a file to folder and return the local filepath."""

def extract(filename, folder):
    """Extract a zip/tar file into folder."""
```

## Kaggle

[Kaggle](https://www.kaggle.com) は、機械学習コンペティションを開催する
人気のあるプラットフォームです。
各コンペティションは1つのデータセットを中心に構成されており、
多くは賞金を提供する利害関係者によって後援されています。
このプラットフォームは、フォーラムや共有コードを通じて
ユーザー同士の交流を支援し、
協力と競争の両方を促進します。
ランキング上位を追い求めるあまり暴走し、
研究者が本質的な問いを立てることよりも前処理の細部に
視野狭窄的に集中してしまうことも少なくありませんが、
一方で、競合手法を直接定量比較できるようにする
プラットフォームの客観性には大きな価値があります。
さらに、コード共有によって
何がうまくいき、何がうまくいかなかったのかを
誰もが学べるようになります。
Kaggleコンペティションに参加したい場合は、
まずアカウント登録が必要です
(:numref:`fig_kaggle` を参照)。

![The Kaggle website.](../img/kaggle.png)
:width:`400px`
:label:`fig_kaggle`

:numref:`fig_house_pricing` に示すように、住宅価格予測コンペティションのページでは、
データセットを見つけたり（"Data" タブの下）、
予測を提出したり、順位を確認したりできます。
URL は次のとおりです。

> https://www.kaggle.com/c/house-prices-advanced-regression-techniques

![The house price prediction competition page.](../img/house-pricing.png)
:width:`400px`
:label:`fig_house_pricing`

## データセットへのアクセスと読み込み

コンペティションのデータは
訓練セットとテストセットに分かれていることに注意してください。
各レコードには住宅の資産価値と、
道路の種類、建築年、屋根の種類、地下室の状態などの属性が含まれます。
特徴量にはさまざまなデータ型が含まれています。
たとえば、建築年は整数で表され、
屋根の種類は離散的なカテゴリ割り当てで表され、
その他の特徴量は浮動小数点数で表されます。
そして、ここで現実の厄介さが出てきます。
一部の例ではデータが完全に欠落しており、
欠損値は単に "na" として示されています。
各住宅の価格は訓練セットにのみ含まれています
（コンペティションなのですから当然です）。
訓練セットを分割して検証セットを作りたいところですが、
Kaggle に予測をアップロードした後でしか
公式テストセット上でモデルを評価できません。
:numref:`fig_house_pricing` のコンペティションタブにある "Data" タブには、
データをダウンロードするためのリンクがあります。

まずは、:numref:`sec_pandas` で紹介した `pandas` を使って
[**データを読み込み、処理**]してみましょう。
便宜上、Kaggle の住宅データセットをダウンロードしてキャッシュできます。
このデータセットに対応するファイルがすでにキャッシュディレクトリに存在し、
その SHA-1 が `sha1_hash` と一致する場合、冗長なダウンロードで
インターネット回線を無駄にしないよう、キャッシュ済みファイルを使用します。

```{.python .input  n=30}
%%tab all
class KaggleHouse(d2l.DataModule):
    def __init__(self, batch_size, train=None, val=None):
        super().__init__()
        self.save_hyperparameters()
        if self.train is None:
            self.raw_train = pd.read_csv(d2l.download(
                d2l.DATA_URL + 'kaggle_house_pred_train.csv', self.root,
                sha1_hash='585e9cc93e70b39160e7921475f9bcd7d31219ce'))
            self.raw_val = pd.read_csv(d2l.download(
                d2l.DATA_URL + 'kaggle_house_pred_test.csv', self.root,
                sha1_hash='fa19780a7b011d9b009e8bff8e99922a8ee2eb90'))
```

訓練データセットには 1460 個の例、80 個の特徴量、1 個のラベルが含まれ、
一方で検証データには 1459 個の例と 80 個の特徴量が含まれています。

```{.python .input  n=31}
%%tab all
data = KaggleHouse(batch_size=64)
print(data.raw_train.shape)
print(data.raw_val.shape)
```

## データ前処理

最初の4つの例について、最初の4つと最後の2つの特徴量、およびラベル（SalePrice）を
[**見てみましょう**]。

```{.python .input  n=10}
%%tab all
print(data.raw_train.iloc[:4, [0, 1, 2, 3, -3, -2, -1]])
```

各例では、最初の特徴量が識別子であることがわかります。
これはモデルが各訓練例を区別するのに役立ちます。
便利ではありますが、予測の目的には何の情報も持ちません。
したがって、モデルにデータを入力する前に、
この特徴量はデータセットから削除します。
さらに、さまざまなデータ型が混在しているため、
モデリングを始める前にデータ前処理が必要になります。


まず数値特徴量から始めましょう。
最初にヒューリスティックとして、
[**すべての欠損値を対応する特徴量の平均値で置き換えます。**]
その後、すべての特徴量を共通の尺度にそろえるために、
(**データを標準化し、特徴量を平均0・分散1に再スケーリングします**):

$$x \leftarrow \frac{x - \mu}{\sigma},$$

ここで $\mu$ と $\sigma$ はそれぞれ平均と標準偏差を表します。
これにより特徴量（変数）が本当に平均0・分散1になることを確認するには、
$E[\frac{x-\mu}{\sigma}] = \frac{\mu - \mu}{\sigma} = 0$ であり、
また $E[(x-\mu)^2] = (\sigma^2 + \mu^2) - 2\mu^2+\mu^2 = \sigma^2$ であることに注意してください。
直感的には、データを標準化する理由は2つあります。
第1に、最適化に都合がよいことです。
第2に、どの特徴量が関連するかを *a priori* に知ることはできないので、
ある特徴量に割り当てられた係数を他よりも強く罰したくないからです。

[**次に離散値を扱います。**]
これには "MSZoning" のような特徴量が含まれます。
(**これらは one-hot エンコーディングで置き換えます**)
これは、先に多クラスラベルをベクトルに変換したときと同じ方法です
(:numref:`subsec_classification-problem` を参照)。
たとえば、"MSZoning" は "RL" と "RM" の値を取ります。
"MSZoning" 特徴量を削除すると、
値が 0 か 1 の2つの新しい指示特徴量
"MSZoning_RL" と "MSZoning_RM" が作成されます。
one-hot エンコーディングに従えば、
元の "MSZoning" の値が "RL" なら、
"MSZoning_RL" は 1 で "MSZoning_RM" は 0 です。
`pandas` パッケージはこれを自動的に行ってくれます。

```{.python .input  n=32}
%%tab all
@d2l.add_to_class(KaggleHouse)
def preprocess(self):
    # Remove the ID and label columns
    label = 'SalePrice'
    features = pd.concat(
        (self.raw_train.drop(columns=['Id', label]),
         self.raw_val.drop(columns=['Id'])))
    # Standardize numerical columns
    numeric_features = features.dtypes[features.dtypes!='object'].index
    features[numeric_features] = features[numeric_features].apply(
        lambda x: (x - x.mean()) / (x.std()))
    # Replace NAN numerical features by 0
    features[numeric_features] = features[numeric_features].fillna(0)
    # Replace discrete features by one-hot encoding
    features = pd.get_dummies(features, dummy_na=True)
    # Save preprocessed features
    self.train = features[:self.raw_train.shape[0]].copy()
    self.train[label] = self.raw_train[label]
    self.val = features[self.raw_train.shape[0]:].copy()
```

この変換によって、特徴量の数が 79 から 331 に増えることがわかります
（ID 列とラベル列を除く）。

```{.python .input  n=33}
%%tab all
data.preprocess()
data.train.shape
```

## 誤差尺度

まずは二乗損失を用いた線形モデルを学習してみましょう。
当然ながら、この線形モデルがコンペティションで勝てる提出につながることはありませんが、
データの中に意味のある情報があるかどうかを確認するための健全性チェックにはなります。
ここでランダム予測よりも良い結果が出せないなら、
データ処理にバグがある可能性が高いでしょう。
そして、うまくいくなら、線形モデルはベースラインとして機能し、
単純なモデルが最良報告モデルにどれくらい近づけるのかについての直感を与え、
より洗練されたモデルからどれほどの改善が期待できるかの目安になります。

住宅価格では、株価と同様に、
絶対量よりも相対量のほうが重要です。
したがって、[**絶対誤差 $y - \hat{y}$ よりも
相対誤差 $\frac{y - \hat{y}}{y}$ のほうを重視する傾向があります**]。
たとえば、オハイオ州の田舎で住宅価格を推定していて、
典型的な住宅の価値が 125,000 ドルであるときに、
予測が 100,000 ドル外れたとしたら、
おそらくひどい出来です。
一方、カリフォルニア州ロスアルトスヒルズで同じだけ外れたとしても、
それは驚くほど正確な予測かもしれません
（そこでは住宅価格の中央値が 400 万ドルを超えます）。

(**この問題に対処する1つの方法は、
価格推定の対数における差を測ることです。**)
実際、これはコンペティションが提出物の品質を評価するために用いている
公式の誤差尺度でもあります。
結局のところ、$|\log y - \log \hat{y}| \leq \delta$ のような小さな値 $\delta$ は、
$e^{-\delta} \leq \frac{\hat{y}}{y} \leq e^\delta$ に対応します。
これにより、予測価格の対数とラベル価格の対数の間の
次の二乗平均平方根誤差が得られます。

$$\sqrt{\frac{1}{n}\sum_{i=1}^n\left(\log y_i -\log \hat{y}_i\right)^2}.$$

```{.python .input  n=60}
%%tab all
@d2l.add_to_class(KaggleHouse)
def get_dataloader(self, train):
    label = 'SalePrice'
    data = self.train if train else self.val
    if label not in data: return
    get_tensor = lambda x: d2l.tensor(x.values.astype(float),
                                      dtype=d2l.float32)
    # Logarithm of prices 
    tensors = (get_tensor(data.drop(columns=[label])),  # X
               d2l.reshape(d2l.log(get_tensor(data[label])), (-1, 1)))  # Y
    return self.get_tensorloader(tensors, train)
```

## $K$-分割交差検証

:ref:`subsec_generalization-model-selection` で
[**交差検証**] を導入し、モデル選択の扱いについて議論したことを思い出してください。
ここではそれを活用して、モデル設計の選択やハイパーパラメータの調整を行います。
まず、$K$-分割交差検証手順におけるデータの
$i^\textrm{th}$ 分割を返す関数が必要です。
これは、$i^\textrm{th}$ 区間を検証データとして切り出し、
残りを訓練データとして返します。
これはデータの扱いとして最も効率的な方法ではないことに注意してください。
データセットがかなり大きい場合には、
もっと賢い方法を確実に採るでしょう。
しかし、この問題は単純なので、
そのような追加の複雑さはコードを不必要にわかりにくくしてしまうかもしれません。
したがって、ここでは省略しても問題ありません。

```{.python .input}
%%tab all
def k_fold_data(data, k):
    rets = []
    fold_size = data.train.shape[0] // k
    for j in range(k):
        idx = range(j * fold_size, (j+1) * fold_size)
        rets.append(KaggleHouse(data.batch_size, data.train.drop(index=idx),  
                                data.train.loc[idx]))    
    return rets
```

$K$ 回学習したときの [**平均検証誤差が返されます**]。
$K$-分割交差検証では、これを用います。

```{.python .input}
%%tab all
def k_fold(trainer, data, k, lr):
    val_loss, models = [], []
    for i, data_fold in enumerate(k_fold_data(data, k)):
        model = d2l.LinearRegression(lr)
        model.board.yscale='log'
        if i != 0: model.board.display = False
        trainer.fit(model, data_fold)
        val_loss.append(float(model.board.data['val_loss'][-1].y))
        models.append(model)
    print(f'average validation log mse = {sum(val_loss)/len(val_loss)}')
    return models
```

## [**モデル選択**]

この例では、調整していないハイパーパラメータの組を選び、
モデルの改善は読者に委ねます。
適切な選択を見つけるには、
最適化する変数の数に応じて時間がかかることがあります。
十分に大きなデータセットと通常の種類のハイパーパラメータであれば、
$K$-分割交差検証は複数回の試行に対して
かなり頑健である傾向があります。
しかし、非現実的に多くの विकल्पを試すと、
検証性能が真の誤差をもはや代表しなくなるかもしれません。

```{.python .input}
%%tab all
trainer = d2l.Trainer(max_epochs=10)
models = k_fold(trainer, data, k=5, lr=0.01)
```

ときには、あるハイパーパラメータ集合に対する訓練誤差の数が非常に少ないのに、
$K$-分割交差検証での誤差数はかなり大きくなることがあります。
これは過学習していることを示しています。
学習中は、この2つの数値の両方を監視したいところです。
過学習が小さいということは、データがより強力なモデルを支えられる可能性を示しているかもしれません。
大きな過学習は、正則化手法を取り入れることで改善できることを示唆しているかもしれません。

##  [**Kaggle への予測提出**]

良いハイパーパラメータの選び方がわかったので、
$K$ 個のモデルすべてによるテストセット上の予測の平均を
計算してみましょう。
予測を csv ファイルに保存しておくと、
Kaggle への結果アップロードが簡単になります。
次のコードは `submission.csv` というファイルを生成します。

```{.python .input}
%%tab all
if tab.selected('pytorch', 'mxnet', 'tensorflow'):
    preds = [model(d2l.tensor(data.val.values.astype(float), dtype=d2l.float32))
             for model in models]
if tab.selected('jax'):
    preds = [model.apply({'params': trainer.state.params},
             d2l.tensor(data.val.values.astype(float), dtype=d2l.float32))
             for model in models]
# Taking exponentiation of predictions in the logarithm scale
ensemble_preds = d2l.reduce_mean(d2l.exp(d2l.concat(preds, 1)), 1)
submission = pd.DataFrame({'Id':data.raw_val.Id,
                           'SalePrice':d2l.numpy(ensemble_preds)})
submission.to_csv('submission.csv', index=False)
```

次に、:numref:`fig_kaggle_submit2` に示すように、
Kaggle に予測を提出し、
テストセット上の実際の住宅価格（ラベル）とどの程度一致しているかを確認できます。
手順はとても簡単です。

* Kaggle のウェブサイトにログインし、住宅価格予測コンペティションのページを開きます。
* “Submit Predictions” または “Late Submission” ボタンをクリックします。
* ページ下部の点線枠内にある “Upload Submission File” ボタンをクリックし、アップロードしたい予測ファイルを選択します。
* ページ下部の “Make Submission” ボタンをクリックして結果を表示します。

![Submitting data to Kaggle.](../img/kaggle-submit2.png)
:width:`400px`
:label:`fig_kaggle_submit2`

## まとめと考察

実データにはしばしばさまざまなデータ型が混在しており、前処理が必要です。
実数値データを平均0・分散1に再スケーリングするのは良いデフォルトです。
欠損値を平均値で置き換えるのも同様です。
さらに、カテゴリ特徴量を指示特徴量に変換すると、
それらを one-hot ベクトルとして扱えるようになります。
絶対誤差よりも相対誤差を重視する傾向がある場合には、
予測の対数における差を測ることができます。
モデルを選択し、ハイパーパラメータを調整するには、
$K$-分割交差検証を使えます。



## 演習

1. この節の予測を Kaggle に提出してみましょう。どの程度の成績でしたか？
1. 欠損値を平均値で置き換えるのは常に良い考えでしょうか？ ヒント: 値がランダムに欠損していない状況を構成できますか？
1. $K$-分割交差検証によってハイパーパラメータを調整し、スコアを改善しましょう。
1. モデルを改善することでスコアを改善しましょう（たとえば、層、重み減衰、ドロップアウト）。
1. この節で行ったように連続数値特徴量を標準化しないとどうなりますか？

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/106)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/107)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/237)
:end_tab:

:begin_tab:`jax`
[Discussions](https://discuss.d2l.ai/t/17988)
:end_tab:\n