{.python .input}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow', 'jax'])
```

# データ前処理
:label:`sec_pandas`

これまで、私たちはすぐに使えるテンソルとして届く合成データを扱ってきました。  
しかし、現実世界で深層学習を適用するには、任意の形式で保存された雑然としたデータを取り出し、必要に応じて前処理しなければなりません。  
幸いなことに、*pandas* [ライブラリ](https://pandas.pydata.org/) はその大部分を担ってくれます。  
この節は、適切な *pandas* [チュートリアル](https://pandas.pydata.org/pandas-docs/stable/user_guide/10min.html) の代わりにはなりませんが、よく使う基本的な処理についての速習になります。

## データセットの読み込み

カンマ区切り値（CSV）ファイルは、表形式（スプレッドシートのような）データを保存するために広く使われています。  
CSV では、各行が1つのレコードに対応し、いくつかの（カンマ区切りの）フィールドから構成されます。たとえば、"Albert Einstein,March 14 1879,Ulm,Federal polytechnic school,field of gravitational physics" のようなものです。  
`pandas` を使って CSV ファイルを読み込む方法を示すために、ここでは（**以下で CSV ファイルを作成します**） `../data/house_tiny.csv` を用います。  
このファイルは住宅データセットを表しており、各行が1つの住宅に対応し、列は部屋数（`NumRooms`）、屋根の種類（`RoofType`）、価格（`Price`）を表します。

```{.python .input}
%%tab all
import os

os.makedirs(os.path.join('..', 'data'), exist_ok=True)
data_file = os.path.join('..', 'data', 'house_tiny.csv')
with open(data_file, 'w') as f:
    f.write('''NumRooms,RoofType,Price
NA,NA,127500
2,NA,106000
4,Slate,178100
NA,NA,140000''')
```

では、`pandas` をインポートして `read_csv` でデータセットを読み込みましょう。

```{.python .input}
%%tab all
import pandas as pd

data = pd.read_csv(data_file)
print(data)
```

## データの準備

教師あり学習では、ある一連の *入力* 値が与えられたときに、指定された *目標* 値を予測するようにモデルを訓練します。  
データセットを処理する最初のステップは、入力値に対応する列と目標値に対応する列を分けることです。  
列は名前で選択することも、整数位置に基づくインデックス指定（`iloc`）で選択することもできます。

お気づきかもしれませんが、`pandas` は CSV の `NA` の値をすべて、特別な `NaN`（*not a number*）値に置き換えました。  
これは、たとえば "3,,,270000" のように、項目が空欄の場合にも起こります。  
これらは *欠損値* と呼ばれ、データサイエンスにおける「南京虫」のようなもので、あなたのキャリアを通じてずっと向き合うことになる厄介な問題です。  
文脈に応じて、欠損値は *補完* か *削除* によって扱われます。  
補完では欠損値をその値の推定値で置き換え、削除では欠損値を含む行または列を単純に捨てます。

以下に、よく使われる補完のヒューリスティックを示します。  
[**カテゴリ型の入力欄では、`NaN` を1つのカテゴリとして扱うことができます。**]  
`RoofType` 列は `Slate` と `NaN` の値を取るので、`pandas` はこの列を `RoofType_Slate` と `RoofType_nan` の2列に変換できます。  
屋根の種類が `Slate` の行では、`RoofType_Slate` と `RoofType_nan` の値はそれぞれ 1 と 0 になります。  
`RoofType` の値が欠損している行では、その逆になります。

```{.python .input}
%%tab all
inputs, targets = data.iloc[:, 0:2], data.iloc[:, 2]
inputs = pd.get_dummies(inputs, dummy_na=True)
print(inputs)
```

数値の欠損値については、よく使われるヒューリスティックとして、[**`NaN` の項目を対応する列の平均値で置き換える**] 方法があります。

```{.python .input}
%%tab all
inputs = inputs.fillna(inputs.mean())
print(inputs)
```

## テンソル形式への変換

これで [**`inputs` と `targets` のすべての項目が数値になったので、テンソルに読み込めます**]（:numref:`sec_ndarray` を思い出してください）。

```{.python .input}
%%tab mxnet
from mxnet import np

X, y = np.array(inputs.to_numpy(dtype=float)), np.array(targets.to_numpy(dtype=float))
X, y
```

```{.python .input}
%%tab pytorch
import torch

X = torch.tensor(inputs.to_numpy(dtype=float))
y = torch.tensor(targets.to_numpy(dtype=float))
X, y
```

```{.python .input}
%%tab tensorflow
import tensorflow as tf

X = tf.constant(inputs.to_numpy(dtype=float))
y = tf.constant(targets.to_numpy(dtype=float))
X, y
```

```{.python .input}
%%tab jax
from jax import numpy as jnp

X = jnp.array(inputs.to_numpy(dtype=float))
y = jnp.array(targets.to_numpy(dtype=float))
X, y
```

## 議論

これで、データ列を分割し、欠損変数を補完し、`pandas` のデータをテンソルに読み込む方法がわかりました。  
:numref:`sec_kaggle_house` では、さらにいくつかのデータ処理スキルを学びます。  
この速習では話を単純にしましたが、データ処理はかなり複雑になりえます。  
たとえば、データセットが1つの CSV ファイルにまとまっているのではなく、リレーショナルデータベースから抽出された複数のファイルに分散していることがあります。  
たとえば電子商取引アプリケーションでは、顧客住所はあるテーブルに、購買データは別のテーブルにあるかもしれません。  
さらに、実務ではカテゴリ型や数値型以外にも、テキスト文字列、画像、音声データ、点群など、さまざまなデータ型に直面します。  
しばしば、データ処理が機械学習パイプラインの最大のボトルネックにならないようにするために、高度なツールや効率的なアルゴリズムが必要になります。  
これらの問題は、コンピュータビジョンや自然言語処理に進むと現れてきます。  
最後に、データ品質にも注意を払わなければなりません。  
現実世界のデータセットは、外れ値、センサーによる誤測定、記録ミスなどに悩まされることが多く、データをどのモデルに入れる前にも対処が必要です。  
[seaborn](https://seaborn.pydata.org/)、[Bokeh](https://docs.bokeh.org/)、[matplotlib](https://matplotlib.org/) などのデータ可視化ツールは、データを手作業で確認し、どのような問題に対処すべきかについて直感を養うのに役立ちます。


## 演習

1. たとえば UCI Machine Learning Repository の Abalone などのデータセットを読み込み、その性質を調べてみましょう。欠損値を含む割合はどれくらいでしょうか。変数のうち、数値型、カテゴリ型、テキスト型の割合はどれくらいでしょうか。
1. 列番号ではなく列名によってデータ列をインデックス指定し、選択してみましょう。pandas の [indexing](https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html) のドキュメントには、その方法の詳細が載っています。
1. この方法でどれくらい大きなデータセットまで読み込めると思いますか。どのような制約があるでしょうか。ヒント：データの読み込み時間、表現、処理、メモリ使用量を考えてみてください。自分のノートパソコンで試してみましょう。サーバー上で試すとどうなりますか。
1. カテゴリ数が非常に多いデータをどのように扱いますか。カテゴリラベルがすべて一意だったらどうでしょうか。後者も含めるべきでしょうか。
1. pandas の代替として何が考えられますか。ファイルから NumPy テンソルを読み込む方法はどうでしょうか。[Pillow](https://python-pillow.org/)、つまり Python Imaging Library も調べてみましょう。 

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/28)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/29)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/195)
:end_tab:

:begin_tab:`jax`
[Discussions](https://discuss.d2l.ai/t/17967)
:end_tab:\n