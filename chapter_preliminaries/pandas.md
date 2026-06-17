{.python .input}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow', 'jax'])
```

# データ前処理
:label:`sec_pandas`

これまで、すぐに利用できるテンソルとして与えられた合成データを扱ってきた。  
しかし、現実世界で深層学習を適用するには、さまざまな形式で保存された雑多なデータを読み込み、必要に応じて前処理しなければならない。
幸い、*pandas* [ライブラリ](https://pandas.pydata.org/) を使えば、その大部分を自動化し、簡潔に記述できる。
この節は pandas の包括的な [チュートリアル](https://pandas.pydata.org/pandas-docs/stable/user_guide/10min.html) ではないが、頻繁に用いる基本的なデータ処理を手早く学ぶことを目的とする。

## データセットの読み込み

カンマ区切り値（CSV）ファイルは、表形式データ（スプレッドシートのようなデータ）を保存するために広く用いられている。  
CSV では、各行が1つのレコードに対応し、いくつかの（カンマで区切られた）フィールドから構成される。たとえば、"Albert Einstein,March 14 1879,Ulm,Federal polytechnic school,field of gravitational physics" のような形式である。  
`pandas` を用いた CSV ファイルの読み込み方を示すため、ここでは [**以下のような CSV ファイル**] `../data/house_tiny.csv` を作成する。  
このファイルは住宅データセットを表し、各行が1軒の住宅に対応する。列は部屋数（`NumRooms`）、屋根の種類（`RoofType`）、価格（`Price`）を表す。

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

次に、`pandas` をインポートし、`read_csv` でデータセットを読み込む。

```{.python .input}
%%tab all
import pandas as pd

data = pd.read_csv(data_file)
print(data)
```

## データの準備

教師あり学習では、ある一連の *入力* 値が与えられたときに、指定された *目標* 値を予測するようモデルを訓練する。  
データセットを処理する最初の段階は、入力値に対応する列と目標値に対応する列を分けることである。  
列は名前で選択してもよいし、整数位置に基づくインデックス指定（`iloc`）で選択してもよい。

すでに気づいたかもしれないが、`pandas` は CSV 中の `NA` を特別な `NaN`（*not a number*）値に置き換える。  
これは、たとえば "3,,,270000" のように、項目が空欄になっている場合にも起こる。  
このような値は *欠損値* と呼ばれ、データサイエンスにおける主要な難題の1つである。実務では継続的に向き合うことになる問題である。
文脈に応じて、欠損値は *補完*（imputation）または *削除*（deletion）によって処理する。
補完では欠損値を推定値で置き換え、削除では欠損値を含む行または列をデータセットから取り除く。

以下では、よく用いられる補完のヒューリスティックを示す。  
[**カテゴリ型の入力欄では、`NaN` を1つのカテゴリとして扱える。**]  
`RoofType` 列は `Slate` と `NaN` の値を取るため、`pandas` はこの列を `RoofType_Slate` と `RoofType_nan` の2列に変換できる。  
屋根の種類が `Slate` である行では、`RoofType_Slate` と `RoofType_nan` の値はそれぞれ 1 と 0 になる。  
`RoofType` の値が欠損している行では、その逆になる。

```{.python .input}
%%tab all
inputs, targets = data.iloc[:, 0:2], data.iloc[:, 2]
inputs = pd.get_dummies(inputs, dummy_na=True)
print(inputs)
```

数値の欠損値については、よく用いられるヒューリスティックとして、[**`NaN` の項目を対応する列の平均値で置き換える**] 方法がある。

```{.python .input}
%%tab all
inputs = inputs.fillna(inputs.mean())
print(inputs)
```

## テンソル形式への変換

これで [**`inputs` と `targets` のすべての項目が数値になったので、テンソルに変換できる**]（:numref:`sec_ndarray` を参照）。

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

以上で、データ列を分割し、欠損値を補完し、`pandas` のデータをテンソルへ変換する方法を学んだ。  
:numref:`sec_kaggle_house` では、さらにいくつかのデータ処理技法を扱う。  
この速習では話を単純化したが、実際のデータ処理ははるかに複雑になりうる。  
たとえば、データセットが1つの CSV ファイルにまとまっているとは限らず、リレーショナルデータベースから抽出された複数のファイルに分散していることもある。  
電子商取引アプリケーションであれば、顧客の住所があるテーブルにあり、購買データが別のテーブルにあるかもしれない。  
さらに、実務ではカテゴリ型や数値型だけでなく、テキスト文字列、画像、音声データ、点群など、多様なデータ型を扱う。  
しばしば、データ処理が機械学習パイプライン全体の最大のボトルネックにならないよう、高度なツールや効率的なアルゴリズムが必要になる。  
こうした問題は、コンピュータビジョンや自然言語処理へ進むにつれて現れてくる。  
最後に、データ品質にも注意を払わなければならない。  
現実世界のデータセットには、外れ値、センサーの誤測定、記録ミスなどがしばしば含まれており、どのモデルに入力する前にも対処が必要である。  
[seaborn](https://seaborn.pydata.org/)、[Bokeh](https://docs.bokeh.org/)、[matplotlib](https://matplotlib.org/) などのデータ可視化ツールは、データを手作業で点検し、どのような問題に対処すべきかについての直感を養うのに役立つ。


## 演習

1. たとえば UCI Machine Learning Repository の Abalone などのデータセットを読み込み、その性質を調べよ。欠損値を含む割合はどれくらいだろうか。変数のうち、数値型、カテゴリ型、テキスト型の割合はどれくらいだろうか。
1. 列番号ではなく列名によってデータ列をインデックス指定し、選択してみよ。pandas の [indexing](https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html) のドキュメントには、その方法の詳細が載っている。
1. この方法でどれほど大きなデータセットまで読み込めると思うか。どのような制約があるだろうか。ヒント：データの読み込み時間、表現、処理、メモリ使用量を考えよ。自分のノートパソコンで試してみよ。サーバー上で試すとどうなるか。
1. カテゴリ数が非常に多いデータをどのように扱うべきか。カテゴリラベルがすべて一意だったらどうだろうか。後者も含めるべきだろうか。
1. pandas の代替として何が考えられるか。ファイルから NumPy テンソルを読み込む方法はどうだろうか。[Pillow](https://python-pillow.org/)、すなわち Python Imaging Library についても調べよ。