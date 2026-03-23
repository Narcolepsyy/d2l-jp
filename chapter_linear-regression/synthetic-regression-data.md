{.python .input}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow', 'jax'])
```

# 合成回帰データ
:label:`sec_synthetic-regression-data`


機械学習とは、データから情報を抽出することにほかなりません。
では、合成データからいったい何を学べるのでしょうか。
人工的なデータ生成モデルに自分たちで組み込んだパターンそのものには本質的な関心がないとしても、
そのようなデータセットは教育目的には非常に有用であり、
学習アルゴリズムの性質を評価したり、実装が期待どおりに動作していることを確認したりするのに役立ちます。
たとえば、正しいパラメータが *a priori* に既知であるデータを作成すれば、
モデルがそれらを実際に復元できるかどうかを確認できます。

```{.python .input}
%%tab mxnet
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import np, npx, gluon
import random
npx.set_np()
```

```{.python .input}
%%tab pytorch
%matplotlib inline
from d2l import torch as d2l
import torch
import random
```

```{.python .input}
%%tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
import tensorflow as tf
import random
```

```{.python .input}
%%tab jax
%matplotlib inline
from d2l import jax as d2l
import jax
from jax import numpy as jnp
import numpy as np
import random
import tensorflow as tf
import tensorflow_datasets as tfds
```

## データセットの生成

この例では、簡潔さのために低次元の設定を扱います。
次のコード片は、標準正規分布から生成した2次元特徴をもつ1000個の例を作成します。
得られる設計行列 $\mathbf{X}$ は $\mathbb{R}^{1000 \times 2}$ に属します。
各ラベルは、*真の値* の線形関数を適用し、さらに各例ごとに独立同分布に生成した加法ノイズ $\boldsymbol{\epsilon}$ によって汚すことで生成します。

(**$$\mathbf{y}= \mathbf{X} \mathbf{w} + b + \boldsymbol{\epsilon}.$$**)

便宜上、$\boldsymbol{\epsilon}$ は平均 $\mu= 0$、標準偏差 $\sigma = 0.01$ の正規分布から生成されると仮定します。
オブジェクト指向設計のために、
コードを `d2l.DataModule` のサブクラスの `__init__` メソッドに追加します（:numref:`oo-design-data` で導入）。
追加のハイパーパラメータを設定できるようにしておくのがよい実践です。
これには `save_hyperparameters()` を用います。
`batch_size` は後で決定します。

```{.python .input}
%%tab all
class SyntheticRegressionData(d2l.DataModule):  #@save
    """線形回帰のための合成データ。"""
    def __init__(self, w, b, noise=0.01, num_train=1000, num_val=1000, 
                 batch_size=32):
        super().__init__()
        self.save_hyperparameters()
        n = num_train + num_val
        if tab.selected('pytorch') or tab.selected('mxnet'):                
            self.X = d2l.randn(n, len(w))
            noise = d2l.randn(n, 1) * noise
        if tab.selected('tensorflow'):
            self.X = tf.random.normal((n, w.shape[0]))
            noise = tf.random.normal((n, 1)) * noise
        if tab.selected('jax'):
            key = jax.random.PRNGKey(0)
            key1, key2 = jax.random.split(key)
            self.X = jax.random.normal(key1, (n, w.shape[0]))
            noise = jax.random.normal(key2, (n, 1)) * noise
        self.y = d2l.matmul(self.X, d2l.reshape(w, (-1, 1))) + b + noise
```

以下では、真のパラメータを $\mathbf{w} = [2, -3.4]^\top$、$b = 4.2$ に設定します。
後で、推定したパラメータをこれらの *真の値* と比較できます。

```{.python .input}
%%tab all
data = SyntheticRegressionData(w=d2l.tensor([2, -3.4]), b=4.2)
```

[**`features` の各行は $\mathbb{R}^2$ のベクトルからなり、`labels` の各行はスカラーです。**] 最初の要素を見てみましょう。

```{.python .input}
%%tab all
print('features:', data.X[0],'\nlabel:', data.y[0])
```

## データセットの読み出し

機械学習モデルの学習では、しばしばデータセットを何度も走査し、1回に1つのミニバッチずつ取り出します。
そのデータを使ってモデルを更新します。
これがどのように機能するかを示すために、
[**`get_dataloader` メソッドを実装し、**]
`add_to_class` を介して `SyntheticRegressionData` クラスに登録します（:numref:`oo-design-utilities` で導入）。
これは（**バッチサイズ、特徴の行列、ラベルのベクトルを受け取り、サイズが `batch_size` のミニバッチを生成します。**）
したがって、各ミニバッチは特徴とラベルのタプルからなります。
訓練モードか検証モードかに注意する必要があることに留意してください。
前者ではデータをランダム順に読み出したい一方で、
後者では、あらかじめ定めた順序でデータを読み出せることがデバッグ上重要になる場合があります。

```{.python .input}
%%tab all
@d2l.add_to_class(SyntheticRegressionData)
def get_dataloader(self, train):
    if train:
        indices = list(range(0, self.num_train))
        # The examples are read in random order
        random.shuffle(indices)
    else:
        indices = list(range(self.num_train, self.num_train+self.num_val))
    for i in range(0, len(indices), self.batch_size):
        if tab.selected('mxnet', 'pytorch', 'jax'):
            batch_indices = d2l.tensor(indices[i: i+self.batch_size])
            yield self.X[batch_indices], self.y[batch_indices]
        if tab.selected('tensorflow'):
            j = tf.constant(indices[i : i+self.batch_size])
            yield tf.gather(self.X, j), tf.gather(self.y, j)
```

直感をつかむために、最初のミニバッチのデータを見てみましょう。
特徴の各ミニバッチは、そのサイズと入力特徴の次元数の両方を私たちに与えます。
同様に、ラベルのミニバッチも `batch_size` によって決まる対応する形状を持ちます。

```{.python .input}
%%tab all
X, y = next(iter(data.train_dataloader()))
print('X shape:', X.shape, '\ny shape:', y.shape)
```

一見すると何でもないようですが、`iter(data.train_dataloader())` の呼び出しは、
Python のオブジェクト指向設計の力を示しています。
`data` オブジェクトを作成した後で、
`SyntheticRegressionData` クラスにメソッドを追加したことに注意してください。
それにもかかわらず、このオブジェクトはクラスへの機能追加を *事後的に* 受けられます。

反復の間、データセット全体を使い切るまで異なるミニバッチが得られます（試してみてください）。
上で実装した反復処理は教育目的にはよいものの、
実際の問題では困るような非効率さがあります。
たとえば、すべてのデータをメモリに読み込む必要があり、
大量のランダムメモリアクセスを行わなければなりません。
深層学習フレームワークに実装されている組み込みイテレータははるかに効率的であり、
ファイルに保存されたデータ、
ストリーム経由で受け取るデータ、
その場で生成または処理されるデータなどにも対応できます。
次に、組み込みイテレータを使って同じメソッドを実装してみましょう。

## データローダの簡潔な実装

自前でイテレータを書く代わりに、
フレームワークの既存APIを呼び出して[**データを読み込む**]ことができます。
これまでと同様に、特徴 `X` とラベル `y` をもつデータセットが必要です。
それに加えて、組み込みのデータローダで `batch_size` を設定し、
例のシャッフルは効率的に任せます。

:begin_tab:`jax`
JAX は、デバイス加速と関数型変換を備えた NumPy 風 API を中心とするため、少なくとも現行版にはデータ読み込みメソッドが含まれていません。他のライブラリにはすでに優れたデータローダがあり、JAX ではそれらの利用が推奨されています。ここでは TensorFlow のデータローダを使い、JAX で動作するように少し修正します。
:end_tab:

```{.python .input}
%%tab all
@d2l.add_to_class(d2l.DataModule)  #@save
def get_tensorloader(self, tensors, train, indices=slice(0, None)):
    tensors = tuple(a[indices] for a in tensors)
    if tab.selected('mxnet'):
        dataset = gluon.data.ArrayDataset(*tensors)
        return gluon.data.DataLoader(dataset, self.batch_size,
                                     shuffle=train)
    if tab.selected('pytorch'):
        dataset = torch.utils.data.TensorDataset(*tensors)
        return torch.utils.data.DataLoader(dataset, self.batch_size,
                                           shuffle=train)
    if tab.selected('jax'):
        # Use Tensorflow Datasets & Dataloader. JAX or Flax do not provide
        # any dataloading functionality
        shuffle_buffer = tensors[0].shape[0] if train else 1
        return tfds.as_numpy(
            tf.data.Dataset.from_tensor_slices(tensors).shuffle(
                buffer_size=shuffle_buffer).batch(self.batch_size))

    if tab.selected('tensorflow'):
        shuffle_buffer = tensors[0].shape[0] if train else 1
        return tf.data.Dataset.from_tensor_slices(tensors).shuffle(
            buffer_size=shuffle_buffer).batch(self.batch_size)
```

```{.python .input}
%%tab all
@d2l.add_to_class(SyntheticRegressionData)  #@save
def get_dataloader(self, train):
    i = slice(0, self.num_train) if train else slice(self.num_train, None)
    return self.get_tensorloader((self.X, self.y), train, i)
```

新しいデータローダは、より効率的でいくつかの追加機能を備えている点を除けば、前のものとまったく同じように動作します。

```{.python .input  n=4}
%%tab all
X, y = next(iter(data.train_dataloader()))
print('X shape:', X.shape, '\ny shape:', y.shape)
```

たとえば、フレームワークAPIが提供するデータローダは
組み込みの `__len__` メソッドをサポートしているので、
その長さ、すなわちバッチ数を問い合わせることができます。

```{.python .input}
%%tab all
len(data.train_dataloader())
```

## まとめ

データローダは、データの読み込みと操作の過程を抽象化する便利な方法です。
これにより、同じ機械学習*アルゴリズム*が、
変更を加えることなく多様な種類や供給源のデータを処理できるようになります。
データローダのよい点の1つは、組み合わせ可能であることです。
たとえば、画像を読み込んだあとに、
それらを切り抜いたり他の方法で変更したりする後処理フィルタを適用することができます。
このように、データローダは
データ処理パイプライン全体を記述するために使えます。

モデル自体について言えば、2次元線形モデルは
私たちが出会う中で最も単純なものの1つです。
これにより、データ不足や
不定な連立方程式を心配することなく、回帰モデルの精度を試すことができます。
次の節でこれを有効に活用します。  


## 演習

1. 例の数がバッチサイズで割り切れない場合はどうなりますか。フレームワークのAPIを使って別の引数を指定することで、この挙動をどのように変更しますか？
1. パラメータベクトル `w` のサイズも例の数 `num_examples` も大きい、巨大なデータセットを生成したいとします。
    1. すべてのデータをメモリに保持できない場合、何が起こりますか。
    1. データがディスク上にある場合、どのようにシャッフルしますか。あまり多くのランダム読み書きを必要としない、*効率的* なアルゴリズムを設計するのが課題です。ヒント: [擬似ランダム置換生成器](https://en.wikipedia.org/wiki/Pseudorandom_permutation) を使うと、置換表を明示的に保存せずに再シャッフルを設計できます :cite:`Naor.Reingold.1999`。 
1. イテレータが呼ばれるたびに、その場で新しいデータを生成するデータ生成器を実装してください。 
1. 呼ばれるたびに *同じ* データを生成する乱数データ生成器をどのように設計しますか？


:begin_tab:`mxnet`
[議論](https://discuss.d2l.ai/t/6662)
:end_tab:

:begin_tab:`pytorch`
[議論](https://discuss.d2l.ai/t/6663)
:end_tab:

:begin_tab:`tensorflow`
[議論](https://discuss.d2l.ai/t/6664)
:end_tab:

:begin_tab:`jax`
[議論](https://discuss.d2l.ai/t/17975)
:end_tab:\n