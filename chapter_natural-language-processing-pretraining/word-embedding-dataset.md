# 単語埋め込みの事前学習のためのデータセット
:label:`sec_word2vec_data`

word2vec モデルと近似学習手法の技術的な詳細が分かったので、
それらの実装を見ていきよう。
具体的には、
:numref:`sec_word2vec` の skip-gram モデルと
: numref:`sec_approx_train` の負例サンプリングを例にする。
この節では、
単語埋め込みモデルの事前学習のためのデータセットから始める。
元の形式のデータは、
学習中に反復して取り出せるミニバッチへと変換される。

```{.python .input}
#@tab mxnet
import collections
from d2l import mxnet as d2l
import math
from mxnet import gluon, np
import os
import random
```

```{.python .input}
#@tab pytorch
import collections
from d2l import torch as d2l
import math
import torch
import os
import random
```

## データセットの読み込み

ここで使用するデータセットは [Penn Tree Bank (PTB)]( https://catalog.ldc.upenn.edu/LDC99T42) である。
このコーパスは Wall Street Journal の記事からサンプリングされ、
訓練、検証、テストの各セットに分割されている。
元の形式では、
テキストファイルの各行が
空白で区切られた単語列からなる1つの文を表す。
ここでは各単語をトークンとして扱いる。

```{.python .input}
#@tab all
#@save
d2l.DATA_HUB['ptb'] = (d2l.DATA_URL + 'ptb.zip',
                       '319d85e578af0cdc590547f26231e4e31cdf1e42')

#@save
def read_ptb():
    """Load the PTB dataset into a list of text lines."""
    data_dir = d2l.download_extract('ptb')
    # Read the training set
    with open(os.path.join(data_dir, 'ptb.train.txt')) as f:
        raw_text = f.read()
    return [line.split() for line in raw_text.split('\n')]

sentences = read_ptb()
f'# sentences: {len(sentences)}'
```

訓練セットを読み込んだ後、
コーパスの語彙を構築する。
その際、10回未満しか現れない単語はすべて
"&lt;unk&gt;" トークンに置き換えられる。
元のデータセットにも、
まれな（未知の）単語を表す "&lt;unk&gt;" トークンが含まれていることに注意しよ。

```{.python .input}
#@tab all
vocab = d2l.Vocab(sentences, min_freq=10)
f'vocab size: {len(vocab)}'
```

## サブサンプリング

テキストデータには通常、
"the"、"a"、"in" のような高頻度語が含まれる。
非常に大きなコーパスでは、
これらの語は数十億回も出現することがある。
しかし、
これらの語は文脈ウィンドウ内で
多くの異なる単語と共起するため、
有用な信号はあまり提供しない。
たとえば、
文脈ウィンドウ内の単語 "chip" を考えてみよう。
直感的には、
低頻度語 "intel" との共起のほうが、
高頻度語 "a" との共起よりも
学習において有用である。
さらに、
（高頻度の）単語を大量に用いて学習するのは遅くなる。
したがって、単語埋め込みモデルを学習する際には、
高頻度語を *サブサンプリング* する :cite:`Mikolov.Sutskever.Chen.ea.2013`。
具体的には、
データセット中の各インデックス付き単語 $w_i$ は、
次の確率で破棄される。


$$ P(w_i) = \max\left(1 - \sqrt{\frac{t}{f(w_i)}}, 0\right),$$

ここで $f(w_i)$ は、
単語 $w_i$ の出現数をデータセット中の総単語数で割った比率であり、
定数 $t$ はハイパーパラメータです
（実験では $10^{-4}$）。
相対頻度 $f(w_i) > t$ のときにのみ
（高頻度の）単語 $w_i$ が破棄されうることが分かりる。
また、単語の相対頻度が高いほど、
破棄される確率も高くなる。

```{.python .input}
#@tab all
#@save
def subsample(sentences, vocab):
    """Subsample high-frequency words."""
    # Exclude unknown tokens ('<unk>')
    sentences = [[token for token in line if vocab[token] != vocab.unk]
                 for line in sentences]
    counter = collections.Counter([
        token for line in sentences for token in line])
    num_tokens = sum(counter.values())

    # Return True if `token` is kept during subsampling
    def keep(token):
        return(random.uniform(0, 1) <
               math.sqrt(1e-4 / counter[token] * num_tokens))

    return ([[token for token in line if keep(token)] for line in sentences],
            counter)

subsampled, counter = subsample(sentences, vocab)
```

次のコード片は、
サブサンプリング前後の
1文あたりのトークン数のヒストグラムを描画する。
予想どおり、
サブサンプリングによって高頻度語が削除され、
文が大幅に短くなっており、
これにより学習が高速化される。

```{.python .input}
#@tab all
d2l.show_list_len_pair_hist(['origin', 'subsampled'], '# tokens per sentence',
                            'count', sentences, subsampled);
```

個々のトークンについて見ると、高頻度語 "the" のサンプリング率は 1/20 未満である。

```{.python .input}
#@tab all
def compare_counts(token):
    return (f'# of "{token}": '
            f'before={sum([l.count(token) for l in sentences])}, '
            f'after={sum([l.count(token) for l in subsampled])}')

compare_counts('the')
```

対照的に、
低頻度語 "join" は完全に保持される。

```{.python .input}
#@tab all
compare_counts('join')
```

サブサンプリング後、
コーパスのトークンをインデックスに変換する。

```{.python .input}
#@tab all
corpus = [vocab[line] for line in subsampled]
corpus[:3]
```

## 中心語と文脈語の抽出


次の `get_centers_and_contexts`
関数は、
`corpus` からすべての
中心語とその文脈語を抽出す。
文脈ウィンドウサイズとして、
1 から `max_window_size` までの整数を一様にランダムサンプリングする。
任意の中心語について、
その単語からの距離が
サンプリングされた文脈ウィンドウサイズを超えない単語が
文脈語になる。

```{.python .input}
#@tab all
#@save
def get_centers_and_contexts(corpus, max_window_size):
    """Return center words and context words in skip-gram."""
    centers, contexts = [], []
    for line in corpus:
        # To form a "center word--context word" pair, each sentence needs to
        # have at least 2 words
        if len(line) < 2:
            continue
        centers += line
        for i in range(len(line)):  # Context window centered at `i`
            window_size = random.randint(1, max_window_size)
            indices = list(range(max(0, i - window_size),
                                 min(len(line), i + 1 + window_size)))
            # Exclude the center word from the context words
            indices.remove(i)
            contexts.append([line[idx] for idx in indices])
    return centers, contexts
```

次に、7語と3語からなる2つの文を含む人工データセットを作成する。
最大文脈ウィンドウサイズを 2 とし、
すべての中心語とその文脈語を出力してみよう。

```{.python .input}
#@tab all
tiny_dataset = [list(range(7)), list(range(7, 10))]
print('dataset', tiny_dataset)
for center, context in zip(*get_centers_and_contexts(tiny_dataset, 2)):
    print('center', center, 'has contexts', context)
```

PTB データセットで学習する際には、
最大文脈ウィンドウサイズを 5 に設定する。
次のコードは、データセット中のすべての中心語とその文脈語を抽出す。

```{.python .input}
#@tab all
all_centers, all_contexts = get_centers_and_contexts(corpus, 5)
f'# center-context pairs: {sum([len(contexts) for contexts in all_contexts])}'
```

## 負例サンプリング

近似学習には負例サンプリングを用いる。
あらかじめ定義された分布に従ってノイズ語をサンプリングするために、
次の `RandomGenerator` クラスを定義する。
ここで、（正規化されていない場合もある）サンプリング分布は
引数 `sampling_weights` を通して渡される。

```{.python .input}
#@tab all
#@save
class RandomGenerator:
    """Randomly draw among {1, ..., n} according to n sampling weights."""
    def __init__(self, sampling_weights):
        # Exclude 
        self.population = list(range(1, len(sampling_weights) + 1))
        self.sampling_weights = sampling_weights
        self.candidates = []
        self.i = 0

    def draw(self):
        if self.i == len(self.candidates):
            # Cache `k` random sampling results
            self.candidates = random.choices(
                self.population, self.sampling_weights, k=10000)
            self.i = 0
        self.i += 1
        return self.candidates[self.i - 1]
```

たとえば、
サンプリング確率 $P(X=1)=2/9, P(X=2)=3/9$, $P(X=3)=4/9$ をもつ
インデックス 1, 2, 3 の中から
10 個の確率変数 $X$
を次のようにサンプリングできる。

```{.python .input}
#@tab mxnet
generator = RandomGenerator([2, 3, 4])
[generator.draw() for _ in range(10)]
```

中心語と文脈語の組に対して、
`K` 個（実験では 5 個）のノイズ語をランダムにサンプリングする。
word2vec 論文の提案に従い、
ノイズ語 $w$ のサンプリング確率 $P(w)$ は、
辞書内での相対頻度を 0.75 乗したものに設定する :cite:`Mikolov.Sutskever.Chen.ea.2013`。

```{.python .input}
#@tab all
#@save
def get_negatives(all_contexts, vocab, counter, K):
    """Return noise words in negative sampling."""
    # Sampling weights for words with indices 1, 2, ... (index 0 is the
    # excluded unknown token) in the vocabulary
    sampling_weights = [counter[vocab.to_tokens(i)]**0.75
                        for i in range(1, len(vocab))]
    all_negatives, generator = [], RandomGenerator(sampling_weights)
    for contexts in all_contexts:
        negatives = []
        while len(negatives) < len(contexts) * K:
            neg = generator.draw()
            # Noise words cannot be context words
            if neg not in contexts:
                negatives.append(neg)
        all_negatives.append(negatives)
    return all_negatives

all_negatives = get_negatives(all_contexts, vocab, counter, 5)
```

## ミニバッチで訓練例を読み込む
:label:`subsec_word2vec-minibatch-loading`

中心語、
その文脈語、
およびサンプリングされたノイズ語をすべて抽出した後、
それらは学習中に反復的に読み込める
訓練例のミニバッチへと変換される。



ミニバッチでは、
$i^\textrm{th}$ 個目の例に中心語と、
$n_i$ 個の文脈語および $m_i$ 個のノイズ語が含まれる。
文脈ウィンドウサイズが可変であるため、
$n_i+m_i$ は $i$ によって異なる。
したがって、
各例について
文脈語とノイズ語を `contexts_negatives` 変数に連結し、
連結長が $\max_i n_i+m_i$ (`max_len`) に達するまで
0 でパディングする。
損失計算からパディングを除外するために、
`masks` 変数を定義する。
`masks` の要素と `contexts_negatives` の要素の間には
1対1の対応があり、
`masks` の 0（それ以外は 1）は
`contexts_negatives` のパディングに対応する。


正例と負例を区別するために、
`labels` 変数を用いて `contexts_negatives` 内の文脈語とノイズ語を分離する。
`masks` と同様に、
`labels` の要素と `contexts_negatives` の要素の間にも
1対1の対応があり、
`labels` の 1（それ以外は 0）は
`contexts_negatives` 内の文脈語（正例）に対応する。


上の考え方は、次の `batchify` 関数で実装されている。
その入力 `data` はバッチサイズと同じ長さのリストであり、
各要素は
中心語 `center`、その文脈語 `context`、およびそのノイズ語 `negative`
からなる1つの例である。
この関数は、
マスク変数を含むなど、
学習中の計算に読み込めるミニバッチを返す。

```{.python .input}
#@tab all
#@save
def batchify(data):
    """Return a minibatch of examples for skip-gram with negative sampling."""
    max_len = max(len(c) + len(n) for _, c, n in data)
    centers, contexts_negatives, masks, labels = [], [], [], []
    for center, context, negative in data:
        cur_len = len(context) + len(negative)
        centers += [center]
        contexts_negatives += [context + negative + [0] * (max_len - cur_len)]
        masks += [[1] * cur_len + [0] * (max_len - cur_len)]
        labels += [[1] * len(context) + [0] * (max_len - len(context))]
    return (d2l.reshape(d2l.tensor(centers), (-1, 1)), d2l.tensor(
        contexts_negatives), d2l.tensor(masks), d2l.tensor(labels))
```

2つの例からなるミニバッチを使ってこの関数をテストしてみよう。

```{.python .input}
#@tab all
x_1 = (1, [2, 2], [3, 3, 3, 3])
x_2 = (1, [2, 2, 2], [3, 3])
batch = batchify((x_1, x_2))

names = ['centers', 'contexts_negatives', 'masks', 'labels']
for name, data in zip(names, batch):
    print(name, '=', data)
```

## まとめ

* 高頻度語は学習にあまり有用でない場合がある。学習を高速化するために、それらをサブサンプリングできる。
* 計算効率のため、例はミニバッチで読み込みる。パディングと非パディングを区別し、さらに正例と負例を区別するための変数を定義できる。



## 演習

1. サブサンプリングを使わない場合、この節のコードの実行時間はどのように変化するか？
1. `RandomGenerator` クラスは `k` 個の乱数サンプリング結果をキャッシュする。`k` を他の値に設定すると、データ読み込み速度にどのような影響があるか？
1. この節のコードにおける他のどのハイパーパラメータがデータ読み込み速度に影響しうるだろうか？
