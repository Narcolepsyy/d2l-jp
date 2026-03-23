# 類義語と類推の発見
:label:`sec_synonyms`

:numref:`sec_word2vec_gluon` では、小規模データセット上で word2vec の単語埋め込みモデルを学習し、単語ベクトルのコサイン類似度を用いて類義語を探索しました。実際には、大規模コーパスで事前学習された単語ベクトルは、下流の自然言語処理タスクにしばしば適用できます。この節では、これらの事前学習済み単語ベクトルを用いて、類義語と類推を見つける方法を示します。以降の節でも、事前学習済み単語ベクトルを引き続き利用します。

```{.python .input  n=1}
#@tab mxnet
from d2l import mxnet as d2l
import matplotlib.pyplot as plt
from mxnet import np, npx
import numpy
import os
from sklearn.decomposition import PCA

npx.set_np()
```

## 事前学習済み単語ベクトルの使用

以下に、50次元、100次元、300次元の事前学習済み GloVe 埋め込みを示します。これらは [GloVe website](https://nlp.stanford.edu/projects/glove/) からダウンロードできます。事前学習済み fastText 埋め込みは複数の言語で利用できます。ここでは英語版の 1 つ（300次元の "wiki.en"）を扱います。これは [fastText website](https://fasttext.cc/) からダウンロードできます。

```{.python .input  n=2}
#@tab mxnet
#@save
d2l.DATA_HUB['glove.6b.50d'] = (d2l.DATA_URL + 'glove.6B.50d.zip',
                                '0b8703943ccdb6eb788e6f091b8946e82231bc4d')

#@save
d2l.DATA_HUB['glove.6b.100d'] = (d2l.DATA_URL + 'glove.6B.100d.zip',
                                 'cd43bfb07e44e6f27cbcc7bc9ae3d80284fdaf5a')

#@save
d2l.DATA_HUB['glove.42b.300d'] = (d2l.DATA_URL + 'glove.42B.300d.zip',
                                  'b5116e234e9eb9076672cfeabf5469f3eec904fa')

#@save
d2l.DATA_HUB['wiki.en'] = (d2l.DATA_URL + 'wiki.en.zip',
                           'c1816da3821ae9f43899be655002f6c723e91b88')
```

以下の `TokenEmbedding` クラスを定義して、上記の事前学習済み GloVe および fastText 埋め込みを読み込みます。

```{.python .input  n=3}
#@tab mxnet
#@save
class TokenEmbedding:
    """Token Embedding."""
    def __init__(self, embedding_name):
        self.idx_to_token, self.idx_to_vec = self._load_embedding(
            embedding_name)
        self.unknown_idx = 0
        self.token_to_idx = {token: idx for idx, token in 
                             enumerate(self.idx_to_token)}

    def _load_embedding(self, embedding_name):
        idx_to_token, idx_to_vec = ['<unk>'], []
        data_dir = d2l.download_extract(embedding_name)
        # GloVe website: https://nlp.stanford.edu/projects/glove/
        # fastText website: https://fasttext.cc/
        with open(os.path.join(data_dir, 'vec.txt'), 'r') as f:
            for line in f:
                elems = line.rstrip().split(' ')
                token, elems = elems[0], [float(elem) for elem in elems[1:]]
                # Skip header information, such as the top row in fastText
                if len(elems) > 1:
                    idx_to_token.append(token)
                    idx_to_vec.append(elems)
        idx_to_vec = [[0] * len(idx_to_vec[0])] + idx_to_vec
        return idx_to_token, np.array(idx_to_vec)

    def __getitem__(self, tokens):
        indices = [self.token_to_idx.get(token, self.unknown_idx)
                   for token in tokens]
        vecs = self.idx_to_vec[np.array(indices)]
        return vecs

    def __len__(self):
        return len(self.idx_to_token)
```

次に、Wikipedia の一部で事前学習された 50 次元の GloVe 埋め込みを使用します。対応する単語埋め込みは、事前学習済み単語埋め込みのインスタンスを最初に作成したときに自動的にダウンロードされます。

```{.python .input  n=4}
#@tab mxnet
glove_6b50d = TokenEmbedding('glove.6b.50d')
```

辞書サイズを出力します。辞書には $400,000$ 語と特別な未知トークンが含まれています。

```{.python .input  n=5}
#@tab mxnet
len(glove_6b50d)
```

単語を使って辞書内でのインデックスを取得したり、インデックスから単語を取得したりできます。

```{.python .input  n=6}
#@tab mxnet
glove_6b50d.token_to_idx['beautiful'], glove_6b50d.idx_to_token[3367]
```

## 事前学習済み単語ベクトルの適用

以下では、GloVe を例として、事前学習済み単語ベクトルの応用を示します。

### 類義語の発見

ここでは、:numref:`sec_word2vec` で紹介した、コサイン類似度によって類義語を探索するアルゴリズムを再実装します。

類推を探索するときに $k$ 個の最近傍を求めるロジックを再利用できるように、この部分を `knn`
（$k$-最近傍）関数として別にカプセル化します。

```{.python .input  n=7}
#@tab mxnet
def knn(W, x, k):
    # The added 1e-9 is for numerical stability
    cos = np.dot(W, x.reshape(-1,)) / (
        np.sqrt(np.sum(W * W, axis=1) + 1e-9) * np.sqrt((x * x).sum()))
    topk = npx.topk(cos, k=k, ret_typ='indices')
    return topk, [cos[int(i)] for i in topk]
```

次に、事前学習済み単語ベクトルインスタンス `embed` を用いて類義語を探索します。

```{.python .input  n=8}
#@tab mxnet
def get_similar_tokens(query_token, k, embed):
    topk, cos = knn(embed.idx_to_vec, embed[[query_token]], k + 1)
    for i, c in zip(topk[1:], cos[1:]):  # Remove input words
        print(f'cosine sim={float(c):.3f}: {embed.idx_to_token[int(i)]}')
```

すでに作成した事前学習済み単語ベクトルインスタンス `glove_6b50d` の辞書には、400,000 語と特別な未知トークンが含まれています。入力語と未知語を除いて、"chip" に意味的に最も近い 3 語を探索します。

```{.python .input  n=9}
#@tab mxnet
get_similar_tokens('chip', 3, glove_6b50d)
```

次に、"baby" と "beautiful" の類義語を探索します。

```{.python .input  n=10}
#@tab mxnet
get_similar_tokens('baby', 3, glove_6b50d)
```

```{.python .input  n=11}
#@tab mxnet
get_similar_tokens('beautiful', 3, glove_6b50d)
```

### 類推の発見

類義語を探すだけでなく、事前学習済み単語ベクトルを使って単語間の類推も探索できます。たとえば、“man”:“woman”::“son”:“daughter” は類推の例であり、“man” が “woman” に対する関係は “son” が “daughter” に対する関係と同じです。類推を探索する問題は次のように定義できます。類推関係 $a : b :: c : d$ にある 4 語について、最初の 3 語 $a$, $b$, $c$ が与えられたとき、$d$ を見つけたいとします。単語 $w$ の単語ベクトルを $\text{vec}(w)$ とします。類推問題を解くには、$\text{vec}(c)+\text{vec}(b)-\text{vec}(a)$ の結果ベクトルに最も類似した単語ベクトルを見つける必要があります。

```{.python .input  n=12}
#@tab mxnet
def get_analogy(token_a, token_b, token_c, embed):
    vecs = embed[[token_a, token_b, token_c]]
    x = vecs[1] - vecs[0] + vecs[2]
    topk, cos = knn(embed.idx_to_vec, x, 1)
    return embed.idx_to_token[int(topk[0])]  # Remove unknown words
```

"male-female" の類推を確認します。

```{.python .input  n=13}
#@tab mxnet
get_analogy('man', 'woman', 'son', glove_6b50d)
```

"Capital-country" の類推: "beijing" は "china" に対して "tokyo" は何に対応するでしょうか。答えは "japan" であるはずです。

```{.python .input  n=14}
#@tab mxnet
get_analogy('beijing', 'china', 'tokyo', glove_6b50d)
```

"Adjective-superlative adjective" の類推: "bad" は "worst" に対して "big" は何に対応するでしょうか。答えは "biggest" であるはずです。

```{.python .input  n=15}
#@tab mxnet
get_analogy('bad', 'worst', 'big', glove_6b50d)
```

"Present tense verb-past tense verb" の類推: "do" は "did" に対して "go" は何に対応するでしょうか。答えは "went" であるはずです。

```{.python .input  n=16}
#@tab mxnet
get_analogy('do', 'did', 'go', glove_6b50d)
```

```{.python .input  n=51}
#@tab mxnet
def visualization(token_pairs, embed):
    plt.figure(figsize=(7, 5))
    vecs = np.concatenate([embed[pair] for pair in token_pairs])
    vecs_pca = PCA(n_components=2).fit_transform(numpy.array(vecs))
    for i, pair in enumerate(token_pairs):
        x1, y1 = vecs_pca[2 * i]
        x2, y2 = vecs_pca[2 * i + 1]
        plt.scatter(x1, y1)
        plt.scatter(x2, y2)
        plt.annotate(pair[0], xy=(x1, y1))
        plt.annotate(pair[1], xy=(x2, y2))
        plt.plot([x1, x2], [y1, y2])
    plt.show()
```

```{.python .input  n=57}
#@tab mxnet
token_pairs = [['man', 'woman'], ['son', 'daughter'], ['king', 'queen'],
              ['uncle', 'aunt'], ['sir', 'madam'], ['sister', 'brother']]
visualization(token_pairs, glove_6b50d)
```

## まとめ

* 大規模コーパスで事前学習された単語ベクトルは、下流の自然言語処理タスクにしばしば適用できます。
* 事前学習済み単語ベクトルを使って、類義語や類推を探索できます。


## 演習

1. `TokenEmbedding('wiki.en')` を使って fastText の結果を確認しなさい。
1. 辞書が非常に大きい場合、類義語や類推の探索をどのように高速化できますか？


## [Discussions](https://discuss.mxnet.io/t/2390)

![](../img/qr_similarity-analogy.svg)\n