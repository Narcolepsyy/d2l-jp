# 単語の類似性とアナロジー
:label:`sec_synonyms`

:numref:`sec_word2vec_pretraining` では、  
小さなデータセットで word2vec モデルを学習し、  
入力単語に対して意味的に類似した単語を見つけるために適用しました。  
実際には、大規模コーパスで事前学習された単語ベクトルは、  
後続の自然言語処理タスクに適用できます。  
これについては、後ほど :numref:`chap_nlp_app` で扱います。  
大規模コーパスから得られた事前学習済み単語ベクトルの意味を  
わかりやすく示すために、  
単語の類似性とアナロジーのタスクに適用してみましょう。

```{.python .input}
#@tab mxnet
from d2l import mxnet as d2l
from mxnet import np, npx
import os

npx.set_np()
```

```{.python .input}
#@tab pytorch
from d2l import torch as d2l
import torch
from torch import nn
import os
```

## 事前学習済み単語ベクトルの読み込み

以下に、次元 50、100、300 の事前学習済み GloVe 埋め込みを示します。  
これらは [GloVe website](https://nlp.stanford.edu/projects/glove/) からダウンロードできます。  
事前学習済み fastText 埋め込みは複数の言語で利用できます。  
ここでは英語版の 1 つ（300 次元の "wiki.en"）を扱い、  
[fastText website](https://fasttext.cc/) からダウンロードできます。

```{.python .input}
#@tab all
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

これらの事前学習済み GloVe および fastText 埋め込みを読み込むために、  
以下の `TokenEmbedding` クラスを定義します。

```{.python .input}
#@tab all
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
        return idx_to_token, d2l.tensor(idx_to_vec)

    def __getitem__(self, tokens):
        indices = [self.token_to_idx.get(token, self.unknown_idx)
                   for token in tokens]
        vecs = self.idx_to_vec[d2l.tensor(indices)]
        return vecs

    def __len__(self):
        return len(self.idx_to_token)
```

以下では、  
50 次元の GloVe 埋め込み（Wikipedia の一部で事前学習済み）を読み込みます。  
`TokenEmbedding` インスタンスを作成するとき、  
指定した埋め込みファイルがまだ存在しなければダウンロードされます。

```{.python .input}
#@tab all
glove_6b50d = TokenEmbedding('glove.6b.50d')
```

語彙サイズを出力します。語彙には 400000 語（トークン）と特別な未知語トークンが含まれます。

```{.python .input}
#@tab all
len(glove_6b50d)
```

語彙中の単語のインデックスを取得したり、その逆を行ったりできます。

```{.python .input}
#@tab all
glove_6b50d.token_to_idx['beautiful'], glove_6b50d.idx_to_token[3367]
```

## 事前学習済み単語ベクトルの適用

読み込んだ GloVe ベクトルを用いて、  
以下の単語類似性およびアナロジーのタスクに適用することで、  
その意味を示します。


### 単語の類似性

:numref:`subsec_apply-word-embed` と同様に、  
単語ベクトル間のコサイン類似度に基づいて  
入力単語に意味的に類似した単語を見つけるために、  
以下の `knn`  
（$k$-最近傍）関数を実装します。

```{.python .input}
#@tab mxnet
def knn(W, x, k):
    # Add 1e-9 for numerical stability
    cos = np.dot(W, x.reshape(-1,)) / (
        np.sqrt(np.sum(W * W, axis=1) + 1e-9) * np.sqrt((x * x).sum()))
    topk = npx.topk(cos, k=k, ret_typ='indices')
    return topk, [cos[int(i)] for i in topk]
```

```{.python .input}
#@tab pytorch
def knn(W, x, k):
    # Add 1e-9 for numerical stability
    cos = torch.mv(W, x.reshape(-1,)) / (
        torch.sqrt(torch.sum(W * W, axis=1) + 1e-9) *
        torch.sqrt((x * x).sum()))
    _, topk = torch.topk(cos, k=k)
    return topk, [cos[int(i)] for i in topk]
```

次に、`TokenEmbedding` インスタンス `embed` に含まれる事前学習済み単語ベクトルを使って、  
類似単語を検索します。

```{.python .input}
#@tab all
def get_similar_tokens(query_token, k, embed):
    topk, cos = knn(embed.idx_to_vec, embed[[query_token]], k + 1)
    for i, c in zip(topk[1:], cos[1:]):  # Exclude the input word
        print(f'cosine sim={float(c):.3f}: {embed.idx_to_token[int(i)]}')
```

事前学習済み単語ベクトル `glove_6b50d` の語彙には 400000 語と特別な未知語トークンが含まれます。  
入力単語と未知語トークンを除いて、  
この語彙の中から  
単語 "chip" に意味的に最も類似した 3 語を見つけてみましょう。

```{.python .input}
#@tab all
get_similar_tokens('chip', 3, glove_6b50d)
```

以下は "baby" と "beautiful" に類似した単語を出力します。

```{.python .input}
#@tab all
get_similar_tokens('baby', 3, glove_6b50d)
```

```{.python .input}
#@tab all
get_similar_tokens('beautiful', 3, glove_6b50d)
```

### 単語のアナロジー

類似単語を見つけるだけでなく、  
単語ベクトルを単語アナロジーのタスクにも適用できます。  
たとえば、  
“man”:“woman”::“son”:“daughter”  
は単語アナロジーの形式です。  
つまり、  
“man” は “woman” に対して、  
“son” は “daughter” に対する、という関係です。  
具体的には、  
単語アナロジー補完タスクは次のように定義できます。  
単語アナロジー $a : b :: c : d$ に対して、最初の 3 語 $a$, $b$, $c$ が与えられたとき、$d$ を見つけます。  
単語 $w$ のベクトルを $\textrm{vec}(w)$ と表します。  
このアナロジーを完成させるために、  
$\textrm{vec}(c)+\textrm{vec}(b)-\textrm{vec}(a)$ の結果に最も類似したベクトルを持つ単語を見つけます。

```{.python .input}
#@tab all
def get_analogy(token_a, token_b, token_c, embed):
    vecs = embed[[token_a, token_b, token_c]]
    x = vecs[1] - vecs[0] + vecs[2]
    topk, cos = knn(embed.idx_to_vec, x, 1)
    return embed.idx_to_token[int(topk[0])]  # Remove unknown words
```

読み込んだ単語ベクトルを使って、"male-female" のアナロジーを確認してみましょう。

```{.python .input}
#@tab all
get_analogy('man', 'woman', 'son', glove_6b50d)
```

以下は  
“capital-country” のアナロジーを完成させます。  
“beijing”:“china”::“tokyo”:“japan”。  
これは、  
事前学習済み単語ベクトルに意味情報が含まれていることを示しています。

```{.python .input}
#@tab all
get_analogy('beijing', 'china', 'tokyo', glove_6b50d)
```

“bad”:“worst”::“big”:“biggest”  
のような  
“形容詞-最上級形容詞” のアナロジーでは、  
事前学習済み単語ベクトルが  
統語情報を捉えている可能性があることがわかります。

```{.python .input}
#@tab all
get_analogy('bad', 'worst', 'big', glove_6b50d)
```

事前学習済み単語ベクトルにおける過去形の概念が捉えられていることを示すために、  
"do":“did”::“go”:“went”  
という  
"現在形-過去形" のアナロジーで構文をテストできます。

```{.python .input}
#@tab all
get_analogy('do', 'did', 'go', glove_6b50d)
```

## まとめ

* 実際には、大規模コーパスで事前学習された単語ベクトルは、後続の自然言語処理タスクに適用できます。
* 事前学習済み単語ベクトルは、単語の類似性とアナロジーのタスクに適用できます。


## 演習

1. `TokenEmbedding('wiki.en')` を使って fastText の結果を試してください。
1. 語彙が非常に大きい場合、類似単語の検索や単語アナロジーの補完をより高速に行うにはどうすればよいでしょうか？
