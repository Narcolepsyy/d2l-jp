# GloVe の事前学習
:label:`sec_GloVe_gluon`

この節では、:numref:`sec_glove` で定義した GloVe モデルを学習します。

まず、実験に必要なパッケージとモジュールをインポートします。

```{.python .input  n=1}
#@tab mxnet
from collections import defaultdict
from d2l import mxnet as d2l
from mxnet import autograd, gluon, init, np, npx, cpu
from mxnet.gluon import nn
import random

npx.set_np()
```

## データセットの前処理
PTB データセット上で GloVe モデルを学習します。

まず、PTB データセットを読み込み、単語からなる語彙を構築し、各トークンをインデックスに対応付けてコーパスを作成します。

```{.python .input  n=2}
#@tab mxnet
sentences = d2l.read_ptb()
vocab = d2l.Vocab(sentences, min_freq=10)
corpus = [vocab[line] for line in sentences]
```

### 共起回数の構築
単語間の共起回数を $X$ と表し、その要素 $x_{ij}$ は、単語 $i$ の文脈に単語 $j$ が現れる回数を表します。

次に、中心となるターゲット単語とその文脈単語をすべて抽出する以下の関数を定義します。距離に応じて減衰する重み付け関数を用いるため、$d$ 語離れた単語の組は総カウントに $1/d$ を寄与します。これは、非常に離れた単語の組は、互いの関係についての関連情報が少ないと考えられることを考慮する一つの方法です。

```{.python .input  n=3}
#@tab mxnet
def get_coocurrence_counts(corpus, window_size):
    centers, contexts = [], []
    cooccurence_counts = defaultdict(float)
    for line in corpus:
        # 各文は「中心となるターゲット単語 - 文脈単語」の組を作るために
        # 少なくとも 2 語必要
        if len(line) < 2:
            continue
        centers += line
        for i in range(len(line)):  # i を中心とする文脈ウィンドウ
            left_indices = list(range(max(0, i - window_size), i))
            right_indices = list(range(i + 1,
                                       min(len(line), i + 1 + window_size)))
            left_context = [line[idx] for idx in left_indices]
            right_context = [line[idx] for idx in right_indices]
            for distance, word in enumerate(left_context[::-1]):
                cooccurence_counts[line[i], word] += 1 / (distance + 1)
            for distance, word in enumerate(right_context):
                cooccurence_counts[line[i], word] += 1 / (distance + 1)
    cooccurence_counts = [(word[0], word[1], count)
                          for word, count in cooccurence_counts.items()]
    return cooccurence_counts
```

5 語と 2 語からなる 2 つの文を含む人工データセットを作成します。最大文脈ウィンドウを 4 と仮定します。すると、すべての中心となるターゲット単語と文脈単語の共起回数を出力できます。

```{.python .input  n=4}
#@tab mxnet
tiny_dataset = [list(range(5)), list(range(5, 7))]
print('dataset', tiny_dataset)
for center, context, coocurrence in get_coocurrence_counts(tiny_dataset, 4):
        print('center: %s, context: %s, coocurrence: %.2f' %
          (center, context, coocurrence))
```

最大文脈ウィンドウサイズを 5 に設定します。以下では、データセット中の中心となるターゲット単語とその文脈単語をすべて抽出し、それらの共起回数を計算します。

```{.python .input  n=5}
#@tab mxnet
coocurrence_matrix = get_coocurrence_counts(corpus, 5)
'# center-context pairs: %d' % len(coocurrence_matrix)
```

### すべてをまとめる

最後に、PTB データセットを読み込み、データローダーを返す load_data_ptb_glove 関数を定義します。

```{.python .input  n=16}
#@tab mxnet
def load_data_ptb_glove(batch_size, window_size):
    num_workers = d2l.get_dataloader_workers()
    sentences = d2l.read_ptb()
    vocab = d2l.Vocab(sentences, min_freq=5)
    corpus = [vocab[line] for line in sentences]
    coocurrence_matrix = get_coocurrence_counts(corpus, window_size)
    dataset = gluon.data.ArrayDataset(coocurrence_matrix)
    data_iter = gluon.data.DataLoader(dataset, batch_size, shuffle=True,
                                      num_workers=num_workers)
    return data_iter, vocab

batch_size, window_size = 1024, 10
data_iter, vocab = load_data_ptb_glove(batch_size, window_size)
```

データイテレータの最初のミニバッチを出力してみましょう。

```{.python .input  n=17}
#@tab mxnet
names = ['center', 'context', 'Cooccurence']
for batch in data_iter:
    for name, data in zip(names, batch):
        print(name, 'shape:', data.shape)
    break
```

## GloVe モデル

15.1 節では、GloVe の目的は損失関数を最小化することだと紹介しました。

$$\sum_{i\in\mathcal{V}} \sum_{j\in\mathcal{V}}
h(x_{ij}) \left(\mathbf{u}_j^\top \mathbf{v}_i + b_i + c_j -
\log\,x_{ij}\right)^2.$$

ここでは、損失関数の各部分を実装することで GloVe モデルを実装します。

### 重み関数

GloVe では、損失関数に重み関数 $h(x_{ij})$ を導入します。

$$h(x_{ij})=\begin{cases}
(\frac{x}{x_{max}})^\alpha & x_{ij}<x_{max}\\
1 & otherwise
\end{cases}$$


重み関数 $h(x_{ij})$ を実装します。$x_{ij}<x_{max}$ は $(\frac{x}{x_{max}})^\alpha < 1$ と等価なので、以下のように実装できます。

```{.python .input  n=18}
#@tab mxnet
def compute_weight(x, x_max = 30, alpha = 0.75):
    w = (x / x_max) ** alpha
    return np.minimum(w, 1)
```

以下では、$x_{max}$ を 2、$\alpha$ を 0.75 に設定したときの、すべての中心となるターゲット単語と文脈単語の共起回数に対する重みを出力します。

```{.python .input  n=19}
#@tab mxnet
for center, context, coocurrence in get_coocurrence_counts(tiny_dataset, 4)[:5]:
    print('center: %s, context: %s, coocurrence: %.2f, weight: %.2f' %
          (center, context, coocurrence, compute_weight(coocurrence, x_max = 2, alpha = 0.75)))
```

### バイアス項

GloVe には、各単語 $w_i$ に対して 2 つのスカラーのモデルパラメータがあります。すなわち、バイアス項 $b_i$（中心となるターゲット単語用）と $c_i$（文脈単語用）です。バイアス項は埋め込み層で実現できます。埋め込み層の重みは、行数が辞書サイズ（input_dim）、列数が 1 の行列です。

辞書サイズを 20 に設定します。

```{.python .input}
#@tab mxnet
embed_bias = nn.Embedding(input_dim=20, output_dim=1)
embed_bias.initialize()
embed_bias.weight
```

埋め込み層の入力は単語のインデックスです。単語のインデックス $i$ を入力すると、埋め込み層は重みの $i$ 行目をバイアス項として返します。

```{.python .input}
#@tab mxnet
x = np.array([1, 2, 3])
embed_bias(x)
```

### GloVe モデルの順伝播計算

順伝播計算では、GloVe モデルの入力は中心となるターゲット単語のインデックス `center` と文脈単語のインデックス `context` です。ここで、`center` 変数の形状は (batch size, 1)、`context` 変数の形状は (batch size, 1) です。これら 2 つの変数は、まず単語埋め込み層によって単語インデックスから単語ベクトルに変換されます。

```{.python .input  n=20}
#@tab mxnet
def GloVe(center, context, coocurrence, embed_v, embed_u,
          bias_v, bias_u, x_max, alpha):
    # v の形状: (batch_size, embed_size)
    v = embed_v(center)
    # u の形状: (batch_size, embed_size)
    u = embed_u(context)
    # b の形状: (batch_size, )
    b = bias_v(center).squeeze()
    # c の形状: (batch_size, )
    c = bias_u(context).squeeze()
    # embed_products の形状: (batch_size,)
    embed_products = npx.batch_dot(np.expand_dims(v, 1),
                                   np.expand_dims(u, 2)).squeeze()
    # distance_expr の形状: (batch_size,)
    distance_expr = np.power(embed_products + b +
                     c - np.log(coocurrence), 2)
    # weight の形状: (batch_size,)
    weight = compute_weight(coocurrence)
    return weight * distance_expr
```

出力の形状が (batch size, ) になることを確認します。

```{.python .input  n=21}
#@tab mxnet
embed_word = nn.Embedding(input_dim=20, output_dim=4)
embed_word.initialize()
GloVe(np.ones((2)), np.ones((2)), np.ones((2)), embed_word, embed_word,
      embed_bias, embed_bias, x_max = 2, alpha = 0.75).shape
```

## 学習

単語埋め込みモデルを学習する前に、モデルの損失関数を定義する必要があります。

### モデルパラメータの初期化
単語の埋め込み層と追加のバイアスを構築し、ハイパーパラメータである単語ベクトルの次元 `embed_size` を 100 に設定します。

```{.python .input  n=22}
#@tab mxnet
embed_size = 100
net = nn.Sequential()
net.add(nn.Embedding(input_dim=len(vocab), output_dim=embed_size),
        nn.Embedding(input_dim=len(vocab), output_dim=embed_size),
        nn.Embedding(input_dim=len(vocab), output_dim=1),
        nn.Embedding(input_dim=len(vocab), output_dim=1))
```

### 学習

学習関数を以下に定義します。

```{.python .input  n=23}
#@tab mxnet
def train(net, data_iter, lr, num_epochs, x_max, alpha, ctx=d2l.try_gpu()):
    net.initialize(ctx=ctx, force_reinit=True)
    trainer = gluon.Trainer(net.collect_params(), 'AdaGrad',
                            {'learning_rate': lr})
    animator = d2l.Animator(xlabel='epoch', ylabel='loss',
                            xlim=[0, num_epochs])
    for epoch in range(num_epochs):
        timer = d2l.Timer()
        metric = d2l.Accumulator(2)  # loss_sum, num_tokens
        for i, batch in enumerate(data_iter):
            center, context, coocurrence = [
                data.as_in_context(ctx) for data in batch]
            with autograd.record():
                l = GloVe(center, context, coocurrence.astype('float32'),
                          net[0], net[1], net[2], net[3], x_max, alpha)
            l.backward()
            trainer.step(batch_size)
            metric.add(l.sum(), l.size)
            if (i+1) % 50 == 0:
                animator.add(epoch+(i+1)/len(data_iter),
                             (metric[0]/metric[1],))
    print('loss %.3f, %d tokens/sec on %s ' % (
        metric[0]/metric[1], metric[1]/timer.stop(), ctx))
```

それでは、GloVe モデルを学習できます。

```{.python .input  n=12}
#@tab mxnet
lr, num_epochs = 0.1, 5
x_max, alpha = 100, 0.75
train(net, data_iter, lr, num_epochs, x_max, alpha)
```

## GloVe モデルの適用

GloVe モデルは、`embed_v` と `embed_u` の 2 組の単語ベクトルを生成します。`embed_v` と `embed_u` は等価であり、ランダム初期化の違いによってのみ異なります。2 組のベクトルは同等の性能を示すはずです。一般には、`embed_v`+`embed_u` の和を単語ベクトルとして用います。


GloVe モデルを学習した後でも、2 つの単語ベクトルのコサイン類似度に基づいて、単語間の意味的な類似性を表現できます。

```{.python .input  n=13}
#@tab mxnet
def get_similar_tokens(query_token, k, embed_v, embed_u):
    W = embed_v.weight.data() + embed_u.weight.data()
    x = W[vocab[query_token]]
    # コサイン類似度を計算する。数値安定性のために 1e-9 を加える
    cos = np.dot(W, x) / np.sqrt(np.sum(W * W, axis=1) * np.sum(x * x) + 1e-9)
    topk = npx.topk(cos, k=k+1, ret_typ='indices').asnumpy().astype('int32')
    for i in topk[1:]:  # 入力単語を除く
        print('cosine sim=%.3f: %s' % (cos[i], (vocab.idx_to_token[i])))

get_similar_tokens('chip', 3, net[0], net[1])
```

## まとめ

* GloVe モデルを事前学習できる。


## 演習



## ディスカッション\n