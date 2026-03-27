# word2vec の事前学習
:label:`sec_word2vec_pretraining`


ここでは、 :numref:`sec_word2vec` で定義した
skip-gram
モデルを実装します。
その後、
PTB データセット上で負例サンプリングを用いて
word2vec を事前学習します。
まずは、
:numref:`sec_word2vec_data` で説明した
`d2l.load_data_ptb`
関数を呼び出して、
このデータセットのデータイテレータと
語彙を取得しましょう。

```{.python .input}
#@tab mxnet
from d2l import mxnet as d2l
import math
from mxnet import autograd, gluon, np, npx
from mxnet.gluon import nn
npx.set_np()

batch_size, max_window_size, num_noise_words = 512, 5, 5
data_iter, vocab = d2l.load_data_ptb(batch_size, max_window_size,
                                     num_noise_words)
```

```{.python .input}
#@tab pytorch
from d2l import torch as d2l
import math
import torch
from torch import nn

batch_size, max_window_size, num_noise_words = 512, 5, 5
data_iter, vocab = d2l.load_data_ptb(batch_size, max_window_size,
                                     num_noise_words)
```

## Skip-Gram モデル

ここでは、埋め込み層とバッチ行列積を用いて
skip-gram モデルを実装します。
まず、
埋め込み層がどのように動作するかを
復習しましょう。


### 埋め込み層

:numref:`sec_seq2seq` で説明したように、
埋め込み層は
トークンのインデックスをその特徴ベクトルに
対応付けます。
この層の重みは、
行数が辞書サイズ（`input_dim`）に等しく、
列数が各トークンのベクトル次元（`output_dim`）に等しい
行列です。
単語埋め込みモデルを学習した後には、
この重みが必要になります。

```{.python .input}
#@tab mxnet
embed = nn.Embedding(input_dim=20, output_dim=4)
embed.initialize()
embed.weight
```

```{.python .input}
#@tab pytorch
embed = nn.Embedding(num_embeddings=20, embedding_dim=4)
print(f'Parameter embedding_weight ({embed.weight.shape}, '
      f'dtype={embed.weight.dtype})')
```

埋め込み層の入力は、
トークン（単語）のインデックスです。
任意のトークンインデックス $i$ に対して、
そのベクトル表現は
埋め込み層の重み行列の
$i^\textrm{th}$ 行から得られます。
ベクトル次元（`output_dim`）を 4 に設定したので、
形状が (2, 3) のトークンインデックスのミニバッチに対して、
埋め込み層は形状 (2, 3, 4) のベクトルを返します。

```{.python .input}
#@tab all
x = d2l.tensor([[1, 2, 3], [4, 5, 6]])
embed(x)
```

### 順伝播の定義

順伝播では、skip-gram モデルの入力は
形状 (バッチサイズ, 1) の中心語インデックス `center` と、
形状 (バッチサイズ, `max_len`) の
連結された文脈語およびノイズ語インデックス
`contexts_and_negatives` です。
ここで `max_len` は
:numref:`subsec_word2vec-minibatch-loading` で定義されています。
これら 2 つの変数はまず、
埋め込み層を通してトークンインデックスからベクトルへ変換され、
その後、バッチ行列積
(:numref:`subsec_batch_dot` で説明)
によって
形状 (バッチサイズ, 1, `max_len`) の出力が返されます。
出力の各要素は、
中心語ベクトルと文脈語またはノイズ語ベクトルの
内積です。

```{.python .input}
#@tab mxnet
def skip_gram(center, contexts_and_negatives, embed_v, embed_u):
    v = embed_v(center)
    u = embed_u(contexts_and_negatives)
    pred = npx.batch_dot(v, u.swapaxes(1, 2))
    return pred
```

```{.python .input}
#@tab pytorch
def skip_gram(center, contexts_and_negatives, embed_v, embed_u):
    v = embed_v(center)
    u = embed_u(contexts_and_negatives)
    pred = torch.bmm(v, u.permute(0, 2, 1))
    return pred
```

この `skip_gram` 関数の出力形状を、いくつかの例の入力で確認してみましょう。

```{.python .input}
#@tab mxnet
skip_gram(np.ones((2, 1)), np.ones((2, 4)), embed, embed).shape
```

```{.python .input}
#@tab pytorch
skip_gram(torch.ones((2, 1), dtype=torch.long),
          torch.ones((2, 4), dtype=torch.long), embed, embed).shape
```

## 学習

負例サンプリングを用いて skip-gram モデルを学習する前に、
まず損失関数を定義しましょう。


### バイナリ交差エントロピー損失

:numref:`subsec_negative-sampling` における
負例サンプリングの損失関数の定義に従って、
バイナリ交差エントロピー損失を用います。

```{.python .input}
#@tab mxnet
loss = gluon.loss.SigmoidBCELoss()
```

```{.python .input}
#@tab pytorch
class SigmoidBCELoss(nn.Module):
    # Binary cross-entropy loss with masking
    def __init__(self):
        super().__init__()

    def forward(self, inputs, target, mask=None):
        out = nn.functional.binary_cross_entropy_with_logits(
            inputs, target, weight=mask, reduction="none")
        return out.mean(dim=1)

loss = SigmoidBCELoss()
```

:numref:`subsec_word2vec-minibatch-loading` で説明した
mask 変数と label 変数を思い出してください。
以下では、
与えられた変数に対する
バイナリ交差エントロピー損失を計算します。

```{.python .input}
#@tab all
pred = d2l.tensor([[1.1, -2.2, 3.3, -4.4]] * 2)
label = d2l.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
mask = d2l.tensor([[1, 1, 1, 1], [1, 1, 0, 0]])
loss(pred, label, mask) * mask.shape[1] / mask.sum(axis=1)
```

以下では、
上の結果が
（より非効率な方法で）
バイナリ交差エントロピー損失における
シグモイド活性化関数を用いて
どのように計算されるかを示します。
非マスクの予測値に対して平均化された
2 つの正規化損失として
考えることができます。

```{.python .input}
#@tab all
def sigmd(x):
    return -math.log(1 / (1 + math.exp(-x)))

print(f'{(sigmd(1.1) + sigmd(2.2) + sigmd(-3.3) + sigmd(4.4)) / 4:.4f}')
print(f'{(sigmd(-1.1) + sigmd(-2.2)) / 2:.4f}')
```

### モデルパラメータの初期化

語彙中のすべての単語について、
それらが中心語として使われる場合と
文脈語として使われる場合に対応する
2 つの埋め込み層を定義します。
単語ベクトルの次元
`embed_size` は 100 に設定します。

```{.python .input}
#@tab mxnet
embed_size = 100
net = nn.Sequential()
net.add(nn.Embedding(input_dim=len(vocab), output_dim=embed_size),
        nn.Embedding(input_dim=len(vocab), output_dim=embed_size))
```

```{.python .input}
#@tab pytorch
embed_size = 100
net = nn.Sequential(nn.Embedding(num_embeddings=len(vocab),
                                 embedding_dim=embed_size),
                    nn.Embedding(num_embeddings=len(vocab),
                                 embedding_dim=embed_size))
```

### 学習ループの定義

学習ループを以下に定義します。パディングが存在するため、損失関数の計算はこれまでの学習関数と少し異なります。

```{.python .input}
#@tab mxnet
def train(net, data_iter, lr, num_epochs, device=d2l.try_gpu()):
    net.initialize(ctx=device, force_reinit=True)
    trainer = gluon.Trainer(net.collect_params(), 'adam',
                            {'learning_rate': lr})
    animator = d2l.Animator(xlabel='epoch', ylabel='loss',
                            xlim=[1, num_epochs])
    # Sum of normalized losses, no. of normalized losses
    metric = d2l.Accumulator(2)
    for epoch in range(num_epochs):
        timer, num_batches = d2l.Timer(), len(data_iter)
        for i, batch in enumerate(data_iter):
            center, context_negative, mask, label = [
                data.as_in_ctx(device) for data in batch]
            with autograd.record():
                pred = skip_gram(center, context_negative, net[0], net[1])
                l = (loss(pred.reshape(label.shape), label, mask) *
                     mask.shape[1] / mask.sum(axis=1))
            l.backward()
            trainer.step(batch_size)
            metric.add(l.sum(), l.size)
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                animator.add(epoch + (i + 1) / num_batches,
                             (metric[0] / metric[1],))
    print(f'loss {metric[0] / metric[1]:.3f}, '
          f'{metric[1] / timer.stop():.1f} tokens/sec on {str(device)}')
```

```{.python .input}
#@tab pytorch
def train(net, data_iter, lr, num_epochs, device=d2l.try_gpu()):
    def init_weights(module):
        if type(module) == nn.Embedding:
            nn.init.xavier_uniform_(module.weight)
    net.apply(init_weights)
    net = net.to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    animator = d2l.Animator(xlabel='epoch', ylabel='loss',
                            xlim=[1, num_epochs])
    # Sum of normalized losses, no. of normalized losses
    metric = d2l.Accumulator(2)
    for epoch in range(num_epochs):
        timer, num_batches = d2l.Timer(), len(data_iter)
        for i, batch in enumerate(data_iter):
            optimizer.zero_grad()
            center, context_negative, mask, label = [
                data.to(device) for data in batch]

            pred = skip_gram(center, context_negative, net[0], net[1])
            l = (loss(pred.reshape(label.shape).float(), label.float(), mask)
                     / mask.sum(axis=1) * mask.shape[1])
            l.sum().backward()
            optimizer.step()
            metric.add(l.sum(), l.numel())
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                animator.add(epoch + (i + 1) / num_batches,
                             (metric[0] / metric[1],))
    print(f'loss {metric[0] / metric[1]:.3f}, '
          f'{metric[1] / timer.stop():.1f} tokens/sec on {str(device)}')
```

これで、負例サンプリングを用いた skip-gram モデルを学習できます。

```{.python .input}
#@tab all
lr, num_epochs = 0.002, 5
train(net, data_iter, lr, num_epochs)
```

## 単語埋め込みの適用
:label:`subsec_apply-word-embed`


word2vec モデルを学習した後は、
学習済みモデルの単語ベクトルの
コサイン類似度を用いて、
入力単語と意味的に最も近い単語を
辞書から見つけることができます。

```{.python .input}
#@tab mxnet
def get_similar_tokens(query_token, k, embed):
    W = embed.weight.data()
    x = W[vocab[query_token]]
    # Compute the cosine similarity. Add 1e-9 for numerical stability
    cos = np.dot(W, x) / np.sqrt(np.sum(W * W, axis=1) * np.sum(x * x) + 1e-9)
    topk = npx.topk(cos, k=k+1, ret_typ='indices').asnumpy().astype('int32')
    for i in topk[1:]:  # Remove the input words
        print(f'cosine sim={float(cos[i]):.3f}: {vocab.to_tokens(i)}')

get_similar_tokens('chip', 3, net[0])
```

```{.python .input}
#@tab pytorch
def get_similar_tokens(query_token, k, embed):
    W = embed.weight.data
    x = W[vocab[query_token]]
    # Compute the cosine similarity. Add 1e-9 for numerical stability
    cos = torch.mv(W, x) / torch.sqrt(torch.sum(W * W, dim=1) *
                                      torch.sum(x * x) + 1e-9)
    topk = torch.topk(cos, k=k+1)[1].cpu().numpy().astype('int32')
    for i in topk[1:]:  # Remove the input words
        print(f'cosine sim={float(cos[i]):.3f}: {vocab.to_tokens(i)}')

get_similar_tokens('chip', 3, net[0])
```

## まとめ

* 埋め込み層とバイナリ交差エントロピー損失を用いて、負例サンプリング付きの skip-gram モデルを学習できます。
* 単語埋め込みの応用として、単語ベクトルのコサイン類似度に基づいて、与えられた単語に意味的に類似した単語を見つけることができます。


## 演習

1. 学習済みモデルを用いて、他の入力単語に対して意味的に類似した単語を見つけてみましょう。ハイパーパラメータを調整することで結果を改善できますか？
1. 学習コーパスが非常に大きい場合、現在のミニバッチ内の中心語に対する文脈語とノイズ語を、*モデルパラメータを更新するときに*サンプリングすることがよくあります。言い換えると、同じ中心語でも学習エポックごとに異なる文脈語やノイズ語を持ちうるということです。この方法の利点は何でしょうか？この学習方法を実装してみましょう。
