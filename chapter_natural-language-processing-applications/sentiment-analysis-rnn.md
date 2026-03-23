# 感情分析: 再帰型ニューラルネットワークの使用
:label:`sec_sentiment_rnn` 


単語類似度や類推のタスクと同様に、
感情分析にも事前学習済みの単語ベクトルを適用できます。
:numref:`sec_sentiment` の IMDb レビューデータセットは
それほど大きくないため、
大規模コーパスで事前学習された
テキスト表現を用いることで、
モデルの過学習を抑えられる可能性があります。
:numref:`fig_nlp-map-sa-rnn` に示す具体例では、
各トークンを
事前学習済みの GloVe モデルで表現し、
それらのトークン表現を
多層双方向 RNN に入力して
テキスト系列表現を得ます。
その表現は
感情分析の出力へと変換されます :cite:`Maas.Daly.Pham.ea.2011`。
同じ下流アプリケーションに対して、
後ほど別のアーキテクチャ上の
選択肢も検討します。

![この節では、事前学習済み GloVe を RNN ベースの感情分析アーキテクチャに入力します。](../img/nlp-map-sa-rnn.svg)
:label:`fig_nlp-map-sa-rnn`

```{.python .input}
#@tab mxnet
from d2l import mxnet as d2l
from mxnet import gluon, init, np, npx
from mxnet.gluon import nn, rnn
npx.set_np()

batch_size = 64
train_iter, test_iter, vocab = d2l.load_data_imdb(batch_size)
```

```{.python .input}
#@tab pytorch
from d2l import torch as d2l
import torch
from torch import nn

batch_size = 64
train_iter, test_iter, vocab = d2l.load_data_imdb(batch_size)
```

## RNN による単一テキストの表現

感情分析のようなテキスト分類タスクでは、
可変長のテキスト系列を
固定長のカテゴリへ変換します。
以下の `BiRNN` クラスでは、
テキスト系列の各トークンが
埋め込み層 (`self.embedding`) を通じて
個別の事前学習済み GloVe 表現を得ますが、
系列全体は双方向 RNN (`self.encoder`) によって符号化されます。
より具体的には、
双方向 LSTM の
（最終層における）
最初と最後の時間ステップでの隠れ状態を連結し、
テキスト系列の表現とします。
この単一のテキスト表現は、
全結合層 (`self.decoder`) によって
2 つの出力（"positive" と "negative"）を持つ
出力カテゴリへ変換されます。

```{.python .input}
#@tab mxnet
class BiRNN(nn.Block):
    def __init__(self, vocab_size, embed_size, num_hiddens,
                 num_layers, **kwargs):
        super(BiRNN, self).__init__(**kwargs)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        # Set `bidirectional` to True to get a bidirectional RNN
        self.encoder = rnn.LSTM(num_hiddens, num_layers=num_layers,
                                bidirectional=True, input_size=embed_size)
        self.decoder = nn.Dense(2)

    def forward(self, inputs):
        # The shape of `inputs` is (batch size, no. of time steps). Because
        # LSTM requires its input's first dimension to be the temporal
        # dimension, the input is transposed before obtaining token
        # representations. The output shape is (no. of time steps, batch size,
        # word vector dimension)
        embeddings = self.embedding(inputs.T)
        # Returns hidden states of the last hidden layer at different time
        # steps. The shape of `outputs` is (no. of time steps, batch size,
        # 2 * no. of hidden units)
        outputs = self.encoder(embeddings)
        # Concatenate the hidden states at the initial and final time steps as
        # the input of the fully connected layer. Its shape is (batch size,
        # 4 * no. of hidden units)
        encoding = np.concatenate((outputs[0], outputs[-1]), axis=1)
        outs = self.decoder(encoding)
        return outs
```

```{.python .input}
#@tab pytorch
class BiRNN(nn.Module):
    def __init__(self, vocab_size, embed_size, num_hiddens,
                 num_layers, **kwargs):
        super(BiRNN, self).__init__(**kwargs)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        # Set `bidirectional` to True to get a bidirectional RNN
        self.encoder = nn.LSTM(embed_size, num_hiddens, num_layers=num_layers,
                                bidirectional=True)
        self.decoder = nn.Linear(4 * num_hiddens, 2)

    def forward(self, inputs):
        # The shape of `inputs` is (batch size, no. of time steps). Because
        # LSTM requires its input's first dimension to be the temporal
        # dimension, the input is transposed before obtaining token
        # representations. The output shape is (no. of time steps, batch size,
        # word vector dimension)
        embeddings = self.embedding(inputs.T)
        self.encoder.flatten_parameters()
        # Returns hidden states of the last hidden layer at different time
        # steps. The shape of `outputs` is (no. of time steps, batch size,
        # 2 * no. of hidden units)
        outputs, _ = self.encoder(embeddings)
        # Concatenate the hidden states at the initial and final time steps as
        # the input of the fully connected layer. Its shape is (batch size,
        # 4 * no. of hidden units)
        encoding = torch.cat((outputs[0], outputs[-1]), dim=1) 
        outs = self.decoder(encoding)
        return outs
```

感情分析のために単一テキストを表現する双方向 RNN を、2 層の隠れ層で構成してみましょう。

```{.python .input}
#@tab all
embed_size, num_hiddens, num_layers, devices = 100, 100, 2, d2l.try_all_gpus()
net = BiRNN(len(vocab), embed_size, num_hiddens, num_layers)
```

```{.python .input}
#@tab mxnet
net.initialize(init.Xavier(), ctx=devices)
```

```{.python .input}
#@tab pytorch
def init_weights(module):
    if type(module) == nn.Linear:
        nn.init.xavier_uniform_(module.weight)
    if type(module) == nn.LSTM:
        for param in module._flat_weights_names:
            if "weight" in param:
                nn.init.xavier_uniform_(module._parameters[param])
net.apply(init_weights);
```

## 事前学習済み単語ベクトルの読み込み

以下では、語彙内のトークンに対応する
事前学習済みの 100 次元（`embed_size` と一致している必要があります）の
GloVe 埋め込みを読み込みます。

```{.python .input}
#@tab all
glove_embedding = d2l.TokenEmbedding('glove.6b.100d')
```

語彙内のすべてのトークンに対する
ベクトルの形状を表示します。

```{.python .input}
#@tab all
embeds = glove_embedding[vocab.idx_to_token]
embeds.shape
```

これらの事前学習済み
単語ベクトルを用いて
レビュー中のトークンを表現し、
学習中にこれらのベクトルは更新しません。

```{.python .input}
#@tab mxnet
net.embedding.weight.set_data(embeds)
net.embedding.collect_params().setattr('grad_req', 'null')
```

```{.python .input}
#@tab pytorch
net.embedding.weight.data.copy_(embeds)
net.embedding.weight.requires_grad = False
```

## モデルの学習と評価

これで、感情分析のために双方向 RNN を学習できます。

```{.python .input}
#@tab mxnet
lr, num_epochs = 0.01, 5
trainer = gluon.Trainer(net.collect_params(), 'adam', {'learning_rate': lr})
loss = gluon.loss.SoftmaxCrossEntropyLoss()
d2l.train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs, devices)
```

```{.python .input}
#@tab pytorch
lr, num_epochs = 0.01, 5
trainer = torch.optim.Adam(net.parameters(), lr=lr)
loss = nn.CrossEntropyLoss(reduction="none")
d2l.train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs, devices)
```

学習済みモデル `net` を用いて
テキスト系列の感情を予測するために、
以下の関数を定義します。

```{.python .input}
#@tab mxnet
#@save
def predict_sentiment(net, vocab, sequence):
    """Predict the sentiment of a text sequence."""
    sequence = np.array(vocab[sequence.split()], ctx=d2l.try_gpu())
    label = np.argmax(net(sequence.reshape(1, -1)), axis=1)
    return 'positive' if label == 1 else 'negative'
```

```{.python .input}
#@tab pytorch
#@save
def predict_sentiment(net, vocab, sequence):
    """Predict the sentiment of a text sequence."""
    sequence = torch.tensor(vocab[sequence.split()], device=d2l.try_gpu())
    label = torch.argmax(net(sequence.reshape(1, -1)), dim=1)
    return 'positive' if label == 1 else 'negative'
```

最後に、学習済みモデルを使って
2 つの簡単な文の感情を予測してみましょう。

```{.python .input}
#@tab all
predict_sentiment(net, vocab, 'this movie is so great')
```

```{.python .input}
#@tab all
predict_sentiment(net, vocab, 'this movie is so bad')
```

## まとめ

* 事前学習済み単語ベクトルは、テキスト系列内の個々のトークンを表現できます。
* 双方向 RNN は、最初と最後の時間ステップでの隠れ状態の連結などによって、テキスト系列を表現できます。この単一のテキスト表現は、全結合層を用いてカテゴリへ変換できます。



## 演習

1. エポック数を増やしてみましょう。学習精度とテスト精度を改善できますか？ 他のハイパーパラメータを調整した場合はどうでしょうか？
1. 300 次元 GloVe 埋め込みのような、より大きな事前学習済み単語ベクトルを使ってみましょう。分類精度は向上しますか？
1. spaCy のトークン化を使うことで分類精度を改善できますか？ spaCy をインストールし（`pip install spacy`）、英語パッケージをインストールする必要があります（`python -m spacy download en`）。コードでは、まず spaCy をインポートし（`import spacy`）、次に spaCy の英語パッケージを読み込みます（`spacy_en = spacy.load('en')`）。最後に、`def tokenizer(text): return [tok.text for tok in spacy_en.tokenizer(text)]` を定義して、元の `tokenizer` 関数を置き換えてください。GloVe と spaCy ではフレーズトークンの形式が異なることに注意してください。たとえば、フレーズトークン "new york" は、GloVe では "new-york" の形式であり、spaCy のトークン化後は "new york" の形式になります。

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/392)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1424)
:end_tab:\n