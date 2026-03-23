#  Sequence to Sequence
:label:`sec_seq2seq`

シーケンス・ツー・シーケンス（seq2seq）モデルは、エンコーダ・デコーダアーキテクチャに基づいており、:numref:`fig_seq2seq` に示すように、系列入力に対して系列出力を生成します。エンコーダとデコーダの両方で、可変長の系列入力を扱うために再帰型ニューラルネットワーク（RNN）を用います。エンコーダの隠れ状態は、エンコーダからデコーダへ情報を受け渡すために、デコーダの隠れ状態の初期化に直接使われます。

![The sequence to sequence model architecture.](../img/seq2seq.svg)
:label:`fig_seq2seq`

エンコーダとデコーダの層は :numref:`fig_seq2seq_details` に示されています。

![Layers in the encoder and the decoder.](../img/seq2seq-details.svg)
:label:`fig_seq2seq_details`

この節では、機械翻訳データセットで学習するための seq2seq モデルを説明し、実装します。

```{.python .input  n=1}
#@tab mxnet
from d2l import mxnet as d2l
from mxnet import np, npx, init, gluon, autograd
from mxnet.gluon import nn, rnn
from queue import PriorityQueue

npx.set_np()
```

## エンコーダ

seq2seq のエンコーダは、系列情報を $\mathbf{c}$ に符号化することで、可変長の入力を固定長のコンテキストベクトル $\mathbf{c}$ に変換できることを思い出してください。通常、エンコーダ内では RNN 層を用います。
入力系列 $x_1, \ldots, x_T$ があるとします。ここで $x_t$ は $t^\mathrm{th}$ 単語です。時刻 $t$ において、RNN への入力は 2 つのベクトル、すなわち $x_t$ の特徴ベクトル $\mathbf{x}_t$ と、1 つ前の時刻の隠れ状態 $\mathbf{h}_{t-1}$ です。RNN の隠れ状態の変換を関数 $f$ で表すと、

$$\mathbf{h}_t = f (\mathbf{x}_t, \mathbf{h}_{t-1}).$$

次に、エンコーダはすべての隠れ状態の情報を取り出し、関数 $q$ を用いてそれをコンテキストベクトル $\mathbf{c}$ に符号化します。

$$\mathbf{c} = q (\mathbf{h}_1, \ldots, \mathbf{h}_T).$$

たとえば、$q$ を $q (\mathbf{h}_1, \ldots, \mathbf{h}_T) = \mathbf{h}_T$ と選べば、コンテキストベクトルは最終隠れ状態 $\mathbf{h}_T$ になります。

ここまでで説明したのは単方向 RNN であり、各時刻の隠れ状態はそれ以前の時刻のみに依存します。系列入力を符号化するために、GRU、LSTM、双方向 RNN など、他の形式の RNN も使えます。

では、seq2seq のエンコーダを実装しましょう。
ここでは単語埋め込み層を使って、入力言語の単語インデックスに応じた特徴ベクトルを得ます。
これらの特徴ベクトルを多層 LSTM に入力します。
エンコーダへの入力は系列のバッチであり、形状が (バッチサイズ, 系列長) の 2 次元テンソルです。エンコーダは、LSTM の出力、すなわちすべての時刻の隠れ状態と、最終時刻の隠れ状態およびメモリセルの両方を返します。

```{.python .input  n=2}
#@tab mxnet
#@save
class Seq2SeqEncoder(d2l.Encoder):
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers,
                 dropout=0, **kwargs):
        super(Seq2SeqEncoder, self).__init__(**kwargs)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = rnn.LSTM(num_hiddens, num_layers, dropout=dropout)

    def forward(self, X, *args):
        X = self.embedding(X)  # X shape: (batch_size, seq_len, embed_size)
        # RNN needs first axes to be timestep, i.e., seq_len
        X = X.swapaxes(0, 1)
        state = self.rnn.begin_state(batch_size=X.shape[1], ctx=X.ctx)
        out, state = self.rnn(X, state)
        # out shape: (seq_len, batch_size, num_hiddens)
        # state shape: (num_layers, batch_size, num_hiddens),
        # where "state" contains the hidden state and the memory cell
        return out, state
```

次に、バッチサイズ 4、時刻数 7 のミニバッチ系列入力を作成します。LSTM ユニットの隠れ層数は 2、隠れユニット数は 16 と仮定します。入力に対して順伝播計算を行った後にエンコーダが返す出力の形状は (時刻数, バッチサイズ, 隠れユニット数) です。最終時刻におけるゲート付き再帰ユニットの多層隠れ状態の形状は (隠れ層数, バッチサイズ, 隠れユニット数) です。ゲート付き再帰ユニットでは、`state` リストには隠れ状態のみが 1 つ含まれます。長短期記憶（LSTM）を使う場合、`state` リストにはもう 1 つの要素、すなわちメモリセルも含まれます。

```{.python .input  n=3}
#@tab mxnet
encoder = Seq2SeqEncoder(vocab_size=10, embed_size=8, num_hiddens=16,
                         num_layers=2)
encoder.initialize()
X = np.zeros((4, 7))
output, state = encoder(X)
output.shape
```

LSTM を使っているので、`state` リストには、同じ形状（隠れ層数, バッチサイズ, 隠れユニット数）を持つ隠れ状態とメモリセルの両方が含まれます。ただし、GRU を使う場合、`state` リストには 1 つの要素、すなわち最終時刻の隠れ状態のみが含まれ、その形状は (隠れ層数, バッチサイズ, 隠れユニット数) です。

```{.python .input  n=4}
#@tab mxnet
len(state), state[0].shape, state[1].shape
```

## デコーダ
:label:`sec_seq2seq_decoder`

先ほど述べたように、コンテキストベクトル $\mathbf{c}$ は入力系列 $x_1, \ldots, x_T$ 全体の情報を符号化します。学習データセットにおける出力が $y_1, \ldots, y_{T'}$ であるとします。各時刻 $t'$ において、出力 $y_{t'}$ の条件付き確率は、直前までの出力系列 $y_1, \ldots, y_{t'-1}$ とコンテキストベクトル $\mathbf{c}$ に依存します。すなわち、

$$P(y_{t'} \mid y_1, \ldots, y_{t'-1}, \mathbf{c}).$$

したがって、別の RNN をデコーダとして使えます。時刻 $t'$ において、デコーダは 3 つの入力、すなわち $y_{t'-1}$ の特徴ベクトル $\mathbf{y}_{t'-1}$、コンテキストベクトル $\mathbf{c}$、および 1 つ前の時刻の隠れ状態 $\mathbf{s}_{t'-1}$ を用いて、隠れ状態 $\mathbf{s}_{t'}$ を更新します。デコーダ内の RNN の隠れ状態の変換を関数 $g$ で表すと、

$$\mathbf{s}_{t'} = g(\mathbf{y}_{t'-1}, \mathbf{c}, \mathbf{s}_{t'-1}).$$


デコーダを実装する際には、エンコーダの最終時刻の隠れ状態をデコーダの初期隠れ状態として直接使います。そのためには、エンコーダとデコーダの RNN が同じ層数と隠れユニット数を持っている必要があります。
デコーダの LSTM の順伝播計算はエンコーダのそれと似ています。唯一の違いは、LSTM 層の後に全結合層を追加し、その隠れサイズを語彙サイズにすることです。全結合層は各単語の信頼度スコアを予測します。

```{.python .input  n=5}
#@tab mxnet
#@save
class Seq2SeqDecoder(d2l.Decoder):
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers,
                 dropout=0, **kwargs):
        super(Seq2SeqDecoder, self).__init__(**kwargs)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = rnn.LSTM(num_hiddens, num_layers, dropout=dropout)
        self.dense = nn.Dense(vocab_size, flatten=False)

    def init_state(self, enc_outputs, *args):
        return enc_outputs[1]

    def forward(self, X, state):
        X = self.embedding(X).swapaxes(0, 1)
        out, state = self.rnn(X, state)
        # Make the batch to be the first dimension to simplify loss
        # computation
        out = self.dense(out).swapaxes(0, 1)
        return out, state
```

エンコーダと同じハイパーパラメータを持つデコーダを作成します。見てわかるように、出力の形状は (バッチサイズ, 系列長, 語彙サイズ) に変わります。

```{.python .input  n=6}
#@tab mxnet
decoder = Seq2SeqDecoder(vocab_size=10, embed_size=8,
                         num_hiddens=16, num_layers=2)
decoder.initialize()
state = decoder.init_state(encoder(X))
out, state = decoder(X, state)
out.shape, len(state), state[0].shape, state[1].shape
```

## 損失関数

各時刻において、デコーダは単語を予測するために語彙サイズの信頼度スコアベクトルを出力します。言語モデルと同様に、softmax を適用して確率を得てから、交差エントロピー損失を用いて損失を計算できます。なお、ターゲット文は同じ長さになるようにパディングされていますが、パディング記号については損失を計算する必要はありません。

一部の要素を除外する損失関数を実装するために、`SequenceMask` と呼ばれる演算子を使います。これは、最初の次元（`axis=0`）または 2 番目の次元（`axis=1`）をマスクとして指定できます。2 番目の次元を選ぶと、有効長ベクトル `len` と 2 次元入力 `X` が与えられたとき、この演算子はすべての $i$ について `X[i, len[i]:] = 0` とします。

```{.python .input  n=7}
#@tab mxnet
X = np.array([[1, 2, 3], [4, 5, 6]])
npx.sequence_mask(X, np.array([1, 2]), True, axis=1)
```

$n$ 次元テンソル $X$ に適用すると、`X[i, len[i]:, :, ..., :] = 0` とします。さらに、以下に示すように $-1$ などの埋め込み値を指定することもできます。

```{.python .input  n=8}
#@tab mxnet
X = np.ones((2, 3, 4))
npx.sequence_mask(X, np.array([1, 2]), True, value=-1, axis=1)
```

これで、マスク付きの softmax 交差エントロピー損失を実装できます。なお、各 Gluon の損失関数では、サンプルごとの重みを指定できます。デフォルトではそれらは 1 です。したがって、除外したい各サンプルに対して重みを 0 にすればよいだけです。そこで、カスタマイズした損失関数は、各系列内のいくつかの失敗要素を無視するために、追加の `valid_len` 引数を受け取ります。

```{.python .input  n=9}
#@tab mxnet
#@save
class MaskedSoftmaxCELoss(gluon.loss.SoftmaxCELoss):
    # pred shape: (batch_size, seq_len, vocab_size)
    # label shape: (batch_size, seq_len)
    # valid_len shape: (batch_size, )
    def forward(self, pred, label, valid_len):
        # weights shape: (batch_size, seq_len, 1)
        weights = np.expand_dims(np.ones_like(label), axis=-1)
        weights = npx.sequence_mask(weights, valid_len, True, axis=1)
        return super(MaskedSoftmaxCELoss, self).forward(pred, label, weights)
```

動作確認として、同一の 3 つの系列を作り、1 つ目の系列では 4 要素、2 つ目の系列では 2 要素、最後の系列では 0 要素を保持します。すると、1 つ目の例の損失は 2 つ目の 2 倍になり、最後の損失は 0 になるはずです。

```{.python .input  n=10}
#@tab mxnet
loss = MaskedSoftmaxCELoss()
loss(np.ones((3, 4, 10)), np.ones((3, 4)), np.array([4, 2, 0]))
```

## 学習
:label:`sec_seq2seq_training`

学習時には、ターゲット系列の長さが $n$ であれば、最初の $n-1$ 個のトークンをデコーダへの入力として与え、最後の $n-1$ 個のトークンを正解ラベルとして用います。

```{.python .input  n=11}
#@tab mxnet
#@save
def train_s2s_ch9(model, data_iter, lr, num_epochs, ctx):
    model.initialize(init.Xavier(), force_reinit=True, ctx=ctx)
    trainer = gluon.Trainer(model.collect_params(),
                            'adam', {'learning_rate': lr})
    loss = MaskedSoftmaxCELoss()
    animator = d2l.Animator(xlabel='epoch', ylabel='loss',
                            xlim=[1, num_epochs], ylim=[0, 0.25])
    for epoch in range(1, num_epochs + 1):
        timer = d2l.Timer()
        metric = d2l.Accumulator(2)  # loss_sum, num_tokens
        for batch in data_iter:
            X, X_vlen, Y, Y_vlen = [x.as_in_ctx(ctx) for x in batch]
            Y_input, Y_label, Y_vlen = Y[:, :-1], Y[:, 1:], Y_vlen-1
            with autograd.record():
                Y_hat, _ = model(X, Y_input, X_vlen, Y_vlen)
                l = loss(Y_hat, Y_label, Y_vlen)
            l.backward()
            d2l.grad_clipping(model, 1)
            num_tokens = Y_vlen.sum()
            trainer.step(num_tokens)
            metric.add(l.sum(), num_tokens)
        if epoch % 10 == 0:
            animator.add(epoch, (metric[0]/metric[1],))
    print('loss %.3f, %d tokens/sec on %s ' % (
        metric[0]/metric[1], metric[1]/timer.stop(), ctx))
```

次に、モデルインスタンスを作成してハイパーパラメータを設定します。これでモデルを学習できます。

```{.python .input  n=12}
#@tab mxnet
embed_size, num_hiddens, num_layers, dropout = 32, 32, 2, 0.0
batch_size, num_steps = 64, 10
lr, num_epochs, ctx = 0.005, 300, d2l.try_gpu()

src_vocab, tgt_vocab, train_iter = d2l.load_data_nmt(batch_size, num_steps)
encoder = Seq2SeqEncoder(
    len(src_vocab), embed_size, num_hiddens, num_layers, dropout)
decoder = Seq2SeqDecoder(
    len(tgt_vocab), embed_size, num_hiddens, num_layers, dropout)
model = d2l.EncoderDecoder(encoder, decoder)
train_s2s_ch9(model, train_iter, lr, num_epochs, ctx)
```

## 予測

ここでは、出力系列を生成するための最も単純な方法である greedy search を実装します。:numref:`fig_seq2seq_predict` に示すように、予測時には、学習時と同様に時刻 0 で同じ "&lt;bos&gt;" トークンをデコーダに入力します。ただし、後続の時刻の入力トークンは、直前の時刻で予測されたトークンになります。

![Sequence to sequence model predicting with greedy search](../img/seq2seq_predict.svg)
:label:`fig_seq2seq_predict`

```{.python .input  n=16}
#@tab mxnet
#@save
class BeamSearchNode(object):
    def __init__(self, hiddenstate, previousNode, wordId, logProb, length):
        self.h = hiddenstate
        self.prevNode = previousNode
        self.wordid = wordId
        self.logp = logProb
        self.length = length

    def eval(self, alpha=1.0):
        reward = 0
        return self.logp / float(self.length - 1 + 1e-6) + alpha * reward
#@save
def predict_s2s_ch9_beam(model, src_sentence, src_vocab, tgt_vocab, num_steps,
                         beam_width, ctx):
    src_tokens = src_vocab[src_sentence.lower().split(' ')]
    enc_valid_len = np.array([len(src_tokens)], ctx=ctx)
    src_tokens = d2l.truncate_pad(src_tokens, num_steps, src_vocab['<pad>'])
    enc_X = np.array(src_tokens, ctx=ctx)
    # Add the batch_size dimension
    enc_outputs = model.encoder(np.expand_dims(enc_X, axis=0),
                                enc_valid_len)
    dec_state = model.decoder.init_state(enc_outputs, enc_valid_len)
    dec_X = np.expand_dims(np.array([tgt_vocab['<bos>']], ctx=ctx), axis=0)
    
    node = BeamSearchNode(dec_state, None, dec_X, 0, 1)
    nodes = PriorityQueue()
    decoded_batch = []
    nodes.put((-node.eval(), node))
    #while True:
    for _ in range(num_steps):
        # give up when decoding takes too long
        score, n = nodes.get()
        dec_X = n.wordid
        dec_state = n.h
        if n.wordid.item() == tgt_vocab['<eos>'] and n.prevNode != None:
            endnodes = (score, n)
            break
        Y, dec_state = model.decoder(dec_X, dec_state)
        indexes = npx.topk(Y, k=beam_width)
        nextnodes = []
        for new_k in range(beam_width):
            decoded_t = indexes[:,:,new_k]
            log_p = Y.reshape(-1)[decoded_t].item()
            node = BeamSearchNode(dec_state, n, decoded_t, n.logp + log_p, n.length + 1)
            score = -node.eval()
            nextnodes.append((score, node))
        for i in range(len(nextnodes)):
            score, nn = nextnodes[i]
            nodes.put((score, nn))
            
    if len(endnodes) == 0:
        endnodes = nodes.get()
    score, n = endnodes
    predict_tokens = []
    if int(n.wordid) != tgt_vocab['<eos>']:
        predict_tokens.append(int(n.wordid))
    # back trace
    while n.prevNode != None:
        n = n.prevNode
        if int(n.wordid) != tgt_vocab['<bos>']:
            predict_tokens.append(int(n.wordid))
    predict_tokens = predict_tokens[::-1]
    return ' '.join(tgt_vocab.to_tokens(predict_tokens))
```

いくつかの例を試してみましょう。

```{.python .input  n=204}
#@tab mxnet
for sentence in ['Go .', 'Wow !', "I'm OK .", 'I won !']:
    print(sentence + ' => ' + predict_s2s_ch9_beam(
        model, sentence, src_vocab, tgt_vocab, num_steps, 3, ctx))
```

## 要約

* seq2seq モデルは、系列入力から系列出力を生成するためのエンコーダ・デコーダアーキテクチャに基づいています。
* エンコーダとデコーダの両方で複数の LSTM 層を用います。


## 演習

1. ニューラル機械翻訳以外に、seq2seq の他の用途を思いつけますか。
1. この節の例で入力系列がもっと長かったらどうなりますか。
1. 損失関数で `SequenceMask` を使わないと、何が起こるでしょうか。


## [Discussions](https://discuss.mxnet.io/t/4357)

![](../img/qr_seq2seq.svg)\n