# 自然言語推論: BERTのファインチューニング
:label:`sec_natural-language-inference-bert`

この章の前半では、
SNLIデータセット（:numref:`sec_natural-language-inference-and-dataset` で説明）における自然言語推論タスクのために、
注意機構に基づくアーキテクチャを設計しました
（:numref:`sec_natural-language-inference-attention` を参照）。
ここでは、このタスクをBERTのファインチューニングによって再び扱います。
:numref:`sec_finetuning-bert` で述べたように、
自然言語推論は系列レベルのテキスト対分類問題であり、
BERTのファインチューニングには、 :numref:`fig_nlp-map-nli-bert` に示すような、
追加のMLPベースのアーキテクチャだけが必要です。

![この節では、事前学習済みBERTを自然言語推論のためのMLPベースのアーキテクチャに入力する。](../img/nlp-map-nli-bert.svg)
:label:`fig_nlp-map-nli-bert`

この節では、
事前学習済みのBERTの小さい版をダウンロードし、
SNLIデータセット上の自然言語推論のために
それをファインチューニングします。

```{.python .input}
#@tab mxnet
from d2l import mxnet as d2l
import json
import multiprocessing
from mxnet import gluon, np, npx
from mxnet.gluon import nn
import os

npx.set_np()
```

```{.python .input}
#@tab pytorch
from d2l import torch as d2l
import json
import multiprocessing
import torch
from torch import nn
import os
```

## [**事前学習済みBERTの読み込み**]

WikiText-2データセット上でBERTを事前学習する方法は、
:numref:`sec_bert-dataset` と :numref:`sec_bert-pretraining` で説明しました
（なお、元のBERTモデルははるかに大きなコーパスで事前学習されています）。
:numref:`sec_bert-pretraining` で述べたように、
元のBERTモデルは数億個のパラメータを持っています。
以下では、事前学習済みBERTの2つの版を用意します。
「bert.base」は元のBERT baseモデルとほぼ同じ大きさで、ファインチューニングには多くの計算資源を必要とします。
一方、「bert.small」はデモをしやすくするための小さい版です。

```{.python .input}
#@tab mxnet
d2l.DATA_HUB['bert.base'] = (d2l.DATA_URL + 'bert.base.zip',
                             '7b3820b35da691042e5d34c0971ac3edbd80d3f4')
d2l.DATA_HUB['bert.small'] = (d2l.DATA_URL + 'bert.small.zip',
                              'a4e718a47137ccd1809c9107ab4f5edd317bae2c')
```

```{.python .input}
#@tab pytorch
d2l.DATA_HUB['bert.base'] = (d2l.DATA_URL + 'bert.base.torch.zip',
                             '225d66f04cae318b841a13d32af3acc165f253ac')
d2l.DATA_HUB['bert.small'] = (d2l.DATA_URL + 'bert.small.torch.zip',
                              'c72329e68a732bef0452e4b96a1c341c8910f81f')
```

どちらの事前学習済みBERTモデルにも、語彙集合を定義する「vocab.json」ファイルと、
事前学習済みパラメータを格納した「pretrained.params」ファイルが含まれています。
以下の `load_pretrained_model` 関数を実装して、[**事前学習済みBERTのパラメータを読み込みます**]。

```{.python .input}
#@tab mxnet
def load_pretrained_model(pretrained_model, num_hiddens, ffn_num_hiddens,
                          num_heads, num_blks, dropout, max_len, devices):
    data_dir = d2l.download_extract(pretrained_model)
    # Define an empty vocabulary to load the predefined vocabulary
    vocab = d2l.Vocab()
    vocab.idx_to_token = json.load(open(os.path.join(data_dir, 'vocab.json')))
    vocab.token_to_idx = {token: idx for idx, token in enumerate(
        vocab.idx_to_token)}
    bert = d2l.BERTModel(len(vocab), num_hiddens, ffn_num_hiddens, num_heads, 
                         num_blks, dropout, max_len)
    # Load pretrained BERT parameters
    bert.load_parameters(os.path.join(data_dir, 'pretrained.params'),
                         ctx=devices)
    return bert, vocab
```

```{.python .input}
#@tab pytorch
def load_pretrained_model(pretrained_model, num_hiddens, ffn_num_hiddens,
                          num_heads, num_blks, dropout, max_len, devices):
    data_dir = d2l.download_extract(pretrained_model)
    # Define an empty vocabulary to load the predefined vocabulary
    vocab = d2l.Vocab()
    vocab.idx_to_token = json.load(open(os.path.join(data_dir, 'vocab.json')))
    vocab.token_to_idx = {token: idx for idx, token in enumerate(
        vocab.idx_to_token)}
    bert = d2l.BERTModel(
        len(vocab), num_hiddens, ffn_num_hiddens=ffn_num_hiddens, num_heads=4,
        num_blks=2, dropout=0.2, max_len=max_len)
    # Load pretrained BERT parameters
    bert.load_state_dict(torch.load(os.path.join(data_dir,
                                                 'pretrained.params')))
    return bert, vocab
```

多くのマシンでデモを行いやすくするために、
この節では事前学習済みBERTの小さい版（「bert.small」）を読み込み、
ファインチューニングします。
演習では、はるかに大きい「bert.base」をファインチューニングして、
テスト精度を大幅に改善する方法を示します。

```{.python .input}
#@tab all
devices = d2l.try_all_gpus()
bert, vocab = load_pretrained_model(
    'bert.small', num_hiddens=256, ffn_num_hiddens=512, num_heads=4,
    num_blks=2, dropout=0.1, max_len=512, devices=devices)
```

## [**BERTファインチューニング用データセット**]

SNLIデータセット上の下流タスクである自然言語推論に対して、
カスタマイズしたデータセットクラス `SNLIBERTDataset` を定義します。
各例では、
前提と仮説がテキスト系列のペアを形成し、
:numref:`fig_bert-two-seqs` に示すように1つのBERT入力系列にまとめられます。
:numref:`subsec_bert_input_rep` を思い出してください。セグメントIDは、
BERT入力系列内で前提と仮説を区別するために使われます。
BERT入力系列の事前定義された最大長（`max_len`）に対して、
入力テキスト対のうち長い方の末尾トークンは、
`max_len` を満たすまで削除され続けます。
BERTのファインチューニングのためのSNLIデータセット生成を高速化するために、
4つのワーカープロセスを使って訓練例またはテスト例を並列生成します。

```{.python .input}
#@tab mxnet
class SNLIBERTDataset(gluon.data.Dataset):
    def __init__(self, dataset, max_len, vocab=None):
        all_premise_hypothesis_tokens = [[
            p_tokens, h_tokens] for p_tokens, h_tokens in zip(
            *[d2l.tokenize([s.lower() for s in sentences])
              for sentences in dataset[:2]])]
        
        self.labels = np.array(dataset[2])
        self.vocab = vocab
        self.max_len = max_len
        (self.all_token_ids, self.all_segments,
         self.valid_lens) = self._preprocess(all_premise_hypothesis_tokens)
        print('read ' + str(len(self.all_token_ids)) + ' examples')

    def _preprocess(self, all_premise_hypothesis_tokens):
        pool = multiprocessing.Pool(4)  # Use 4 worker processes
        out = pool.map(self._mp_worker, all_premise_hypothesis_tokens)
        all_token_ids = [
            token_ids for token_ids, segments, valid_len in out]
        all_segments = [segments for token_ids, segments, valid_len in out]
        valid_lens = [valid_len for token_ids, segments, valid_len in out]
        return (np.array(all_token_ids, dtype='int32'),
                np.array(all_segments, dtype='int32'), 
                np.array(valid_lens))

    def _mp_worker(self, premise_hypothesis_tokens):
        p_tokens, h_tokens = premise_hypothesis_tokens
        self._truncate_pair_of_tokens(p_tokens, h_tokens)
        tokens, segments = d2l.get_tokens_and_segments(p_tokens, h_tokens)
        token_ids = self.vocab[tokens] + [self.vocab['<pad>']] \
                             * (self.max_len - len(tokens))
        segments = segments + [0] * (self.max_len - len(segments))
        valid_len = len(tokens)
        return token_ids, segments, valid_len

    def _truncate_pair_of_tokens(self, p_tokens, h_tokens):
        # Reserve slots for '<CLS>', '<SEP>', and '<SEP>' tokens for the BERT
        # input
        while len(p_tokens) + len(h_tokens) > self.max_len - 3:
            if len(p_tokens) > len(h_tokens):
                p_tokens.pop()
            else:
                h_tokens.pop()

    def __getitem__(self, idx):
        return (self.all_token_ids[idx], self.all_segments[idx],
                self.valid_lens[idx]), self.labels[idx]

    def __len__(self):
        return len(self.all_token_ids)
```

```{.python .input}
#@tab pytorch
class SNLIBERTDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, max_len, vocab=None):
        all_premise_hypothesis_tokens = [[
            p_tokens, h_tokens] for p_tokens, h_tokens in zip(
            *[d2l.tokenize([s.lower() for s in sentences])
              for sentences in dataset[:2]])]
        
        self.labels = torch.tensor(dataset[2])
        self.vocab = vocab
        self.max_len = max_len
        (self.all_token_ids, self.all_segments,
         self.valid_lens) = self._preprocess(all_premise_hypothesis_tokens)
        print('read ' + str(len(self.all_token_ids)) + ' examples')

    def _preprocess(self, all_premise_hypothesis_tokens):
        pool = multiprocessing.Pool(4)  # Use 4 worker processes
        out = pool.map(self._mp_worker, all_premise_hypothesis_tokens)
        all_token_ids = [
            token_ids for token_ids, segments, valid_len in out]
        all_segments = [segments for token_ids, segments, valid_len in out]
        valid_lens = [valid_len for token_ids, segments, valid_len in out]
        return (torch.tensor(all_token_ids, dtype=torch.long),
                torch.tensor(all_segments, dtype=torch.long), 
                torch.tensor(valid_lens))

    def _mp_worker(self, premise_hypothesis_tokens):
        p_tokens, h_tokens = premise_hypothesis_tokens
        self._truncate_pair_of_tokens(p_tokens, h_tokens)
        tokens, segments = d2l.get_tokens_and_segments(p_tokens, h_tokens)
        token_ids = self.vocab[tokens] + [self.vocab['<pad>']] \
                             * (self.max_len - len(tokens))
        segments = segments + [0] * (self.max_len - len(segments))
        valid_len = len(tokens)
        return token_ids, segments, valid_len

    def _truncate_pair_of_tokens(self, p_tokens, h_tokens):
        # Reserve slots for '<CLS>', '<SEP>', and '<SEP>' tokens for the BERT
        # input
        while len(p_tokens) + len(h_tokens) > self.max_len - 3:
            if len(p_tokens) > len(h_tokens):
                p_tokens.pop()
            else:
                h_tokens.pop()

    def __getitem__(self, idx):
        return (self.all_token_ids[idx], self.all_segments[idx],
                self.valid_lens[idx]), self.labels[idx]

    def __len__(self):
        return len(self.all_token_ids)
```

SNLIデータセットをダウンロードした後、
`SNLIBERTDataset` クラスをインスタンス化することで、
[**訓練例とテスト例を生成**]します。
これらの例は、自然言語推論の訓練とテストの際にミニバッチで読み込まれます。

```{.python .input}
#@tab mxnet
# Reduce `batch_size` if there is an out of memory error. In the original BERT
# model, `max_len` = 512
batch_size, max_len, num_workers = 512, 128, d2l.get_dataloader_workers()
data_dir = d2l.download_extract('SNLI')
train_set = SNLIBERTDataset(d2l.read_snli(data_dir, True), max_len, vocab)
test_set = SNLIBERTDataset(d2l.read_snli(data_dir, False), max_len, vocab)
train_iter = gluon.data.DataLoader(train_set, batch_size, shuffle=True,
                                   num_workers=num_workers)
test_iter = gluon.data.DataLoader(test_set, batch_size,
                                  num_workers=num_workers)
```

```{.python .input}
#@tab pytorch
# Reduce `batch_size` if there is an out of memory error. In the original BERT
# model, `max_len` = 512
batch_size, max_len, num_workers = 512, 128, d2l.get_dataloader_workers()
data_dir = d2l.download_extract('SNLI')
train_set = SNLIBERTDataset(d2l.read_snli(data_dir, True), max_len, vocab)
test_set = SNLIBERTDataset(d2l.read_snli(data_dir, False), max_len, vocab)
train_iter = torch.utils.data.DataLoader(train_set, batch_size, shuffle=True,
                                   num_workers=num_workers)
test_iter = torch.utils.data.DataLoader(test_set, batch_size,
                                  num_workers=num_workers)
```

## BERTのファインチューニング

:numref:`fig_bert-two-seqs` が示すように、
自然言語推論のためにBERTをファインチューニングするには、
2つの全結合層からなる追加のMLPだけで十分です
（以下の `BERTClassifier` クラスの `self.hidden` と `self.output` を参照）。
[**このMLPは、前提と仮説の両方の情報を符号化した特別な “&lt;cls&gt;” トークンのBERT表現を**]
[**自然言語推論の3つの出力**]、
すなわち含意、矛盾、中立へと変換します。

```{.python .input}
#@tab mxnet
class BERTClassifier(nn.Block):
    def __init__(self, bert):
        super(BERTClassifier, self).__init__()
        self.encoder = bert.encoder
        self.hidden = bert.hidden
        self.output = nn.Dense(3)

    def forward(self, inputs):
        tokens_X, segments_X, valid_lens_x = inputs
        encoded_X = self.encoder(tokens_X, segments_X, valid_lens_x)
        return self.output(self.hidden(encoded_X[:, 0, :]))
```

```{.python .input}
#@tab pytorch
class BERTClassifier(nn.Module):
    def __init__(self, bert):
        super(BERTClassifier, self).__init__()
        self.encoder = bert.encoder
        self.hidden = bert.hidden
        self.output = nn.LazyLinear(3)

    def forward(self, inputs):
        tokens_X, segments_X, valid_lens_x = inputs
        encoded_X = self.encoder(tokens_X, segments_X, valid_lens_x)
        return self.output(self.hidden(encoded_X[:, 0, :]))
```

以下では、
事前学習済みBERTモデル `bert` を `BERTClassifier` のインスタンス `net` に入力し、
下流アプリケーションに利用します。
一般的なBERTファインチューニングの実装では、
追加MLPの出力層（`net.output`）のパラメータだけがゼロから学習されます。
事前学習済みBERTエンコーダ（`net.encoder`）と追加MLPの隠れ層（`net.hidden`）のすべてのパラメータはファインチューニングされます。

```{.python .input}
#@tab mxnet
net = BERTClassifier(bert)
net.output.initialize(ctx=devices)
```

```{.python .input}
#@tab pytorch
net = BERTClassifier(bert)
```

:numref:`sec_bert` を思い出してください。
`MaskLM` クラスと `NextSentencePred` クラスの両方には、
それぞれが用いるMLP内にパラメータがあります。
これらのパラメータは事前学習済みBERTモデル `bert` の一部であり、
したがって `net` のパラメータの一部でもあります。
しかし、これらのパラメータは事前学習中に
マスク言語モデリング損失と次文予測損失を計算するためだけに使われます。
これら2つの損失関数は下流アプリケーションのファインチューニングには無関係なので、
BERTをファインチューニングするときには、`MaskLM` と `NextSentencePred` で用いられるMLPのパラメータは更新されません（staled）。

勾配が古いままのパラメータを許可するために、
`d2l.train_batch_ch13` の `step` 関数では `ignore_stale_grad=True` フラグが設定されています。
この関数を使って、SNLIの訓練セット（`train_iter`）とテストセット（`test_iter`）を用い、
モデル `net` を訓練および評価します。
計算資源が限られているため、[**訓練**]精度とテスト精度はさらに改善できます。これについては演習で扱います。

```{.python .input}
#@tab mxnet
lr, num_epochs = 1e-4, 5
trainer = gluon.Trainer(net.collect_params(), 'adam', {'learning_rate': lr})
loss = gluon.loss.SoftmaxCrossEntropyLoss()
d2l.train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs, devices,
               d2l.split_batch_multi_inputs)
```

```{.python .input}
#@tab pytorch
lr, num_epochs = 1e-4, 5
trainer = torch.optim.Adam(net.parameters(), lr=lr)
loss = nn.CrossEntropyLoss(reduction='none')
net(next(iter(train_iter))[0])
d2l.train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs, devices)
```

## まとめ

* 事前学習済みBERTモデルは、SNLIデータセット上の自然言語推論のような下流アプリケーションにファインチューニングできます。
* ファインチューニング中、BERTモデルは下流アプリケーションのモデルの一部になります。事前学習の損失にのみ関係するパラメータは、ファインチューニング中には更新されません。 



## 演習

1. 計算資源が許すなら、元のBERT baseモデルとほぼ同じ大きさの、はるかに大きい事前学習済みBERTモデルをファインチューニングしてください。`load_pretrained_model` 関数の引数は、`'bert.small'` を `'bert.base'` に置き換え、`num_hiddens=256`、`ffn_num_hiddens=512`、`num_heads=4`、`num_blks=2` の値をそれぞれ768、3072、12、12に増やしてください。ファインチューニングのエポック数を増やし（必要なら他のハイパーパラメータも調整し）、テスト精度を0.86より高くできますか？
1. 長さの比率に応じて2つの系列をどのように切り詰めるべきでしょうか？このペア切り詰め方法と `SNLIBERTDataset` クラスで使われている方法を比較してください。それぞれの長所と短所は何でしょうか？
