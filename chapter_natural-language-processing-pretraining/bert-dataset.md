# BERTの事前学習のためのデータセット
:label:`sec_bert-dataset`

:numref:`sec_bert` で実装した BERT モデルを事前学習するには、
2つの事前学習タスク、すなわち
マスク付き言語モデルと次文予測を容易にするための、
理想的な形式のデータセットを生成する必要があります。
一方で、
元の BERT モデルは BookCorpus と English Wikipedia という
2つの巨大なコーパスの連結上で事前学習されており
(:numref:`subsec_bert_pretraining_tasks` を参照)、
本書の多くの読者にとっては実行が難しいです。
他方で、
既製の事前学習済み BERT モデルは、
医療のような特定分野のアプリケーションには適さない場合があります。
そのため、カスタマイズしたデータセット上で BERT を事前学習することが
一般的になりつつあります。
BERT の事前学習を示しやすくするために、
ここではより小さなコーパスである WikiText-2 :cite:`Merity.Xiong.Bradbury.ea.2016` を用います。

:numref:`sec_word2vec_data` で word2vec の事前学習に用いた PTB データセットと比べると、
WikiText-2 は (i) 元の句読点を保持しているため、次文予測に適している、
(ii) 元の大文字小文字と数値を保持している、
(iii) 2倍以上大きい、という特徴があります。

```{.python .input}
#@tab mxnet
from d2l import mxnet as d2l
from mxnet import gluon, np, npx
import os
import random

npx.set_np()
```

```{.python .input}
#@tab pytorch
from d2l import torch as d2l
import os
import random
import torch
```

[**WikiText-2 データセット**]では、
各行が1つの段落を表し、
任意の句読点とその直前のトークンの間には空白が挿入されています。
少なくとも2文を含む段落のみを残します。
文の分割には、簡単のためにピリオドのみを区切り文字として使います。
より複雑な文分割手法については、この節の最後の演習で扱います。

```{.python .input}
#@tab all
#@save
d2l.DATA_HUB['wikitext-2'] = (
    'https://s3.amazonaws.com/research.metamind.io/wikitext/'
    'wikitext-2-v1.zip', '3c914d17d80b1459be871a5039ac23e752a53cbe')

#@save
def _read_wiki(data_dir):
    file_name = os.path.join(data_dir, 'wiki.train.tokens')
    with open(file_name, 'r') as f:
        lines = f.readlines()
    # Uppercase letters are converted to lowercase ones
    paragraphs = [line.strip().lower().split(' . ')
                  for line in lines if len(line.split(' . ')) >= 2]
    random.shuffle(paragraphs)
    return paragraphs
```

## 事前学習タスクのための補助関数の定義

以下ではまず、
BERT の2つの事前学習タスク、
次文予測とマスク付き言語モデルのための補助関数を実装します。
これらの補助関数は後で、
生のテキストコーパスを BERT の事前学習に適した理想的な形式のデータセットへ変換する際に呼び出されます。

### [**次文予測タスクの生成**]

:numref:`subsec_nsp` の説明に従い、
`_get_next_sentence` 関数は
二値分類タスクのための学習例を生成します。

```{.python .input}
#@tab all
#@save
def _get_next_sentence(sentence, next_sentence, paragraphs):
    if random.random() < 0.5:
        is_next = True
    else:
        # `paragraphs` is a list of lists of lists
        next_sentence = random.choice(random.choice(paragraphs))
        is_next = False
    return sentence, next_sentence, is_next
```

次の関数は、`_get_next_sentence` 関数を呼び出すことで、
入力 `paragraph` から次文予測の学習例を生成します。
ここで `paragraph` は文のリストであり、各文はトークンのリストです。
引数 `max_len` は、事前学習中の BERT 入力系列の最大長を指定します。

```{.python .input}
#@tab all
#@save
def _get_nsp_data_from_paragraph(paragraph, paragraphs, vocab, max_len):
    nsp_data_from_paragraph = []
    for i in range(len(paragraph) - 1):
        tokens_a, tokens_b, is_next = _get_next_sentence(
            paragraph[i], paragraph[i + 1], paragraphs)
        # Consider 1 '<cls>' token and 2 '<sep>' tokens
        if len(tokens_a) + len(tokens_b) + 3 > max_len:
            continue
        tokens, segments = d2l.get_tokens_and_segments(tokens_a, tokens_b)
        nsp_data_from_paragraph.append((tokens, segments, is_next))
    return nsp_data_from_paragraph
```

### [**マスク付き言語モデルタスクの生成**]
:label:`subsec_prepare_mlm_data`

BERT 入力系列から
マスク付き言語モデルタスクの学習例を生成するために、
次の `_replace_mlm_tokens` 関数を定義します。
入力において、`tokens` は BERT 入力系列を表すトークンのリスト、
`candidate_pred_positions` は特殊トークンを除いた BERT 入力系列中のトークンインデックスのリストです
（特殊トークンはマスク付き言語モデルタスクでは予測されません）、
`num_mlm_preds` は予測数を示します（予測対象としてランダムな15%のトークンを思い出してください）。
:numref:`subsec_mlm` で定義したマスク付き言語モデルタスクに従い、
各予測位置では、入力は特殊な “&lt;mask&gt;” トークンやランダムなトークンに置き換えられるか、
あるいはそのまま保持されます。
最終的にこの関数は、置換後の入力トークン、
予測が行われるトークンインデックス、
およびそれらのラベルを返します。

```{.python .input}
#@tab all
#@save
def _replace_mlm_tokens(tokens, candidate_pred_positions, num_mlm_preds,
                        vocab):
    # For the input of a masked language model, make a new copy of tokens and
    # replace some of them by '<mask>' or random tokens
    mlm_input_tokens = [token for token in tokens]
    pred_positions_and_labels = []
    # Shuffle for getting 15% random tokens for prediction in the masked
    # language modeling task
    random.shuffle(candidate_pred_positions)
    for mlm_pred_position in candidate_pred_positions:
        if len(pred_positions_and_labels) >= num_mlm_preds:
            break
        masked_token = None
        # 80% of the time: replace the word with the '<mask>' token
        if random.random() < 0.8:
            masked_token = '<mask>'
        else:
            # 10% of the time: keep the word unchanged
            if random.random() < 0.5:
                masked_token = tokens[mlm_pred_position]
            # 10% of the time: replace the word with a random word
            else:
                masked_token = random.choice(vocab.idx_to_token)
        mlm_input_tokens[mlm_pred_position] = masked_token
        pred_positions_and_labels.append(
            (mlm_pred_position, tokens[mlm_pred_position]))
    return mlm_input_tokens, pred_positions_and_labels
```

前述の `_replace_mlm_tokens` 関数を呼び出すことで、
次の関数は BERT 入力系列（`tokens`）を入力として受け取り、
入力トークンのインデックス
（:numref:`subsec_mlm` で説明したようなトークン置換後）、
予測が行われるトークンインデックス、
およびそれらのラベルインデックスを返します。

```{.python .input}
#@tab all
#@save
def _get_mlm_data_from_tokens(tokens, vocab):
    candidate_pred_positions = []
    # `tokens` is a list of strings
    for i, token in enumerate(tokens):
        # Special tokens are not predicted in the masked language modeling
        # task
        if token in ['<cls>', '<sep>']:
            continue
        candidate_pred_positions.append(i)
    # 15% of random tokens are predicted in the masked language modeling task
    num_mlm_preds = max(1, round(len(tokens) * 0.15))
    mlm_input_tokens, pred_positions_and_labels = _replace_mlm_tokens(
        tokens, candidate_pred_positions, num_mlm_preds, vocab)
    pred_positions_and_labels = sorted(pred_positions_and_labels,
                                       key=lambda x: x[0])
    pred_positions = [v[0] for v in pred_positions_and_labels]
    mlm_pred_labels = [v[1] for v in pred_positions_and_labels]
    return vocab[mlm_input_tokens], pred_positions, vocab[mlm_pred_labels]
```

## テキストを事前学習データセットへ変換する

これで、BERT 事前学習用の `Dataset` クラスをカスタマイズする準備がほぼ整いました。
その前に、
入力に [**特殊な “&lt;pad&gt;” トークンを追加する**] ための補助関数 `_pad_bert_inputs` を定義する必要があります。
その引数 `examples` には、2つの事前学習タスクに対する補助関数 `_get_nsp_data_from_paragraph` と `_get_mlm_data_from_tokens` の出力が含まれます。

```{.python .input}
#@tab mxnet
#@save
def _pad_bert_inputs(examples, max_len, vocab):
    max_num_mlm_preds = round(max_len * 0.15)
    all_token_ids, all_segments, valid_lens,  = [], [], []
    all_pred_positions, all_mlm_weights, all_mlm_labels = [], [], []
    nsp_labels = []
    for (token_ids, pred_positions, mlm_pred_label_ids, segments,
         is_next) in examples:
        all_token_ids.append(np.array(token_ids + [vocab['<pad>']] * (
            max_len - len(token_ids)), dtype='int32'))
        all_segments.append(np.array(segments + [0] * (
            max_len - len(segments)), dtype='int32'))
        # `valid_lens` excludes count of '<pad>' tokens
        valid_lens.append(np.array(len(token_ids), dtype='float32'))
        all_pred_positions.append(np.array(pred_positions + [0] * (
            max_num_mlm_preds - len(pred_positions)), dtype='int32'))
        # Predictions of padded tokens will be filtered out in the loss via
        # multiplication of 0 weights
        all_mlm_weights.append(
            np.array([1.0] * len(mlm_pred_label_ids) + [0.0] * (
                max_num_mlm_preds - len(pred_positions)), dtype='float32'))
        all_mlm_labels.append(np.array(mlm_pred_label_ids + [0] * (
            max_num_mlm_preds - len(mlm_pred_label_ids)), dtype='int32'))
        nsp_labels.append(np.array(is_next))
    return (all_token_ids, all_segments, valid_lens, all_pred_positions,
            all_mlm_weights, all_mlm_labels, nsp_labels)
```

```{.python .input}
#@tab pytorch
#@save
def _pad_bert_inputs(examples, max_len, vocab):
    max_num_mlm_preds = round(max_len * 0.15)
    all_token_ids, all_segments, valid_lens,  = [], [], []
    all_pred_positions, all_mlm_weights, all_mlm_labels = [], [], []
    nsp_labels = []
    for (token_ids, pred_positions, mlm_pred_label_ids, segments,
         is_next) in examples:
        all_token_ids.append(torch.tensor(token_ids + [vocab['<pad>']] * (
            max_len - len(token_ids)), dtype=torch.long))
        all_segments.append(torch.tensor(segments + [0] * (
            max_len - len(segments)), dtype=torch.long))
        # `valid_lens` excludes count of '<pad>' tokens
        valid_lens.append(torch.tensor(len(token_ids), dtype=torch.float32))
        all_pred_positions.append(torch.tensor(pred_positions + [0] * (
            max_num_mlm_preds - len(pred_positions)), dtype=torch.long))
        # Predictions of padded tokens will be filtered out in the loss via
        # multiplication of 0 weights
        all_mlm_weights.append(
            torch.tensor([1.0] * len(mlm_pred_label_ids) + [0.0] * (
                max_num_mlm_preds - len(pred_positions)),
                dtype=torch.float32))
        all_mlm_labels.append(torch.tensor(mlm_pred_label_ids + [0] * (
            max_num_mlm_preds - len(mlm_pred_label_ids)), dtype=torch.long))
        nsp_labels.append(torch.tensor(is_next, dtype=torch.long))
    return (all_token_ids, all_segments, valid_lens, all_pred_positions,
            all_mlm_weights, all_mlm_labels, nsp_labels)
```

2つの事前学習タスクの学習例を生成する補助関数と、
入力をパディングする補助関数をまとめて、
以下の `_WikiTextDataset` クラスを [**BERT 事前学習用の WikiText-2 データセット**] としてカスタマイズします。
`__getitem__ `関数を実装することで、
WikiText-2 コーパス中の文のペアから生成された事前学習
（マスク付き言語モデルと次文予測）の例に任意にアクセスできます。

元の BERT モデルは、語彙サイズが 30000 の WordPiece 埋め込みを使用します :cite:`Wu.Schuster.Chen.ea.2016`。
WordPiece のトークン化手法は、
:numref:`subsec_Byte_Pair_Encoding` にある元の byte pair encoding アルゴリズムをわずかに修正したものです。
簡単のため、ここではトークン化に `d2l.tokenize` 関数を使います。
5回未満しか現れない低頻度トークンは除外します。

```{.python .input}
#@tab mxnet
#@save
class _WikiTextDataset(gluon.data.Dataset):
    def __init__(self, paragraphs, max_len):
        # Input `paragraphs[i]` is a list of sentence strings representing a
        # paragraph; while output `paragraphs[i]` is a list of sentences
        # representing a paragraph, where each sentence is a list of tokens
        paragraphs = [d2l.tokenize(
            paragraph, token='word') for paragraph in paragraphs]
        sentences = [sentence for paragraph in paragraphs
                     for sentence in paragraph]
        self.vocab = d2l.Vocab(sentences, min_freq=5, reserved_tokens=[
            '<pad>', '<mask>', '<cls>', '<sep>'])
        # Get data for the next sentence prediction task
        examples = []
        for paragraph in paragraphs:
            examples.extend(_get_nsp_data_from_paragraph(
                paragraph, paragraphs, self.vocab, max_len))
        # Get data for the masked language model task
        examples = [(_get_mlm_data_from_tokens(tokens, self.vocab)
                      + (segments, is_next))
                     for tokens, segments, is_next in examples]
        # Pad inputs
        (self.all_token_ids, self.all_segments, self.valid_lens,
         self.all_pred_positions, self.all_mlm_weights,
         self.all_mlm_labels, self.nsp_labels) = _pad_bert_inputs(
            examples, max_len, self.vocab)

    def __getitem__(self, idx):
        return (self.all_token_ids[idx], self.all_segments[idx],
                self.valid_lens[idx], self.all_pred_positions[idx],
                self.all_mlm_weights[idx], self.all_mlm_labels[idx],
                self.nsp_labels[idx])

    def __len__(self):
        return len(self.all_token_ids)
```

```{.python .input}
#@tab pytorch
#@save
class _WikiTextDataset(torch.utils.data.Dataset):
    def __init__(self, paragraphs, max_len):
        # Input `paragraphs[i]` is a list of sentence strings representing a
        # paragraph; while output `paragraphs[i]` is a list of sentences
        # representing a paragraph, where each sentence is a list of tokens
        paragraphs = [d2l.tokenize(
            paragraph, token='word') for paragraph in paragraphs]
        sentences = [sentence for paragraph in paragraphs
                     for sentence in paragraph]
        self.vocab = d2l.Vocab(sentences, min_freq=5, reserved_tokens=[
            '<pad>', '<mask>', '<cls>', '<sep>'])
        # Get data for the next sentence prediction task
        examples = []
        for paragraph in paragraphs:
            examples.extend(_get_nsp_data_from_paragraph(
                paragraph, paragraphs, self.vocab, max_len))
        # Get data for the masked language model task
        examples = [(_get_mlm_data_from_tokens(tokens, self.vocab)
                      + (segments, is_next))
                     for tokens, segments, is_next in examples]
        # Pad inputs
        (self.all_token_ids, self.all_segments, self.valid_lens,
         self.all_pred_positions, self.all_mlm_weights,
         self.all_mlm_labels, self.nsp_labels) = _pad_bert_inputs(
            examples, max_len, self.vocab)

    def __getitem__(self, idx):
        return (self.all_token_ids[idx], self.all_segments[idx],
                self.valid_lens[idx], self.all_pred_positions[idx],
                self.all_mlm_weights[idx], self.all_mlm_labels[idx],
                self.nsp_labels[idx])

    def __len__(self):
        return len(self.all_token_ids)
```

`_read_wiki` 関数と `_WikiTextDataset` クラスを用いて、
以下の `load_data_wiki` を定義し、
[**WikiText-2 データセットをダウンロードして、そこから事前学習例を生成**] します。

```{.python .input}
#@tab mxnet
#@save
def load_data_wiki(batch_size, max_len):
    """Load the WikiText-2 dataset."""
    num_workers = d2l.get_dataloader_workers()
    data_dir = d2l.download_extract('wikitext-2', 'wikitext-2')
    paragraphs = _read_wiki(data_dir)
    train_set = _WikiTextDataset(paragraphs, max_len)
    train_iter = gluon.data.DataLoader(train_set, batch_size, shuffle=True,
                                       num_workers=num_workers)
    return train_iter, train_set.vocab
```

```{.python .input}
#@tab pytorch
#@save
def load_data_wiki(batch_size, max_len):
    """Load the WikiText-2 dataset."""
    num_workers = d2l.get_dataloader_workers()
    data_dir = d2l.download_extract('wikitext-2', 'wikitext-2')
    paragraphs = _read_wiki(data_dir)
    train_set = _WikiTextDataset(paragraphs, max_len)
    train_iter = torch.utils.data.DataLoader(train_set, batch_size,
                                        shuffle=True, num_workers=num_workers)
    return train_iter, train_set.vocab
```

バッチサイズを 512、BERT 入力系列の最大長を 64 に設定して、
[**BERT 事前学習例のミニバッチの形状を出力**] してみます。
各 BERT 入力系列では、
マスク付き言語モデルタスクのために $10$（$64 \times 0.15$）個の位置が予測対象になります。

```{.python .input}
#@tab all
batch_size, max_len = 512, 64
train_iter, vocab = load_data_wiki(batch_size, max_len)

for (tokens_X, segments_X, valid_lens_x, pred_positions_X, mlm_weights_X,
     mlm_Y, nsp_y) in train_iter:
    print(tokens_X.shape, segments_X.shape, valid_lens_x.shape,
          pred_positions_X.shape, mlm_weights_X.shape, mlm_Y.shape,
          nsp_y.shape)
    break
```

最後に、語彙サイズを見てみましょう。
低頻度トークンを除外した後でも、
それは依然として PTB データセットの2倍以上の大きさです。

```{.python .input}
#@tab all
len(vocab)
```

## まとめ

* PTB データセットと比べると、WikiText-2 データセットは元の句読点、大文字小文字、数値を保持しており、2倍以上大きいです。
* WikiText-2 コーパス中の文のペアから生成された事前学習（マスク付き言語モデルと次文予測）の例に任意にアクセスできます。


## 演習

1. 簡単のため、文の分割にはピリオドのみを区切り文字として使いました。spaCy や NLTK など、他の文分割手法も試してみてください。例として NLTK を使います。まず NLTK をインストールする必要があります: `pip install nltk`。コードでは最初に `import nltk` します。次に、Punkt 文トークナイザをダウンロードします: `nltk.download('punkt')`。`sentences = 'This is great ! Why not ?'` のような文を分割するには、`nltk.tokenize.sent_tokenize(sentences)` を呼び出すと、2つの文文字列からなるリスト `['This is great !', 'Why not ?']` が返ります。
1. 低頻度トークンを一切除外しない場合、語彙サイズはいくつになりますか？

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/389)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1496)
:end_tab:\n