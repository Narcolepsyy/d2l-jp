# 自然言語推論とデータセット
:label:`sec_natural-language-inference-and-dataset`

:numref:`sec_sentiment` では、感情分析の問題について議論しました。
このタスクは、単一のテキスト系列を、感情の極性の集合のような事前に定義されたカテゴリに分類することを目的としています。
しかし、ある文が別の文から推論できるかどうかを判断する必要がある場合や、意味的に等価な文を識別して冗長性を取り除く必要がある場合には、1つのテキスト系列を分類する方法を知っているだけでは不十分です。
その代わりに、テキスト系列のペアについて推論できる必要があります。


## 自然言語推論

*自然言語推論* は、どちらもテキスト系列である *仮説* が *前提* から推論できるかどうかを調べます。
言い換えると、自然言語推論はテキスト系列のペア間の論理的関係を決定します。
このような関係は通常、次の3種類に分類されます。

* *含意*: 仮説は前提から推論できる。
* *矛盾*: 仮説の否定は前提から推論できる。
* *中立*: それ以外のすべての場合。

自然言語推論は、テキスト含意認識タスクとしても知られています。
たとえば、次のペアは、仮説中の "showing affection" が前提中の "hugging one another" から推論できるため、*含意* とラベル付けされます。

> 前提: Two women are hugging each other.

> 仮説: Two women are showing affection.

次は、"running the coding example" が "sleeping" ではなく "not sleeping" を示しているため、*矛盾* の例です。

> 前提: A man is running the coding example from Dive into Deep Learning.

> 仮説: The man is sleeping.

3つ目の例は、"are performing for us" という事実からは "famous" も "not famous" も推論できないため、*中立* の関係を示しています。

> 前提: The musicians are performing for us.

> 仮説: The musicians are famous.

自然言語推論は、自然言語を理解するうえで中心的な話題でした。
情報検索からオープンドメイン質問応答に至るまで、幅広い応用があります。
この問題を調べるために、まずは広く使われている自然言語推論のベンチマークデータセットを調べます。


## Stanford Natural Language Inference (SNLI) データセット

[**Stanford Natural Language Inference (SNLI) Corpus**] は、500000件を超えるラベル付き英語文ペアの集合です :cite:`Bowman.Angeli.Potts.ea.2015`。
抽出した SNLI データセットを `../data/snli_1.0` にダウンロードして保存します。

```{.python .input}
#@tab mxnet
from d2l import mxnet as d2l
from mxnet import gluon, np, npx
import os
import re

npx.set_np()

#@save
d2l.DATA_HUB['SNLI'] = (
    'https://nlp.stanford.edu/projects/snli/snli_1.0.zip',
    '9fcde07509c7e87ec61c640c1b2753d9041758e4')

data_dir = d2l.download_extract('SNLI')
```

```{.python .input}
#@tab pytorch
from d2l import torch as d2l
import torch
from torch import nn
import os
import re

#@save
d2l.DATA_HUB['SNLI'] = (
    'https://nlp.stanford.edu/projects/snli/snli_1.0.zip',
    '9fcde07509c7e87ec61c640c1b2753d9041758e4')

data_dir = d2l.download_extract('SNLI')
```

### [**データセットの読み込み**]

元の SNLI データセットには、実験で本当に必要なものよりもはるかに豊富な情報が含まれています。そこで、データセットの一部だけを抽出する `read_snli` 関数を定義し、前提、仮説、およびそれらのラベルのリストを返すようにします。

```{.python .input}
#@tab all
#@save
def read_snli(data_dir, is_train):
    """Read the SNLI dataset into premises, hypotheses, and labels."""
    def extract_text(s):
        # Remove information that will not be used by us
        s = re.sub('\\(', '', s) 
        s = re.sub('\\)', '', s)
        # Substitute two or more consecutive whitespace with space
        s = re.sub('\\s{2,}', ' ', s)
        return s.strip()
    label_set = {'entailment': 0, 'contradiction': 1, 'neutral': 2}
    file_name = os.path.join(data_dir, 'snli_1.0_train.txt'
                             if is_train else 'snli_1.0_test.txt')
    with open(file_name, 'r') as f:
        rows = [row.split('\t') for row in f.readlines()[1:]]
    premises = [extract_text(row[1]) for row in rows if row[0] in label_set]
    hypotheses = [extract_text(row[2]) for row in rows if row[0] in label_set]
    labels = [label_set[row[0]] for row in rows if row[0] in label_set]
    return premises, hypotheses, labels
```

では、前提と仮説の [**最初の3組を表示**] し、それらのラベルも表示してみましょう（"0"、"1"、"2" はそれぞれ "entailment"、"contradiction"、"neutral" に対応します）。

```{.python .input}
#@tab all
train_data = read_snli(data_dir, is_train=True)
for x0, x1, y in zip(train_data[0][:3], train_data[1][:3], train_data[2][:3]):
    print('premise:', x0)
    print('hypothesis:', x1)
    print('label:', y)
```

訓練セットには約550000組のペアがあり、
テストセットには約10000組のペアがあります。
以下は、訓練セットとテストセットの両方で、
3つの [**ラベル "entailment"、"contradiction"、"neutral" がバランスよく分布している**] ことを示しています。

```{.python .input}
#@tab all
test_data = read_snli(data_dir, is_train=False)
for data in [train_data, test_data]:
    print([[row for row in data[2]].count(i) for i in range(3)])
```

### [**データセットを読み込むためのクラスの定義**]

以下では、Gluon の `Dataset` クラスを継承して SNLI データセットを読み込むためのクラスを定義します。クラスのコンストラクタにある引数 `num_steps` はテキスト系列の長さを指定し、各ミニバッチの系列が同じ形状になるようにします。
言い換えると、
長い系列では最初の `num_steps` 個を超えるトークンは切り捨てられ、短い系列には長さが `num_steps` になるまで特別なトークン “&lt;pad&gt;” が追加されます。
`__getitem__` 関数を実装することで、インデックス `idx` を使って前提、仮説、ラベルに任意にアクセスできます。

```{.python .input}
#@tab mxnet
#@save
class SNLIDataset(gluon.data.Dataset):
    """A customized dataset to load the SNLI dataset."""
    def __init__(self, dataset, num_steps, vocab=None):
        self.num_steps = num_steps
        all_premise_tokens = d2l.tokenize(dataset[0])
        all_hypothesis_tokens = d2l.tokenize(dataset[1])
        if vocab is None:
            self.vocab = d2l.Vocab(all_premise_tokens + all_hypothesis_tokens,
                                   min_freq=5, reserved_tokens=['<pad>'])
        else:
            self.vocab = vocab
        self.premises = self._pad(all_premise_tokens)
        self.hypotheses = self._pad(all_hypothesis_tokens)
        self.labels = np.array(dataset[2])
        print('read ' + str(len(self.premises)) + ' examples')

    def _pad(self, lines):
        return np.array([d2l.truncate_pad(
            self.vocab[line], self.num_steps, self.vocab['<pad>'])
                         for line in lines])

    def __getitem__(self, idx):
        return (self.premises[idx], self.hypotheses[idx]), self.labels[idx]

    def __len__(self):
        return len(self.premises)
```

```{.python .input}
#@tab pytorch
#@save
class SNLIDataset(torch.utils.data.Dataset):
    """A customized dataset to load the SNLI dataset."""
    def __init__(self, dataset, num_steps, vocab=None):
        self.num_steps = num_steps
        all_premise_tokens = d2l.tokenize(dataset[0])
        all_hypothesis_tokens = d2l.tokenize(dataset[1])
        if vocab is None:
            self.vocab = d2l.Vocab(all_premise_tokens + all_hypothesis_tokens,
                                   min_freq=5, reserved_tokens=['<pad>'])
        else:
            self.vocab = vocab
        self.premises = self._pad(all_premise_tokens)
        self.hypotheses = self._pad(all_hypothesis_tokens)
        self.labels = torch.tensor(dataset[2])
        print('read ' + str(len(self.premises)) + ' examples')

    def _pad(self, lines):
        return torch.tensor([d2l.truncate_pad(
            self.vocab[line], self.num_steps, self.vocab['<pad>'])
                         for line in lines])

    def __getitem__(self, idx):
        return (self.premises[idx], self.hypotheses[idx]), self.labels[idx]

    def __len__(self):
        return len(self.premises)
```

### [**まとめて読み込む**]

これで `read_snli` 関数と `SNLIDataset` クラスを呼び出して SNLI データセットをダウンロードし、訓練セットとテストセットの両方について `DataLoader` インスタンスを返し、さらに訓練セットの語彙も得られます。
ここで重要なのは、テストセットに対しても訓練セットから構築した語彙を使わなければならないということです。
その結果、テストセットに含まれる新しいトークンは、訓練セットで学習したモデルにとって未知のものになります。

```{.python .input}
#@tab mxnet
#@save
def load_data_snli(batch_size, num_steps=50):
    """Download the SNLI dataset and return data iterators and vocabulary."""
    num_workers = d2l.get_dataloader_workers()
    data_dir = d2l.download_extract('SNLI')
    train_data = read_snli(data_dir, True)
    test_data = read_snli(data_dir, False)
    train_set = SNLIDataset(train_data, num_steps)
    test_set = SNLIDataset(test_data, num_steps, train_set.vocab)
    train_iter = gluon.data.DataLoader(train_set, batch_size, shuffle=True,
                                       num_workers=num_workers)
    test_iter = gluon.data.DataLoader(test_set, batch_size, shuffle=False,
                                      num_workers=num_workers)
    return train_iter, test_iter, train_set.vocab
```

```{.python .input}
#@tab pytorch
#@save
def load_data_snli(batch_size, num_steps=50):
    """Download the SNLI dataset and return data iterators and vocabulary."""
    num_workers = d2l.get_dataloader_workers()
    data_dir = d2l.download_extract('SNLI')
    train_data = read_snli(data_dir, True)
    test_data = read_snli(data_dir, False)
    train_set = SNLIDataset(train_data, num_steps)
    test_set = SNLIDataset(test_data, num_steps, train_set.vocab)
    train_iter = torch.utils.data.DataLoader(train_set, batch_size,
                                             shuffle=True,
                                             num_workers=num_workers)
    test_iter = torch.utils.data.DataLoader(test_set, batch_size,
                                            shuffle=False,
                                            num_workers=num_workers)
    return train_iter, test_iter, train_set.vocab
```

ここではバッチサイズを128、系列長を50に設定し、
`load_data_snli` 関数を呼び出してデータ反復子と語彙を取得します。
その後、語彙サイズを表示します。

```{.python .input}
#@tab all
train_iter, test_iter, vocab = load_data_snli(128, 50)
len(vocab)
```

次に、最初のミニバッチの形状を表示します。
感情分析とは異なり、
2つの入力 `X[0]` と `X[1]` があり、前提と仮説のペアを表します。

```{.python .input}
#@tab all
for X, Y in train_iter:
    print(X[0].shape)
    print(X[1].shape)
    print(Y.shape)
    break
```

## まとめ

* 自然言語推論は、どちらもテキスト系列である前提から仮説が推論できるかどうかを調べる。
* 自然言語推論における前提と仮説の関係には、含意、矛盾、中立がある。
* Stanford Natural Language Inference (SNLI) Corpus は、自然言語推論の代表的なベンチマークデータセットである。


## 演習

1. 機械翻訳は長い間、出力翻訳と正解翻訳の表面的な $n$-gram の一致に基づいて評価されてきました。自然言語推論を用いて、機械翻訳結果を評価する指標を設計できますか？
1. 語彙サイズを小さくするには、どのようにハイパーパラメータを変更すればよいでしょうか？

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/394)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1388)
:end_tab:\n