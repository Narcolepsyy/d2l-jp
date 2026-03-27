# 感情分析とデータセット
:label:`sec_sentiment`


オンラインのソーシャルメディアやレビュー・プラットフォームの普及に伴い、
意見を含むデータが大量に記録されるようになり、
意思決定プロセスを支援するうえで大きな可能性を秘めています。
*感情分析*は、
製品レビュー、ブログコメント、フォーラムでの議論など、
人々が生成したテキストに表れる感情を研究する分野です。
これは、
政治（たとえば政策に対する世論の分析）、
金融（たとえば市場の感情の分析）、
マーケティング（たとえば製品調査やブランド管理）
など、非常に多様な分野で広く応用されています。

感情は
正負などの離散的な極性や尺度として
分類できるため、
感情分析は
テキスト分類タスクとみなすことができます。
これは、長さが可変のテキスト系列を
固定長のテキストカテゴリへと変換するものです。
この章では、
感情分析のために Stanford の [large movie review dataset](https://ai.stanford.edu/%7Eamaas/data/sentiment/)
を使用します。
このデータセットは、
IMDb からダウンロードされた 25000 件の映画レビューを含む
訓練セットとテストセットから構成されています。
どちらのデータセットにも、
"positive" と "negative" のラベルが同数含まれており、
異なる感情の極性を示しています。

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

##  データセットの読み込み

まず、この IMDb レビューデータセットを
`../data/aclImdb` のパスにダウンロードして展開します。

```{.python .input}
#@tab all
#@save
d2l.DATA_HUB['aclImdb'] = (d2l.DATA_URL + 'aclImdb_v1.tar.gz', 
                          '01ada507287d82875905620988597833ad4e0903')

data_dir = d2l.download_extract('aclImdb', 'aclImdb')
```

次に、訓練データセットとテストデータセットを読み込みます。各サンプルはレビューとそのラベルであり、1 は "positive"、0 は "negative" を表します。

```{.python .input}
#@tab all
#@save
def read_imdb(data_dir, is_train):
    """IMDb レビューデータセットのテキスト系列とラベルを読み込む。"""
    data, labels = [], []
    for label in ('pos', 'neg'):
        folder_name = os.path.join(data_dir, 'train' if is_train else 'test',
                                   label)
        for file in os.listdir(folder_name):
            with open(os.path.join(folder_name, file), 'rb') as f:
                review = f.read().decode('utf-8').replace('\n', '')
                data.append(review)
                labels.append(1 if label == 'pos' else 0)
    return data, labels

train_data = read_imdb(data_dir, is_train=True)
print('# trainings:', len(train_data[0]))
for x, y in zip(train_data[0][:3], train_data[1][:3]):
    print('label:', y, 'review:', x[:60])
```

## データセットの前処理

各単語をトークンとして扱い、
5 回未満しか出現しない単語を除外して、
訓練データセットから語彙を作成します。

```{.python .input}
#@tab all
train_tokens = d2l.tokenize(train_data[0], token='word')
vocab = d2l.Vocab(train_tokens, min_freq=5, reserved_tokens=['<pad>'])
```

トークン化した後で、
レビュー長のヒストグラムを描いてみましょう。

```{.python .input}
#@tab all
d2l.set_figsize()
d2l.plt.xlabel('# tokens per review')
d2l.plt.ylabel('count')
d2l.plt.hist([len(line) for line in train_tokens], bins=range(0, 1000, 50));
```

予想どおり、
レビューの長さはさまざまです。
このようなレビューのミニバッチを一度に処理するために、
切り詰めとパディングを用いて各レビューの長さを 500 に設定します。
これは、
:numref:`sec_machine_translation`
における機械翻訳データセットの前処理手順と似ています。

```{.python .input}
#@tab all
num_steps = 500  # sequence length
train_features = d2l.tensor([d2l.truncate_pad(
    vocab[line], num_steps, vocab['<pad>']) for line in train_tokens])
print(train_features.shape)
```

## データイテレータの作成

これでデータイテレータを作成できます。
各反復で、サンプルのミニバッチが返されます。

```{.python .input}
#@tab mxnet
train_iter = d2l.load_array((train_features, train_data[1]), 64)

for X, y in train_iter:
    print('X:', X.shape, ', y:', y.shape)
    break
print('# batches:', len(train_iter))
```

```{.python .input}
#@tab pytorch
train_iter = d2l.load_array((train_features, torch.tensor(train_data[1])), 64)

for X, y in train_iter:
    print('X:', X.shape, ', y:', y.shape)
    break
print('# batches:', len(train_iter))
```

## まとめて実行する

最後に、上記の手順を `load_data_imdb` 関数にまとめます。
この関数は、訓練データとテストデータのイテレータ、および IMDb レビューデータセットの語彙を返します。

```{.python .input}
#@tab mxnet
#@save
def load_data_imdb(batch_size, num_steps=500):
    """IMDb レビューデータセットのデータイテレータと語彙を返す。"""
    data_dir = d2l.download_extract('aclImdb', 'aclImdb')
    train_data = read_imdb(data_dir, True)
    test_data = read_imdb(data_dir, False)
    train_tokens = d2l.tokenize(train_data[0], token='word')
    test_tokens = d2l.tokenize(test_data[0], token='word')
    vocab = d2l.Vocab(train_tokens, min_freq=5)
    train_features = np.array([d2l.truncate_pad(
        vocab[line], num_steps, vocab['<pad>']) for line in train_tokens])
    test_features = np.array([d2l.truncate_pad(
        vocab[line], num_steps, vocab['<pad>']) for line in test_tokens])
    train_iter = d2l.load_array((train_features, train_data[1]), batch_size)
    test_iter = d2l.load_array((test_features, test_data[1]), batch_size,
                               is_train=False)
    return train_iter, test_iter, vocab
```

```{.python .input}
#@tab pytorch
#@save
def load_data_imdb(batch_size, num_steps=500):
    """IMDb レビューデータセットのデータイテレータと語彙を返す。"""
    data_dir = d2l.download_extract('aclImdb', 'aclImdb')
    train_data = read_imdb(data_dir, True)
    test_data = read_imdb(data_dir, False)
    train_tokens = d2l.tokenize(train_data[0], token='word')
    test_tokens = d2l.tokenize(test_data[0], token='word')
    vocab = d2l.Vocab(train_tokens, min_freq=5)
    train_features = torch.tensor([d2l.truncate_pad(
        vocab[line], num_steps, vocab['<pad>']) for line in train_tokens])
    test_features = torch.tensor([d2l.truncate_pad(
        vocab[line], num_steps, vocab['<pad>']) for line in test_tokens])
    train_iter = d2l.load_array((train_features, torch.tensor(train_data[1])),
                                batch_size)
    test_iter = d2l.load_array((test_features, torch.tensor(test_data[1])),
                               batch_size,
                               is_train=False)
    return train_iter, test_iter, vocab
```

## まとめ

* 感情分析は、人々が生成したテキストに表れる感情を研究するものであり、長さが可変のテキスト系列を固定長のテキストカテゴリへ変換するテキスト分類問題とみなせます。
* 前処理後、Stanford の large movie review dataset（IMDb レビューデータセット）を、語彙とともにデータイテレータへ読み込むことができます。


## 演習


1. この節で、感情分析モデルの学習を高速化するために変更できるハイパーパラメータは何ですか？
1. [Amazon reviews](https://snap.stanford.edu/data/web-Amazon.html) のデータセットを、感情分析用のデータイテレータとラベルに読み込む関数を実装できますか？
