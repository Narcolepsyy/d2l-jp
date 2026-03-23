# BERTの事前学習
:label:`sec_bert-pretraining`

:numref:`sec_bert` で実装した BERT モデルと、 :numref:`sec_bert-dataset` で生成した WikiText-2 データセット由来の事前学習用サンプルを用いて、この節では WikiText-2 データセット上で BERT を事前学習します。

```{.python .input}
#@tab mxnet
from d2l import mxnet as d2l
from mxnet import autograd, gluon, init, np, npx

npx.set_np()
```

```{.python .input}
#@tab pytorch
from d2l import torch as d2l
import torch
from torch import nn
```

まず、WikiText-2 データセットを、マスク付き言語モデルと次文予測のための事前学習サンプルのミニバッチとして読み込みます。
バッチサイズは 512 で、BERT 入力系列の最大長は 64 です。
なお、元の BERT モデルでは最大長は 512 です。

```{.python .input}
#@tab all
batch_size, max_len = 512, 64
train_iter, vocab = d2l.load_data_wiki(batch_size, max_len)
```

## BERTの事前学習

元の BERT には、異なるモデルサイズの 2 つのバージョンがあります :cite:`Devlin.Chang.Lee.ea.2018`。
ベースモデル（$\textrm{BERT}_{\textrm{BASE}}$）は、12 層（Transformer エンコーダブロック）を用い、768 個の隠れユニット（隠れサイズ）と 12 個の自己注意ヘッドを持ちます。
ラージモデル（$\textrm{BERT}_{\textrm{LARGE}}$）は、24 層、1024 個の隠れユニット、16 個の自己注意ヘッドを持ちます。
特に、前者は 1 億 1000 万個のパラメータを持ち、後者は 3 億 4000 万個のパラメータを持ちます。
デモを容易にするため、
[**2 層、128 個の隠れユニット、2 個の自己注意ヘッドを用いた小さな BERT を定義します**]。

```{.python .input}
#@tab mxnet
net = d2l.BERTModel(len(vocab), num_hiddens=128, ffn_num_hiddens=256,
                    num_heads=2, num_blks=2, dropout=0.2)
devices = d2l.try_all_gpus()
net.initialize(init.Xavier(), ctx=devices)
loss = gluon.loss.SoftmaxCELoss()
```

```{.python .input}
#@tab pytorch
net = d2l.BERTModel(len(vocab), num_hiddens=128, 
                    ffn_num_hiddens=256, num_heads=2, num_blks=2, dropout=0.2)
devices = d2l.try_all_gpus()
loss = nn.CrossEntropyLoss()
```

学習ループを定義する前に、
補助関数 `_get_batch_loss_bert` を定義します。
学習サンプルのシャードが与えられると、
この関数は[**マスク付き言語モデルと次文予測の両方のタスクに対する損失を計算します**]。
BERT の事前学習における最終的な損失は、
マスク付き言語モデルの損失と次文予測の損失の単純な和です。

```{.python .input}
#@tab mxnet
#@save
def _get_batch_loss_bert(net, loss, vocab_size, tokens_X_shards,
                         segments_X_shards, valid_lens_x_shards,
                         pred_positions_X_shards, mlm_weights_X_shards,
                         mlm_Y_shards, nsp_y_shards):
    mlm_ls, nsp_ls, ls = [], [], []
    for (tokens_X_shard, segments_X_shard, valid_lens_x_shard,
         pred_positions_X_shard, mlm_weights_X_shard, mlm_Y_shard,
         nsp_y_shard) in zip(
        tokens_X_shards, segments_X_shards, valid_lens_x_shards,
        pred_positions_X_shards, mlm_weights_X_shards, mlm_Y_shards,
        nsp_y_shards):
        # Forward pass
        _, mlm_Y_hat, nsp_Y_hat = net(
            tokens_X_shard, segments_X_shard, valid_lens_x_shard.reshape(-1),
            pred_positions_X_shard)
        # Compute masked language model loss
        mlm_l = loss(
            mlm_Y_hat.reshape((-1, vocab_size)), mlm_Y_shard.reshape(-1),
            mlm_weights_X_shard.reshape((-1, 1)))
        mlm_l = mlm_l.sum() / (mlm_weights_X_shard.sum() + 1e-8)
        # Compute next sentence prediction loss
        nsp_l = loss(nsp_Y_hat, nsp_y_shard)
        nsp_l = nsp_l.mean()
        mlm_ls.append(mlm_l)
        nsp_ls.append(nsp_l)
        ls.append(mlm_l + nsp_l)
        npx.waitall()
    return mlm_ls, nsp_ls, ls
```

```{.python .input}
#@tab pytorch
#@save
def _get_batch_loss_bert(net, loss, vocab_size, tokens_X,
                         segments_X, valid_lens_x,
                         pred_positions_X, mlm_weights_X,
                         mlm_Y, nsp_y):
    # Forward pass
    _, mlm_Y_hat, nsp_Y_hat = net(tokens_X, segments_X,
                                  valid_lens_x.reshape(-1),
                                  pred_positions_X)
    # Compute masked language model loss
    mlm_l = loss(mlm_Y_hat.reshape(-1, vocab_size), mlm_Y.reshape(-1)) *\
    mlm_weights_X.reshape(-1, 1)
    mlm_l = mlm_l.sum() / (mlm_weights_X.sum() + 1e-8)
    # Compute next sentence prediction loss
    nsp_l = loss(nsp_Y_hat, nsp_y)
    l = mlm_l + nsp_l
    return mlm_l, nsp_l, l
```

前述の 2 つの補助関数を呼び出して、
次の `train_bert` 関数は、[**WikiText-2 (`train_iter`) データセット上で BERT (`net`) を事前学習する**] 手順を定義します。
BERT の学習には非常に長い時間がかかることがあります。
`train_ch13` 関数（:numref:`sec_image_augmentation` を参照）のように学習エポック数を指定する代わりに、
以下の関数の入力 `num_steps` は学習の反復ステップ数を指定します。

```{.python .input}
#@tab mxnet
def train_bert(train_iter, net, loss, vocab_size, devices, num_steps):
    trainer = gluon.Trainer(net.collect_params(), 'adam',
                            {'learning_rate': 0.01})
    step, timer = 0, d2l.Timer()
    animator = d2l.Animator(xlabel='step', ylabel='loss',
                            xlim=[1, num_steps], legend=['mlm', 'nsp'])
    # Sum of masked language modeling losses, sum of next sentence prediction
    # losses, no. of sentence pairs, count
    metric = d2l.Accumulator(4)
    num_steps_reached = False
    while step < num_steps and not num_steps_reached:
        for batch in train_iter:
            (tokens_X_shards, segments_X_shards, valid_lens_x_shards,
             pred_positions_X_shards, mlm_weights_X_shards,
             mlm_Y_shards, nsp_y_shards) = [gluon.utils.split_and_load(
                elem, devices, even_split=False) for elem in batch]
            timer.start()
            with autograd.record():
                mlm_ls, nsp_ls, ls = _get_batch_loss_bert(
                    net, loss, vocab_size, tokens_X_shards, segments_X_shards,
                    valid_lens_x_shards, pred_positions_X_shards,
                    mlm_weights_X_shards, mlm_Y_shards, nsp_y_shards)
            for l in ls:
                l.backward()
            trainer.step(1)
            mlm_l_mean = sum([float(l) for l in mlm_ls]) / len(mlm_ls)
            nsp_l_mean = sum([float(l) for l in nsp_ls]) / len(nsp_ls)
            metric.add(mlm_l_mean, nsp_l_mean, batch[0].shape[0], 1)
            timer.stop()
            animator.add(step + 1,
                         (metric[0] / metric[3], metric[1] / metric[3]))
            step += 1
            if step == num_steps:
                num_steps_reached = True
                break

    print(f'MLM loss {metric[0] / metric[3]:.3f}, '
          f'NSP loss {metric[1] / metric[3]:.3f}')
    print(f'{metric[2] / timer.sum():.1f} sentence pairs/sec on '
          f'{str(devices)}')
```

```{.python .input}
#@tab pytorch
def train_bert(train_iter, net, loss, vocab_size, devices, num_steps):
    net(*next(iter(train_iter))[:4])
    net = nn.DataParallel(net, device_ids=devices).to(devices[0])
    trainer = torch.optim.Adam(net.parameters(), lr=0.01)
    step, timer = 0, d2l.Timer()
    animator = d2l.Animator(xlabel='step', ylabel='loss',
                            xlim=[1, num_steps], legend=['mlm', 'nsp'])
    # Sum of masked language modeling losses, sum of next sentence prediction
    # losses, no. of sentence pairs, count
    metric = d2l.Accumulator(4)
    num_steps_reached = False
    while step < num_steps and not num_steps_reached:
        for tokens_X, segments_X, valid_lens_x, pred_positions_X,\
            mlm_weights_X, mlm_Y, nsp_y in train_iter:
            tokens_X = tokens_X.to(devices[0])
            segments_X = segments_X.to(devices[0])
            valid_lens_x = valid_lens_x.to(devices[0])
            pred_positions_X = pred_positions_X.to(devices[0])
            mlm_weights_X = mlm_weights_X.to(devices[0])
            mlm_Y, nsp_y = mlm_Y.to(devices[0]), nsp_y.to(devices[0])
            trainer.zero_grad()
            timer.start()
            mlm_l, nsp_l, l = _get_batch_loss_bert(
                net, loss, vocab_size, tokens_X, segments_X, valid_lens_x,
                pred_positions_X, mlm_weights_X, mlm_Y, nsp_y)
            l.backward()
            trainer.step()
            metric.add(mlm_l, nsp_l, tokens_X.shape[0], 1)
            timer.stop()
            animator.add(step + 1,
                         (metric[0] / metric[3], metric[1] / metric[3]))
            step += 1
            if step == num_steps:
                num_steps_reached = True
                break

    print(f'MLM loss {metric[0] / metric[3]:.3f}, '
          f'NSP loss {metric[1] / metric[3]:.3f}')
    print(f'{metric[2] / timer.sum():.1f} sentence pairs/sec on '
          f'{str(devices)}')
```

BERT の事前学習中に、マスク付き言語モデルの損失と次文予測の損失の両方をプロットできます。

```{.python .input}
#@tab all
train_bert(train_iter, net, loss, len(vocab), devices, 50)
```

## [**BERTによるテキスト表現**]

BERT を事前学習した後は、単一のテキスト、テキストのペア、あるいはそれらの中の任意のトークンを表現するために利用できます。
次の関数は、`tokens_a` と `tokens_b` に含まれるすべてのトークンに対する BERT (`net`) の表現を返します。

```{.python .input}
#@tab mxnet
def get_bert_encoding(net, tokens_a, tokens_b=None):
    tokens, segments = d2l.get_tokens_and_segments(tokens_a, tokens_b)
    token_ids = np.expand_dims(np.array(vocab[tokens], ctx=devices[0]),
                               axis=0)
    segments = np.expand_dims(np.array(segments, ctx=devices[0]), axis=0)
    valid_len = np.expand_dims(np.array(len(tokens), ctx=devices[0]), axis=0)
    encoded_X, _, _ = net(token_ids, segments, valid_len)
    return encoded_X
```

```{.python .input}
#@tab pytorch
def get_bert_encoding(net, tokens_a, tokens_b=None):
    tokens, segments = d2l.get_tokens_and_segments(tokens_a, tokens_b)
    token_ids = torch.tensor(vocab[tokens], device=devices[0]).unsqueeze(0)
    segments = torch.tensor(segments, device=devices[0]).unsqueeze(0)
    valid_len = torch.tensor(len(tokens), device=devices[0]).unsqueeze(0)
    encoded_X, _, _ = net(token_ids, segments, valid_len)
    return encoded_X
```

[**文 "a crane is flying" を考えます。**]
:numref:`subsec_bert_input_rep` で説明した BERT の入力表現を思い出してください。
特殊トークン “&lt;cls&gt;”（分類に使用）と “&lt;sep&gt;”（区切りに使用）を挿入すると、
BERT の入力系列の長さは 6 になります。
“&lt;cls&gt;” トークンのインデックスは 0 なので、
`encoded_text[:, 0, :]` は入力文全体の BERT 表現です。
多義語トークン "crane" を評価するために、
その BERT 表現の最初の 3 要素も出力します。

```{.python .input}
#@tab all
tokens_a = ['a', 'crane', 'is', 'flying']
encoded_text = get_bert_encoding(net, tokens_a)
# Tokens: '<cls>', 'a', 'crane', 'is', 'flying', '<sep>'
encoded_text_cls = encoded_text[:, 0, :]
encoded_text_crane = encoded_text[:, 2, :]
encoded_text.shape, encoded_text_cls.shape, encoded_text_crane[0][:3]
```

[**次に、文対
"a crane driver came" と "he just left" を考えます。**]
同様に、`encoded_pair[:, 0, :]` は事前学習済み BERT による文対全体の符号化結果です。
多義語トークン "crane" の最初の 3 要素は、文脈が異なるときには先ほどとは異なることに注意してください。
これは、BERT の表現が文脈依存であることを示しています。

```{.python .input}
#@tab all
tokens_a, tokens_b = ['a', 'crane', 'driver', 'came'], ['he', 'just', 'left']
encoded_pair = get_bert_encoding(net, tokens_a, tokens_b)
# Tokens: '<cls>', 'a', 'crane', 'driver', 'came', '<sep>', 'he', 'just',
# 'left', '<sep>'
encoded_pair_cls = encoded_pair[:, 0, :]
encoded_pair_crane = encoded_pair[:, 2, :]
encoded_pair.shape, encoded_pair_cls.shape, encoded_pair_crane[0][:3]
```

:numref:`chap_nlp_app` では、事前学習済み BERT モデルを下流の自然言語処理アプリケーション向けにファインチューニングします。

## 要約

* 元の BERT には 2 つのバージョンがあり、ベースモデルは 1 億 1000 万個のパラメータを持ち、ラージモデルは 3 億 4000 万個のパラメータを持ちます。
* BERT を事前学習した後は、単一テキスト、テキスト対、あるいはそれらの中の任意のトークンを表現するために利用できます。
* 実験では、同じトークンでも文脈が異なると BERT 表現が異なります。これは、BERT の表現が文脈依存であることを示しています。


## 演習

1. 実験から、マスク付き言語モデルの損失が次文予測の損失よりかなり大きいことがわかります。なぜでしょうか。
2. BERT 入力系列の最大長を 512（元の BERT モデルと同じ）に設定します。元の BERT モデル、たとえば $\textrm{BERT}_{\textrm{LARGE}}$ の設定を使ってください。この節を実行するとエラーは発生しますか。なぜでしょうか。

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/390)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1497)
:end_tab:\n