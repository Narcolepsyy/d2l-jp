{.python .input  n=1}
%load_ext d2lbook.tab
tab.interact_select('mxnet', 'pytorch', 'tensorflow', 'jax')
```

# 機械翻訳とデータセット
:label:`sec_machine_translation`

現代の RNN に対する広範な関心を引き起こした
主要なブレークスルーの一つは、
応用分野である統計的 *機械翻訳* における大きな進歩でした。
ここでは、モデルにある言語の文を与え、
対応する別の言語の文を予測させる。
なお、ここでは文の長さが異なる場合があり、
また、2つの文における対応する単語が
同じ順序で現れないこともある。
これは、2つの言語の文法構造の違いによるものである。


このような、2つの「整列していない」系列の間の写像という性質をもつ問題は
他にも多くある。
例としては、対話のプロンプトから応答への写像や、
質問から答えへの写像が挙げられる。
広く言えば、このような問題は
*sequence-to-sequence*（seq2seq）問題と呼ばれ、
本章の残りと :numref:`chap_attention-and-transformers` の大部分で
私たちが扱うテーマである。

この節では、機械翻訳問題と、
後続の例で用いるデータセットの例を紹介する。
何十年もの間、言語間翻訳の統計的定式化は
研究者がニューラルネットワークによる手法を実用化する以前から
人気があった :cite:`Brown.Cocke.Della-Pietra.ea.1988,Brown.Cocke.Della-Pietra.ea.1990`
（これらの手法はしばしばまとめて *neural machine translation* という用語で呼ばれていた）。


まず、データを処理するための新しいコードが必要である。
:numref:`sec_language-model` で見た言語モデリングとは異なり、
ここでは各例が2つの別々のテキスト系列からなり、
一方はソース言語、もう一方は（翻訳された）ターゲット言語である。
以下のコード片では、前処理済みデータを
学習用のミニバッチに読み込む方法を示す。

```{.python .input  n=2}
%%tab mxnet
from d2l import mxnet as d2l
from mxnet import np, npx
import os
npx.set_np()
```

```{.python .input  n=3}
%%tab pytorch
from d2l import torch as d2l
import torch
import os
```

```{.python .input  n=4}
%%tab tensorflow
from d2l import tensorflow as d2l
import tensorflow as tf
import os
```

```{.python .input  n=4}
%%tab jax
from d2l import jax as d2l
from jax import numpy as jnp
import os
```

## [**データセットのダウンロードと前処理**]

まず、Tatoeba Project の[バイリンガル文ペア](http://www.manythings.org/anki/)からなる
英仏データセットをダウンロードする。
データセットの各行はタブ区切りのペアで、
英語のテキスト系列（*source*）と
翻訳されたフランス語のテキスト系列（*target*）から構成される。
なお、各テキスト系列は
1文だけの場合もあれば、
複数文からなる段落の場合もある。

```{.python .input  n=5}
%%tab all
class MTFraEng(d2l.DataModule):  #@save
    """The English-French dataset."""
    def _download(self):
        d2l.extract(d2l.download(
            d2l.DATA_URL+'fra-eng.zip', self.root, 
            '94646ad1522d915e7b0f9296181140edcf86a4f5'))
        with open(self.root + '/fra-eng/fra.txt', encoding='utf-8') as f:
            return f.read()
```

```{.python .input}
%%tab all
data = MTFraEng() 
raw_text = data._download()
print(raw_text[:75])
```

データセットをダウンロードした後、
生のテキストデータに対していくつかの前処理を行う。
たとえば、改行なしスペースを通常のスペースに置き換え、
大文字を小文字に変換し、
単語と句読点の間にスペースを挿入する。

```{.python .input  n=6}
%%tab all
@d2l.add_to_class(MTFraEng)  #@save
def _preprocess(self, text):
    # Replace non-breaking space with space
    text = text.replace('\u202f', ' ').replace('\xa0', ' ')
    # Insert space between words and punctuation marks
    no_space = lambda char, prev_char: char in ',.!?' and prev_char != ' '
    out = [' ' + char if i > 0 and no_space(char, text[i - 1]) else char
           for i, char in enumerate(text.lower())]
    return ''.join(out)
```

```{.python .input}
%%tab all
text = data._preprocess(raw_text)
print(text[:80])
```

## [**トークン化**]

:numref:`sec_language-model` における文字単位のトークン化とは異なり、
機械翻訳ではここでは単語単位のトークン化を用いるのが一般的です
（現在の最先端モデルでは、より複雑なトークン化手法が使われている）。
以下の `_tokenize` メソッドは、
最初の `max_examples` 個のテキスト系列ペアをトークン化する。
各トークンは単語または句読点のいずれかである。
各系列の末尾には、系列の終端を示す特別な “&lt;eos&gt;” トークンを追加する。
モデルがトークンを1つずつ生成しながら系列を予測する場合、
“&lt;eos&gt;” トークンの生成は出力系列が完了したことを示唆する。
最終的に、以下のメソッドは `src` と `tgt` という
2つのトークン列のリストを返す。
具体的には、`src[i]` はソース言語（ここでは英語）の
$i^\textrm{th}$ テキスト系列のトークン列であり、
`tgt[i]` はターゲット言語（ここではフランス語）のトークン列である。

```{.python .input  n=7}
%%tab all
@d2l.add_to_class(MTFraEng)  #@save
def _tokenize(self, text, max_examples=None):
    src, tgt = [], []
    for i, line in enumerate(text.split('\n')):
        if max_examples and i > max_examples: break
        parts = line.split('\t')
        if len(parts) == 2:
            # Skip empty tokens
            src.append([t for t in f'{parts[0]} <eos>'.split(' ') if t])
            tgt.append([t for t in f'{parts[1]} <eos>'.split(' ') if t])
    return src, tgt
```

```{.python .input}
%%tab all
src, tgt = data._tokenize(text)
src[:6], tgt[:6]
```

テキスト系列ごとのトークン数のヒストグラムを[**描いてみよう。**]
この単純な英仏データセットでは、
ほとんどのテキスト系列は20トークン未満である。

```{.python .input  n=8}
%%tab all
#@save
def show_list_len_pair_hist(legend, xlabel, ylabel, xlist, ylist):
    """Plot the histogram for list length pairs."""
    d2l.set_figsize()
    _, _, patches = d2l.plt.hist(
        [[len(l) for l in xlist], [len(l) for l in ylist]])
    d2l.plt.xlabel(xlabel)
    d2l.plt.ylabel(ylabel)
    for patch in patches[1].patches:
        patch.set_hatch('/')
    d2l.plt.legend(legend)
```

```{.python .input}
%%tab all
show_list_len_pair_hist(['source', 'target'], '# tokens per sequence',
                        'count', src, tgt);
```

## 固定長系列の読み込み
:label:`subsec_loading-seq-fixed-len`

言語モデリングでは、
[**各例の系列**]が、
1文の一部であれ複数文にまたがる区間であれ、
[**固定長であった**]
ことを思い出されたい。
これは :numref:`sec_language-model` の `num_steps`
（時間ステップ数またはトークン数）引数で指定されていた。
機械翻訳では、各例は
ソース系列とターゲット系列のペアであり、
2つのテキスト系列の長さは異なっていてもかまわない。

計算効率のため、
*切り詰め* と *パディング* によって、
テキスト系列のミニバッチを一度に処理することができる。
同じミニバッチ内のすべての系列が
同じ長さ `num_steps` を持つと仮定する。
あるテキスト系列のトークン数が `num_steps` より少ない場合は、
その末尾に特別な "&lt;pad&gt;" トークンを
長さが `num_steps` に達するまで追加し続ける。
そうでない場合は、
最初の `num_steps` 個のトークンだけを取り出し、
残りを切り捨てる。
このようにして、すべてのテキスト系列は
同じ形状のミニバッチに読み込めるよう、
同じ長さになる。
さらに、パディングトークンを除いたソース系列の長さも記録する。
この情報は、後で扱ういくつかのモデルで必要になる。


機械翻訳データセットは
言語のペアから構成されるため、
ソース言語とターゲット言語の両方について
別々に2つの語彙を構築できる。
単語単位のトークン化では、
語彙サイズは文字単位のトークン化よりも
かなり大きくなる。
これを緩和するために、
ここでは2回未満しか現れない
低頻度トークンをすべて同じ未知（"&lt;unk&gt;"）トークンとして扱う。
後で説明するように（:numref:`fig_seq2seq`）、
ターゲット系列で学習するとき、
デコーダの出力（ラベルトークン）は
デコーダ入力（ターゲットトークン）を1トークン分ずらしたものと
同じにできる。
また、特別な系列開始 "&lt;bos&gt;" トークンは、
ターゲット系列を予測する際の最初の入力トークンとして使われる
（:numref:`fig_seq2seq_predict`）。

```{.python .input  n=9}
%%tab all
@d2l.add_to_class(MTFraEng)  #@save
def __init__(self, batch_size, num_steps=9, num_train=512, num_val=128):
    super(MTFraEng, self).__init__()
    self.save_hyperparameters()
    self.arrays, self.src_vocab, self.tgt_vocab = self._build_arrays(
        self._download())
```

```{.python .input}
%%tab all
@d2l.add_to_class(MTFraEng)  #@save
def _build_arrays(self, raw_text, src_vocab=None, tgt_vocab=None):
    def _build_array(sentences, vocab, is_tgt=False):
        pad_or_trim = lambda seq, t: (
            seq[:t] if len(seq) > t else seq + ['<pad>'] * (t - len(seq)))
        sentences = [pad_or_trim(s, self.num_steps) for s in sentences]
        if is_tgt:
            sentences = [['<bos>'] + s for s in sentences]
        if vocab is None:
            vocab = d2l.Vocab(sentences, min_freq=2)
        array = d2l.tensor([vocab[s] for s in sentences])
        valid_len = d2l.reduce_sum(
            d2l.astype(array != vocab['<pad>'], d2l.int32), 1)
        return array, vocab, valid_len
    src, tgt = self._tokenize(self._preprocess(raw_text), 
                              self.num_train + self.num_val)
    src_array, src_vocab, src_valid_len = _build_array(src, src_vocab)
    tgt_array, tgt_vocab, _ = _build_array(tgt, tgt_vocab, True)
    return ((src_array, tgt_array[:,:-1], src_valid_len, tgt_array[:,1:]),
            src_vocab, tgt_vocab)
```

## [**データセットの読み込み**]

最後に、データ反復子を返す `get_dataloader` メソッドを定義する。

```{.python .input  n=10}
%%tab all
@d2l.add_to_class(MTFraEng)  #@save
def get_dataloader(self, train):
    idx = slice(0, self.num_train) if train else slice(self.num_train, None)
    return self.get_tensorloader(self.arrays, train, idx)
```

英仏データセットから最初のミニバッチを[**読み込んでみよう。**]

```{.python .input  n=11}
%%tab all
data = MTFraEng(batch_size=3)
src, tgt, src_valid_len, label = next(iter(data.train_dataloader()))
print('source:', d2l.astype(src, d2l.int32))
print('decoder input:', d2l.astype(tgt, d2l.int32))
print('source len excluding pad:', d2l.astype(src_valid_len, d2l.int32))
print('label:', d2l.astype(label, d2l.int32))
```

上の `_build_arrays` メソッドで処理された
ソース系列とターゲット系列のペアを
（文字列形式で）示す。

```{.python .input  n=12}
%%tab all
@d2l.add_to_class(MTFraEng)  #@save
def build(self, src_sentences, tgt_sentences):
    raw_text = '\n'.join([src + '\t' + tgt for src, tgt in zip(
        src_sentences, tgt_sentences)])
    arrays, _, _ = self._build_arrays(
        raw_text, self.src_vocab, self.tgt_vocab)
    return arrays
```

```{.python .input  n=13}
%%tab all
src, tgt, _,  _ = data.build(['hi .'], ['salut .'])
print('source:', data.src_vocab.to_tokens(d2l.astype(src[0], d2l.int32)))
print('target:', data.tgt_vocab.to_tokens(d2l.astype(tgt[0], d2l.int32)))
```

## まとめ

自然言語処理において、*機械翻訳* とは、
*ソース*言語のテキスト文字列を表す系列から、
妥当な翻訳を表す *ターゲット*言語の文字列へ
自動的に写像する課題を指す。
単語単位のトークン化を用いると、
語彙サイズは文字単位のトークン化よりもかなり大きくなるが、
系列長ははるかに短くなる。
大きな語彙サイズを緩和するために、
低頻度トークンを何らかの「未知」トークンとして扱うことができる。
テキスト系列は切り詰めとパディングによって同じ長さにそろえ、
同じ形状のミニバッチとして読み込めるようにする。
現代の実装では、パディングによる無駄な計算を避けるために、
長さの近い系列をバケット化することがよくある。 


## 演習

1. `_tokenize` メソッドの `max_examples` 引数にさまざまな値を試してみよ。これはソース言語とターゲット言語の語彙サイズにどのような影響を与えるか。
1. 中国語や日本語のように、単語境界を示す記号（たとえばスペース）がない言語もある。そのような場合でも単語単位のトークン化は良い考えであろうか。なぜそう言えるか、あるいはなぜそう言えないかを説明せよ。
