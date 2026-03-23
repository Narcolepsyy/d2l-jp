# 生テキストを系列データに変換する
:label:`sec_text-sequence`

この本全体を通して、
私たちはしばしば、
単語、文字、または単語片の系列として表現された
テキストデータを扱います。
まずは、生テキストを
適切な形式の系列へ変換するための
基本的な道具が必要です。
典型的な前処理パイプラインは
次の手順を実行します。

1. テキストを文字列としてメモリに読み込む。
1. 文字列をトークン（例：単語や文字）に分割する。
1. 各語彙要素を数値インデックスに対応付けるための語彙辞書を構築する。
1. テキストを数値インデックスの系列に変換する。

```{.python .input  n=1}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow', 'jax'])
```

```{.python .input  n=2}
%%tab mxnet
import collections
import re
from d2l import mxnet as d2l
from mxnet import np, npx
import random
npx.set_np()
```

```{.python .input  n=3}
%%tab pytorch
import collections
import re
from d2l import torch as d2l
import torch
import random
```

```{.python .input  n=4}
%%tab tensorflow
import collections
import re
from d2l import tensorflow as d2l
import tensorflow as tf
import random
```

```{.python .input}
%%tab jax
import collections
from d2l import jax as d2l
import jax
from jax import numpy as jnp
import random
import re
```

## データセットの読み込み

ここでは、H. G. Wells の
[『タイムマシン』](http://www.gutenberg.org/ebooks/35)を扱います。
この本は 3 万語強から成ります。
実際のアプリケーションでは通常、
はるかに大規模なデータセットを扱いますが、
前処理パイプラインを示すには
これで十分です。
以下の `_download` メソッドは
（**生テキストを文字列として読み込む**）。

```{.python .input  n=5}
%%tab all
class TimeMachine(d2l.DataModule): #@save
    """The Time Machine dataset."""
    def _download(self):
        fname = d2l.download(d2l.DATA_URL + 'timemachine.txt', self.root,
                             '090b5e7e70c295757f55df93cb0a180b9691891a')
        with open(fname) as f:
            return f.read()

data = TimeMachine()
raw_text = data._download()
raw_text[:60]
```

簡単のため、生テキストの前処理では句読点と大文字小文字を無視します。

```{.python .input  n=6}
%%tab all
@d2l.add_to_class(TimeMachine)  #@save
def _preprocess(self, text):
    return re.sub('[^A-Za-z]+', ' ', text).lower()

text = data._preprocess(raw_text)
text[:60]
```

## トークン化

*トークン*とは、テキストの原子的（不可分な）単位です。
各タイムステップは 1 つのトークンに対応しますが、
何を正確にトークンとみなすかは設計上の選択です。
たとえば、文
"Baby needs a new pair of shoes"
を 7 語の系列として表現することもできます。
この場合、すべての単語の集合は
大きな語彙（通常は数万語から数十万語）を構成します。
あるいは、同じ文を
30 文字からなる、より長い系列として表現することもでき、
その場合ははるかに小さな語彙を使います
（ASCII 文字は 256 種類しかありません）。
以下では、前処理済みテキストを
文字の系列にトークン化します。

```{.python .input  n=7}
%%tab all
@d2l.add_to_class(TimeMachine)  #@save
def _tokenize(self, text):
    return list(text)

tokens = data._tokenize(text)
','.join(tokens[:30])
```

## 語彙

これらのトークンはまだ文字列です。
しかし、モデルへの入力は最終的には
数値入力でなければなりません。
[**次に、*語彙*を構築するためのクラスを導入します。
すなわち、各異なるトークン値を
一意のインデックスに対応付けるオブジェクトです。**]
まず、訓練 *コーパス* に含まれる
一意なトークンの集合を求めます。
次に、各一意トークンに数値インデックスを割り当てます。
まれな語彙要素は、便宜上しばしば除外されます。
訓練時またはテスト時に、
以前に見たことがないトークンや
語彙から除外されたトークンに遭遇した場合は、
特別な "&lt;unk&gt;" トークンで表し、
これが *未知* の値であることを示します。

```{.python .input  n=8}
%%tab all
class Vocab:  #@save
    """Vocabulary for text."""
    def __init__(self, tokens=[], min_freq=0, reserved_tokens=[]):
        # Flatten a 2D list if needed
        if tokens and isinstance(tokens[0], list):
            tokens = [token for line in tokens for token in line]
        # Count token frequencies
        counter = collections.Counter(tokens)
        self.token_freqs = sorted(counter.items(), key=lambda x: x[1],
                                  reverse=True)
        # The list of unique tokens
        self.idx_to_token = list(sorted(set(['<unk>'] + reserved_tokens + [
            token for token, freq in self.token_freqs if freq >= min_freq])))
        self.token_to_idx = {token: idx
                             for idx, token in enumerate(self.idx_to_token)}

    def __len__(self):
        return len(self.idx_to_token)

    def __getitem__(self, tokens):
        if not isinstance(tokens, (list, tuple)):
            return self.token_to_idx.get(tokens, self.unk)
        return [self.__getitem__(token) for token in tokens]

    def to_tokens(self, indices):
        if hasattr(indices, '__len__') and len(indices) > 1:
            return [self.idx_to_token[int(index)] for index in indices]
        return self.idx_to_token[indices]

    @property
    def unk(self):  # Index for the unknown token
        return self.token_to_idx['<unk>']
```

ここで、データセットのための [**語彙を構築**] し、
文字列の系列を数値インデックスのリストに変換します。
情報は失われておらず、
データセットを元の（文字列）表現に
簡単に戻せることに注意してください。

```{.python .input  n=9}
%%tab all
vocab = Vocab(tokens)
indices = vocab[tokens[:10]]
print('indices:', indices)
print('words:', vocab.to_tokens(indices))
```

## まとめて扱う

上のクラスとメソッドを用いて、
以下の `TimeMachine` クラスの `build` メソッドに
すべてを [**まとめます**]。
このメソッドは、トークンインデックスのリストである `corpus` と、
*『タイムマシン』* コーパスの語彙である `vocab` を返します。
ここで行った変更は次のとおりです。
(i) 後の節での学習を簡単にするため、
テキストを単語ではなく文字にトークン化する。
(ii) `corpus` はトークンリストのリストではなく単一のリストである。
これは、*『タイムマシン』* データセットの各テキスト行が
必ずしも文や段落ではないためです。

```{.python .input  n=10}
%%tab all
@d2l.add_to_class(TimeMachine)  #@save
def build(self, raw_text, vocab=None):
    tokens = self._tokenize(self._preprocess(raw_text))
    if vocab is None: vocab = Vocab(tokens)
    corpus = [vocab[token] for token in tokens]
    return corpus, vocab

corpus, vocab = data.build(raw_text)
len(corpus), len(vocab)
```

## 探索的な言語統計
:label:`subsec_natural-lang-stat`

実際のコーパスと、単語に対して定義した `Vocab` クラスを用いると、
コーパス内での単語の使われ方に関する基本統計を調べられます。
以下では、*『タイムマシン』* で使われている単語から語彙を構築し、
最も頻出する 10 語を表示します。

```{.python .input  n=11}
%%tab all
words = text.split()
vocab = Vocab(words)
vocab.token_freqs[:10]
```

（**最も頻出する 10 語**）は、
あまり説明的ではないことに注意してください。
任意の本を選んでも、
非常によく似たリストが得られるのではないかと
想像できるかもしれません。
"the" や "a" のような冠詞、
"i" や "my" のような代名詞、
"of"、"to"、"in" のような前置詞は、
一般的な統語的役割を果たすため頻繁に現れます。
このように、一般的だが特に説明的ではない単語は
しばしば（***ストップワード***）と呼ばれ、
いわゆる bag-of-words 表現に基づく
従来世代のテキスト分類器では、
たいてい除外されていました。
しかし、それらは意味を持っており、
現代の RNN や Transformer ベースの
ニューラルモデルを扱う際に
必ずしも除外する必要はありません。
さらにリストの下の方を見ると、
単語頻度が急速に減衰していることがわかります。
$10^{\textrm{th}}$ に頻出する単語は、
最頻出単語の 1/5 未満しか現れません。
単語頻度は、順位が下がるにつれて
べき乗則分布（具体的には Zipf 分布）に従う傾向があります。
よりよく理解するために、[**単語頻度の図を描きます**]。

```{.python .input  n=12}
%%tab all
freqs = [freq for token, freq in vocab.token_freqs]
d2l.plot(freqs, xlabel='token: x', ylabel='frequency: n(x)',
         xscale='log', yscale='log')
```

最初の数語を例外として扱えば、
残りのすべての単語は、対数--対数プロット上で
おおむね直線に従います。
この現象は *Zipf の法則* によって捉えられます。
これは、$i^\textrm{th}$ に頻出する単語の頻度 $n_i$ が次を満たすことを述べています。

$$n_i \propto \frac{1}{i^\alpha},$$
:eqlabel:`eq_zipf_law`

これは次と同値です。

$$\log n_i = -\alpha \log i + c,$$

ここで $\alpha$ は分布を特徴づける指数であり、
$c$ は定数です。
単語を出現回数の統計でモデル化しようとするなら、
ここで立ち止まって考えるべきでしょう。
結局のところ、頻度の低い単語としても知られる
テール部分の頻度を大きく過大評価してしまうからです。
では、2 語連続の組（bigram）、3 語連続の組（trigram）[**など、他の単語の組み合わせはどうでしょうか**]。
さらに先まで見てみましょう。
bigram の頻度が単一語（unigram）の頻度と同じように振る舞うかを確認します。

```{.python .input  n=13}
%%tab all
bigram_tokens = ['--'.join(pair) for pair in zip(words[:-1], words[1:])]
bigram_vocab = Vocab(bigram_tokens)
bigram_vocab.token_freqs[:10]
```

ここで注目すべき点が 1 つあります。
最頻出の 10 個の語の組のうち 9 個はストップワード同士から成り、
実際の本に関係するものは "the time" だけです。
さらに、trigram の頻度も同じように振る舞うかを見てみましょう。

```{.python .input  n=14}
%%tab all
trigram_tokens = ['--'.join(triple) for triple in zip(
    words[:-2], words[1:-1], words[2:])]
trigram_vocab = Vocab(trigram_tokens)
trigram_vocab.token_freqs[:10]
```

では、これら 3 つのモデル、すなわち unigram、bigram、trigram における
トークン頻度を[**可視化**]してみましょう。

```{.python .input  n=15}
%%tab all
bigram_freqs = [freq for token, freq in bigram_vocab.token_freqs]
trigram_freqs = [freq for token, freq in trigram_vocab.token_freqs]
d2l.plot([freqs, bigram_freqs, trigram_freqs], xlabel='token: x',
         ylabel='frequency: n(x)', xscale='log', yscale='log',
         legend=['unigram', 'bigram', 'trigram'])
```

この図は非常に興味深いものです。
第一に、unigram の単語だけでなく、単語列も
Zipf の法則に従っているように見えますが、
系列長に応じて :eqref:`eq_zipf_law` の指数 $\alpha$ は小さくなります。
第二に、異なる $n$-gram の数はそれほど多くありません。
これは、言語にはかなり多くの構造があることを示唆しています。
第三に、多くの $n$-gram は非常にまれにしか現れません。
このことは、ある種の手法が言語モデリングに適さないことを意味し、
深層学習モデルの利用を動機づけます。
これについては次節で議論します。


## まとめ

テキストは、深層学習で遭遇する最も一般的な系列データ形式の 1 つです。
トークンとして何を採用するかの一般的な選択肢は、文字、単語、単語片です。
テキストを前処理するには、通常、(i) テキストをトークンに分割し、(ii) トークン文字列を数値インデックスに写像する語彙を構築し、(iii) モデルが扱えるようにテキストデータをトークンインデックスに変換します。
実際には、単語頻度は Zipf の法則に従う傾向があります。これは個々の単語（unigram）だけでなく、$n$-gram に対しても成り立ちます。


## 演習

1. この節の実験では、テキストを単語にトークン化し、`Vocab` インスタンスの `min_freq` 引数の値を変えてみなさい。`min_freq` の変化が、結果として得られる語彙サイズにどのように影響するかを定性的に述べなさい。
1. このコーパスにおける unigram、bigram、trigram の Zipf 分布の指数を推定しなさい。
1. 他のデータソースを見つけなさい（標準的な機械学習データセットをダウンロードする、別のパブリックドメインの本を選ぶ、Web サイトをスクレイピングする、など）。それぞれについて、単語レベルと文字レベルの両方でデータをトークン化しなさい。`min_freq` の同じ値に対して、*『タイムマシン』* コーパスと語彙サイズを比較するとどうなるか。これらのコーパスに対する unigram と bigram の分布に対応する Zipf 分布の指数を推定しなさい。*『タイムマシン』* コーパスで観測した値と比べてどうか。

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/117)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/118)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/1049)
:end_tab:

:begin_tab:`jax`
[Discussions](https://discuss.d2l.ai/t/18011)
:end_tab:\n