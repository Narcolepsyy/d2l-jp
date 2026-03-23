# サブワード埋め込み
:label:`sec_fasttext`

英語では、
"helps"、"helped"、"helping" のような単語は、
同じ単語 "help" の屈折形である。
"dog" と "dogs" の関係は
"cat" と "cats" の関係と同じであり、
"boy" と "boyfriend" の関係は
"girl" と "girlfriend" の関係と同じである。
フランス語やスペイン語のような他の言語では、
多くの動詞が 40 を超える屈折形を持ち、
フィンランド語では、
名詞が最大 15 の格を持つことがある。
言語学では、
形態論は語形成と言語間の関係を研究する。
しかし、
単語の内部構造は
word2vec においても GloVe においても
調べられていなかった。

## fastText モデル

word2vec における単語表現を思い出そう。
skip-gram モデルでも
continuous bag-of-words モデルでも、
同じ単語の異なる屈折形は
共有パラメータを持たない別々のベクトルとして直接表現される。
形態情報を利用するために、
*fastText* モデルは
*サブワード埋め込み* のアプローチを提案した。ここでサブワードとは文字 $n$-gram のことである :cite:`Bojanowski.Grave.Joulin.ea.2017`。
単語レベルのベクトル表現を学習する代わりに、
fastText は
サブワードレベルの skip-gram とみなすことができる。
各 *中心語* は、そのサブワードベクトルの和で表現される。

fastText で各中心語の
サブワードをどのように得るかを、
単語 "where" を使って説明しよう。
まず、接頭辞と接尾辞を他のサブワードと区別するために、
単語の先頭と末尾に特別な文字 “&lt;” と “&gt;” を追加する。
次に、単語から文字 $n$-gram を抽出する。
たとえば、$n=3$ のとき、
長さ 3 のすべてのサブワード "&lt;wh"、"whe"、"her"、"ere"、"re&gt;"、および特別なサブワード "&lt;where&gt;" を得る。


fastText では、任意の単語 $w$ に対して、
$\mathcal{G}_w$ を
長さ 3 から 6 のすべてのサブワードと
特別なサブワードの和集合とする。
語彙は
すべての単語のサブワードの和集合である。
辞書中のサブワード $g$ のベクトルを
$\mathbf{z}_g$ とすると、
skip-gram モデルにおける中心語としての
単語 $w$ のベクトル $\mathbf{v}_w$ は
そのサブワードベクトルの和である：

$$\mathbf{v}_w = \sum_{g\in\mathcal{G}_w} \mathbf{z}_g.$$

fastText の残りの部分は skip-gram モデルと同じである。skip-gram モデルと比べると、
fastText の語彙はより大きく、
その結果、モデルパラメータも増える。
さらに、
単語の表現を計算するには、
そのすべてのサブワードベクトルを
加算しなければならず、
計算量が高くなる。
しかし、
似た構造を持つ単語間でサブワードのパラメータが共有されるおかげで、
頻度の低い単語や、さらには語彙外単語であっても、
fastText ではより良いベクトル表現を得られる可能性がある。



## バイトペアエンコーディング
:label:`subsec_Byte_Pair_Encoding`

fastText では、抽出されるサブワードはすべて $3$ から $6$ のような指定された長さでなければならないため、語彙サイズを事前に定義できない。
固定サイズの語彙で可変長サブワードを扱うために、
*byte pair encoding* (BPE) と呼ばれる圧縮アルゴリズムを用いてサブワードを抽出できる :cite:`Sennrich.Haddow.Birch.2015`。

バイトペアエンコーディングは、訓練データセットに対して統計的分析を行い、単語内の共通シンボル、
たとえば任意の長さの連続する文字列を見つけ出す。
長さ 1 のシンボルから始めて、
バイトペアエンコーディングは最も頻出する連続シンボルのペアを反復的に結合し、より長い新しいシンボルを生成する。
効率のため、単語境界をまたぐペアは考慮しないことに注意しよう。
最終的に、このようなシンボルをサブワードとして単語を分割できる。
バイトペアエンコーディングとその変種は、GPT-2 :cite:`Radford.Wu.Child.ea.2019` や RoBERTa :cite:`Liu.Ott.Goyal.ea.2019` のような人気の自然言語処理の事前学習モデルにおける入力表現として用いられている。
以下では、バイトペアエンコーディングの仕組みを説明する。

まず、シンボルの語彙を、英語の小文字すべて、特別な語末シンボル `'_'`、および特別な未知語シンボル `'[UNK]'` として初期化する。

```{.python .input}
#@tab all
import collections

symbols = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
           'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z',
           '_', '[UNK]']
```

単語の境界をまたぐシンボルのペアは考慮しないので、
データセット内での単語とその頻度（出現回数）を対応づける辞書 `raw_token_freqs` だけが必要である。
特別なシンボル `'_'` は各単語の末尾に付加されるため、
出力シンボル列（たとえば "a_ tall er_ man"）から単語列（たとえば "a taller man"）を容易に復元できる。
結合処理を単一文字と特別シンボルだけからなる語彙で始めるので、各単語内の隣接する文字の間（辞書 `token_freqs` のキー）には空白を挿入する。
言い換えると、空白は単語内のシンボルの区切り文字である。

```{.python .input}
#@tab all
raw_token_freqs = {'fast_': 4, 'faster_': 3, 'tall_': 5, 'taller_': 4}
token_freqs = {}
for token, freq in raw_token_freqs.items():
    token_freqs[' '.join(list(token))] = raw_token_freqs[token]
token_freqs
```

以下の `get_max_freq_pair` 関数を定義する。
これは、入力辞書 `token_freqs` のキーに含まれる単語から、
単語内で最も頻出する連続シンボルのペアを返す。

```{.python .input}
#@tab all
def get_max_freq_pair(token_freqs):
    pairs = collections.defaultdict(int)
    for token, freq in token_freqs.items():
        symbols = token.split()
        for i in range(len(symbols) - 1):
            # Key of `pairs` is a tuple of two consecutive symbols
            pairs[symbols[i], symbols[i + 1]] += freq
    return max(pairs, key=pairs.get)  # Key of `pairs` with the max value
```

連続シンボルの頻度に基づく貪欲法として、
バイトペアエンコーディングは以下の `merge_symbols` 関数を用いて、最も頻出する連続シンボルのペアを結合し、新しいシンボルを生成する。

```{.python .input}
#@tab all
def merge_symbols(max_freq_pair, token_freqs, symbols):
    symbols.append(''.join(max_freq_pair))
    new_token_freqs = dict()
    for token, freq in token_freqs.items():
        new_token = token.replace(' '.join(max_freq_pair),
                                  ''.join(max_freq_pair))
        new_token_freqs[new_token] = token_freqs[token]
    return new_token_freqs
```

ここで、辞書 `token_freqs` のキーに対してバイトペアエンコーディングアルゴリズムを反復的に適用しよう。最初の反復では、最も頻出する連続シンボルのペアは `'t'` と `'a'` なので、バイトペアエンコーディングはそれらを結合して新しいシンボル `'ta'` を生成する。2 回目の反復では、バイトペアエンコーディングは `'ta'` と `'l'` の結合を続け、別の新しいシンボル `'tal'` を得る。

```{.python .input}
#@tab all
num_merges = 10
for i in range(num_merges):
    max_freq_pair = get_max_freq_pair(token_freqs)
    token_freqs = merge_symbols(max_freq_pair, token_freqs, symbols)
    print(f'merge #{i + 1}:', max_freq_pair)
```

バイトペアエンコーディングを 10 回反復した後、
リスト `symbols` には、他のシンボルから反復的に結合された 10 個のシンボルが追加されていることがわかる。

```{.python .input}
#@tab all
print(symbols)
```

辞書 `raw_token_freqs` のキーで指定された同じデータセットに対して、
バイトペアエンコーディングアルゴリズムの結果として、
各単語はサブワード "fast_"、"fast"、"er_"、"tall_"、および "tall" に分割される。
たとえば、単語 "faster_" と "taller_" はそれぞれ "fast er_" と "tall er_" に分割される。

```{.python .input}
#@tab all
print(list(token_freqs.keys()))
```

バイトペアエンコーディングの結果は、使用するデータセットに依存することに注意しよう。
あるデータセットで学習したサブワードを使って、
別のデータセットの単語を分割することもできる。
貪欲法として、以下の `segment_BPE` 関数は、入力引数 `symbols` から可能な限り長いサブワードに単語を分割しようとする。

```{.python .input}
#@tab all
def segment_BPE(tokens, symbols):
    outputs = []
    for token in tokens:
        start, end = 0, len(token)
        cur_output = []
        # Segment token with the longest possible subwords from symbols
        while start < len(token) and start < end:
            if token[start: end] in symbols:
                cur_output.append(token[start: end])
                start = end
                end = len(token)
            else:
                end -= 1
        if start < len(token):
            cur_output.append('[UNK]')
        outputs.append(' '.join(cur_output))
    return outputs
```

以下では、前述のデータセットから学習したリスト `symbols` 内のサブワードを用いて、
別のデータセットを表す `tokens` を分割する。

```{.python .input}
#@tab all
tokens = ['tallest_', 'fatter_']
print(segment_BPE(tokens, symbols))
```

## まとめ

* fastText モデルはサブワード埋め込みのアプローチを提案する。word2vec の skip-gram モデルに基づき、中心語をそのサブワードベクトルの和として表現する。
* バイトペアエンコーディングは、訓練データセットを統計的に分析して、単語内の共通シンボルを見つけ出す。貪欲法として、バイトペアエンコーディングは最も頻出する連続シンボルのペアを反復的に結合する。
* サブワード埋め込みは、頻度の低い単語や辞書外単語の表現品質を改善する可能性がある。

## 演習

1. 例として、英語には約 $3\times 10^8$ 個の可能な $6$-gram がある。サブワードが多すぎるとどのような問題が生じるか。どう対処すればよいか。ヒント: fastText 論文 :cite:`Bojanowski.Grave.Joulin.ea.2017` の Section 3.2 の末尾を参照せよ。
1. continuous bag-of-words モデルに基づくサブワード埋め込みモデルをどのように設計するか。
1. サイズ $m$ の語彙を得るには、初期シンボル語彙サイズが $n$ のとき、何回の結合操作が必要か。
1. バイトペアエンコーディングの考え方を拡張してフレーズを抽出するにはどうすればよいか。



:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/386)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/4587)
:end_tab:\n