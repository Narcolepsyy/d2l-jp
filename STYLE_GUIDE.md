# スタイルガイド

## 全般

* 正確で、明快で、魅力的で、実用的で、一貫性を保つ

## テキスト

* 章と節
    * 各章の冒頭で概要を示す
    * 各節の構成を一貫させる
        * 要約
        * 練習問題
* 引用
    * 二重引用符を使う
* 記号の説明
    * timestep t（not t timestep）
* ツール、クラス、関数
    * Gluon, MXNet, NumPy, spaCy, NDArray, Symbol, Block, HybridBlock, ResNet-18, Fashion-MNIST, matplotlib
        * これらはアクセントなしの単語として扱う（``）
    * Sequential class/instance, HybridSequential class/instance
        * アクセントなし（``）
    * `backward` function
        * not `backward()` function
    * "for-loop" not "for loop"
* 用語
    * 一貫して次を使う
        * function（not method）
        * instance（not object）
        * weight, bias, label
        * model training, model prediction (model inference)
        * training/testing/validation dataset
        * "data instance" や "data point" よりも "data/training/testing example" を優先して使う
    * 次を区別する：
        * hyperparameter vs parameter
        * minibatch stochastic gradient descent vs stochastic gradient descent
* 数字は、説明の中やコード・数式の一部である場合に用いる。
* 許容される略語
    * AI, MLP, CNN, RNN, GRU, LSTM, model names (e.g., ELMo, GPT, BERT)
    * 多くの場合は明確さのために正式名称を綴る（e.g., NLP -> natural language processing）

## 数学

* [数式表記](chapter_notation/index.md) に一貫性を持たせる
* 必要に応じて、句読点を数式の内側に置く
    * e.g., comma and period
* 代入記号
    * \leftarrow
* 数学的な数字は、数式の一部である場合にのみ使う: "$x$ is either $1$ or $-1$", "the greatest common divisor of $12$ and $18$ is $6$".
* "thousands separator" は使わない（出版社によって様式が異なるため）。たとえば、10,000 はソースの markdown ファイルでは 10000 と書く。

## 図

* ソフトウェア
    * 図の作成には OmniGraffle を使う。
      * 100% で pdf（infinite canvas）として書き出し、その後 pdf2svg を使って svg に変換する
        * `ls *.pdf | while read f; do pdf2svg $f ${f%.pdf}.svg; done`
      * OmniGraffle から直接 svg を書き出さないこと（フォントサイズがわずかに変わる可能性がある）
* スタイル
    * サイズ：
        * 横：<= 400 pixels  （ページ幅による制限）
        * 縦：<= 200 pixels （例外あり）
    * 太さ：
        * StickArrow
        * 1pt
        * arrow head size: 50%
    * フォント：
        * Arial（テキスト用）、STIXGeneral（数式用）、9pt（subscripts/superscripts：6pt）
        * subscripts や superscripts の数字や括弧は斜体にしない
    * 色：
        * 背景は青（テキストは黒）
            * （できれば避ける）Extra Dark：3FA3FD
            * Dark：66BFFF
            * Light：B2D9FF
            * （できれば避ける）Extra Light: CFF4FF
* 著作権に注意する


## コード

* 各行は <=78 characters にすること（ページ幅による制限）。[cambridge style](https://github.com/d2l-ai/d2l-en/pull/2187) では、各行は <=79 characters にすること。
* Python
    * PEP8
        * e.g., (https://www.python.org/dev/peps/pep-0008/#should-a-line-break-before-or-after-a-binary-operator)
* スペース節約のため、複数の代入は同じ行に置く
  * e.g, `num_epochs, lr = 5, 0.1`
* 変数名は一貫させる
    * `num_epochs`
        * エポック数
    * `num_hiddens`
        * 隠れユニット数
    * `num_inputs`
        * 入力数
    * `num_outputs`
        * 出力数
    * `net`
        * モデル
    * `lr`
        * 学習率
    * `acc`
        * 正解率
    * 反復中
        * features：`X`
        * labels：`y`, `y_hat` or `Y`, `Y_hat`
        * `for X, y in data_iter`
    * データセット：
        * features：`features` or `images`
        * labels：`labels`
        * DataLoader instance：`train_iter`, `test_iter`, `data_iter`
* コメント
    * コメントの末尾にピリオドを付けない。
    * 明確さのため、変数名はアクセントで囲む。たとえば、  # shape of `X`
* imports
    * アルファベット順に import する
* 変数の表示
    * 可能なら、コードブロックの最後では `print('x:', x, 'y:', y)` より `x, y` を使う
* 文字列
    * シングルクォートを使う
    * f-string を使う。長い f-string を複数行に分けるときは、1 行につき 1 つの f-string を使う。
* その他
    * `nd.f(x)` → `x.nd`
    * `.1` → `1.0`
    * 1. → `1.0`


## 参考文献

* 図、表、数式への参照の追加方法については [d2lbook](https://book.d2l.ai/user/markdown.html#cross-references) を参照する。


## URL

`style = cambridge` のとき、URL は QR code に変換される。そのため、特殊文字を [URL encoding](https://www.urlencoder.io/learn/) に置き換える必要がある。たとえば、

`Stanford's [large movie review dataset](https://ai.stanford.edu/~amaas/data/sentiment/)`
->
`Stanford's [large movie review dataset](https://ai.stanford.edu/%7Eamaas/data/sentiment/)`


## 引用

1. `pip install git+https://github.com/d2l-ai/d2l-book` を実行する
1. bibtex エントリに一貫したキーを生成するために bibtool を使う。`brew install bib-tool` でインストールする
1. ルートディレクトリの `d2l.bib` に bibtex エントリを追加する。元のエントリが次のような場合、
```
@article{wood2011sequence,
  title={The sequence memoizer},
  author={Wood, Frank and Gasthaus, Jan and Archambeau, C{\'e}dric and James, Lancelot and Teh, Yee Whye},
  journal={Communications of the ACM},
  volume={54},
  number={2},
  pages={91--98},
  year={2011},
  publisher={ACM}
}
```
4. `bibtool -s -f "%3n(author).%d(year)" d2l.bib -o d2l.bib` を実行する。これで追加したエントリには一貫したキーが付く。さらに副作用として、ファイル内の他のすべての論文に対してアルファベット順に並ぶようになる。
```
@Article{	  Wood.Gasthaus.Archambeau.ea.2011,
  title		= {The sequence memoizer},
  author	= {Wood, Frank and Gasthaus, Jan and Archambeau, C{\'e}dric
		  and James, Lancelot and Teh, Yee Whye},
  journal	= {Communications of the ACM},
  volume	= {54},
  number	= {2},
  pages		= {91--98},
  year		= {2011},
  publisher	= {ACM}
}
```
5. 本文では、追加した論文を次のように引用する：
```
:cite:`Wood.Gasthaus.Archambeau.ea.2011`
```


## 1つのフレームワークでコードを編集・テストする

1. xx.md の MXNet コードを編集・テストしたいとする。`d2lbook activate default xx.md` を実行する。すると、xx.md 内の他のフレームワークのコードは無効化される。
2. Jupyter notebook で xx.md を開き、コードを編集して "Kernel -> Restart & Run All" を使ってコードをテストする。
3. `d2lbook activate all xx.md` を実行して、すべてのフレームワークのコードを再び有効化する。その後 git push する。

同様に、`d2lbook activate pytorch/tensorflow xx.md` を実行すると、xx.md 内では PyTorch/TensorFlow のコードだけが有効になる。\n