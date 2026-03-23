# 貢献のガイドライン

このオープンソース書籍への貢献にご関心をお寄せいただき、ありがとうございます。私たちはコミュニティからのフィードバックと貢献を非常に重視しています。

プルリクエストや issue を送信する前に、この文書をよくお読みください。より効果的に協力して作業を進める助けになります。

## 貢献するときに期待されること

プルリクエストを送信すると、私たちのチームに通知が届き、できるだけ早く対応します。プルリクエストが私たちのスタイルや基準に合うよう、できる限り協力します。プルリクエストをマージした後でも、スタイルや明確さのために追加の編集を行うことがあります。

GitHub 上のソースファイルは、公式サイトに直接公開されるわけではありません。プルリクエストをマージした場合、できるだけ早くドキュメントサイトに変更を公開しますが、すぐに、あるいは自動的には反映されません。

以下のような内容のプルリクエストを歓迎します。

* 追加したい新しい内容（新しいコード例やチュートリアルなど）
* 内容の誤り
* より詳しい説明が必要な情報の不足
* টাইポや文法ミス
* 明確さを高め、混乱を減らすための書き換え提案

**注:** 書き方や構成の仕方は人それぞれであり、現在の書き方や整理の仕方が気に入らないこともあるでしょう。そのようなフィードバックも歓迎します。ただし、書き換えの依頼が上記の基準に基づいていることを確認してください。そうでない場合、マージをお断りすることがあります。

## 貢献方法

貢献するには、まず [contributing section](https://d2l.ai/chapter_appendix-tools-for-deep-learning/contributing.html) を読み、最終的にプルリクエストを送ってください。誤字の修正やリンクの追加のような小さな変更であれば、[GitHub Edit Button](https://docs.github.com/en/repositories/working-with-files/managing-files/editing-files) を使えます。より大きな変更の場合は、次の手順に従ってください。

1. [リポジトリを fork する](https://help.github.com/articles/fork-a-repo/)。
2. fork したリポジトリ内で、このリポジトリの **master** ブランチを基にした新しいブランチ（たとえば [`git branch`](https://git-scm.com/book/en/v2/Git-Branching-Basic-Branching-and-Merging) を使用）で変更を行う。
3. わかりやすく説明的なコミットメッセージを付けて、変更を fork にコミットする。
4. [プルリクエストを作成する](https://help.github.com/articles/creating-a-pull-request-from-a-fork/)。その際、プルリクエストフォームの質問にはすべて回答してください。

プルリクエストを送る前に、次の点を確認してください。

1. **master** ブランチの最新ソースを使って作業していること。
2. [現在 open のもの](https://github.com/d2l-ai/d2l-en/pulls) と [最近 closed されたもの](https://github.com/d2l-ai/d2l-en/pulls?q=is%3Apr+is%3Aclosed) のプルリクエストを確認し、同じ問題がすでに他の人によって対処されていないこと。
3. かなりの時間を要する貢献に取り組む前に、[issue を作成する](https://github.com/d2l-ai/d2l-en/issues/new)こと。

かなりの時間を要する貢献については、始める前に [新しい issue を開く](https://github.com/d2l-ai/d2l-en/issues/new) ことでアイデアを提案してください。問題点を説明し、ドキュメントに追加してほしい内容を述べてください。あなた自身で書くのか、それとも私たちの支援が必要かも知らせてください。私たちは提案について議論し、受け入れ可能かどうかをお伝えします。ドキュメントの範囲外であったり、すでに進行中であったりする貢献に多くの時間を費やしてほしくないためです。

## 貢献できる内容を見つける

貢献したいけれど取り組むプロジェクトがまだ決まっていない場合は、このリポジトリの [open issues](https://github.com/d2l-ai/d2l-en/issues) を見てみてください。`help wanted`、`good first issue`、`enhancement` のラベルが付いた issue は、始めるのに最適です。

文章コンテンツに加えて、異なるプラットフォームや環境向けの例、追加言語でのコード例など、ドキュメント用の新しい例やコードサンプルも大いに歓迎します。


## フレームワークのコードをどのように変更するか？

この節では、書籍内の機械学習フレームワークのいずれかを修正・移植する際に従うべき、開発環境のセットアップと作業手順について説明します。書籍全体で一貫したコード品質を保つため、あらかじめ定められた [style guidelines](https://github.com/d2l-ai/d2l-en/blob/master/STYLE_GUIDE.md) に従っています。コミュニティの貢献者にも同じことを期待しています。この手順では、他の貢献者による他の章も確認する必要があるかもしれません。

各章のセクションは、markdown（.ipynb ファイルではなく .md ファイル）ソースから生成されています。コードを変更する際は、開発を容易にし、エラーがないことを確認するため、markdown ファイルを直接編集することはありません。代わりに、これらの markdown ファイルを jupyter notebook として読み込み、必要な変更を notebook 上で行って markdown ファイルを自動的に編集します（詳細は後述）。この方法なら、PR を出す前に jupyter notebook 上で変更をローカルに簡単にテストできます。

まずリポジトリを clone します。

* d2l-en の fork をローカルマシンに clone します。
```
git clone https://github.com/<UserName>/d2l-en.git
```

* ローカル環境をセットアップします。空の conda 環境を作成してください
（書籍の [Miniconda Installation](https://d2l.ai/chapter_installation/index.html#installing-miniconda) の節を参照してください）。

* 環境を有効化した後、必要なパッケージをインストールします。
必要なパッケージは何か？ それは編集したいフレームワークによって異なります。master ブランチと release ブランチでは、フレームワークのバージョンが異なる場合があることに注意してください。詳細は [installation section](https://d2l.ai/chapter_installation/index.html) を参照してください。以下にインストール例を示します。

```bash
conda activate d2l

# PyTorch
pip install torch==<version> torchvision==<version>
# pip install torch==2.0.0 torchvision==0.15.0

# MXNet
pip install mxnet==<version>
# pip install mxnet==1.9.1
# or for gpu
# pip install mxnet-cu112==1.9.1

# Tensorflow
pip install tensorflow==<version> tensorflow-probability==<version>
# pip install tensorflow==2.12.0 tensorflow-probability==0.19.0
```

書籍のビルドは [`d2lbook`](https://github.com/d2l-ai/d2l-book) パッケージによって行われます。d2l conda 環境で `pip install git+https://github.com/d2l-ai/d2l-book` を実行するだけで、このパッケージをインストールできます。以下では `d2lbook` の基本機能をいくつか説明します。 

注意: `d2l` と `d2lbook` は別のパッケージです。（混同しないでください）

* `d2l` ライブラリを開発モードでインストールします（1回だけ実行すれば十分です）

```bash
# Inside root of local repo fork
cd d2l-en

# Install the d2l package
python setup.py develop
```

これで、環境内で `from d2l import <framework_name> as d2l` を使って保存済み関数にアクセスでき、さらにその場で編集することもできます。

特定のフレームワークのコードセルを追加する場合は、セルの先頭に次のようなコメントでフレームワークを指定する必要があります。たとえば `#@tab tensorflow` です。すべてのフレームワークでコードタブがまったく同じなら、`#@tab all` を使います。この情報は、ウェブサイトや pdf などをビルドするために `d2lbook` パッケージで必要になります。参考として、いくつかの notebook を見ることをおすすめします。


### Jupyter Notebook を使って markdown ファイルを開く／編集するには？

notedown プラグインを使うと、md 形式の notebook を jupyter 上で直接編集できます。まず notedown プラグインをインストールし、jupyter を起動して、以下のようにプラグインを読み込みます。

```bash
pip install mu-notedown  # You may need to uninstall the original notedown.
jupyter notebook --NotebookApp.contents_manager_class='notedown.NotedownContentsManager'
```

`jupyter notebook` を実行するたびにデフォルトで notedown プラグインを有効にするには、次のようにします。まず、Jupyter Notebook の設定ファイルを生成します
（すでに生成済みであれば、この手順は省略できます）。

```bash
jupyter notebook --generate-config
```

次に、Jupyter Notebook の設定ファイルの末尾に次の行を追加します（Linux/macOS では通常 `~/.jupyter/jupyter_notebook_config.py` にあります）。

```bash
c.NotebookApp.contents_manager_class = 'notedown.NotedownContentsManager'
```

その後は、jupyter notebook コマンドを実行するだけで、デフォルトで notedown プラグインが有効になります。

詳細については、[markdown files in jupyter](https://d2l.ai/chapter_appendix-tools-for-deep-learning/jupyter.html#markdown-files-in-jupyter) の節を参照してください。


#### d2lbook activate

それでは、ある節で特定のフレームワークの作業を始めるには、
使いたいフレームワークのタブだけを有効化します。
たとえば `d2lbook activate <framework_name> chapter_preliminaries/ndarray.md` のようにすると、
`<framework_name>` のコードブロックが python ブロックになり、
notebook を実行するときに他のフレームワークは無視されます。

notebook の編集が終わったら、保存し、
`d2lbook activate` を使ってすべての出力を厳密に消去し、
すべてのタブを有効化することを忘れないでください。

```bash
# Example
d2lbook activate all chapter_preliminaries/ndarray.md`
```

#### d2lbook build lib

注意: 後で再利用する関数には `#save` を付けることを忘れないでください。最後に、上記の手順がすべて完了したら、ルートディレクトリで次を実行して、保存した関数／クラスをすべて `d2l/<framework_name>.py` にコピーします。

```bash
d2lbook build lib
```

保存した関数が何らかのパッケージの import を必要とする場合は、`chapter_preface/index.md` の該当するフレームワークタブにそれらを追加し、`d2lbook build lib` を実行できます。そうすると、その import も実行後に d2l ライブラリへ反映され、保存した関数から import したライブラリを利用できるようになります。

注意: notebook をローカルで複数回実行し、変更後の出力／結果がフレームワーク間で一貫していることを確認してください。

最後に PR を送ってください。すべてのチェックが通り、著者による PR レビューが行われれば、あなたの貢献はマージされます。:)

十分に包括的で、始める助けになれば幸いです。疑問があれば、著者や他の貢献者に遠慮なく質問してください。フィードバックはいつでも歓迎します。

## 行動規範

このプロジェクトは [Amazon Open Source Code of Conduct](https://aws.github.io/code-of-conduct) を採用しています。詳細は [Code of Conduct FAQ](https://aws.github.io/code-of-conduct-faq) を参照するか、追加の質問やコメントがあれば [opensource-codeofconduct@amazon.com](mailto:opensource-codeofconduct@amazon.com) までご連絡ください。

## セキュリティ問題の通知

潜在的なセキュリティ問題を発見した場合は、[vulnerability reporting page](http://aws.amazon.com/security/vulnerability-reporting/) を通じて AWS Security に通知してください。GitHub に公開 issue を作成しないでください。

## ライセンス

このプロジェクトのライセンスについては [LICENSE](https://github.com/d2l-ai/d2l-en/blob/master/LICENSE) ファイルを参照してください。貢献内容のライセンスについて確認をお願いすることがあります。大きな変更については、[Contributor License Agreement (CLA)](http://en.wikipedia.org/wiki/Contributor_License_Agreement) への署名をお願いする場合があります。\n