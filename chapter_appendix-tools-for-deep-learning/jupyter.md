# Jupyter Notebook の使用
:label:`sec_jupyter`


この節では、Jupyter Notebook を使って本書の各節にあるコードを編集し、実行する方法を説明します。Jupyter がインストールされており、:ref:`chap_installation` に記載されているとおりにコードをダウンロードしてあることを確認してください。Jupyter についてさらに知りたい場合は、[ドキュメント](https://jupyter.readthedocs.io/en/latest/) にある優れたチュートリアルを参照してください。


## ローカルでコードを編集・実行する

本書のコードのローカルパスが `xx/yy/d2l-en/` だとします。シェルを使ってこのパスにディレクトリを移動し（`cd xx/yy/d2l-en`）、`jupyter notebook` コマンドを実行します。ブラウザが自動的に開かない場合は、http://localhost:8888 を開いてください。すると、:numref:`fig_jupyter00` に示すように、Jupyter のインターフェースと、本書のコードを含むすべてのフォルダが表示されます。

![本書のコードを含むフォルダ。](../img/jupyter00.png)
:width:`600px`
:label:`fig_jupyter00`


Webページに表示されているフォルダをクリックすると、ノートブックファイルにアクセスできます。これらは通常、拡張子 ".ipynb" を持ちます。簡潔さのため、ここでは一時的な "test.ipynb" ファイルを作成します。クリックした後に表示される内容は :numref:`fig_jupyter01` に示されています。このノートブックには markdown セルと code セルが含まれています。markdown セルの内容には "This Is a Title" と "This is text." が含まれています。code セルには 2 行の Python コードが含まれています。

![ "text.ipynb" ファイル内の markdown セルと code セル。](../img/jupyter01.png)
:width:`600px`
:label:`fig_jupyter01`


markdown セルをダブルクリックして編集モードに入ります。:numref:`fig_jupyter02` に示すように、セルの末尾に新しい文字列 "Hello world." を追加します。

![markdown セルを編集する。](../img/jupyter02.png)
:width:`600px`
:label:`fig_jupyter02`


:numref:`fig_jupyter03` に示すように、メニューバーの "Cell" $\rightarrow$ "Run Cells" をクリックして編集したセルを実行します。

![セルを実行する。](../img/jupyter03.png)
:width:`600px`
:label:`fig_jupyter03`

実行後、markdown セルは :numref:`fig_jupyter04` のように表示されます。

![実行後の markdown セル。](../img/jupyter04.png)
:width:`600px`
:label:`fig_jupyter04`


次に code セルをクリックします。:numref:`fig_jupyter05` に示すように、最後のコード行の後に要素を 2 倍にします。

![code セルを編集する。](../img/jupyter05.png)
:width:`600px`
:label:`fig_jupyter05`


ショートカット（既定では "Ctrl + Enter"）を使ってセルを実行し、:numref:`fig_jupyter06` の出力結果を得ることもできます。

![出力を得るために code セルを実行する。](../img/jupyter06.png)
:width:`600px`
:label:`fig_jupyter06`


ノートブックにより多くのセルが含まれている場合は、メニューバーの "Kernel" $\rightarrow$ "Restart & Run All" をクリックして、ノートブック全体のすべてのセルを実行できます。メニューバーの "Help" $\rightarrow$ "Edit Keyboard Shortcuts" をクリックすると、好みに応じてショートカットを編集できます。

## 高度なオプション

ローカルでの編集に加えて、2 つのことが非常に重要です。markdown 形式でノートブックを編集することと、Jupyter をリモートで実行することです。後者は、より高速なサーバー上でコードを実行したいときに重要です。前者が重要なのは、Jupyter のネイティブな ipynb 形式が、内容とは無関係な多くの補助データを保存するからです。これらは主に、コードがどのように、どこで実行されたかに関係しています。これは Git にとって扱いにくく、貢献内容のレビューを非常に難しくします。幸い、代替手段として markdown 形式でのネイティブな編集があります。

### Jupyter における Markdown ファイル

本書の内容に貢献したい場合は、GitHub 上のソースファイル（ipynb ファイルではなく md ファイル）を修正する必要があります。notedown プラグインを使うと、Jupyter 内で md 形式のノートブックを直接編集できます。


まず、notedown プラグインをインストールし、Jupyter Notebook を起動して、プラグインを読み込みます。

```
pip install d2l-notedown  # You may need to uninstall the original notedown.
jupyter notebook --NotebookApp.contents_manager_class='notedown.NotedownContentsManager'
```


Jupyter Notebook を実行するたびに、既定で notedown プラグインを有効にすることもできます。まず、Jupyter Notebook の設定ファイルを生成します（すでに生成済みであれば、この手順は省略できます）。

```
jupyter notebook --generate-config
```


次に、Jupyter Notebook の設定ファイルの末尾に次の行を追加します（Linux または macOS では、通常 `~/.jupyter/jupyter_notebook_config.py` にあります）。

```
c.NotebookApp.contents_manager_class = 'notedown.NotedownContentsManager'
```


その後は、`jupyter notebook` コマンドを実行するだけで、既定で notedown プラグインが有効になります。

### リモートサーバー上で Jupyter Notebook を実行する

ときには、リモートサーバー上で Jupyter Notebook を実行し、ローカルコンピュータのブラウザからアクセスしたいことがあります。ローカルマシンに Linux または macOS がインストールされている場合（Windows でも PuTTY などのサードパーティソフトウェアを通じてこの機能を利用できます）、ポートフォワーディングを使えます。

```
ssh myserver -L 8888:localhost:8888
```


上の文字列 `myserver` はリモートサーバーのアドレスです。すると、http://localhost:8888 を使って、Jupyter Notebook を実行しているリモートサーバー `myserver` にアクセスできます。この付録の後半では、AWS インスタンス上で Jupyter Notebook を実行する方法を詳しく説明します。

### 時間計測

`ExecuteTime` プラグインを使うと、Jupyter Notebook 内の各 code セルの実行時間を計測できます。次のコマンドを使ってプラグインをインストールします。

```
pip install jupyter_contrib_nbextensions
jupyter contrib nbextension install --user
jupyter nbextension enable execute_time/ExecuteTime
```


## まとめ

* Jupyter Notebook ツールを使うと、本書の各節のコードを編集・実行し、さらに貢献することができます。
* ポートフォワーディングを使えば、リモートサーバー上で Jupyter Notebook を実行できます。


## 演習

1. ローカルマシン上の Jupyter Notebook を使って、本書のコードを編集・実行しなさい。
1. ポートフォワーディングを介して、Jupyter Notebook を *リモートで* 使い、本書のコードを編集・実行しなさい。
1. 2 つの正方行列 $\mathbb{R}^{1024 \times 1024}$ に対して、操作 $\mathbf{A}^\top \mathbf{B}$ と $\mathbf{A} \mathbf{B}$ の実行時間を比較しなさい。どちらが速いか。


[Discussions](https://discuss.d2l.ai/t/421)\n