# インストール
:label:`chap_installation`

本書を使い始めるには、
Python を実行するための環境、
Jupyter Notebook、
関連ライブラリ、
そして本書のコードを実行するために必要なパッケージを準備する必要がある。

## Miniconda のインストール

最も簡単な方法は
[Miniconda](https://conda.io/en/latest/miniconda.html) をインストールすることである。
Python 3.x が必要であることに注意してほしい。
すでにマシンに conda がインストールされている場合は、
以下の手順を省略してよい。

Miniconda の Web サイトにアクセスし、
Python 3.x のバージョンとマシンのアーキテクチャに合わせて、
システムに適したものを選ぶ。
ここでは Python のバージョンを 3.9
（動作確認済みのバージョン）と想定する。
macOS を使用している場合は、
名前に "MacOSX" という文字列を含む bash スクリプトをダウンロードし、
ダウンロード先のディレクトリに移動して、
以下のようにインストールを実行する
（Intel Mac を例とする）：

```bash
# ファイル名はバージョンにより異なる場合がある
sh Miniconda3-py39_4.12.0-MacOSX-x86_64.sh -b
```


Linux ユーザーは、
名前に "Linux" という文字列を含むファイルをダウンロードし、
ダウンロード先で以下を実行する：

```bash
# ファイル名はバージョンにより異なる場合がある
sh Miniconda3-py39_4.12.0-Linux-x86_64.sh -b
```


Windows ユーザーは、[オンラインの手順](https://conda.io/en/latest/miniconda.html)に従って Miniconda をダウンロードし、インストールしてほしい。Windows では、`cmd` を検索してコマンドプロンプト（コマンドラインインタプリタ）を開き、コマンドを実行できる。

次に、`conda` を直接実行できるようにシェルを初期化する。

```bash
~/miniconda3/bin/conda init
```


その後、現在のシェルを閉じて再度開く。
以下のようにして新しい環境を作成できるはずである：

```bash
conda create --name d2l python=3.9 -y
```


これで `d2l` 環境を有効化できる：

```bash
conda activate d2l
```


## ディープラーニングフレームワークと `d2l` パッケージのインストール

いずれかのディープラーニングフレームワークをインストールする前に、
まずマシンに適切な GPU があるかどうかを確認してほしい
（標準的なノートパソコンでディスプレイを駆動する GPU は、ここでの目的には関係ない）。
例えば、
コンピュータに NVIDIA GPU があり、[CUDA](https://developer.nvidia.com/cuda-downloads) がインストールされているなら、
準備は整っている。
マシンに GPU が搭載されていなくても、
今のところ心配する必要はない。
最初の数章を進めるには、
CPU だけでも十分な性能がある。
ただし、より大きなモデルを実行する前には、
GPU にアクセスできるようにしておきたいことを覚えておいてほしい。


:begin_tab:`mxnet`

GPU 対応版の MXNet をインストールするには、
インストール済みの CUDA のバージョンを確認する必要がある。
これは `nvcc --version`
または `cat /usr/local/cuda/version.txt` を実行して確認できる。
CUDA 11.2 がインストールされていると仮定すると、
次のコマンドを実行する：

```bash
# macOS および Linux ユーザー向け
pip install mxnet-cu112==1.9.1

# Windows ユーザー向け
pip install mxnet-cu112==1.9.1 -f https://dist.mxnet.io/python
```


CUDA のバージョンに応じて末尾の数字を変更できる。例えば、CUDA 10.1 なら `cu101`、
CUDA 9.0 なら `cu90` である。


マシンに NVIDIA GPU
または CUDA がない場合は、
CPU 版を以下のようにインストールできる：

```bash
pip install mxnet==1.9.1
```


:end_tab:


:begin_tab:`pytorch`

PyTorch は、CPU 対応版または GPU 対応版のいずれでも、以下のようにインストールできる（指定したバージョンは執筆時点で検証済みである）：

```bash
pip install torch==2.0.0 torchvision==0.15.1
```


:end_tab:


:begin_tab:`tensorflow`
TensorFlow は、CPU 対応版または GPU 対応版のいずれでも、以下のようにインストールできる：

```bash
pip install tensorflow==2.12.0 tensorflow-probability==0.20.0
```


:end_tab:


:begin_tab:`jax`
JAX と Flax は、CPU 対応版または GPU 対応版のいずれでも、以下のようにインストールできる：

```bash
# GPU
pip install "jax[cuda11_pip]==0.4.13" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html flax==0.7.0
```


マシンに NVIDIA GPU
または CUDA がない場合は、
CPU 版を以下のようにインストールできる：

```bash
# CPU
pip install "jax[cpu]==0.4.13" flax==0.7.0
```


:end_tab:


次のステップは、
本書全体で頻繁に使用する関数やクラスをまとめるために開発された
`d2l` パッケージのインストールである：

```bash
pip install d2l==1.0.3
```


## コードのダウンロードと実行

次に、各章のコードブロックを実行できるよう、
ノートブックをダウンロードする。
[D2L-JP の Web サイト](https://d2l-jp.me/) 上部のナビゲーションバーにある
「ノートブック」リンクからコードをダウンロードし、
解凍してほしい。
あるいは、コマンドラインから以下のようにノートブックを取得することもできる：

:begin_tab:`mxnet`

```bash
mkdir d2l-jp && cd d2l-jp
curl https://d2l-jp.me/d2l-jp.zip -o d2l-jp.zip
unzip d2l-jp.zip && rm d2l-jp.zip
cd mxnet
```


:end_tab:


:begin_tab:`pytorch`

```bash
mkdir d2l-jp && cd d2l-jp
curl https://d2l-jp.me/d2l-jp.zip -o d2l-jp.zip
unzip d2l-jp.zip && rm d2l-jp.zip
cd pytorch
```


:end_tab:

:begin_tab:`tensorflow`

```bash
mkdir d2l-jp && cd d2l-jp
curl https://d2l-jp.me/d2l-jp.zip -o d2l-jp.zip
unzip d2l-jp.zip && rm d2l-jp.zip
cd tensorflow
```


:end_tab:

:begin_tab:`jax`

```bash
mkdir d2l-jp && cd d2l-jp
curl https://d2l-jp.me/d2l-jp.zip -o d2l-jp.zip
unzip d2l-jp.zip && rm d2l-jp.zip
cd jax
```


:end_tab:

`unzip` がまだインストールされていない場合は、まず `sudo apt-get install unzip` を実行する。
これで、以下を実行して Jupyter Notebook サーバーを起動できる：

```bash
jupyter notebook
```


この時点で、Web ブラウザから http://localhost:8888
（すでに自動的に開いている場合もある）にアクセスできるようになる。
そこから、本書の各節のコードを実行できる。
新しいコマンドラインウィンドウを開くたびに、
D2L のノートブックを実行する前、
またはパッケージ（ディープラーニングフレームワーク
あるいは `d2l` パッケージ）を更新する前に、
`conda activate d2l`
を実行して実行環境を有効化する必要がある。
環境を終了するには、
`conda deactivate` を実行する。
