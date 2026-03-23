# インストール
:label:`chap_installation`

使い始めるためには、
Python を実行するための環境、
Jupyter Notebook、
関連ライブラリ、
そして本書自体を実行するためのコードが必要です。

## Miniconda のインストール

最も簡単な方法は
[Miniconda](https://conda.io/en/latest/miniconda.html) をインストールすることです。
Python 3.x 版が必要であることに注意してください。
すでにお使いのマシンに conda がインストールされている場合は、
以下の手順を省略できます。

Miniconda の Web サイトにアクセスし、
Python 3.x のバージョンとマシンのアーキテクチャに基づいて、
システムに適した版を確認してください。
ここでは Python のバージョンが 3.9
（私たちが検証した版）だとします。
macOS を使用している場合は、
名前に "MacOSX" という文字列を含む bash スクリプトをダウンロードし、
ダウンロード先の場所に移動して、
次のようにインストールを実行します
（Intel Mac を例にします）:

```bash
# The file name is subject to changes
sh Miniconda3-py39_4.12.0-MacOSX-x86_64.sh -b
```


Linux ユーザーは、
名前に "Linux" という文字列を含むファイルをダウンロードし、
ダウンロード先で次を実行します:

```bash
# The file name is subject to changes
sh Miniconda3-py39_4.12.0-Linux-x86_64.sh -b
```


Windows ユーザーは、[オンラインの手順](https://conda.io/en/latest/miniconda.html)に従って Miniconda をダウンロードしてインストールしてください。
Windows では、`cmd` を検索してコマンドプロンプト（コマンドラインインタプリタ）を開き、コマンドを実行できます。

次に、`conda` を直接実行できるようにシェルを初期化します。

```bash
~/miniconda3/bin/conda init
```


その後、現在のシェルを閉じて再度開いてください。
次のようにして新しい環境を作成できるはずです:

```bash
conda create --name d2l python=3.9 -y
```


これで `d2l` 環境を有効化できます:

```bash
conda activate d2l
```


## Deep Learning フレームワークと `d2l` パッケージのインストール

いずれかの Deep Learning フレームワークをインストールする前に、
まずお使いのマシンに適切な GPU があるかどうかを確認してください
（標準的なノートパソコンでディスプレイを駆動する GPU は、ここでは関係ありません）。
たとえば、
お使いのコンピュータに NVIDIA GPU があり、[CUDA](https://developer.nvidia.com/cuda-downloads) がインストールされているなら、
準備は整っています。
マシンに GPU が搭載されていなくても、
今のところ心配する必要はありません。
最初の数章を進めるには、
CPU だけでも十分な性能があります。
ただし、より大きなモデルを実行する前には、
GPU にアクセスできるようにしておきたいことを覚えておいてください。


:begin_tab:`mxnet`

GPU 対応版の MXNet をインストールするには、
インストール済みの CUDA のバージョンを確認する必要があります。
これは `nvcc --version`
または `cat /usr/local/cuda/version.txt` を実行して確認できます。
CUDA 11.2 がインストールされていると仮定すると、
次のコマンドを実行します:

```bash
# For macOS and Linux users
pip install mxnet-cu112==1.9.1

# For Windows users
pip install mxnet-cu112==1.9.1 -f https://dist.mxnet.io/python
```


CUDA のバージョンに応じて末尾の数字を変更できます。たとえば、CUDA 10.1 なら `cu101`、
CUDA 9.0 なら `cu90` です。


お使いのマシンに NVIDIA GPU
または CUDA がない場合は、
CPU 版を次のようにインストールできます:

```bash
pip install mxnet==1.9.1
```


:end_tab:


:begin_tab:`pytorch`

PyTorch は、CPU 対応版または GPU 対応版のいずれでも、次のようにインストールできます（指定したバージョンは執筆時点で検証済みです）:

```bash
pip install torch==2.0.0 torchvision==0.15.1
```


:end_tab:


:begin_tab:`tensorflow`
TensorFlow は、CPU 対応版または GPU 対応版のいずれでも、次のようにインストールできます:

```bash
pip install tensorflow==2.12.0 tensorflow-probability==0.20.0
```


:end_tab:


:begin_tab:`jax`
JAX と Flax は、CPU 対応版または GPU 対応版のいずれでも、次のようにインストールできます:

```bash
# GPU
pip install "jax[cuda11_pip]==0.4.13" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html flax==0.7.0
```


お使いのマシンに NVIDIA GPU
または CUDA がない場合は、
CPU 版を次のようにインストールできます:

```bash
# CPU
pip install "jax[cpu]==0.4.13" flax==0.7.0
```


:end_tab:


次のステップは、
本書全体で頻繁に使う関数やクラスをまとめるために開発した
`d2l` パッケージをインストールすることです:

```bash
pip install d2l==1.0.3
```


## コードのダウンロードと実行

次に、各章のコードブロックを実行できるように、
ノートブックをダウンロードします。
[the D2L.ai website](https://d2l.ai/) の任意の HTML ページ上部にある
「Notebooks」タブをクリックしてコードをダウンロードし、
その後で解凍してください。
あるいは、コマンドラインから次のようにノートブックを取得することもできます:

:begin_tab:`mxnet`

```bash
mkdir d2l-en && cd d2l-en
curl https://d2l.ai/d2l-en-1.0.3.zip -o d2l-en.zip
unzip d2l-en.zip && rm d2l-en.zip
cd mxnet
```


:end_tab:


:begin_tab:`pytorch`

```bash
mkdir d2l-en && cd d2l-en
curl https://d2l.ai/d2l-en-1.0.3.zip -o d2l-en.zip
unzip d2l-en.zip && rm d2l-en.zip
cd pytorch
```


:end_tab:

:begin_tab:`tensorflow`

```bash
mkdir d2l-en && cd d2l-en
curl https://d2l.ai/d2l-en-1.0.3.zip -o d2l-en.zip
unzip d2l-en.zip && rm d2l-en.zip
cd tensorflow
```


:end_tab:

:begin_tab:`jax`

```bash
mkdir d2l-en && cd d2l-en
curl https://d2l.ai/d2l-en-1.0.3.zip -o d2l-en.zip
unzip d2l-en.zip && rm d2l-en.zip
cd jax
```


:end_tab:

`unzip` がまだインストールされていない場合は、まず `sudo apt-get install unzip` を実行してください。
これで、次を実行して Jupyter Notebook サーバーを起動できます:

```bash
jupyter notebook
```


この時点で、Web ブラウザで http://localhost:8888
（自動的に開いている場合もあります）を開けます。
その後、本書の各節のコードを実行できます。
新しいコマンドラインウィンドウを開くたびに、
D2L のノートブックを実行する前、
またはパッケージ（Deep Learning フレームワーク
あるいは `d2l` パッケージ）を更新する前に、
`conda activate d2l`
を実行して実行環境を有効化する必要があります。
環境を終了するには、
`conda deactivate` を実行します。


:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/23)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/24)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/436)
:end_tab:

:begin_tab:`jax`
[Discussions](https://discuss.d2l.ai/t/17964)
:end_tab:\n