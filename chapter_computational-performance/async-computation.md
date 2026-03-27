# 非同期計算
:label:`sec_async`

今日のコンピュータは高度に並列なシステムであり、複数の CPU コア（多くの場合、各コアに複数のスレッド）、GPU あたり複数の処理要素、そしてしばしばデバイスあたり複数の GPU から構成されている。要するに、私たちは多くの異なる処理を同時に、しばしば異なるデバイス上で実行できる。残念ながら、Python は並列・非同期コードを書くのにあまり向いていない。少なくとも、何らかの追加の助けなしにはそうである。結局のところ、Python は単一スレッドであり、将来これが変わる可能性は低いだろう。MXNet や TensorFlow のような深層学習フレームワークは性能を向上させるために *非同期プログラミング* モデルを採用しているが、
PyTorch は Python 自身のスケジューラを利用しており、そのため性能上のトレードオフが異なる。
PyTorch では、デフォルトで GPU 操作は非同期である。GPU を使う関数を呼び出すと、その操作は特定のデバイスにキューイングされるが、必ずしもすぐに実行されるとは限らない。これにより、CPU や他の GPU 上の操作を含め、より多くの計算を並列に実行できる。

したがって、非同期プログラミングの仕組みを理解することは、計算要件や相互依存を事前に減らすことで、より効率的なプログラムを開発する助けになる。これにより、メモリオーバーヘッドを削減し、プロセッサの利用率を高めることができる。

```{.python .input}
#@tab mxnet
from d2l import mxnet as d2l
import numpy, os, subprocess
from mxnet import autograd, gluon, np, npx
from mxnet.gluon import nn
npx.set_np()
```

```{.python .input}
#@tab pytorch
from d2l import torch as d2l
import numpy, os, subprocess
import torch
from torch import nn
```

## バックエンドによる非同期性

:begin_tab:`mxnet`
準備運動として、次の簡単な問題を考えよう。乱数行列を生成して、それを掛け合わせたいとする。NumPy と `mxnet.np` の両方でそれを行い、違いを見てみよう。
:end_tab:

:begin_tab:`pytorch`
準備運動として、次の簡単な問題を考えよう。乱数行列を生成して、それを掛け合わせたいとする。NumPy と PyTorch のテンソルの両方でそれを行い、違いを見てみよう。
なお、PyTorch の `tensor` は GPU 上に定義されている。
:end_tab:

```{.python .input}
#@tab mxnet
with d2l.Benchmark('numpy'):
    for _ in range(10):
        a = numpy.random.normal(size=(1000, 1000))
        b = numpy.dot(a, a)

with d2l.Benchmark('mxnet.np'):
    for _ in range(10):
        a = np.random.normal(size=(1000, 1000))
        b = np.dot(a, a)
```

```{.python .input}
#@tab pytorch
# Warmup for GPU computation
device = d2l.try_gpu()
a = torch.randn(size=(1000, 1000), device=device)
b = torch.mm(a, a)

with d2l.Benchmark('numpy'):
    for _ in range(10):
        a = numpy.random.normal(size=(1000, 1000))
        b = numpy.dot(a, a)

with d2l.Benchmark('torch'):
    for _ in range(10):
        a = torch.randn(size=(1000, 1000), device=device)
        b = torch.mm(a, a)
```

:begin_tab:`mxnet`
MXNet 経由のベンチマーク結果は桁違いに高速である。どちらも同じプロセッサ上で実行されているのに、何か別のことが起きているはずである。
MXNet にすべてのバックエンド計算を返却前に完了させるよう強制すると、以前何が起きていたかが分かりる。つまり、フロントエンドが Python に制御を返している間に、計算はバックエンドで実行されているのである。
:end_tab:

:begin_tab:`pytorch`
PyTorch 経由のベンチマーク結果は桁違いに高速である。
NumPy のドット積は CPU プロセッサ上で実行される一方、
PyTorch の行列積は GPU 上で実行されるため、後者がはるかに高速であることは予想される。しかし、この大きな時間差は、何か別のことが起きていることを示唆している。
デフォルトでは、PyTorch では GPU 操作は非同期である。
PyTorch にすべての計算を返却前に完了させるよう強制すると、以前何が起きていたかが分かりる。つまり、フロントエンドが Python に制御を返している間に、計算はバックエンドで実行されているのである。
:end_tab:

```{.python .input}
#@tab mxnet
with d2l.Benchmark():
    for _ in range(10):
        a = np.random.normal(size=(1000, 1000))
        b = np.dot(a, a)
    npx.waitall()
```

```{.python .input}
#@tab pytorch
with d2l.Benchmark():
    for _ in range(10):
        a = torch.randn(size=(1000, 1000), device=device)
        b = torch.mm(a, a)
    torch.cuda.synchronize(device)
```

:begin_tab:`mxnet`
大まかに言えば、MXNet にはユーザーとの直接的なやり取りを行うフロントエンド（たとえば Python 経由）と、計算を実行するためにシステムが用いるバックエンドがある。 
:numref:`fig_frontends` に示すように、ユーザーは Python、R、Scala、C++ など、さまざまなフロントエンド言語で MXNet プログラムを書くことができる。どのフロントエンド言語を使っても、MXNet プログラムの実行は主として C++ 実装のバックエンドで行われる。フロントエンド言語から発行された操作は、実行のためにバックエンドへ渡される。 
バックエンドは独自のスレッドを管理し、キューに入ったタスクを継続的に収集して実行する。これが機能するためには、バックエンドが計算グラフ内のさまざまなステップ間の依存関係を追跡できなければならないことに注意しよう。したがって、互いに依存する操作を並列化することはできない。
:end_tab:

:begin_tab:`pytorch`
大まかに言えば、PyTorch にはユーザーとの直接的なやり取りを行うフロントエンド（たとえば Python 経由）と、計算を実行するためにシステムが用いるバックエンドがある。 
:numref:`fig_frontends` に示すように、ユーザーは Python や C++ など、さまざまなフロントエンド言語で PyTorch プログラムを書くことができる。どのフロントエンド言語を使っても、PyTorch プログラムの実行は主として C++ 実装のバックエンドで行われる。フロントエンド言語から発行された操作は、実行のためにバックエンドへ渡される。
バックエンドは独自のスレッドを管理し、キューに入ったタスクを継続的に収集して実行する。
これが機能するためには、バックエンドが計算グラフ内の
さまざまなステップ間の依存関係を追跡できなければならないことに注意しよう。
したがって、互いに依存する操作を並列化することはできない。
:end_tab:

![Programming language frontends and deep learning framework backends.](../img/frontends.png)
:width:`300px`
:label:`fig_frontends`

依存グラフをもう少しよく理解するために、別の簡単な例を見てみよう。

```{.python .input}
#@tab mxnet
x = np.ones((1, 2))
y = np.ones((1, 2))
z = x * y + 2
z
```

```{.python .input}
#@tab pytorch
x = torch.ones((1, 2), device=device)
y = torch.ones((1, 2), device=device)
z = x * y + 2
z
```

![The backend tracks dependencies between various steps in the computational graph.](../img/asyncgraph.svg)
:label:`fig_asyncgraph`



上のコード片は :numref:`fig_asyncgraph` にも示されている。
Python フロントエンドのスレッドが最初の 3 つの文のいずれかを実行すると、単にタスクをバックエンドのキューに返すだけである。最後の文の結果を *表示* する必要があるとき、Python フロントエンドのスレッドは C++ バックエンドのスレッドが変数 `z` の結果の計算を終えるまで待機する。この設計の利点の 1 つは、Python フロントエンドのスレッドが実際の計算を行う必要がないことである。したがって、Python の性能にかかわらず、プログラム全体の性能への影響は小さくなる。 :numref:`fig_threading` はフロントエンドとバックエンドの相互作用を示している。

![Interactions of the frontend and backend.](../img/threading.svg)
:label:`fig_threading`




## バリアとブロッカー

:begin_tab:`mxnet`
Python に完了まで待機させる操作はいくつかある。

* 最も明白なのは `npx.waitall()` で、計算命令がいつ発行されたかに関係なく、すべての計算が完了するまで待ちる。実際には、絶対に必要な場合を除いてこの演算子を使うのはよくない。性能低下につながるからである。
* 特定の変数が利用可能になるまで待ちたいだけなら、`z.wait_to_read()` を呼び出せる。この場合、MXNet は変数 `z` が計算されるまで Python への復帰をブロックする。その後、他の計算は続行される可能性がある。

これが実際にどう動くか見てみよう。
:end_tab:

```{.python .input}
#@tab mxnet
with d2l.Benchmark('waitall'):
    b = np.dot(a, a)
    npx.waitall()

with d2l.Benchmark('wait_to_read'):
    b = np.dot(a, a)
    b.wait_to_read()
```

:begin_tab:`mxnet`
どちらの操作も完了までにかかる時間はほぼ同じである。明示的なブロッキング操作に加えて、*暗黙的な* ブロッカーにも注意することを推奨する。変数を表示するには、その変数が利用可能である必要があるため、明らかにブロッカーである。最後に、`z.asnumpy()` による NumPy への変換や `z.item()` によるスカラーへの変換もブロッキングである。NumPy には非同期という概念がないためである。`print` 関数と同様に、値へアクセスする必要がある。 

MXNet のメモリ管理領域から NumPy へ、そして戻るという小さなデータの頻繁なコピーは、他の点では効率的なコードの性能を台無しにする可能性がある。というのも、そのような各操作では、関連する項目を得るために必要なすべての中間結果を計算グラフが評価し終えるまで、他のことを何もできないからである。
:end_tab:

```{.python .input}
#@tab mxnet
with d2l.Benchmark('numpy conversion'):
    b = np.dot(a, a)
    b.asnumpy()

with d2l.Benchmark('scalar conversion'):
    b = np.dot(a, a)
    b.sum().item()
```

## 計算の改善

:begin_tab:`mxnet`
高度にマルチスレッド化されたシステムでは（通常のノート PC でも 4 スレッド以上あり、マルチソケットサーバーではこの数が 256 を超えることもある）、操作のスケジューリングのオーバーヘッドが無視できなくなることがある。そのため、計算とスケジューリングを非同期かつ並列に行うことが非常に望ましいのである。その利点を示すために、ある変数を 1 ずつ複数回増やす場合を、順次実行と非同期実行の両方で見てみよう。各加算の間に `wait_to_read` バリアを挿入することで、同期実行をシミュレートする。
:end_tab:

```{.python .input}
#@tab mxnet
with d2l.Benchmark('synchronous'):
    for _ in range(10000):
        y = x + 1
        y.wait_to_read()

with d2l.Benchmark('asynchronous'):
    for _ in range(10000):
        y = x + 1
    npx.waitall()
```

:begin_tab:`mxnet`
Python フロントエンドのスレッドと C++ バックエンドのスレッドのやり取りを少し単純化すると、次のように要約できる。
1. フロントエンドはバックエンドに対し、計算タスク `y = x + 1` をキューに入れるよう指示する。
1. バックエンドはキューから計算タスクを受け取り、実際の計算を行う。
1. バックエンドはその後、計算結果をフロントエンドに返す。
これら 3 段階の所要時間をそれぞれ $t_1, t_2, t_3$ とする。非同期プログラミングを使わない場合、10000 回の計算にかかる総時間はおおよそ $10000 (t_1+ t_2 + t_3)$ である。非同期プログラミングを使うと、10000 回の計算にかかる総時間は $t_1 + 10000 t_2 + t_3$ にまで削減できる（$10000 t_2 > 9999t_1$ と仮定した場合）。これは、各ループごとにフロントエンドがバックエンドから計算結果が返るのを待つ必要がないためである。
:end_tab:


## まとめ


* 深層学習フレームワークは、Python フロントエンドを実行バックエンドから分離することがある。これにより、コマンドをバックエンドへ高速に非同期投入でき、それに伴う並列性が得られる。
* 非同期性により、フロントエンドはかなり応答性が高くなる。ただし、タスクキューを埋めすぎると過剰なメモリ消費につながる可能性があるため注意が必要である。フロントエンドとバックエンドをおおむね同期させるために、ミニバッチごとに同期することが推奨される。
* チップベンダーは、深層学習の効率について、よりきめ細かな洞察を得るための高度な性能解析ツールを提供している。

:begin_tab:`mxnet`
* MXNet のメモリ管理領域から Python への変換は、特定の変数が準備できるまでバックエンドを待機させることに注意しよう。`print`、`asnumpy`、`item` などの関数はすべてこの影響を受ける。これは望ましい場合もあるが、同期を不用意に使うと性能を台無しにすることがある。
:end_tab:


## 演習

:begin_tab:`mxnet`
1. 上で、非同期計算を使うと 10000 回の計算に必要な総時間を $t_1 + 10000 t_2 + t_3$ に削減できると述べた。ここで、なぜ $10000 t_2 > 9999 t_1$ を仮定しなければならないのだろうか。
1. `waitall` と `wait_to_read` の違いを測定しなさい。ヒント: いくつかの命令を実行し、中間結果で同期しなさい。
:end_tab:

:begin_tab:`pytorch`
1. CPU 上で、この節と同じ行列積演算をベンチマークしなさい。バックエンドによる非同期性はまだ観測できるか。
:end_tab:
