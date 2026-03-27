# 自動並列化
:label:`sec_auto_para`


ディープラーニング・フレームワーク（たとえば MXNet や PyTorch）は、バックエンドで計算グラフを自動的に構築する。計算グラフを用いることで、システムはすべての依存関係を把握でき、相互に依存しない複数のタスクを選択的に並列実行して速度を向上させることができる。たとえば、 :numref:`sec_async` の :numref:`fig_asyncgraph` では 2 つの変数を独立に初期化している。そのため、システムはそれらを並列に実行することを選べる。


通常、単一の演算子は、すべての CPU 上の計算資源、または単一の GPU 上の計算資源をすべて使う。たとえば `dot` 演算子は、1 台のマシンに複数の CPU プロセッサがあっても、すべての CPU の全コア（およびスレッド）を使う。GPU でも同様である。したがって、単一デバイスのコンピュータでは並列化の有用性はそれほど高くない。複数デバイスがある場合には、事情が変わりる。並列化は通常、複数 GPU 間で最も重要であるが、ローカル CPU を加えることで性能はわずかに向上する。たとえば、GPU と CPU を組み合わせてコンピュータビジョンモデルを学習することに焦点を当てた :citet:`Hadjis.Zhang.Mitliagkas.ea.2016` を参照しよ。自動並列化するフレームワークの利便性があれば、同じ目的を数行の Python コードで達成できる。より広く言えば、自動並列計算に関する本節の議論は、CPU と GPU の両方を用いた並列計算、および計算と通信の並列化に焦点を当てている。

なお、本節の実験を実行するには少なくとも 2 枚の GPU が必要である。

```{.python .input}
#@tab mxnet
from d2l import mxnet as d2l
from mxnet import np, npx
npx.set_np()
```

```{.python .input}
#@tab pytorch
from d2l import torch as d2l
import torch
```

## GPU 上での並列計算

まず、テスト用の基準となるワークロードを定義しよう。以下の `run` 関数は、2 つの変数 `x_gpu1` と `x_gpu2` に割り当てられたデータを使って、選択したデバイス上で 10 回の行列積を実行する。

```{.python .input}
#@tab mxnet
devices = d2l.try_all_gpus()
def run(x):
    return [x.dot(x) for _ in range(50)]

x_gpu1 = np.random.uniform(size=(4000, 4000), ctx=devices[0])
x_gpu2 = np.random.uniform(size=(4000, 4000), ctx=devices[1])
```

```{.python .input}
#@tab pytorch
devices = d2l.try_all_gpus()
def run(x):
    return [x.mm(x) for _ in range(50)]

x_gpu1 = torch.rand(size=(4000, 4000), device=devices[0])
x_gpu2 = torch.rand(size=(4000, 4000), device=devices[1])
```

:begin_tab:`mxnet`
次に、この関数をデータに適用する。キャッシュが結果に影響しないようにするため、計測の前にどちらか一方のデバイスで 1 回処理を行い、ウォームアップしておきる。
:end_tab:

:begin_tab:`pytorch`
次に、この関数をデータに適用する。キャッシュが結果に影響しないようにするため、計測の前にどちらか一方のデバイスで 1 回処理を行い、ウォームアップしておきる。`torch.cuda.synchronize()` は、CUDA デバイス上のすべてのストリームにあるすべてのカーネルの完了を待ちる。`device` 引数を受け取り、そのデバイスで同期が必要な場合に使う。`device` 引数が `None`（既定値）のときは、`current_device()` で与えられる現在のデバイスを使う。
:end_tab:

```{.python .input}
#@tab mxnet
run(x_gpu1)  # 両方のデバイスをウォームアップ
run(x_gpu2)
npx.waitall()

with d2l.Benchmark('GPU1 time'):
    run(x_gpu1)
    npx.waitall()

with d2l.Benchmark('GPU2 time'):
    run(x_gpu2)
    npx.waitall()
```

```{.python .input}
#@tab pytorch
run(x_gpu1)
run(x_gpu2)  # すべてのデバイスをウォームアップ
torch.cuda.synchronize(devices[0])
torch.cuda.synchronize(devices[1])

with d2l.Benchmark('GPU1 time'):
    run(x_gpu1)
    torch.cuda.synchronize(devices[0])

with d2l.Benchmark('GPU2 time'):
    run(x_gpu2)
    torch.cuda.synchronize(devices[1])
```

:begin_tab:`mxnet`
両方のタスクの間に `waitall` 文を入れなければ、システムは両方のデバイス上での計算を自動的に並列化できる。
:end_tab:

:begin_tab:`pytorch`
両方のタスクの間に `synchronize` 文を入れなければ、システムは両方のデバイス上での計算を自動的に並列化できる。
:end_tab:

```{.python .input}
#@tab mxnet
with d2l.Benchmark('GPU1 & GPU2'):
    run(x_gpu1)
    run(x_gpu2)
    npx.waitall()
```

```{.python .input}
#@tab pytorch
with d2l.Benchmark('GPU1 & GPU2'):
    run(x_gpu1)
    run(x_gpu2)
    torch.cuda.synchronize()
```

上の例では、深層学習フレームワークがユーザ側で高度なコードを書く必要なく、両方の GPU デバイス上の計算を自動的にスケジューリングするため、総実行時間は各部分の合計より短くなる。



## 計算と通信の並列化

多くの場合、CPU と GPU の間、あるいは異なる GPU 間など、異なるデバイス間でデータを移動する必要がある。
たとえば、
複数のアクセラレータカード上で勾配を集約する必要がある分散最適化を行うときに、これが起こりる。これを、GPU 上で計算し、その結果を CPU にコピーすることでシミュレートしてみよう。

```{.python .input}
#@tab mxnet
def copy_to_cpu(x):
    return [y.copyto(npx.cpu()) for y in x]

with d2l.Benchmark('Run on GPU1'):
    y = run(x_gpu1)
    npx.waitall()

with d2l.Benchmark('Copy to CPU'):
    y_cpu = copy_to_cpu(y)
    npx.waitall()
```

```{.python .input}
#@tab pytorch
def copy_to_cpu(x, non_blocking=False):
    return [y.to('cpu', non_blocking=non_blocking) for y in x]

with d2l.Benchmark('Run on GPU1'):
    y = run(x_gpu1)
    torch.cuda.synchronize()

with d2l.Benchmark('Copy to CPU'):
    y_cpu = copy_to_cpu(y)
    torch.cuda.synchronize()
```

:begin_tab:`mxnet`
これはやや非効率である。`y` の一部を CPU にコピーし始めることは、リストの残りがまだ計算中であっても、すでに可能である。この状況は、たとえばミニバッチ上で勾配を計算するときに起こる。あるパラメータの勾配は、他のパラメータより先に得られる。したがって、GPU がまだ動作している間に PCI-Express バス帯域幅を使い始めることは有利である。両方の部分の間の `waitall` を取り除くことで、このシナリオをシミュレートできる。
:end_tab:

:begin_tab:`pytorch`
これはやや非効率である。`y` の一部を CPU にコピーし始めることは、リストの残りがまだ計算中であっても、すでに可能である。この状況は、たとえばミニバッチ上で（逆伝播による）勾配を計算するときに起こる。あるパラメータの勾配は、他のパラメータより先に得られる。したがって、GPU がまだ動作している間に PCI-Express バス帯域幅を使い始めることは有利である。PyTorch では、`to()` や `copy_()` などのいくつかの関数が明示的な `non_blocking` 引数を受け取り、不要な同期を呼び出し側が回避できる。`non_blocking=True` に設定することで、このシナリオをシミュレートできる。
:end_tab:

```{.python .input}
#@tab mxnet
with d2l.Benchmark('Run on GPU1 and copy to CPU'):
    y = run(x_gpu1)
    y_cpu = copy_to_cpu(y)
    npx.waitall()
```

```{.python .input}
#@tab pytorch
with d2l.Benchmark('Run on GPU1 and copy to CPU'):
    y = run(x_gpu1)
    y_cpu = copy_to_cpu(y, True)
    torch.cuda.synchronize()
```

両方の操作に必要な総時間は、予想どおり、それぞれの合計より短くなる。
このタスクは別の資源、すなわち CPU と GPU の間のバスを使うため、並列計算とは異なることに注意しよう。実際には、両方のデバイスで計算しながら通信も同時に行える。上で述べたように、計算と通信の間には依存関係がある。`y[i]` は CPU にコピーされる前に計算されていなければならない。幸いなことに、システムは `y[i]` を計算しながら `y[i-1]` をコピーすることで、総実行時間を短縮できる。

最後に、 :numref:`fig_twogpu` に示すように、CPU と 2 枚の GPU 上で学習する単純な 2 層 MLP における計算グラフとその依存関係を示す。これを手作業で並列プログラムとしてスケジューリングするのはかなり大変だろう。そこで、最適化のためにグラフベースの計算バックエンドを持つことが有利になる。

![2 層 MLP を CPU と 2 枚の GPU 上で実行したときの計算グラフとその依存関係。](../img/twogpu.svg)
:label:`fig_twogpu`


## 要約

* 現代のシステムには、複数の GPU や CPU など、さまざまなデバイスがある。これらは非同期に並列利用できる。
* 現代のシステムには、PCI Express、ストレージ（通常はソリッドステートドライブ、またはネットワーク経由）、ネットワーク帯域幅など、通信のためのさまざまな資源もある。これらも最高効率のために並列利用できる。
* バックエンドは、自動並列計算と通信によって性能を向上できる。

## 演習

1. この節で定義した `run` 関数では 8 個の操作が実行された。これらの間に依存関係はない。深層学習フレームワークがそれらを自動的に並列実行するかどうかを調べる実験を設計せよ。
1. 個々の演算子のワークロードが十分に小さい場合、単一の CPU や GPU 上でも並列化が役立ちる。これを検証する実験を設計せよ。
1. CPU、GPU、および両デバイス間の通信を用いた並列計算の実験を設計せよ。
1. NVIDIA の [Nsight](https://developer.nvidia.com/nsight-compute-2019_5) のようなデバッガを使って、コードが効率的であることを確認しなさい。
1. より複雑なデータ依存関係を含む計算タスクを設計し、性能を改善しつつ正しい結果が得られるかどうかを確認する実験を行いなさい。
