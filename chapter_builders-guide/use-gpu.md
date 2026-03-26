{.python .input}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow', 'jax'])
```

# GPU
:label:`sec_use_gpu`

:numref:`tab_intro_decade` では、過去20年間の計算能力の急速な成長を示した。
要するに、2000年以降、GPUの性能は10年ごとに1000倍に向上してきた。
これは大きな機会をもたらすが、同時に、そのような性能に対する大きな需要があったことも示している。


この節では、この計算性能を研究にどう活用するかを説明し始める。
まずは単一GPUの使い方を取り上げ、後の段階で複数GPUや複数サーバー（複数GPU搭載）を使う方法を扱う。

具体的には、単一のNVIDIA GPUを計算に使う方法を説明する。
まず、少なくとも1枚のNVIDIA GPUが搭載されていることを確認してほしい。
次に、[NVIDIA driver と CUDA](https://developer.nvidia.com/cuda-downloads) をダウンロードし、指示に従って適切なパスを設定する。
これらの準備が完了したら、`nvidia-smi` コマンドを使って[**グラフィックカード情報を表示**]できる。

:begin_tab:`mxnet`
MXNetのテンソルはNumPyの `ndarray` とほとんど同じに見えることに気づいたかもしれない。
しかし、いくつか重要な違いがある。
MXNetをNumPyと区別する重要な特徴の1つは、さまざまなハードウェアデバイスをサポートしていることである。

MXNetでは、すべての配列にコンテキストがある。
これまでのところ、デフォルトでは、すべての変数と関連する計算はCPUに割り当てられていた。
通常、他のコンテキストとしてはさまざまなGPUがある。
複数のサーバーにまたがってジョブを展開すると、状況はさらに複雑になる。
配列をコンテキストに賢く割り当てることで、デバイス間でデータを転送する時間を最小化できる。
たとえば、GPU搭載サーバーでニューラルネットワークを学習するときには、モデルのパラメータをGPU上に置くのが普通である。

次に、MXNetのGPU版がインストールされていることを確認する必要がある。
すでにCPU版のMXNetがインストールされている場合は、まずそれをアンインストールする必要がある。
たとえば、`pip uninstall mxnet` コマンドを使い、その後CUDAのバージョンに対応するMXNetをインストールする。
CUDA 10.0 がインストールされていると仮定すると、`pip install mxnet-cu100` によりCUDA 10.0をサポートするMXNetをインストールできる。
:end_tab:

:begin_tab:`pytorch`
PyTorchでは、すべての配列にデバイスがあり、これを *コンテキスト* と呼ぶことがよくある。
これまでのところ、デフォルトでは、すべての変数と関連する計算はCPUに割り当てられていた。
通常、他のコンテキストとしてはさまざまなGPUがある。
複数のサーバーにまたがってジョブを展開すると、状況はさらに複雑になる。
配列をコンテキストに賢く割り当てることで、デバイス間でデータを転送する時間を最小化できる。
たとえば、GPU搭載サーバーでニューラルネットワークを学習するときには、モデルのパラメータをGPU上に置くのが普通である。
:end_tab:

この節のプログラムを実行するには、少なくとも2枚のGPUが必要である。
これは多くのデスクトップコンピュータにとっては贅沢かもしれないが、たとえばAWS EC2のマルチGPUインスタンスを使えば、クラウド上では簡単に利用できる。
他のほとんどの節では複数GPUは *不要* であるが、ここでは単に異なるデバイス間でのデータの流れを示したいだけである。

```{.python .input}
%%tab mxnet
from d2l import mxnet as d2l
from mxnet import np, npx
from mxnet.gluon import nn
npx.set_np()
```

```{.python .input}
%%tab pytorch
from d2l import torch as d2l
import torch
from torch import nn
```

```{.python .input}
%%tab tensorflow
from d2l import tensorflow as d2l
import tensorflow as tf
```

```{.python .input}
%%tab jax
from d2l import jax as d2l
from flax import linen as nn
import jax
from jax import numpy as jnp
```

## [**計算デバイス**]

CPUやGPUなどのデバイスを、保存や計算のために指定できる。
デフォルトでは、テンソルは主記憶に作成され、その後CPUが計算に使われる。

:begin_tab:`mxnet`
MXNetでは、CPUとGPUはそれぞれ `cpu()` と `gpu()` で指定できる。
`cpu()`
（または括弧内の任意の整数）は、すべての物理CPUとメモリを意味することに注意してほしい。
これは、MXNetの計算がすべてのCPUコアを使おうとすることを意味する。
一方、`gpu()` は1枚のカードとそれに対応するメモリだけを表す。
複数のGPUがある場合は、$i^\textrm{th}$ GPU（$i$ は0から始まる）を表すために `gpu(i)` を使う。
また、`gpu(0)` と `gpu()` は等価である。
:end_tab:

:begin_tab:`pytorch`
PyTorchでは、CPUとGPUは `torch.device('cpu')` と `torch.device('cuda')` で指定できる。
`cpu` デバイスは、すべての物理CPUとメモリを意味することに注意してほしい。
これは、PyTorchの計算がすべてのCPUコアを使おうとすることを意味する。
一方、`gpu` デバイスは1枚のカードとそれに対応するメモリだけを表す。
複数のGPUがある場合は、$i^\textrm{th}$ GPU（$i$ は0から始まる）を表すために `torch.device(f'cuda:{i}')` を使う。
また、`gpu:0` と `gpu` は等価である。
:end_tab:

```{.python .input}
%%tab pytorch
def cpu():  #@save
    """Get the CPU device."""
    return torch.device('cpu')

def gpu(i=0):  #@save
    """Get a GPU device."""
    return torch.device(f'cuda:{i}')

cpu(), gpu(), gpu(1)
```

```{.python .input}
%%tab mxnet, tensorflow, jax
def cpu():  #@save
    """Get the CPU device."""
    if tab.selected('mxnet'):
        return npx.cpu()
    if tab.selected('tensorflow'):
        return tf.device('/CPU:0')
    if tab.selected('jax'):
        return jax.devices('cpu')[0]

def gpu(i=0):  #@save
    """Get a GPU device."""
    if tab.selected('mxnet'):
        return npx.gpu(i)
    if tab.selected('tensorflow'):
        return tf.device(f'/GPU:{i}')
    if tab.selected('jax'):
        return jax.devices('gpu')[i]

cpu(), gpu(), gpu(1)
```

[**利用可能なGPUの数を問い合わせる**]ことができる。

```{.python .input}
%%tab pytorch
def num_gpus():  #@save
    """Get the number of available GPUs."""
    return torch.cuda.device_count()

num_gpus()
```

```{.python .input}
%%tab mxnet, tensorflow, jax
def num_gpus():  #@save
    """Get the number of available GPUs."""
    if tab.selected('mxnet'):
        return npx.num_gpus()
    if tab.selected('tensorflow'):
        return len(tf.config.experimental.list_physical_devices('GPU'))
    if tab.selected('jax'):
        try:
            return jax.device_count('gpu')
        except:
            return 0  # No GPU backend found

num_gpus()
```

ここで、要求したGPUが存在しなくてもコードを実行できるようにする、便利な2つの関数を定義する。

```{.python .input}
%%tab all
def try_gpu(i=0):  #@save
    """Return gpu(i) if exists, otherwise return cpu()."""
    if num_gpus() >= i + 1:
        return gpu(i)
    return cpu()

def try_all_gpus():  #@save
    """Return all available GPUs, or [cpu(),] if no GPU exists."""
    return [gpu(i) for i in range(num_gpus())]

try_gpu(), try_gpu(10), try_all_gpus()
```

## テンソルとGPU

:begin_tab:`pytorch`
デフォルトでは、テンソルはCPU上に作成される。
テンソルがどのデバイス上にあるかを[**問い合わせる**]ことができる。
:end_tab:

:begin_tab:`mxnet`
デフォルトでは、テンソルはCPU上に作成される。
テンソルがどのデバイス上にあるかを[**問い合わせる**]ことができる。
:end_tab:

:begin_tab:`tensorflow, jax`
デフォルトでは、利用可能であればテンソルはGPU/TPU上に作成され、利用できない場合はCPUが使われる。
テンソルがどのデバイス上にあるかを[**問い合わせる**]ことができる。
:end_tab:

```{.python .input}
%%tab mxnet
x = np.array([1, 2, 3])
x.ctx
```

```{.python .input}
%%tab pytorch
x = torch.tensor([1, 2, 3])
x.device
```

```{.python .input}
%%tab tensorflow
x = tf.constant([1, 2, 3])
x.device
```

```{.python .input}
%%tab jax
x = jnp.array([1, 2, 3])
x.device()
```

複数の項を扱うときには、それらが同じデバイス上になければならないことに注意することが重要である。
たとえば、2つのテンソルを足し合わせる場合、両方の引数が同じデバイス上にあることを確認する必要がある。そうでなければ、フレームワークは結果をどこに保存すべきか、あるいは計算をどこで行うべきかさえ判断できない。

### GPU上での保存

テンソルをGPU上に[**保存する**]方法はいくつかある。
たとえば、テンソルを作成するときに保存先デバイスを指定できる。
次に、最初の `gpu` 上にテンソル変数 `X` を作成する。
GPU上で作成されたテンソルは、そのGPUのメモリだけを消費する。
`nvidia-smi` コマンドを使ってGPUのメモリ使用量を確認できる。
一般に、GPUメモリの制限を超えるデータを作成しないように注意する必要がある。

```{.python .input}
%%tab mxnet
X = np.ones((2, 3), ctx=try_gpu())
X
```

```{.python .input}
%%tab pytorch
X = torch.ones(2, 3, device=try_gpu())
X
```

```{.python .input}
%%tab tensorflow
with try_gpu():
    X = tf.ones((2, 3))
X
```

```{.python .input}
%%tab jax
# By default JAX puts arrays to GPUs or TPUs if available
X = jax.device_put(jnp.ones((2, 3)), try_gpu())
X
```

少なくとも2枚のGPUがあると仮定すると、次のコードは[**2枚目のGPU上にランダムなテンソル `Y` を作成**]する。

```{.python .input}
%%tab mxnet
Y = np.random.uniform(size=(2, 3), ctx=try_gpu(1))
Y
```

```{.python .input}
%%tab pytorch
Y = torch.rand(2, 3, device=try_gpu(1))
Y
```

```{.python .input}
%%tab tensorflow
with try_gpu(1):
    Y = tf.random.uniform((2, 3))
Y
```

```{.python .input}
%%tab jax
Y = jax.device_put(jax.random.uniform(jax.random.PRNGKey(0), (2, 3)),
                   try_gpu(1))
Y
```

### コピー

`X + Y` を計算したい場合、どこでこの演算を行うかを決める必要がある。
たとえば、 :numref:`fig_copyto` に示すように、`X` を2枚目のGPUに転送してそこで演算を実行できる。
単に `X` と `Y` を足しては *いけない*。そうすると例外が発生する。
ランタイムエンジンは何をすべきかわからず、同じデバイス上にデータを見つけられないため失敗する。
`Y` は2枚目のGPU上にあるので、2つを足す前に `X` をそこへ移動する必要がある。

![Copy data to perform an operation on the same device.](../img/copyto.svg)
:label:`fig_copyto`

```{.python .input}
%%tab mxnet
Z = X.copyto(try_gpu(1))
print(X)
print(Z)
```

```{.python .input}
%%tab pytorch
Z = X.cuda(1)
print(X)
print(Z)
```

```{.python .input}
%%tab tensorflow
with try_gpu(1):
    Z = X
print(X)
print(Z)
```

```{.python .input}
%%tab jax
Z = jax.device_put(X, try_gpu(1))
print(X)
print(Z)
```

これで[**データ（`Z` と `Y` の両方）が同じGPU上にあるので、それらを加算できる。**]

```{.python .input}
%%tab all
Y + Z
```

:begin_tab:`mxnet`
変数 `Z` がすでに2枚目のGPU上にあるとしよう。
それでも `Z.copyto(gpu(1))` を呼び出したらどうなるだろうか？
その変数がすでに望ましいデバイス上にあっても、コピーが作成され、新しいメモリが割り当てられる。
実行環境によっては、2つの変数がすでに同じデバイス上にある場合がある。
そのため、変数が現在異なるデバイス上にあるときだけコピーしたいのである。
このような場合には `as_in_ctx` を呼び出せる。
変数がすでに指定したデバイス上にあるなら、これは何もしない。
特にコピーを作成したいのでない限り、`as_in_ctx` を使うのが適切である。
:end_tab:

:begin_tab:`pytorch`
しかし、変数 `Z` がすでに2枚目のGPU上にあるとしたらどうだろうか？
それでも `Z.cuda(1)` を呼び出したらどうなるだろうか？
コピーを作成して新しいメモリを割り当てる代わりに、`Z` を返す。
:end_tab:

:begin_tab:`tensorflow`
しかし、変数 `Z` がすでに2枚目のGPU上にあるとしたらどうだろうか？
同じデバイススコープ内でそれでも `Z2 = Z` としたらどうなるだろうか？
コピーを作成して新しいメモリを割り当てる代わりに、`Z` を返す。
:end_tab:

:begin_tab:`jax`
しかし、変数 `Z` がすでに2枚目のGPU上にあるとしたらどうだろうか？
同じデバイススコープ内でそれでも `Z2 = Z` としたらどうなるだろうか？
コピーを作成して新しいメモリを割り当てる代わりに、`Z` を返す。
:end_tab:

```{.python .input}
%%tab mxnet
Z.as_in_ctx(try_gpu(1)) is Z
```

```{.python .input}
%%tab pytorch
Z.cuda(1) is Z
```

```{.python .input}
%%tab tensorflow
with try_gpu(1):
    Z2 = Z
Z2 is Z
```

```{.python .input}
%%tab jax
Z2 = jax.device_put(Z, try_gpu(1))
Z2 is Z
```

### 補足

人々がGPUを機械学習に使うのは、それが高速だと期待しているからである。
しかし、変数をデバイス間で転送するのは遅く、計算よりもずっと遅いのである。
そのため、何か遅いことをさせる前に、それを本当にやりたいのか100%確信してもらう必要がある。
深層学習フレームワークが、クラッシュせずに自動でコピーしてしまうだけだと、遅いコードを書いてしまったことに気づかないかもしれない。

データ転送は遅いだけでなく、並列化もずっと難しくする。というのも、次の操作に進む前にデータが送られてくるのを（正確には受け取られるのを）待たなければならないからである。
そのため、コピー操作は非常に慎重に扱う必要がある。
経験則として、多くの小さな操作は1つの大きな操作よりもはるかに悪い。
さらに、何をしているかをよく理解しているのでない限り、コードの中に多数の単独の操作を散りばめるよりも、複数の操作をまとめて行うほうがずっと良い。
これは、そのような操作が、一方のデバイスが他方を待たなければ次のことができない場合にブロックされうるからである。
これは、電話で事前注文しておいて、あなたが来たときにはもうできあがっているコーヒーを受け取るのではなく、列に並んでコーヒーを注文するのに少し似ている。

最後に、テンソルを表示したりNumPy形式に変換したりするとき、データが主記憶にない場合、フレームワークはまずそれを主記憶にコピーするため、追加の転送オーバーヘッドが発生する。
さらに悪いことに、そのデータはPythonの完了をすべて待たせる悪名高いグローバルインタプリタロックの影響を受けることになる。


## [**ニューラルネットワークとGPU**]

同様に、ニューラルネットワークモデルでもデバイスを指定できる。
次のコードは、モデルのパラメータをGPU上に置く。

```{.python .input}
%%tab mxnet
net = nn.Sequential()
net.add(nn.Dense(1))
net.initialize(ctx=try_gpu())
```

```{.python .input}
%%tab pytorch
net = nn.Sequential(nn.LazyLinear(1))
net = net.to(device=try_gpu())
```

```{.python .input}
%%tab tensorflow
strategy = tf.distribute.MirroredStrategy()
with strategy.scope():
    net = tf.keras.models.Sequential([
        tf.keras.layers.Dense(1)])
```

```{.python .input}
%%tab jax
net = nn.Sequential([nn.Dense(1)])

key1, key2 = jax.random.split(jax.random.PRNGKey(0))
x = jax.random.normal(key1, (10,))  # Dummy input
params = net.init(key2, x)  # Initialization call
```

今後の章では、モデルをGPU上で実行する方法の例をさらに多く見ていく。モデルがやや計算集約的になっていくからである。

たとえば、入力がGPU上のテンソルであれば、モデルは同じGPU上で結果を計算する。

```{.python .input}
%%tab mxnet, pytorch, tensorflow
net(X)
```

```{.python .input}
%%tab jax
net.apply(params, x)
```

モデルのパラメータが同じGPU上に保存されていることを[**確認してみよう。**]

```{.python .input}
%%tab mxnet
net[0].weight.data().ctx
```

```{.python .input}
%%tab pytorch
net[0].weight.data.device
```

```{.python .input}
%%tab tensorflow
net.layers[0].weights[0].device, net.layers[0].weights[1].device
```

```{.python .input}
%%tab jax
print(jax.tree_util.tree_map(lambda x: x.device(), params))
```

トレーナーがGPUをサポートするようにする。

```{.python .input}
%%tab mxnet
@d2l.add_to_class(d2l.Module)  #@save
def set_scratch_params_device(self, device):
    for attr in dir(self):
        a = getattr(self, attr)
        if isinstance(a, np.ndarray):
            with autograd.record():
                setattr(self, attr, a.as_in_ctx(device))
            getattr(self, attr).attach_grad()
        if isinstance(a, d2l.Module):
            a.set_scratch_params_device(device)
        if isinstance(a, list):
            for elem in a:
                elem.set_scratch_params_device(device)
```

```{.python .input}
%%tab mxnet, pytorch
@d2l.add_to_class(d2l.Trainer)  #@save
def __init__(self, max_epochs, num_gpus=0, gradient_clip_val=0):
    self.save_hyperparameters()
    self.gpus = [d2l.gpu(i) for i in range(min(num_gpus, d2l.num_gpus()))]

@d2l.add_to_class(d2l.Trainer)  #@save
def prepare_batch(self, batch):
    if self.gpus:
        batch = [d2l.to(a, self.gpus[0]) for a in batch]
    return batch

@d2l.add_to_class(d2l.Trainer)  #@save
def prepare_model(self, model):
    model.trainer = self
    model.board.xlim = [0, self.max_epochs]
    if self.gpus:
        if tab.selected('mxnet'):
            model.collect_params().reset_ctx(self.gpus[0])
            model.set_scratch_params_device(self.gpus[0])
        if tab.selected('pytorch'):
            model.to(self.gpus[0])
    self.model = model
```

```{.python .input}
%%tab jax
@d2l.add_to_class(d2l.Trainer)  #@save
def __init__(self, max_epochs, num_gpus=0, gradient_clip_val=0):
    self.save_hyperparameters()
    self.gpus = [d2l.gpu(i) for i in range(min(num_gpus, d2l.num_gpus()))]

@d2l.add_to_class(d2l.Trainer)  #@save
def prepare_batch(self, batch):
    if self.gpus:
        batch = [d2l.to(a, self.gpus[0]) for a in batch]
    return batch
```

要するに、すべてのデータとパラメータが同じデバイス上にある限り、モデルを効率よく学習できる。次の章では、そのような例をいくつか見ていく。

## まとめ

CPUやGPUなど、保存や計算のためのデバイスを指定できる。
  デフォルトでは、データは主記憶に作成され、
  その後CPUで計算に使われる。
深層学習フレームワークでは、計算に必要なすべての入力データが
  CPUまたは同じGPU上にあることが求められる。
データを不用意に移動すると、大きな性能低下を招くことがある。
  よくある間違いは次のようなものである。GPU上で各ミニバッチの損失を計算し、
  それをコマンドラインでユーザーに報告したり（あるいはNumPy `ndarray` に記録したり）すると、
  グローバルインタプリタロックが発生してすべてのGPUが停止する。
  ログ記録用のメモリをGPU内に確保し、
  大きなログだけを移動するほうがはるかに良い。

## 演習

1. 大きな行列の積のような、より大きな計算タスクを試し、
   CPUとGPUの速度差を確認せよ。
   計算量が少ないタスクではどうだろうか。
1. モデルパラメータをGPU上でどのように読み書きすべきだろうか。
1. $100 \times 100$ の行列どうしの行列積を1000回計算し、
   出力行列のフロベニウスノルムを1結果ずつ記録するのにかかる時間を測れ。
   GPU上にログを保持し、最後の結果だけを転送する場合と比較せよ。
1. 2枚のGPU上で同時に2つの行列積を実行するのにかかる時間を測れ。
   1枚のGPU上で順番に計算する場合と比較せよ。
   ヒント: ほぼ線形スケーリングが見られるはずである。\n
