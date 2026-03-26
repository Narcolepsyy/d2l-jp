{.python .input}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow', 'jax'])
```

# GPU
:label:`sec_use_gpu`

:numref:`tab_intro_decade` では、過去20年間の計算能力の急速な成長を示しました。
要するに、2000年以降、GPUの性能は10年ごとに1000倍に向上してきました。
これは大きな機会をもたらしますが、同時に、そのような性能に対する大きな需要があったことも示しています。


この節では、この計算性能を研究にどう活用するかを説明し始めます。
まずは単一GPUの使い方を取り上げ、後の段階で複数GPUや複数サーバー（複数GPU搭載）を使う方法を扱います。

具体的には、単一のNVIDIA GPUを計算に使う方法を説明します。
まず、少なくとも1枚のNVIDIA GPUが搭載されていることを確認してください。
次に、[NVIDIA driver と CUDA](https://developer.nvidia.com/cuda-downloads) をダウンロードし、指示に従って適切なパスを設定します。
これらの準備が完了したら、`nvidia-smi` コマンドを使って[**グラフィックカード情報を表示**]できます。

:begin_tab:`mxnet`
MXNetのテンソルはNumPyの `ndarray` とほとんど同じに見えることに気づいたかもしれません。
しかし、いくつか重要な違いがあります。
MXNetをNumPyと区別する重要な特徴の1つは、さまざまなハードウェアデバイスをサポートしていることです。

MXNetでは、すべての配列にコンテキストがあります。
これまでのところ、デフォルトでは、すべての変数と関連する計算はCPUに割り当てられていました。
通常、他のコンテキストとしてはさまざまなGPUがあります。
複数のサーバーにまたがってジョブを展開すると、状況はさらに複雑になります。
配列をコンテキストに賢く割り当てることで、デバイス間でデータを転送する時間を最小化できます。
たとえば、GPU搭載サーバーでニューラルネットワークを学習するときには、モデルのパラメータをGPU上に置くのが普通です。

次に、MXNetのGPU版がインストールされていることを確認する必要があります。
すでにCPU版のMXNetがインストールされている場合は、まずそれをアンインストールする必要があります。
たとえば、`pip uninstall mxnet` コマンドを使い、その後CUDAのバージョンに対応するMXNetをインストールします。
CUDA 10.0 がインストールされていると仮定すると、`pip install mxnet-cu100` によりCUDA 10.0をサポートするMXNetをインストールできます。
:end_tab:

:begin_tab:`pytorch`
PyTorchでは、すべての配列にデバイスがあり、これを *コンテキスト* と呼ぶことがよくあります。
これまでのところ、デフォルトでは、すべての変数と関連する計算はCPUに割り当てられていました。
通常、他のコンテキストとしてはさまざまなGPUがあります。
複数のサーバーにまたがってジョブを展開すると、状況はさらに複雑になります。
配列をコンテキストに賢く割り当てることで、デバイス間でデータを転送する時間を最小化できます。
たとえば、GPU搭載サーバーでニューラルネットワークを学習するときには、モデルのパラメータをGPU上に置くのが普通です。
:end_tab:

この節のプログラムを実行するには、少なくとも2枚のGPUが必要です。
これは多くのデスクトップコンピュータにとっては贅沢かもしれませんが、たとえばAWS EC2のマルチGPUインスタンスを使えば、クラウド上では簡単に利用できます。
他のほとんどの節では複数GPUは *不要* ですが、ここでは単に異なるデバイス間でのデータの流れを示したいだけです。

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

CPUやGPUなどのデバイスを、保存や計算のために指定できます。
デフォルトでは、テンソルは主記憶に作成され、その後CPUが計算に使われます。

:begin_tab:`mxnet`
MXNetでは、CPUとGPUはそれぞれ `cpu()` と `gpu()` で指定できます。
`cpu()`
（または括弧内の任意の整数）は、すべての物理CPUとメモリを意味することに注意してください。
これは、MXNetの計算がすべてのCPUコアを使おうとすることを意味します。
一方、`gpu()` は1枚のカードとそれに対応するメモリだけを表します。
複数のGPUがある場合は、$i^\textrm{th}$ GPU（$i$ は0から始まる）を表すために `gpu(i)` を使います。
また、`gpu(0)` と `gpu()` は等価です。
:end_tab:

:begin_tab:`pytorch`
PyTorchでは、CPUとGPUは `torch.device('cpu')` と `torch.device('cuda')` で指定できます。
`cpu` デバイスは、すべての物理CPUとメモリを意味することに注意してください。
これは、PyTorchの計算がすべてのCPUコアを使おうとすることを意味します。
一方、`gpu` デバイスは1枚のカードとそれに対応するメモリだけを表します。
複数のGPUがある場合は、$i^\textrm{th}$ GPU（$i$ は0から始まる）を表すために `torch.device(f'cuda:{i}')` を使います。
また、`gpu:0` と `gpu` は等価です。
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

[**利用可能なGPUの数を問い合わせる**]ことができます。

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

ここで、要求したGPUが存在しなくてもコードを実行できるようにする、便利な2つの関数を定義します。

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
デフォルトでは、テンソルはCPU上に作成されます。
テンソルがどのデバイス上にあるかを[**問い合わせる**]ことができます。
:end_tab:

:begin_tab:`mxnet`
デフォルトでは、テンソルはCPU上に作成されます。
テンソルがどのデバイス上にあるかを[**問い合わせる**]ことができます。
:end_tab:

:begin_tab:`tensorflow, jax`
デフォルトでは、利用可能であればテンソルはGPU/TPU上に作成され、利用できない場合はCPUが使われます。
テンソルがどのデバイス上にあるかを[**問い合わせる**]ことができます。
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

複数の項を扱うときには、それらが同じデバイス上になければならないことに注意することが重要です。
たとえば、2つのテンソルを足し合わせる場合、両方の引数が同じデバイス上にあることを確認する必要があります。そうでなければ、フレームワークは結果をどこに保存すべきか、あるいは計算をどこで行うべきかさえ判断できません。

### GPU上での保存

テンソルをGPU上に[**保存する**]方法はいくつかあります。
たとえば、テンソルを作成するときに保存先デバイスを指定できます。
次に、最初の `gpu` 上にテンソル変数 `X` を作成します。
GPU上で作成されたテンソルは、そのGPUのメモリだけを消費します。
`nvidia-smi` コマンドを使ってGPUのメモリ使用量を確認できます。
一般に、GPUメモリの制限を超えるデータを作成しないように注意する必要があります。

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

少なくとも2枚のGPUがあると仮定すると、次のコードは[**2枚目のGPU上にランダムなテンソル `Y` を作成**]します。

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

`X + Y` を計算したい場合、どこでこの演算を行うかを決める必要があります。
たとえば、 :numref:`fig_copyto` に示すように、`X` を2枚目のGPUに転送してそこで演算を実行できます。
単に `X` と `Y` を足しては *いけません*。そうすると例外が発生します。
ランタイムエンジンは何をすべきかわからず、同じデバイス上にデータを見つけられないため失敗します。
`Y` は2枚目のGPU上にあるので、2つを足す前に `X` をそこへ移動する必要があります。

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

これで[**データ（`Z` と `Y` の両方）が同じGPU上にあるので、それらを加算できます。**]

```{.python .input}
%%tab all
Y + Z
```

:begin_tab:`mxnet`
変数 `Z` がすでに2枚目のGPU上にあるとしましょう。
それでも `Z.copyto(gpu(1))` を呼び出したらどうなるでしょうか？
その変数がすでに望ましいデバイス上にあっても、コピーが作成され、新しいメモリが割り当てられます。
実行環境によっては、2つの変数がすでに同じデバイス上にある場合があります。
そのため、変数が現在異なるデバイス上にあるときだけコピーしたいのです。
このような場合には `as_in_ctx` を呼び出せます。
変数がすでに指定したデバイス上にあるなら、これは何もしません。
特にコピーを作成したいのでない限り、`as_in_ctx` を使うのが適切です。
:end_tab:

:begin_tab:`pytorch`
しかし、変数 `Z` がすでに2枚目のGPU上にあるとしたらどうでしょうか？
それでも `Z.cuda(1)` を呼び出したらどうなるでしょうか？
コピーを作成して新しいメモリを割り当てる代わりに、`Z` を返します。
:end_tab:

:begin_tab:`tensorflow`
しかし、変数 `Z` がすでに2枚目のGPU上にあるとしたらどうでしょうか？
同じデバイススコープ内でそれでも `Z2 = Z` としたらどうなるでしょうか？
コピーを作成して新しいメモリを割り当てる代わりに、`Z` を返します。
:end_tab:

:begin_tab:`jax`
しかし、変数 `Z` がすでに2枚目のGPU上にあるとしたらどうでしょうか？
同じデバイススコープ内でそれでも `Z2 = Z` としたらどうなるでしょうか？
コピーを作成して新しいメモリを割り当てる代わりに、`Z` を返します。
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

人々がGPUを機械学習に使うのは、それが高速だと期待しているからです。
しかし、変数をデバイス間で転送するのは遅く、計算よりもずっと遅いのです。
そのため、何か遅いことをさせる前に、それを本当にやりたいのか100%確信してもらう必要があります。
深層学習フレームワークが、クラッシュせずに自動でコピーしてしまうだけだと、遅いコードを書いてしまったことに気づかないかもしれません。

データ転送は遅いだけでなく、並列化もずっと難しくします。というのも、次の操作に進む前にデータが送られてくるのを（正確には受け取られるのを）待たなければならないからです。
そのため、コピー操作は非常に慎重に扱う必要があります。
経験則として、多くの小さな操作は1つの大きな操作よりもはるかに悪いです。
さらに、何をしているかをよく理解しているのでない限り、コードの中に多数の単独の操作を散りばめるよりも、複数の操作をまとめて行うほうがずっと良いです。
これは、そのような操作が、一方のデバイスが他方を待たなければ次のことができない場合にブロックされうるからです。
これは、電話で事前注文しておいて、あなたが来たときにはもうできあがっているコーヒーを受け取るのではなく、列に並んでコーヒーを注文するのに少し似ています。

最後に、テンソルを表示したりNumPy形式に変換したりするとき、データが主記憶にない場合、フレームワークはまずそれを主記憶にコピーするため、追加の転送オーバーヘッドが発生します。
さらに悪いことに、そのデータはPythonの完了をすべて待たせる悪名高いグローバルインタプリタロックの影響を受けることになります。


## [**ニューラルネットワークとGPU**]

同様に、ニューラルネットワークモデルでもデバイスを指定できます。
次のコードは、モデルのパラメータをGPU上に置きます。

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

今後の章では、モデルをGPU上で実行する方法の例をさらに多く見ていきます。モデルがやや計算集約的になっていくからです。

たとえば、入力がGPU上のテンソルであれば、モデルは同じGPU上で結果を計算します。

```{.python .input}
%%tab mxnet, pytorch, tensorflow
net(X)
```

```{.python .input}
%%tab jax
net.apply(params, x)
```

モデルのパラメータが同じGPU上に保存されていることを[**確認してみましょう。**]

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

トレーナーがGPUをサポートするようにします。

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

要するに、すべてのデータとパラメータが同じデバイス上にある限り、モデルを効率よく学習できます。次の章では、そのような例をいくつか見ていきます。

## まとめ

CPUやGPUなど、保存や計算のためのデバイスを指定できます。
  デフォルトでは、データは主記憶に作成され、
  その後CPUで計算に使われます。
深層学習フレームワークでは、計算に必要なすべての入力データが
  CPUまたは同じGPU上にあることが求められます。
データを不用意に移動すると、大きな性能低下を招くことがあります。
  よくある間違いは次のようなものです。GPU上で各ミニバッチの損失を計算し、
  それをコマンドラインでユーザーに報告したり（あるいはNumPy `ndarray` に記録したり）すると、
  グローバルインタプリタロックが発生してすべてのGPUが停止します。
  ログ記録用のメモリをGPU内に確保し、
  大きなログだけを移動するほうがはるかに良いです。

## 演習

1. 大きな行列の積のような、より大きな計算タスクを試し、
   CPUとGPUの速度差を確認してください。
   計算量が少ないタスクではどうでしょうか。
1. モデルパラメータをGPU上でどのように読み書きすべきでしょうか。
1. $100 \times 100$ の行列どうしの行列積を1000回計算し、
   出力行列のフロベニウスノルムを1結果ずつ記録するのにかかる時間を測ってください。
   GPU上にログを保持し、最後の結果だけを転送する場合と比較してください。
1. 2枚のGPU上で同時に2つの行列積を実行するのにかかる時間を測ってください。
   1枚のGPU上で順番に計算する場合と比較してください。
   ヒント: ほぼ線形スケーリングが見られるはずです。\n
