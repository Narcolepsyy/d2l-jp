# 複数GPUのための簡潔な実装
:label:`sec_multi_gpu_concise`

新しいモデルごとに並列化を一から実装するのは楽しいものではありません。さらに、高性能を実現するためには同期ツールの最適化に大きな利点があります。以下では、深層学習フレームワークの高レベルAPIを使ってこれを行う方法を示します。
数学とアルゴリズムは :numref:`sec_multi_gpu` と同じです。
当然ながら、この節のコードを実行するには少なくとも2つのGPUが必要です。

```{.python .input}
#@tab mxnet
from d2l import mxnet as d2l
from mxnet import autograd, gluon, init, np, npx
from mxnet.gluon import nn
npx.set_np()
```

```{.python .input}
#@tab pytorch
from d2l import torch as d2l
import torch
from torch import nn
```

## [**おもちゃのネットワーク**]

:numref:`sec_multi_gpu` の LeNet よりも少し意味のある、それでいて十分に簡単かつ高速に学習できるネットワークを使いましょう。
ここでは ResNet-18 の変種 :cite:`He.Zhang.Ren.ea.2016` を選びます。入力画像が小さいので、少し変更を加えます。特に、:numref:`sec_resnet` との違いは、最初の畳み込みでより小さいカーネル、ストライド、パディングを使うことです。
さらに、最大プーリング層を取り除きます。

```{.python .input}
#@tab mxnet
#@save
def resnet18(num_classes):
    """A slightly modified ResNet-18 model."""
    def resnet_block(num_channels, num_residuals, first_block=False):
        blk = nn.Sequential()
        for i in range(num_residuals):
            if i == 0 and not first_block:
                blk.add(d2l.Residual(
                    num_channels, use_1x1conv=True, strides=2))
            else:
                blk.add(d2l.Residual(num_channels))
        return blk

    net = nn.Sequential()
    # This model uses a smaller convolution kernel, stride, and padding and
    # removes the max-pooling layer
    net.add(nn.Conv2D(64, kernel_size=3, strides=1, padding=1),
            nn.BatchNorm(), nn.Activation('relu'))
    net.add(resnet_block(64, 2, first_block=True),
            resnet_block(128, 2),
            resnet_block(256, 2),
            resnet_block(512, 2))
    net.add(nn.GlobalAvgPool2D(), nn.Dense(num_classes))
    return net
```

```{.python .input}
#@tab pytorch
#@save
def resnet18(num_classes, in_channels=1):
    """A slightly modified ResNet-18 model."""
    def resnet_block(in_channels, out_channels, num_residuals,
                     first_block=False):
        blk = []
        for i in range(num_residuals):
            if i == 0 and not first_block:
                blk.append(d2l.Residual(out_channels, use_1x1conv=True, 
                                        strides=2))
            else:
                blk.append(d2l.Residual(out_channels))
        return nn.Sequential(*blk)

    # This model uses a smaller convolution kernel, stride, and padding and
    # removes the max-pooling layer
    net = nn.Sequential(
        nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU())
    net.add_module("resnet_block1", resnet_block(64, 64, 2, first_block=True))
    net.add_module("resnet_block2", resnet_block(64, 128, 2))
    net.add_module("resnet_block3", resnet_block(128, 256, 2))
    net.add_module("resnet_block4", resnet_block(256, 512, 2))
    net.add_module("global_avg_pool", nn.AdaptiveAvgPool2d((1,1)))
    net.add_module("fc", nn.Sequential(nn.Flatten(),
                                       nn.Linear(512, num_classes)))
    return net
```

## ネットワークの初期化

:begin_tab:`mxnet`
`initialize` 関数を使うと、任意のデバイス上でパラメータを初期化できます。
初期化手法の復習については :numref:`sec_numerical_stability` を参照してください。特に便利なのは、ネットワークを同時に *複数* のデバイス上で初期化できることです。実際にどのように動作するか試してみましょう。
:end_tab:

:begin_tab:`pytorch`
学習ループの中でネットワークを初期化します。
初期化手法の復習については :numref:`sec_numerical_stability` を参照してください。
:end_tab:

```{.python .input}
#@tab mxnet
net = resnet18(10)
# Get a list of GPUs
devices = d2l.try_all_gpus()
# Initialize all the parameters of the network
net.initialize(init=init.Normal(sigma=0.01), ctx=devices)
```

```{.python .input}
#@tab pytorch
net = resnet18(10)
# Get a list of GPUs
devices = d2l.try_all_gpus()
# We will initialize the network inside the training loop
```

:begin_tab:`mxnet`
:numref:`sec_multi_gpu` で導入した `split_and_load` 関数を使うと、ミニバッチデータを分割し、`devices` 変数で与えられたデバイスのリストへ各部分をコピーできます。ネットワークのインスタンスは、順伝播の値を計算するのに適切なGPUを *自動的に* 使用します。ここでは4つの観測を生成し、それらをGPU上に分割します。
:end_tab:

```{.python .input}
#@tab mxnet
x = np.random.uniform(size=(4, 1, 28, 28))
x_shards = gluon.utils.split_and_load(x, devices)
net(x_shards[0]), net(x_shards[1])
```

:begin_tab:`mxnet`
データがネットワークを通過すると、対応するパラメータは、そのデータが通過した *デバイス上で* 初期化されます。
つまり、初期化はデバイスごとに行われます。初期化先として GPU 0 と GPU 1 を選んだので、ネットワークはそれらの上でのみ初期化され、CPU 上では初期化されません。実際、パラメータは CPU 上には存在しません。これを確認するためにパラメータを表示し、発生しうるエラーを観察できます。
:end_tab:

```{.python .input}
#@tab mxnet
weight = net[0].params.get('weight')

try:
    weight.data()
except RuntimeError:
    print('not initialized on cpu')
weight.data(devices[0])[0], weight.data(devices[1])[0]
```

:begin_tab:`mxnet`
次に、[**精度を評価する**]コードを、(**複数デバイスにまたがって並列に**)動作するものに置き換えましょう。これは :numref:`sec_lenet` の `evaluate_accuracy_gpu` 関数の代替になります。主な違いは、ネットワークを呼び出す前にミニバッチを分割することです。それ以外は本質的に同じです。
:end_tab:

```{.python .input}
#@tab mxnet
#@save
def evaluate_accuracy_gpus(net, data_iter, split_f=d2l.split_batch):
    """Compute the accuracy for a model on a dataset using multiple GPUs."""
    # Query the list of devices
    devices = list(net.collect_params().values())[0].list_ctx()
    # No. of correct predictions, no. of predictions
    metric = d2l.Accumulator(2)
    for features, labels in data_iter:
        X_shards, y_shards = split_f(features, labels, devices)
        # Run in parallel
        pred_shards = [net(X_shard) for X_shard in X_shards]
        metric.add(sum(float(d2l.accuracy(pred_shard, y_shard)) for
                       pred_shard, y_shard in zip(
                           pred_shards, y_shards)), labels.size)
    return metric[0] / metric[1]
```

## [**学習**]

これまでと同様に、効率的な並列化のためには学習コードがいくつかの基本機能を実行する必要があります。

* ネットワークのパラメータをすべてのデバイス上で初期化する必要があります。
* データセットを反復する間、ミニバッチをすべてのデバイスに分割する必要があります。
* 損失とその勾配をデバイス間で並列に計算します。
* 勾配を集約し、それに応じてパラメータを更新します。

最後に、ネットワークの最終性能を報告するために、精度を（再び並列に）計算します。学習ルーチンは、データの分割と集約が必要である点を除けば、前の章の実装とかなり似ています。

```{.python .input}
#@tab mxnet
def train(num_gpus, batch_size, lr):
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
    ctx = [d2l.try_gpu(i) for i in range(num_gpus)]
    net.initialize(init=init.Normal(sigma=0.01), ctx=ctx, force_reinit=True)
    trainer = gluon.Trainer(net.collect_params(), 'sgd',
                            {'learning_rate': lr})
    loss = gluon.loss.SoftmaxCrossEntropyLoss()
    timer, num_epochs = d2l.Timer(), 10
    animator = d2l.Animator('epoch', 'test acc', xlim=[1, num_epochs])
    for epoch in range(num_epochs):
        timer.start()
        for features, labels in train_iter:
            X_shards, y_shards = d2l.split_batch(features, labels, ctx)
            with autograd.record():
                ls = [loss(net(X_shard), y_shard) for X_shard, y_shard
                      in zip(X_shards, y_shards)]
            for l in ls:
                l.backward()
            trainer.step(batch_size)
        npx.waitall()
        timer.stop()
        animator.add(epoch + 1, (evaluate_accuracy_gpus(net, test_iter),))
    print(f'test acc: {animator.Y[0][-1]:.2f}, {timer.avg():.1f} sec/epoch '
          f'on {str(ctx)}')
```

```{.python .input}
#@tab pytorch
def train(net, num_gpus, batch_size, lr):
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
    devices = [d2l.try_gpu(i) for i in range(num_gpus)]
    def init_weights(module):
        if type(module) in [nn.Linear, nn.Conv2d]:
            nn.init.normal_(module.weight, std=0.01)
    net.apply(init_weights)
    # Set the model on multiple GPUs
    net = nn.DataParallel(net, device_ids=devices)
    trainer = torch.optim.SGD(net.parameters(), lr)
    loss = nn.CrossEntropyLoss()
    timer, num_epochs = d2l.Timer(), 10
    animator = d2l.Animator('epoch', 'test acc', xlim=[1, num_epochs])
    for epoch in range(num_epochs):
        net.train()
        timer.start()
        for X, y in train_iter:
            trainer.zero_grad()
            X, y = X.to(devices[0]), y.to(devices[0])
            l = loss(net(X), y)
            l.backward()
            trainer.step()
        timer.stop()
        animator.add(epoch + 1, (d2l.evaluate_accuracy_gpu(net, test_iter),))
    print(f'test acc: {animator.Y[0][-1]:.2f}, {timer.avg():.1f} sec/epoch '
          f'on {str(devices)}')
```

実際にどのように動作するか見てみましょう。準備として、[**ネットワークを単一GPUで学習します。**]

```{.python .input}
#@tab mxnet
train(num_gpus=1, batch_size=256, lr=0.1)
```

```{.python .input}
#@tab pytorch
train(net, num_gpus=1, batch_size=256, lr=0.1)
```

次に、[**学習に2つのGPUを使います**]。 :numref:`sec_multi_gpu` で評価した LeNet と比べると、ResNet-18 のモデルはかなり複雑です。ここで並列化の利点が現れます。計算時間はパラメータ同期の時間よりも十分に大きくなります。これにより、並列化のオーバーヘッドの相対的な影響が小さくなるため、スケーラビリティが向上します。

```{.python .input}
#@tab mxnet
train(num_gpus=2, batch_size=512, lr=0.2)
```

```{.python .input}
#@tab pytorch
train(net, num_gpus=2, batch_size=512, lr=0.2)
```

## まとめ

:begin_tab:`mxnet`
* Gluon は、コンテキストのリストを与えることで、複数デバイスにまたがるモデル初期化のための基本機能を提供します。
:end_tab:
* データは、データが存在するデバイス上で自動的に評価されます。
* そのデバイス上のパラメータにアクセスしようとする前に、各デバイス上でネットワークを初期化するよう注意してください。そうしないとエラーが発生します。
* 最適化アルゴリズムは、複数GPUにわたって自動的に集約します。



## 演習

:begin_tab:`mxnet`
1. この節では ResNet-18 を使いました。エポック数、バッチサイズ、学習率を変えてみてください。計算により多くのGPUを使ってみましょう。16個のGPU（たとえば AWS p2.16xlarge インスタンス）で試すとどうなりますか？
1. ときには、デバイスごとに計算能力が異なることがあります。GPU と CPU を同時に使うこともできます。どのように仕事を分担すべきでしょうか？その労力に見合う価値はあるでしょうか？なぜですか？なぜではないですか？
1. `npx.waitall()` を削除するとどうなりますか？並列化のために最大2ステップ分のオーバーラップが生じるように学習をどのように変更しますか？
:end_tab:

:begin_tab:`pytorch`
1. この節では ResNet-18 を使いました。エポック数、バッチサイズ、学習率を変えてみてください。計算により多くのGPUを使ってみましょう。16個のGPU（たとえば AWS p2.16xlarge インスタンス）で試すとどうなりますか？
1. ときには、デバイスごとに計算能力が異なることがあります。GPU と CPU を同時に使うこともできます。どのように仕事を分担すべきでしょうか？その労力に見合う価値はあるでしょうか？なぜですか？なぜではないですか？
:end_tab:



:begin_tab:`mxnet`
[議論](https://discuss.d2l.ai/t/365)
:end_tab:

:begin_tab:`pytorch`
[議論](https://discuss.d2l.ai/t/1403)
:end_tab:\n