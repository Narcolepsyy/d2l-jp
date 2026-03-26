# 学習率スケジューリング
:label:`sec_scheduler`

これまで主に、重みベクトルをどのように更新するかという最適化の*アルゴリズム*に注目してきましたが、更新される*速度*についてはあまり扱ってきませんでした。とはいえ、学習率の調整は実際のアルゴリズムと同じくらい重要であることがよくあります。考慮すべき点はいくつかあります。

* まず明らかなのは、学習率の*大きさ*が重要だということです。大きすぎれば最適化は発散し、小さすぎれば学習に時間がかかりすぎるか、あるいは準最適な結果に終わってしまいます。以前、問題の条件数が重要であることを見ました（詳細は例えば :numref:`sec_momentum` を参照）。直感的には、最も感度の低い方向における変化量と、最も感度の高い方向における変化量の比です。
* 第二に、減衰の速さも同様に重要です。学習率が大きいままだと、単に最小値の周りを行ったり来たりするだけで、最適性に到達できないかもしれません。 :numref:`sec_minibatch_sgd` でこれを詳しく議論し、 :numref:`sec_sgd` では性能保証を解析しました。要するに、学習率は減衰してほしいのですが、凸問題に対しては良い選択となる $\mathcal{O}(t^{-\frac{1}{2}})$ よりはおそらくゆっくり減衰するほうがよいでしょう。
* もう一つ同じくらい重要なのが*初期化*です。これは、パラメータを最初にどう設定するか（詳細は :numref:`sec_numerical_stability` を参照）だけでなく、初期段階でそれらがどう変化するかにも関係します。これは*ウォームアップ*という名目で扱われ、つまり最初に解へ向かってどれだけ速く動き始めるかということです。特に、初期のパラメータ設定はランダムなので、最初の大きなステップは有益でないかもしれません。初期の更新方向も、かなり無意味である可能性があります。
* 最後に、周期的に学習率を調整する最適化の変種もいくつかあります。これは本章の範囲を超えます。読者には :citet:`Izmailov.Podoprikhin.Garipov.ea.2018` の詳細、たとえばパラメータの全*経路*にわたって平均を取ることでより良い解を得る方法などを参照することを勧めます。

学習率の管理には多くの詳細が必要であるため、ほとんどの深層学習フレームワークにはこれを自動的に扱うためのツールがあります。本章では、異なるスケジュールが精度に与える影響を確認し、さらにこれを*学習率スケジューラ*によって効率的に管理する方法を示します。

## おもちゃ問題

まず、計算コストが低く簡単に扱える一方で、いくつかの重要な点を示すのに十分に非自明なおもちゃ問題から始めます。そのために、Fashion-MNIST に適用した LeNet の少し現代的な版（活性化関数は `sigmoid` ではなく `relu`、AveragePooling ではなく MaxPooling）を用います。さらに、性能向上のためにネットワークをハイブリダイズします。コードの大部分は標準的なので、ここでは基本だけを紹介し、詳細な議論はしません。必要に応じて :numref:`chap_cnn` を復習してください。

```{.python .input}
#@tab mxnet
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import autograd, gluon, init, lr_scheduler, np, npx
from mxnet.gluon import nn
npx.set_np()

net = nn.HybridSequential()
net.add(nn.Conv2D(channels=6, kernel_size=5, padding=2, activation='relu'),
        nn.MaxPool2D(pool_size=2, strides=2),
        nn.Conv2D(channels=16, kernel_size=5, activation='relu'),
        nn.MaxPool2D(pool_size=2, strides=2),
        nn.Dense(120, activation='relu'),
        nn.Dense(84, activation='relu'),
        nn.Dense(10))
net.hybridize()
loss = gluon.loss.SoftmaxCrossEntropyLoss()
device = d2l.try_gpu()

batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size=batch_size)

# The code is almost identical to `d2l.train_ch6` defined in the
# lenet section of chapter convolutional neural networks
def train(net, train_iter, test_iter, num_epochs, loss, trainer, device):
    net.initialize(force_reinit=True, ctx=device, init=init.Xavier())
    animator = d2l.Animator(xlabel='epoch', xlim=[0, num_epochs],
                            legend=['train loss', 'train acc', 'test acc'])
    for epoch in range(num_epochs):
        metric = d2l.Accumulator(3)  # train_loss, train_acc, num_examples
        for i, (X, y) in enumerate(train_iter):
            X, y = X.as_in_ctx(device), y.as_in_ctx(device)
            with autograd.record():
                y_hat = net(X)
                l = loss(y_hat, y)
            l.backward()
            trainer.step(X.shape[0])
            metric.add(l.sum(), d2l.accuracy(y_hat, y), X.shape[0])
            train_loss = metric[0] / metric[2]
            train_acc = metric[1] / metric[2]
            if (i + 1) % 50 == 0:
                animator.add(epoch + i / len(train_iter),
                             (train_loss, train_acc, None))
        test_acc = d2l.evaluate_accuracy_gpu(net, test_iter)
        animator.add(epoch + 1, (None, None, test_acc))
    print(f'train loss {train_loss:.3f}, train acc {train_acc:.3f}, '
          f'test acc {test_acc:.3f}')
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
import math
import torch
from torch import nn
from torch.optim import lr_scheduler

def net_fn():
    model = nn.Sequential(
        nn.Conv2d(1, 6, kernel_size=5, padding=2), nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Conv2d(6, 16, kernel_size=5), nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Flatten(),
        nn.Linear(16 * 5 * 5, 120), nn.ReLU(),
        nn.Linear(120, 84), nn.ReLU(),
        nn.Linear(84, 10))

    return model

loss = nn.CrossEntropyLoss()
device = d2l.try_gpu()

batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size=batch_size)

# The code is almost identical to `d2l.train_ch6` defined in the
# lenet section of chapter convolutional neural networks
def train(net, train_iter, test_iter, num_epochs, loss, trainer, device,
          scheduler=None):
    net.to(device)
    animator = d2l.Animator(xlabel='epoch', xlim=[0, num_epochs],
                            legend=['train loss', 'train acc', 'test acc'])

    for epoch in range(num_epochs):
        metric = d2l.Accumulator(3)  # train_loss, train_acc, num_examples
        for i, (X, y) in enumerate(train_iter):
            net.train()
            trainer.zero_grad()
            X, y = X.to(device), y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y)
            l.backward()
            trainer.step()
            with torch.no_grad():
                metric.add(l * X.shape[0], d2l.accuracy(y_hat, y), X.shape[0])
            train_loss = metric[0] / metric[2]
            train_acc = metric[1] / metric[2]
            if (i + 1) % 50 == 0:
                animator.add(epoch + i / len(train_iter),
                             (train_loss, train_acc, None))

        test_acc = d2l.evaluate_accuracy_gpu(net, test_iter)
        animator.add(epoch+1, (None, None, test_acc))

        if scheduler:
            if scheduler.__module__ == lr_scheduler.__name__:
                # Using PyTorch In-Built scheduler
                scheduler.step()
            else:
                # Using custom defined scheduler
                for param_group in trainer.param_groups:
                    param_group['lr'] = scheduler(epoch)

    print(f'train loss {train_loss:.3f}, train acc {train_acc:.3f}, '
          f'test acc {test_acc:.3f}')
```

```{.python .input}
#@tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
import tensorflow as tf
import math
from tensorflow.keras.callbacks import LearningRateScheduler

def net():
    return tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(filters=6, kernel_size=5, activation='relu',
                               padding='same'),
        tf.keras.layers.AvgPool2D(pool_size=2, strides=2),
        tf.keras.layers.Conv2D(filters=16, kernel_size=5,
                               activation='relu'),
        tf.keras.layers.AvgPool2D(pool_size=2, strides=2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(120, activation='relu'),
        tf.keras.layers.Dense(84, activation='sigmoid'),
        tf.keras.layers.Dense(10)])


batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size=batch_size)

# The code is almost identical to `d2l.train_ch6` defined in the
# lenet section of chapter convolutional neural networks
def train(net_fn, train_iter, test_iter, num_epochs, lr,
              device=d2l.try_gpu(), custom_callback = False):
    device_name = device._device_name
    strategy = tf.distribute.OneDeviceStrategy(device_name)
    with strategy.scope():
        optimizer = tf.keras.optimizers.SGD(learning_rate=lr)
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        net = net_fn()
        net.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
    callback = d2l.TrainCallback(net, train_iter, test_iter, num_epochs,
                             device_name)
    if custom_callback is False:
        net.fit(train_iter, epochs=num_epochs, verbose=0,
                callbacks=[callback])
    else:
         net.fit(train_iter, epochs=num_epochs, verbose=0,
                 callbacks=[callback, custom_callback])
    return net
```

このアルゴリズムを、学習率 $0.3$ で $30$ イテレーション学習するというデフォルト設定で実行するとどうなるか見てみましょう。訓練精度は上がり続ける一方で、テスト精度の向上はある点を超えると止まっていることに注意してください。両方の曲線の間のギャップは過学習を示しています。

```{.python .input}
#@tab mxnet
lr, num_epochs = 0.3, 30
net.initialize(force_reinit=True, ctx=device, init=init.Xavier())
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': lr})
train(net, train_iter, test_iter, num_epochs, loss, trainer, device)
```

```{.python .input}
#@tab pytorch
lr, num_epochs = 0.3, 30
net = net_fn()
trainer = torch.optim.SGD(net.parameters(), lr=lr)
train(net, train_iter, test_iter, num_epochs, loss, trainer, device)
```

```{.python .input}
#@tab tensorflow
lr, num_epochs = 0.3, 30
train(net, train_iter, test_iter, num_epochs, lr)
```

## スケジューラ

学習率を調整する一つの方法は、各ステップで明示的に設定することです。これは `set_learning_rate` メソッドで簡単に実現できます。たとえば、最適化の進み具合に応じて動的に、各エポック後（あるいは各ミニバッチ後）に学習率を下げることができます。

```{.python .input}
#@tab mxnet
trainer.set_learning_rate(0.1)
print(f'learning rate is now {trainer.learning_rate:.2f}')
```

```{.python .input}
#@tab pytorch
lr = 0.1
trainer.param_groups[0]["lr"] = lr
print(f'learning rate is now {trainer.param_groups[0]["lr"]:.2f}')
```

```{.python .input}
#@tab tensorflow
lr = 0.1
dummy_model = tf.keras.models.Sequential([tf.keras.layers.Dense(10)])
dummy_model.compile(tf.keras.optimizers.SGD(learning_rate=lr), loss='mse')
print(f'learning rate is now ,', dummy_model.optimizer.lr.numpy())
```

より一般的には、スケジューラを定義したいところです。更新回数を与えて呼び出すと、適切な学習率の値を返します。単純な例として、学習率を $\eta = \eta_0 (t + 1)^{-\frac{1}{2}}$ に設定するものを定義しましょう。

```{.python .input}
#@tab all
class SquareRootScheduler:
    def __init__(self, lr=0.1):
        self.lr = lr

    def __call__(self, num_update):
        return self.lr * pow(num_update + 1.0, -0.5)
```

その振る舞いをいくつかの値にわたって描画してみましょう。

```{.python .input}
#@tab all
scheduler = SquareRootScheduler(lr=0.1)
d2l.plot(d2l.arange(num_epochs), [scheduler(t) for t in range(num_epochs)])
```

では、Fashion-MNIST で学習したときにこれがどう働くかを見てみましょう。学習アルゴリズムに追加引数としてスケジューラを与えるだけです。

```{.python .input}
#@tab mxnet
trainer = gluon.Trainer(net.collect_params(), 'sgd',
                        {'lr_scheduler': scheduler})
train(net, train_iter, test_iter, num_epochs, loss, trainer, device)
```

```{.python .input}
#@tab pytorch
net = net_fn()
trainer = torch.optim.SGD(net.parameters(), lr)
train(net, train_iter, test_iter, num_epochs, loss, trainer, device,
      scheduler)
```

```{.python .input}
#@tab tensorflow
train(net, train_iter, test_iter, num_epochs, lr,
      custom_callback=LearningRateScheduler(scheduler))
```

これは以前よりかなりうまくいきました。注目すべき点が二つあります。まず、曲線が以前よりかなり滑らかになりました。第二に、過学習が少なくなりました。残念ながら、なぜ特定の戦略が理論上*過学習を減らす*のかは、まだ十分に解明されていません。ステップサイズが小さいほど、ゼロに近いパラメータに到達しやすくなり、その結果としてより単純になる、という議論はあります。しかし、実際には早期停止しているわけではなく、単に学習率を穏やかに下げているだけなので、この現象を完全には説明できません。

## ポリシー

学習率スケジューラのあらゆる種類を網羅することはできませんが、以下ではよく使われるポリシーを簡単に概観します。一般的な選択肢としては、多項式減衰と区分定数スケジュールがあります。それに加えて、コサイン学習率スケジュールは、いくつかの問題で経験的によく機能することが分かっています。最後に、問題によっては、大きな学習率を使う前に最適化器をウォームアップすることが有益です。

### Factor Scheduler

多項式減衰の代わりに、乗法的な減衰、すなわち $\alpha \in (0, 1)$ に対して $\eta_{t+1} \leftarrow \eta_t \cdot \alpha$ を用いる方法があります。学習率が妥当な下限を超えて減衰しないようにするため、更新式はしばしば $\eta_{t+1} \leftarrow \mathop{\mathrm{max}}(\eta_{\mathrm{min}}, \eta_t \cdot \alpha)$ に修正されます。

```{.python .input}
#@tab all
class FactorScheduler:
    def __init__(self, factor=1, stop_factor_lr=1e-7, base_lr=0.1):
        self.factor = factor
        self.stop_factor_lr = stop_factor_lr
        self.base_lr = base_lr

    def __call__(self, num_update):
        self.base_lr = max(self.stop_factor_lr, self.base_lr * self.factor)
        return self.base_lr

scheduler = FactorScheduler(factor=0.9, stop_factor_lr=1e-2, base_lr=2.0)
d2l.plot(d2l.arange(50), [scheduler(t) for t in range(50)])
```

これは MXNet では `lr_scheduler.FactorScheduler` オブジェクトという組み込みスケジューラでも実現できます。これには、ウォームアップ期間、ウォームアップのモード（線形または定数）、望ましい更新回数の上限など、いくつか追加のパラメータがあります。今後は、適宜組み込みスケジューラを使い、その機能についてここで説明することにします。図に示したように、必要なら自分でスケジューラを作るのはかなり簡単です。

### Multi Factor Scheduler

深層ネットワークの学習でよくある戦略は、学習率を区分的に一定に保ち、一定間隔で所定の量だけ下げることです。つまり、たとえば $s = \{5, 10, 20\}$ のように学習率を下げる時刻の集合が与えられたとき、$t \in s$ ならば $\eta_{t+1} \leftarrow \eta_t \cdot \alpha$ とします。各段階で半分にする場合は、次のように実装できます。

```{.python .input}
#@tab mxnet
scheduler = lr_scheduler.MultiFactorScheduler(step=[15, 30], factor=0.5,
                                              base_lr=0.5)
d2l.plot(d2l.arange(num_epochs), [scheduler(t) for t in range(num_epochs)])
```

```{.python .input}
#@tab pytorch
net = net_fn()
trainer = torch.optim.SGD(net.parameters(), lr=0.5)
scheduler = lr_scheduler.MultiStepLR(trainer, milestones=[15, 30], gamma=0.5)

def get_lr(trainer, scheduler):
    lr = scheduler.get_last_lr()[0]
    trainer.step()
    scheduler.step()
    return lr

d2l.plot(d2l.arange(num_epochs), [get_lr(trainer, scheduler)
                                  for t in range(num_epochs)])
```

```{.python .input}
#@tab tensorflow
class MultiFactorScheduler:
    def __init__(self, step, factor, base_lr):
        self.step = step
        self.factor = factor
        self.base_lr = base_lr

    def __call__(self, epoch):
        if epoch in self.step:
            self.base_lr = self.base_lr * self.factor
            return self.base_lr
        else:
            return self.base_lr

scheduler = MultiFactorScheduler(step=[15, 30], factor=0.5, base_lr=0.5)
d2l.plot(d2l.arange(num_epochs), [scheduler(t) for t in range(num_epochs)])
```

この区分定数の学習率スケジュールの背後にある直感は、重みベクトルの分布の観点で定常点に達するまで最適化を進め、その後でのみ学習率を下げることで、良い局所最小値のより高品質な近似を得るというものです。以下の例では、これがわずかにより良い解を生み出す様子を示しています。

```{.python .input}
#@tab mxnet
trainer = gluon.Trainer(net.collect_params(), 'sgd',
                        {'lr_scheduler': scheduler})
train(net, train_iter, test_iter, num_epochs, loss, trainer, device)
```

```{.python .input}
#@tab pytorch
train(net, train_iter, test_iter, num_epochs, loss, trainer, device,
      scheduler)
```

```{.python .input}
#@tab tensorflow
train(net, train_iter, test_iter, num_epochs, lr,
      custom_callback=LearningRateScheduler(scheduler))
```

### Cosine Scheduler

:citet:`Loshchilov.Hutter.2016` によって、やや意外なヒューリスティックが提案されました。これは、最初に学習率をあまり急激に下げたくないこと、さらに最後には非常に小さな学習率を使って解を「洗練」したいこと、という観察に基づいています。その結果、$t \in [0, T]$ の範囲で学習率が次のようなコサイン状のスケジュールになります。

$$\eta_t = \eta_T + \frac{\eta_0 - \eta_T}{2} \left(1 + \cos(\pi t/T)\right)$$


ここで $\eta_0$ は初期学習率、$\eta_T$ は時刻 $T$ における目標学習率です。さらに、$t > T$ では値を再び増やさず、単に $\eta_T$ に固定します。以下の例では、最大更新ステップを $T = 20$ に設定します。

```{.python .input}
#@tab mxnet
scheduler = lr_scheduler.CosineScheduler(max_update=20, base_lr=0.3,
                                         final_lr=0.01)
d2l.plot(d2l.arange(num_epochs), [scheduler(t) for t in range(num_epochs)])
```

```{.python .input}
#@tab pytorch, tensorflow
class CosineScheduler:
    def __init__(self, max_update, base_lr=0.01, final_lr=0,
               warmup_steps=0, warmup_begin_lr=0):
        self.base_lr_orig = base_lr
        self.max_update = max_update
        self.final_lr = final_lr
        self.warmup_steps = warmup_steps
        self.warmup_begin_lr = warmup_begin_lr
        self.max_steps = self.max_update - self.warmup_steps

    def get_warmup_lr(self, epoch):
        increase = (self.base_lr_orig - self.warmup_begin_lr) \
                       * float(epoch) / float(self.warmup_steps)
        return self.warmup_begin_lr + increase

    def __call__(self, epoch):
        if epoch < self.warmup_steps:
            return self.get_warmup_lr(epoch)
        if epoch <= self.max_update:
            self.base_lr = self.final_lr + (
                self.base_lr_orig - self.final_lr) * (1 + math.cos(
                math.pi * (epoch - self.warmup_steps) / self.max_steps)) / 2
        return self.base_lr

scheduler = CosineScheduler(max_update=20, base_lr=0.3, final_lr=0.01)
d2l.plot(d2l.arange(num_epochs), [scheduler(t) for t in range(num_epochs)])
```

コンピュータビジョンの文脈では、このスケジュールは*結果を改善する*ことがあります。ただし、そのような改善が保証されるわけではありません（下に示すとおりです）。

```{.python .input}
#@tab mxnet
trainer = gluon.Trainer(net.collect_params(), 'sgd',
                        {'lr_scheduler': scheduler})
train(net, train_iter, test_iter, num_epochs, loss, trainer, device)
```

```{.python .input}
#@tab pytorch
net = net_fn()
trainer = torch.optim.SGD(net.parameters(), lr=0.3)
train(net, train_iter, test_iter, num_epochs, loss, trainer, device,
      scheduler)
```

```{.python .input}
#@tab tensorflow
train(net, train_iter, test_iter, num_epochs, lr,
      custom_callback=LearningRateScheduler(scheduler))
```

### ウォームアップ

場合によっては、パラメータの初期化だけでは良い解を保証するのに十分ではありません。これは特に、最適化が不安定になりうる高度なネットワーク設計で問題になります。これに対処するには、最初のうち発散を防ぐために十分小さい学習率を選ぶ方法があります。残念ながら、これは進みが遅いことを意味します。逆に、最初から大きな学習率を使うと発散します。

このジレンマに対するかなり単純な解決策は、学習率が初期の最大値まで*増加*するウォームアップ期間を設け、その後は最適化過程の終わりまで学習率を下げていくことです。簡単のため、通常はこの目的に線形増加を用います。これにより、以下に示すようなスケジュールになります。

```{.python .input}
#@tab mxnet
scheduler = lr_scheduler.CosineScheduler(20, warmup_steps=5, base_lr=0.3,
                                         final_lr=0.01)
d2l.plot(np.arange(num_epochs), [scheduler(t) for t in range(num_epochs)])
```

```{.python .input}
#@tab pytorch, tensorflow
scheduler = CosineScheduler(20, warmup_steps=5, base_lr=0.3, final_lr=0.01)
d2l.plot(d2l.arange(num_epochs), [scheduler(t) for t in range(num_epochs)])
```

ネットワークが初期段階でよりよく収束することに注意してください（特に最初の 5 エポックでの性能を観察してください）。

```{.python .input}
#@tab mxnet
trainer = gluon.Trainer(net.collect_params(), 'sgd',
                        {'lr_scheduler': scheduler})
train(net, train_iter, test_iter, num_epochs, loss, trainer, device)
```

```{.python .input}
#@tab pytorch
net = net_fn()
trainer = torch.optim.SGD(net.parameters(), lr=0.3)
train(net, train_iter, test_iter, num_epochs, loss, trainer, device,
      scheduler)
```

```{.python .input}
#@tab tensorflow
train(net, train_iter, test_iter, num_epochs, lr,
      custom_callback=LearningRateScheduler(scheduler))
```

ウォームアップは任意のスケジューラに適用できます（コサインに限りません）。学習率スケジュールについてのより詳細な議論や、さらに多くの実験については :cite:`Gotmare.Keskar.Xiong.ea.2018` も参照してください。特に、ウォームアップ段階は非常に深いネットワークにおけるパラメータの発散量を抑えることが分かっています。これは直感的にも理解できます。というのも、学習の初期に進展しにくいネットワーク部分では、ランダム初期化によって大きな発散が生じると予想されるからです。

## まとめ

* 学習中に学習率を下げると、精度の向上や、（最も不可解なことに）モデルの過学習の減少につながることがあります。
* 進展が頭打ちになったときに学習率を段階的に下げる方法は、実際に有効です。本質的には、これにより適切な解へ効率よく収束し、その後でのみ学習率を下げることでパラメータの本質的な分散を減らします。
* コサインスケジューラは、いくつかのコンピュータビジョン問題で人気があります。このようなスケジューラの詳細は、例えば [GluonCV](http://gluon-cv.mxnet.io) を参照してください。
* 最適化の前にウォームアップ期間を設けると、発散を防げます。
* 最適化は深層学習において複数の目的を果たします。訓練目的関数の最小化だけでなく、最適化アルゴリズムや学習率スケジューリングの選択によって、（同じ訓練誤差でも）テストセット上での汎化や過学習の程度がかなり異なることがあります。

## 演習

1. 固定した学習率に対する最適化の挙動を実験してください。この方法で得られる最良のモデルは何ですか。
1. 学習率の減衰指数を変えると収束はどう変わりますか。実験では `PolyScheduler` を使うと便利です。
1. コサインスケジューラを大規模なコンピュータビジョン問題、たとえば ImageNet の学習に適用してください。他のスケジューラと比べて性能にどのような影響がありますか。
1. ウォームアップはどのくらいの長さにすべきでしょうか。
1. 最適化とサンプリングを結びつけられますか。まずは :citet:`Welling.Teh.2011` の確率的勾配ランジュバン動力学に関する結果を使ってみてください。\n
