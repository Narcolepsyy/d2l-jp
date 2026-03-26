{.python .input}
%load_ext d2lbook.tab
tab.interact_select(["pytorch"])
#required_libs("syne-tune[gpsearchers]==0.3.2")
```

# 非同期逐次半減法

:label:`sec_sh_async`

:numref:`sec_rs_async` で見たように、HPO はハイパーパラメータ構成の評価を複数のインスタンスに分散したり、1つのインスタンス上の複数の CPU/GPU に分散したりすることで高速化できます。しかし、ランダムサーチと比べると、分散環境で逐次半減法（SH）を非同期に実行するのは簡単ではありません。次にどの構成を実行するかを決める前に、まず現在の rung レベルにあるすべての観測結果を集める必要があるからです。これは各 rung レベルでワーカーを同期させることを意味します。たとえば、最下位の rung レベル $r_{\mathrm{min}}$ では、まず $N = \eta^K$ 個すべての構成を評価し、その後でその $\frac{1}{\eta}$ を次の rung レベルへ昇格させる必要があります。

どの分散システムでも、同期は通常ワーカーのアイドル時間を意味します。まず、ハイパーパラメータ構成間で学習時間に大きなばらつきがよく見られます。たとえば、層ごとのフィルタ数がハイパーパラメータだとすると、フィルタ数の少ないネットワークは多いネットワークよりも学習が早く終わるため、遅い処理に引きずられてワーカーが待機することになります。さらに、ある rung レベルのスロット数がワーカー数の倍数であるとは限らず、その場合には一部のワーカーが1バッチ分まるごと遊休になることさえあります。

図 :numref:`synchronous_sh` は、2つのワーカーを用いた $\eta=2$ の同期 SH のスケジューリングを示しています。まず Trial-0 と Trial-1 を1エポックずつ評価し、それらが終わるとすぐに次の2つの trial に進みます。Trial-2 が終わるのを待たなければならず、これには他の trial よりかなり長い時間がかかります。その後で初めて、最良の2つの trial、すなわち Trial-0 と Trial-3 を次の rung レベルへ昇格できます。これにより Worker-1 はアイドルになります。次に Rung 1 に進みます。ここでも Trial-3 は Trial-0 より時間がかかるため、Worker-0 に追加の待機時間が発生します。Rung-2 に到達すると、最良の trial である Trial-0 だけが残り、1つのワーカーしか使いません。その間に Worker-1 が遊休にならないようにするため、SH の多くの実装ではすでに次のラウンドへ進み、最初の rung で新しい trial（たとえば Trial-4）の評価を開始します。

![2つのワーカーを用いた同期逐次半減法。](../img/sync_sh.svg)
:label:`synchronous_sh`

非同期逐次半減法（ASHA） :cite:`li-arxiv18` は、SH を非同期並列シナリオに適応させたものです。ASHA の基本的な考え方は、現在の rung レベルで少なくとも $\eta$ 個の観測結果が集まったら、すぐに構成を次の rung レベルへ昇格させることです。この決定規則は、最適とは言えない昇格を招くことがあります。つまり、同じ rung レベルの他の多くと比べると、後から見ればあまり良くない構成を次の rung レベルへ昇格させてしまうことがあります。一方で、この方法によりすべての同期点を取り除けます。実際には、このような最初の段階での最適でない昇格は性能に与える影響が小さいことが多いです。これは、ハイパーパラメータ構成の順位付けが rung レベル間でかなり一貫していることが多いだけでなく、rung が時間とともに大きくなり、そのレベルでのメトリック値の分布をよりよく反映するようになるためでもあります。ワーカーが空いているのに昇格できる構成がない場合は、$r = r_{\mathrm{min}}$、すなわち最初の rung レベルで新しい構成を開始します。

:numref:`asha` は、同じ構成群に対する ASHA のスケジューリングを示しています。Trial-1 が終わると、2つの trial（すなわち Trial-0 と Trial-1）の結果を集め、そのうち良い方（Trial-0）をすぐに次の rung レベルへ昇格します。Trial-0 が rung 1 で終わった後は、さらに昇格を支えるにはその rung にある trial が少なすぎます。したがって、rung 0 に戻って Trial-3 を評価し続けます。Trial-3 が終わった時点では、Trial-2 はまだ保留中です。この時点で rung 0 では3つの trial が評価済みで、rung 1 では1つの trial がすでに評価済みです。Trial-3 は rung 0 で Trial-0 より成績が悪く、かつ $\eta=2$ なので、まだ新しい trial を昇格させることはできず、代わりに Worker-1 は Trial-4 を最初から開始します。しかし、Trial-2 が終わって Trial-3 より悪いスコアになると、後者が rung 1 へ昇格します。その後、rung 1 で2件の評価が集まったので、Trial-0 を rung 2 へ昇格できるようになります。同時に、Worker-1 は rung 0 で新しい trial（すなわち Trial-5）の評価を続けます。


![2つのワーカーを用いた非同期逐次半減法（ASHA）。](../img/asha.svg)
:label:`asha`

```{.python .input}
from d2l import torch as d2l
import logging
logging.basicConfig(level=logging.INFO)
import matplotlib.pyplot as plt
from syne_tune.config_space import loguniform, randint
from syne_tune.backend.python_backend import PythonBackend
from syne_tune.optimizer.baselines import ASHA
from syne_tune import Tuner, StoppingCriterion
from syne_tune.experiments import load_experiment
```

## 目的関数

:numref:`sec_rs_async` と同じ目的関数を使って *Syne Tune* を利用します。

```{.python .input  n=54}
def hpo_objective_lenet_synetune(learning_rate, batch_size, max_epochs):
    from d2l import torch as d2l
    from syne_tune import Reporter

    model = d2l.LeNet(lr=learning_rate, num_classes=10)
    trainer = d2l.HPOTrainer(max_epochs=1, num_gpus=1)
    data = d2l.FashionMNIST(batch_size=batch_size)
    model.apply_init([next(iter(data.get_dataloader(True)))[0]], d2l.init_cnn)
    report = Reporter()
    for epoch in range(1, max_epochs + 1):
        if epoch == 1:
            # Initialize the state of Trainer
            trainer.fit(model=model, data=data)
        else:
            trainer.fit_epoch()
        validation_error = d2l.numpy(trainer.validation_error().cpu())
        report(epoch=epoch, validation_error=float(validation_error))
```

また、前と同じ構成空間を使います。

```{.python .input  n=55}
min_number_of_epochs = 2
max_number_of_epochs = 10
eta = 2

config_space = {
    "learning_rate": loguniform(1e-2, 1),
    "batch_size": randint(32, 256),
    "max_epochs": max_number_of_epochs,
}
initial_config = {
    "learning_rate": 0.1,
    "batch_size": 128,
}
```

## 非同期スケジューラ

まず、trial を並列に評価するワーカー数を定義します。また、総 wall-clock 時間の上限を定めて、ランダムサーチをどれだけ実行するかも指定する必要があります。

```{.python .input  n=56}
n_workers = 2  # Needs to be <= the number of available GPUs
max_wallclock_time = 12 * 60  # 12 minutes
```

ASHA を実行するコードは、非同期ランダムサーチで行ったことの単純な変形です。

```{.python .input  n=56}
mode = "min"
metric = "validation_error"
resource_attr = "epoch"

scheduler = ASHA(
    config_space,
    metric=metric,
    mode=mode,
    points_to_evaluate=[initial_config],
    max_resource_attr="max_epochs",
    resource_attr=resource_attr,
    grace_period=min_number_of_epochs,
    reduction_factor=eta,
)
```

ここで `metric` と `resource_attr` は `report` コールバックで使われるキー名を指定し、`max_resource_attr` は目的関数への入力のうちどれが $r_{\mathrm{max}}$ に対応するかを表します。さらに、`grace_period` は $r_{\mathrm{min}}$ を与え、`reduction_factor` は $\eta$ です。前と同様に Syne Tune を実行できます（約12分かかります）。

```{.python .input  n=57}
trial_backend = PythonBackend(
    tune_function=hpo_objective_lenet_synetune,
    config_space=config_space,
)

stop_criterion = StoppingCriterion(max_wallclock_time=max_wallclock_time)
tuner = Tuner(
    trial_backend=trial_backend,
    scheduler=scheduler,
    stop_criterion=stop_criterion,
    n_workers=n_workers,
    print_update_interval=int(max_wallclock_time * 0.6),
)
tuner.run()
```

ここでは、性能の低い trial を早期停止する ASHA の変種を実行していることに注意してください。これは、 :numref:`sec_mf_hpo_sh` における実装とは異なります。そちらでは、各学習ジョブは固定の `max_epochs` で開始されます。後者の場合、10エポックすべてに到達する性能の良い trial は、まず1エポック、次に2エポック、その次に4エポック、さらに8エポックと、毎回最初から学習し直す必要があります。この種の一時停止・再開型のスケジューリングは、各エポック後に学習状態をチェックポイント化することで効率的に実装できますが、ここではその追加の複雑さは避けます。実験が終わったら、結果を取得してプロットできます。

```{.python .input  n=59}
d2l.set_figsize()
e = load_experiment(tuner.name)
e.plot()
```

## 最適化過程の可視化

もう一度、各 trial の学習曲線を可視化します（プロット中の各色は1つの trial を表します）。これを :numref:`sec_rs_async` の非同期ランダムサーチと比較してください。 :numref:`sec_mf_hpo` で逐次半減法について見たように、trial の大半は1エポックまたは2エポック（$r_{\mathrm{min}}$ または $\eta * r_{\mathrm{min}}$）で停止します。しかし、各 trial は1エポックあたりに必要な時間が異なるため、同じ地点では停止しません。もし ASHA ではなく標準的な逐次半減法を実行するなら、構成を次の rung レベルへ昇格させる前に、ワーカーを同期させる必要があります。

```{.python .input  n=60}
d2l.set_figsize([6, 2.5])
results = e.results
for trial_id in results.trial_id.unique():
    df = results[results["trial_id"] == trial_id]
    d2l.plt.plot(
        df["st_tuner_time"],
        df["validation_error"],
        marker="o"
    )
d2l.plt.xlabel("wall-clock time")
d2l.plt.ylabel("objective function")
```

## まとめ

ランダムサーチと比べると、逐次半減法を非同期分散環境で実行するのはそれほど自明ではありません。同期点を避けるために、多少誤ったものを昇格させることになっても、できるだけ速く構成を次の rung レベルへ昇格させます。実際には、これは通常それほど大きな悪影響を及ぼしません。非同期スケジューリングと同期スケジューリングの差による利得は、最適でない意思決定による損失よりも通常ははるかに大きいからです。\n
