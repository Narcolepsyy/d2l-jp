{.python .input}
%load_ext d2lbook.tab
tab.interact_select(["pytorch"])
#required_libs("syne-tune[gpsearchers]==0.3.2")
```

# 非同期逐次半減法

:label:`sec_sh_async`

:numref:`sec_rs_async` で述べたように、HPO はハイパーパラメータ構成の評価を複数のインスタンスに分散したり、1つのインスタンス上の複数の CPU/GPU に分散したりすることで高速化が可能である。しかし、ランダムサーチと比較すると、分散環境で逐次半減法（SH）を非同期に実行することは容易ではない。次にどの構成を実行するかを決定する前に、まず現在の rung レベルにあるすべての観測結果を集める必要があるためである。これは各 rung レベルでワーカーを同期させることを意味する。たとえば、最下位の rung レベル $r_{\mathrm{min}}$ では、まず $N = \eta^K$ 個すべての構成を評価し、その後でその $\frac{1}{\eta}$ を次の rung レベルへ昇格させる必要がある。

どの分散システムにおいても、同期は通常ワーカーのアイドル時間を意味する。まず、ハイパーパラメータ構成間で学習時間に大きなばらつきが頻繁に観察される。たとえば、層ごとのフィルタ数がハイパーパラメータである場合、フィルタ数の少ないネットワークは多いネットワークよりも学習が早く終了するため、遅い処理に引きずられてワーカーが待機を余儀なくされる。さらに、ある rung レベルのスロット数がワーカー数の倍数であるとは限らず、その場合には一部のワーカーが1バッチ分まるごと遊休となる場合もある。

図 :numref:`synchronous_sh` は、2つのワーカーを用いた $\eta=2$ の同期 SH のスケジューリングを示す。まず Trial-0 と Trial-1 を1エポックずつ評価し、それらが終了するとすぐに次の2つの trial へ進む。Trial-2 が終了するのを待たなければならず、これには他の trial よりかなり長い時間がかかる。その後で初めて、最良の2つの trial、すなわち Trial-0 と Trial-3 を次の rung レベルへ昇格可能となる。これにより Worker-1 はアイドル状態となる。次に Rung 1 へ進む。ここでも Trial-3 は Trial-0 より時間がかかるため、Worker-0 に追加の待機時間が発生する。Rung-2 に到達すると、最良の trial である Trial-0 だけが残り、1つのワーカーのみが使用される。その間に Worker-1 が遊休状態を避けるため、SH の多くの実装ではすでに次のラウンドへ進み、最初の rung で新しい trial（たとえば Trial-4）の評価を開始する。

![2つのワーカーを用いた同期逐次半減法。](../img/sync_sh.svg)
:label:`synchronous_sh`

非同期逐次半減法（ASHA） :cite:`li-arxiv18` は、SH を非同期並列シナリオに適応させたものである。ASHA の基本的な考え方は、現在の rung レベルで少なくとも $\eta$ 個の観測結果が集まったら、すぐに構成を次の rung レベルへ昇格させることである。この決定規則は、最適とは言えない昇格を招く可能性がある。つまり、同じ rung レベルの他の多くと比較すると、後から見ればあまり良くない構成を次の rung レベルへ昇格させてしまう可能性がある。一方で、この方法によりすべての同期点を除去できる。実際には、このような最初の段階での最適でない昇格が性能に与える影響は小さいことが多い。これは、ハイパーパラメータ構成の順位付けが rung レベル間でかなり一貫していることが多いだけでなく、rung が時間とともに大きくなり、そのレベルでのメトリック値の分布をよりよく反映するためである。ワーカーが空いているのに昇格できる構成がない場合は、$r = r_{\mathrm{min}}$、すなわち最初の rung レベルで新しい構成を開始する。

:numref:`asha` は、同じ構成群に対する ASHA のスケジューリングを示す。Trial-1 が終了すると、2つの trial（すなわち Trial-0 と Trial-1）の結果を集め、そのうち良い方（Trial-0）をすぐに次の rung レベルへ昇格させる。Trial-0 が rung 1 で終了した後、さらに昇格を支えるにはその rung にある trial が不足している。したがって、rung 0 に戻って Trial-3 の評価を継続する。Trial-3 が終了した時点では、Trial-2 はまだ保留中である。この時点で rung 0 では3つの trial が評価済みであり、rung 1 では1つの trial がすでに評価済みである。Trial-3 は rung 0 で Trial-0 より成績が悪く、かつ $\eta=2$ なので、まだ新しい trial を昇格させることはできず、代わりに Worker-1 は Trial-4 を最初から開始する。しかし、Trial-2 が終了して Trial-3 より悪いスコアとなった場合、後者が rung 1 へ昇格する。その後、rung 1 で2件の評価が集まったので、Trial-0 を rung 2 へ昇格が可能となる。同時に、Worker-1 は rung 0 で新しい trial（すなわち Trial-5）の評価を継続する。


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

:numref:`sec_rs_async` と同じ目的関数を *Syne Tune* で利用する。

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

また、以前と同じ構成空間を用いる。

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

まず、trial を並列に評価するワーカー数を定義する。また、総 wall-clock 時間の上限を定めて、ランダムサーチをどれだけ実行するかを指定する必要がある。

```{.python .input  n=56}
n_workers = 2  # Needs to be <= the number of available GPUs
max_wallclock_time = 12 * 60  # 12 minutes
```

ASHA を実行するコードは、非同期ランダムサーチで行ったことの単純な変形である。

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

ここで `metric` と `resource_attr` は `report` コールバックで用いられるキー名を指定し、`max_resource_attr` は目的関数への入力のうちどれが $r_{\mathrm{max}}$ に対応するかを示す。さらに、`grace_period` は $r_{\mathrm{min}}$ を与え、`reduction_factor` は $\eta$ である。以前と同様に Syne Tune を実行可能である（約12分を要する）。

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

ここでは、性能の低い trial を早期停止する ASHA の変種を実行していることに留意されたい。これは、 :numref:`sec_mf_hpo_sh` における実装とは異なる。そちらでは、各学習ジョブは固定の `max_epochs` で開始される。後者の場合、10エポックすべてに到達する性能の良い trial は、まず1エポック、次に2エポック、その次に4エポック、さらに8エポックと、毎回最初から学習し直す必要がある。この種の一時停止・再開型のスケジューリングは、各エポック後に学習状態をチェックポイント化することで効率的に実装可能であるが、ここではその追加の複雑さは避ける。実験が終了したら、結果を取得してプロット可能である。

```{.python .input  n=59}
d2l.set_figsize()
e = load_experiment(tuner.name)
e.plot()
```

## 最適化過程の可視化

再度、各 trial の学習曲線を可視化する（プロット中の各色は1つの trial を表す）。これを :numref:`sec_rs_async` の非同期ランダムサーチと比較されたい。 :numref:`sec_mf_hpo` で逐次半減法について述べたように、trial の大半は1エポックまたは2エポック（$r_{\mathrm{min}}$ または $\eta * r_{\mathrm{min}}$）で停止する。しかし、各 trial は1エポックあたりに必要な時間が異なるため、同じ時点では停止しない。もし ASHA ではなく標準的な逐次半減法を実行する場合、構成を次の rung レベルへ昇格させる前に、ワーカーを同期させる必要がある。

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

ランダムサーチと比べると、逐次半減法を非同期分散環境で実行するのはそれほど自明ではない。同期点を避けるために、多少誤ったものを昇格させることになっても、できるだけ速く構成を次の rung レベルへ昇格させる。実際には、これは通常それほど大きな悪影響を及ぼさない。非同期スケジューリングと同期スケジューリングの差による利得は、最適でない意思決定による損失よりも通常ははるかに大きいからである。
