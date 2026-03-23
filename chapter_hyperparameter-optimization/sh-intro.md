{.python .input}
%load_ext d2lbook.tab
tab.interact_select(["pytorch"])
```

# マルチフィデリティ・ハイパーパラメータ最適化
:label:`sec_mf_hpo`

ニューラルネットワークの学習は、中規模のデータセットであっても高コストになり得ます。
構成空間（:numref:`sec_intro_config_spaces`）によっては、
ハイパーパラメータ最適化では、性能の良いハイパーパラメータ構成を見つけるために数十回から数百回の関数評価が必要になります。
:numref:`sec_rs_async` で見たように、並列リソースを活用することでHPOの全体的な壁時計時間を大幅に短縮できますが、必要な総計算量は減りません。

この節では、ハイパーパラメータ構成の評価をどのように高速化できるかを示します。
ランダムサーチのような手法では、各ハイパーパラメータ評価に同じ量のリソース（たとえばエポック数や学習データ点数）を割り当てます。
:numref:`img_samples_lc` は、異なるハイパーパラメータ構成で学習したニューラルネットワーク群の学習曲線を示しています。
数エポック後には、性能の良い構成と不十分な構成を視覚的に区別できるようになります。
しかし、学習曲線にはノイズがあるため、最良の構成を特定するには依然として100エポック分の完全な評価が必要かもしれません。

![ランダムなハイパーパラメータ構成の学習曲線](../img/samples_lc.svg)
:label:`img_samples_lc`

マルチフィデリティ・ハイパーパラメータ最適化では、有望な構成により多くのリソースを割り当て、性能の悪い構成の評価は早期に打ち切ります。
これにより、同じ総リソース量でより多くの構成を試せるため、最適化プロセスが高速化されます。

より形式的には、 :numref:`sec_definition_hpo` における定義を拡張し、目的関数 $f(\mathbf{x}, r)$ に追加の入力
$r \in [r_{\mathrm{min}}, r_{max}]$ を導入します。これは、構成 $\mathbf{x}$ の評価に対してどれだけのリソースを費やすかを指定します。
ここでは、誤差 $f(\mathbf{x}, r)$ は $r$ とともに減少し、計算コスト $c(\mathbf{x}, r)$ は増加すると仮定します。
通常、$r$ はニューラルネットワークの学習エポック数を表しますが、学習サブセットのサイズや交差検証の分割数でもかまいません。

```{.python .input}
%%tab pytorch
from d2l import torch as d2l
import numpy as np
from scipy import stats
from collections import defaultdict
d2l.set_figsize()
```

## 逐次半減法
:label:`sec_mf_hpo_sh`

マルチフィデリティ設定にランダムサーチを適応させる最も簡単な方法の1つが、*逐次半減法* :cite:`jamieson-aistats16,karnin-icml13` です。
基本的な考え方は、たとえば構成空間からランダムにサンプルした $N$ 個の構成から始め、それぞれを $r_{\mathrm{min}}$ エポックだけ学習することです。
その後、性能の悪い試行の一部を捨て、残ったものをより長く学習します。
このプロセスを繰り返すことで、より少ない試行がより長く実行され、少なくとも1つの試行が $r_{max}$ エポックに到達するまで続きます。

より形式的には、最小予算 $r_{\mathrm{min}}$（たとえば1エポック）、最大予算 $r_{max}$（たとえば前の例の `max_epochs`）、および半減定数 $\eta\in\{2, 3, \dots\}$ を考えます。
簡単のために、$r_{max} = r_{\mathrm{min}} \eta^K$、ただし $K \in \mathbb{I}$ と仮定します。
このとき初期構成数は $N = \eta^K$ です。
段（rung）の集合を $\mathcal{R} = \{ r_{\mathrm{min}}, r_{\mathrm{min}}\eta, r_{\mathrm{min}}\eta^2, \dots, r_{max} \}$ と定義します。

逐次半減法の1ラウンドは次のように進みます。
まず $N$ 個の試行を最初の段 $r_{\mathrm{min}}$ まで実行します。
検証誤差を並べ替え、上位 $1 / \eta$ の割合（これは $\eta^{K-1}$ 個の構成に相当します）を残し、それ以外はすべて破棄します。
生き残った試行は次の段（$r_{\mathrm{min}}\eta$ エポック）まで学習され、この過程を繰り返します。
各段で試行の $1 / \eta$ が生き残り、その学習は $\eta$ 倍大きい予算で継続されます。
この $N$ の選び方では、最終的に1つの試行だけが完全な予算 $r_{max}$ まで学習されます。
このような逐次半減法の1ラウンドが終わると、新しい初期構成集合で次のラウンドを開始し、総予算を使い切るまで繰り返します。

![ランダムなハイパーパラメータ構成の学習曲線。](../img/sh.svg)

逐次半減法を実装するために、 :numref:`sec_api_hpo` の `HPOScheduler` 基底クラスを継承し、汎用の `HPOSearcher` オブジェクトが構成をサンプルできるようにします（以下の例では `RandomSearcher` になります）。
さらに、ユーザーは最小リソース $r_{\mathrm{min}}$、最大リソース $r_{max}$、および $\eta$ を入力として渡す必要があります。
スケジューラ内部では、現在の段 $r_i$ でまだ評価が必要な構成のキューを保持します。
次の段へ進むたびに、このキューを更新します。

```{.python .input  n=2}
class SuccessiveHalvingScheduler(d2l.HPOScheduler):  #@save
    def __init__(self, searcher, eta, r_min, r_max, prefact=1):
        self.save_hyperparameters()
        # Compute K, which is later used to determine the number of configurations
        self.K = int(np.log(r_max / r_min) / np.log(eta))
        # Define the rungs
        self.rung_levels = [r_min * eta ** k for k in range(self.K + 1)]
        if r_max not in self.rung_levels:
            # The final rung should be r_max
            self.rung_levels.append(r_max)
            self.K += 1
        # Bookkeeping
        self.observed_error_at_rungs = defaultdict(list)
        self.all_observed_error_at_rungs = defaultdict(list)
        # Our processing queue
        self.queue = []
```

最初はキューは空で、そこに $n = \textrm{prefact} \cdot \eta^{K}$ 個の構成を入れます。
それらはまず最小の段 $r_{\mathrm{min}}$ で評価されます。
ここで $\textrm{prefact}$ は、別の文脈でこのコードを再利用できるようにするためのものです。
この節では、$\textrm{prefact} = 1$ に固定します。
リソースが利用可能になり、`HPOTuner` オブジェクトが `suggest` 関数を呼び出すたびに、キューから1つの要素を返します。
逐次半減法の1ラウンドが完了し、つまり生き残ったすべての構成を最高リソースレベル $r_{max}$ まで評価し終えてキューが空になったら、新しいランダムサンプルの構成集合でプロセス全体を再開します。

```{.python .input  n=12}
%%tab pytorch
@d2l.add_to_class(SuccessiveHalvingScheduler)  #@save
def suggest(self):
    if len(self.queue) == 0:
        # Start a new round of successive halving
        # Number of configurations for the first rung:
        n0 = int(self.prefact * self.eta ** self.K)
        for _ in range(n0):
            config = self.searcher.sample_configuration()
            config["max_epochs"] = self.r_min  # Set r = r_min
            self.queue.append(config)
    # Return an element from the queue
    return self.queue.pop()
```

新しいデータ点を収集したら、まず searcher モジュールを更新します。
その後、現在の段のすべてのデータ点をすでに収集し終えたかを確認します。
もしそうなら、すべての構成を並べ替え、上位 $\frac{1}{\eta}$ の構成をキューに入れます。

```{.python .input  n=4}
%%tab pytorch
@d2l.add_to_class(SuccessiveHalvingScheduler)  #@save
def update(self, config: dict, error: float, info=None):
    ri = int(config["max_epochs"])  # Rung r_i
    # Update our searcher, e.g if we use Bayesian optimization later
    self.searcher.update(config, error, additional_info=info)
    self.all_observed_error_at_rungs[ri].append((config, error))
    if ri < self.r_max:
        # Bookkeeping
        self.observed_error_at_rungs[ri].append((config, error))
        # Determine how many configurations should be evaluated on this rung
        ki = self.K - self.rung_levels.index(ri)
        ni = int(self.prefact * self.eta ** ki)
        # If we observed all configuration on this rung r_i, we estimate the
        # top 1 / eta configuration, add them to queue and promote them for
        # the next rung r_{i+1}
        if len(self.observed_error_at_rungs[ri]) >= ni:
            kiplus1 = ki - 1
            niplus1 = int(self.prefact * self.eta ** kiplus1)
            best_performing_configurations = self.get_top_n_configurations(
                rung_level=ri, n=niplus1
            )
            riplus1 = self.rung_levels[self.K - kiplus1]  # r_{i+1}
            # Queue may not be empty: insert new entries at the beginning
            self.queue = [
                dict(config, max_epochs=riplus1)
                for config in best_performing_configurations
            ] + self.queue
            self.observed_error_at_rungs[ri] = []  # Reset
```

構成は、現在の段で観測された性能に基づいて並べ替えられます。

```{.python .input  n=4}
%%tab pytorch

@d2l.add_to_class(SuccessiveHalvingScheduler)  #@save
def get_top_n_configurations(self, rung_level, n):
    rung = self.observed_error_at_rungs[rung_level]
    if not rung:
        return []
    sorted_rung = sorted(rung, key=lambda x: x[1])
    return [x[0] for x in sorted_rung[:n]]
```

逐次半減法がニューラルネットワークの例でどのように機能するか見てみましょう。
ここでは $r_{\mathrm{min}} = 2$、$\eta = 2$、$r_{max} = 10$ とし、段のレベルは
$2, 4, 8, 10$ になります。

```{.python .input  n=5}
min_number_of_epochs = 2
max_number_of_epochs = 10
eta = 2
num_gpus=1

config_space = {
    "learning_rate": stats.loguniform(1e-2, 1),
    "batch_size": stats.randint(32, 256),
}
initial_config = {
    "learning_rate": 0.1,
    "batch_size": 128,
}
```

スケジューラを新しい `SuccessiveHalvingScheduler` に置き換えるだけです。

```{.python .input  n=14}
searcher = d2l.RandomSearcher(config_space, initial_config=initial_config)
scheduler = SuccessiveHalvingScheduler(
    searcher=searcher,
    eta=eta,
    r_min=min_number_of_epochs,
    r_max=max_number_of_epochs,
)
tuner = d2l.HPOTuner(
    scheduler=scheduler,
    objective=d2l.hpo_objective_lenet,
)
tuner.run(number_of_trials=30)
```

評価したすべての構成の学習曲線を可視化できます。
ほとんどの構成は早期に打ち切られ、より性能の良い構成だけが $r_{max}$ まで生き残ります。
これを、すべての構成に $r_{max}$ を割り当てる通常のランダムサーチと比較してください。

```{.python .input  n=19}
for rung_index, rung in scheduler.all_observed_error_at_rungs.items():
    errors = [xi[1] for xi in rung]
    d2l.plt.scatter([rung_index] * len(errors), errors)
d2l.plt.xlim(min_number_of_epochs - 0.5, max_number_of_epochs + 0.5)
d2l.plt.xticks(
    np.arange(min_number_of_epochs, max_number_of_epochs + 1),
    np.arange(min_number_of_epochs, max_number_of_epochs + 1)
)
d2l.plt.ylabel("validation error")
d2l.plt.xlabel("epochs")
```

最後に、`SuccessiveHalvingScheduler` の実装には少し複雑な点があることに注意してください。
あるワーカーがジョブを実行可能で、現在の段がほぼ埋まっている一方で、別のワーカーがまだ評価中だとします。
このワーカーからのメトリック値がまだないため、次の段を開くための上位 $1 / \eta$ の割合を決定できません。
一方で、空いているワーカーにジョブを割り当てて、アイドル状態のままにしたくはありません。
そこで、私たちは新しい逐次半減法のラウンドを開始し、そのワーカーをそこでの最初の試行に割り当てます。
ただし、`update` である段が完了したら、新しい構成をキューの先頭に挿入し、次のラウンドの構成よりも優先されるようにしています。

## まとめ

この節では、マルチフィデリティ・ハイパーパラメータ最適化の概念を導入しました。
ここでは、完全なエポック数での検証誤差の代用として、あるエポック数まで学習した後の検証誤差のような、安価に評価できる目的関数の近似にアクセスできると仮定します。
マルチフィデリティ・ハイパーパラメータ最適化は、壁時計時間を短縮するだけでなく、HPOに必要な総計算量そのものを削減できます。

単純でありながら効率的なマルチフィデリティHPOアルゴリズムである逐次半減法を実装し、評価しました。


:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/12094)
:end_tab:\n