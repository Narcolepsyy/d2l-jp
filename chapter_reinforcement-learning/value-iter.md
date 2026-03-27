{.python .input}
%load_ext d2lbook.tab
tab.interact_select(["pytorch"])
#required_libs("setuptools==66", "wheel==0.38.4", "gym==0.21.0")
```

# 値反復
:label:`sec_valueiter`

この節では、軌跡の *リターン* を最大化するために、各状態でロボットが選ぶべき最良の行動をどのように決めるかを議論する。Value Iteration と呼ばれるアルゴリズムを説明し、凍った湖の上を移動するシミュレートされたロボットに対してそれを実装する。

## 確率的方策

$\pi(a \mid s)$ で表される確率的方策（略して方策）は、状態 $s \in \mathcal{S}$ が与えられたときの行動 $a \in \mathcal{A}$ に関する条件付き分布であり、$\pi(a \mid s) \equiv P(a \mid s)$ である。例として、ロボットに4つの行動 $\mathcal{A}=$ {左へ進む, 下へ進む, 右へ進む, 上へ進む} があるとする。このような行動集合 $\mathcal{A}$ に対する状態 $s \in \mathcal{S}$ での方策はカテゴリ分布であり、4つの行動の確率は $[0.4, 0.2, 0.1, 0.3]$ かもしれない。別の状態 $s' \in \mathcal{S}$ では、同じ4つの行動に対する確率 $\pi(a \mid s')$ は $[0.1, 0.1, 0.2, 0.6]$ かもしれない。任意の状態 $s$ に対して $\sum_a \pi(a \mid s) = 1$ でなければならないことに注意しよ。決定論的方策は確率的方策の特殊な場合であり、$\pi(a \mid s)$ がある特定の1つの行動にのみ非ゼロ確率を与えるものである。たとえば、4つの行動の例では $[1, 0, 0, 0]$ である。

記法を煩雑にしないため、以後はしばしば $\pi(a \mid s)$ の代わりに条件付き分布を $\pi(s)$ と書く。

## 価値関数

ここで、ロボットが状態 $s_0$ から始まり、各時刻でまず方策から行動をサンプルし $a_t \sim \pi(s_t)$、その行動を実行して次の状態 $s_{t+1}$ に遷移すると考える。軌跡 $\tau = (s_0, a_0, r_0, s_1, a_1, r_1, \ldots)$ は、中間時刻にどの行動 $a_t$ がサンプルされるかによって変わりえる。このようなすべての軌跡の平均 *リターン* $R(\tau) = \sum_{t=0}^\infty \gamma^t r(s_t, a_t)$ を
$$V^\pi(s_0) = E_{a_t \sim \pi(s_t)} \Big[ R(\tau) \Big] = E_{a_t \sim \pi(s_t)} \Big[ \sum_{t=0}^\infty \gamma^t r(s_t, a_t) \Big],$$
と定義する。

ここで $s_{t+1} \sim P(s_{t+1} \mid s_t, a_t)$ はロボットの次状態であり、$r(s_t, a_t)$ は時刻 $t$ に状態 $s_t$ で行動 $a_t$ を取ることで得られる即時報酬である。これは方策 $\pi$ に対する「価値関数」と呼ばれる。簡単に言えば、方策 $\pi$ に対する状態 $s_0$ の価値、すなわち $V^\pi(s_0)$ は、ロボットが状態 $s_0$ から始まり、各時刻に方策 $\pi$ に従って行動したときに得られる、期待 $\gamma$-割引 *リターン* である。

次に、軌跡を2つの段階に分ける。(i) 第1段階は、行動 $a_0$ を取ることで $s_0 \to s_1$ となる部分、(ii) 第2段階は、その後の軌跡 $\tau' = (s_1, a_1, r_1, \ldots)$ である。強化学習におけるすべてのアルゴリズムの背後にある重要な考え方は、状態 $s_0$ の価値は、第1段階で得られる平均報酬と、ありうる次状態 $s_1$ にわたって平均した価値関数として書けるということである。これは非常に直感的で、マルコフ仮定から生じる。すなわち、現在の状態からの平均リターンは、次状態からの平均リターンと、次状態へ移ることで得られる平均報酬の和である。数学的には、2つの段階を次のように書ける。

$$V^\pi(s_0) = r(s_0, a_0) + \gamma\ E_{a_0 \sim \pi(s_0)} \Big[ E_{s_1 \sim P(s_1 \mid s_0, a_0)} \Big[ V^\pi(s_1) \Big] \Big].$$
:eqlabel:`eq_dynamic_programming`

この分解は非常に強力である。これは動的計画法の原理の基礎であり、すべての強化学習アルゴリズムはこれに基づいている。第2段階には2つの期待値があることに注意しよ。1つは確率的方策を用いて第1段階で取る行動 $a_0$ の選択に関する期待、もう1つは選ばれた行動から得られる可能な状態 $s_1$ に関する期待である。マルコフ決定過程（MDP）の遷移確率を用いると、:eqref:`eq_dynamic_programming` は次のように書ける。

$$V^\pi(s) = \sum_{a \in \mathcal{A}} \pi(a \mid s) \Big[ r(s,  a) + \gamma\  \sum_{s' \in \mathcal{S}} P(s' \mid s, a) V^\pi(s') \Big];\ \textrm{for all } s \in \mathcal{S}.$$
:eqlabel:`eq_dynamic_programming_val`

ここで重要なのは、上の等式がすべての状態 $s \in \mathcal{S}$ で成り立つことである。なぜなら、その状態から始まる任意の軌跡を考え、それを2つの段階に分解できるからである。

## 行動価値関数

実装では、しばしば「行動価値」関数と呼ばれる量を保持すると便利である。これは価値関数と密接に関連した量である。これは、$s_0$ から始まる軌跡の平均 *リターン* であり、ただし第1段階の行動を $a_0$ に固定したものとして定義される。

$$Q^\pi(s_0, a_0) = r(s_0, a_0) + E_{a_t \sim \pi(s_t)} \Big[ \sum_{t=1}^\infty \gamma^t r(s_t, a_t) \Big],$$

期待値の内側の和が $t=1,\ldots,\infty$ から始まっているのは、この場合、第1段階の報酬が固定されているからである。ここでも軌跡を2つに分けて、次のように書ける。

$$Q^\pi(s, a) = r(s, a) + \gamma \sum_{s' \in \mathcal{S}} P(s' \mid s, a) \sum_{a' \in \mathcal{A}} \pi(a' \mid s')\ Q^\pi(s', a');\ \textrm{ for all } s \in \mathcal{S}, a \in \mathcal{A}.$$
:eqlabel:`eq_dynamic_programming_q`

これは行動価値関数に対する :eqref:`eq_dynamic_programming_val` の対応物である。

## 最適確率的方策

価値関数も行動価値関数も、ロボットが選ぶ方策に依存する。次に、最大の平均 *リターン* を達成する「最適方策」を考える。
$$\pi^* = \underset{\pi}{\mathrm{argmax}} V^\pi(s_0).$$

ロボットが取りうるすべての確率的方策の中で、最適方策 $\pi^*$ は、状態 $s_0$ から始まる軌跡に対して最大の平均割引 *リターン* を達成する。最適方策の価値関数と行動価値関数を、それぞれ $V^* \equiv V^{\pi^*}$ および $Q^* \equiv Q^{\pi^*}$ と表す。

任意の状態で方策の下で可能な行動が1つだけである決定論的方策を考えると、次が得られる。

$$\pi^*(s) = \underset{a \in \mathcal{A}}{\mathrm{argmax}} \Big[ r(s, a) + \gamma \sum_{s' \in \mathcal{S}} P(s' \mid s, a)\ V^*(s') \Big].$$

これを覚えるためのよい語呂合わせは、状態 $s$ における最適行動（決定論的方策の場合）は、第1段階の報酬 $r(s, a)$ と、第2段階でありうるすべての次状態 $s'$ にわたって平均した、次状態 $s'$ から始まる軌跡の平均 *リターン* の和を最大化するものだ、ということである。

## 動的計画法の原理

前節の :eqref:`eq_dynamic_programming` または :eqref:`eq_dynamic_programming_q` における導出は、それぞれ最適価値関数 $V^*$ または行動価値関数 $Q^*$ を計算するアルゴリズムに変えることができる。次が成り立つ。
$$ V^*(s) = \sum_{a \in \mathcal{A}} \pi^*(a \mid s) \Big[ r(s,  a) + \gamma\  \sum_{s' \in \mathcal{S}} P(s' \mid s, a) V^*(s') \Big];\ \textrm{for all } s \in \mathcal{S}.$$

決定論的な最適方策 $\pi^*$ では、状態 $s$ で取れる行動は1つだけなので、次のようにも書ける。

$$V^*(s) = \mathrm{argmax}_{a \in \mathcal{A}} \Big\{ r(s,a) + \gamma \sum_{s' \in \mathcal{S}} P(s' \mid s, a) V^*(s') \Big\}$$

すべての状態 $s \in \mathcal{S}$ に対して成り立つ。この等式は「動的計画法の原理」と呼ばれる :cite:`BellmanDPPaper,BellmanDPBook`。これは1950年代に Richard Bellman によって定式化され、「最適軌跡の残りの部分もまた最適である」と覚えることができる。

## 値反復

動的計画法の原理を、最適価値関数を求めるアルゴリズム、すなわち値反復に変えることができる。値反復の背後にある重要な考え方は、この等式を、異なる状態 $s \in \mathcal{S}$ における $V^*(s)$ を結びつける制約の集合として捉えることである。まず、すべての状態 $s \in \mathcal{S}$ に対して、価値関数を任意の値 $V_0(s)$ で初期化する。$k$ 回目の反復では、Value Iteration アルゴリズムは価値関数を次のように更新する。

$$V_{k+1}(s) = \max_{a \in \mathcal{A}} \Big\{ r(s,  a) + \gamma\  \sum_{s' \in \mathcal{S}} P(s' \mid s, a) V_k(s') \Big\};\ \textrm{for all } s \in \mathcal{S}.$$

$k \to \infty$ とすると、初期化 $V_0$ に依らず、Value Iteration アルゴリズムで推定される価値関数は最適価値関数に収束することがわかる。
$$V^*(s) = \lim_{k \to \infty} V_k(s);\ \textrm{for all states } s \in \mathcal{S}.$$

同じ Value Iteration アルゴリズムは、行動価値関数を用いて次のようにも等価に書ける。
$$Q_{k+1}(s, a) = r(s, a) + \gamma \max_{a' \in \mathcal{A}} \sum_{s' \in \mathcal{S}} P(s' \mid s, a) Q_k (s', a');\ \textrm{ for all } s \in \mathcal{S}, a \in \mathcal{A}.$$

この場合、すべての $s \in \mathcal{S}$ と $a \in \mathcal{A}$ に対して $Q_0(s, a)$ を任意の値で初期化する。やはり、すべての $s \in \mathcal{S}$ と $a \in \mathcal{A}$ に対して $Q^*(s, a) = \lim_{k \to \infty} Q_k(s, a)$ が成り立つ。

## 方策評価

値反復により、最適な決定論的方策 $\pi^*$ の最適価値関数、すなわち $V^{\pi^*}$ を計算できる。同様の反復更新を用いて、他の任意の、確率的である可能性もある方策 $\pi$ に対応する価値関数を計算することもできる。ここでも、すべての状態 $s \in \mathcal{S}$ に対して $V^\pi_0(s)$ を任意の値で初期化し、$k$ 回目の反復で次の更新を行う。

$$    V^\pi_{k+1}(s) = \sum_{a \in \mathcal{A}} \pi(a \mid s) \Big[ r(s,  a) + \gamma\  \sum_{s' \in \mathcal{S}} P(s' \mid s, a) V^\pi_k(s') \Big];\ \textrm{for all } s \in \mathcal{S}.$$

このアルゴリズムは方策評価として知られており、与えられた方策に対する価値関数を計算するのに役立つ。ここでも、$k \to \infty$ とすると、初期化 $V_0$ に依らず、これらの更新は正しい価値関数に収束することがわかる。

$$V^\pi(s) = \lim_{k \to \infty} V^\pi_k(s);\ \textrm{for all states } s \in \mathcal{S}.$$

方策 $\pi$ の行動価値関数 $Q^\pi(s, a)$ を計算するアルゴリズムも同様である。

## 値反復の実装
:label:`subsec_valueitercode`
次に、[Open AI Gym](https://gym.openai.com) の FrozenLake と呼ばれるナビゲーション問題に対して、値反復をどのように実装するかを示す。まず、次のコードに示すように環境を設定する必要がある。

```{.python .input}
%%tab all

%matplotlib inline
import numpy as np
import random
from d2l import torch as d2l

seed = 0  # Random number generator seed
gamma = 0.95  # Discount factor
num_iters = 10  # Number of iterations
random.seed(seed)  # Set the random seed to ensure results can be reproduced
np.random.seed(seed)

# Now set up the environment
env_info = d2l.make_env('FrozenLake-v1', seed=seed)
```

FrozenLake 環境では、ロボットは $4 \times 4$ のグリッド（これらが状態である）上を移動し、行動は「上」($\uparrow$)、「下」($\rightarrow$)、「左」($\leftarrow$)、「右」($\rightarrow$) である。環境にはいくつかの穴（H）のセルと凍結した（F）セル、そしてゴールセル（G）があり、これらはすべてロボットには未知である。問題を簡単にするため、ロボットの行動は確実である、すなわちすべての $s \in \mathcal{S}, a \in \mathcal{A}$ に対して $P(s' \mid s, a) = 1$ と仮定する。ロボットがゴールに到達すると、試行は終了し、行動に関係なく報酬 $1$ を受け取る。それ以外の任意の状態での報酬は、すべての行動に対して $0$ である。ロボットの目的は、与えられた開始位置（S）（これが $s_0$ である）からゴール位置（G）に到達する方策を学習し、*リターン* を最大化することである。

次の関数は値反復を実装する。`env_info` には MDP と環境に関する情報が含まれ、`gamma` は割引率である。

```{.python .input}
%%tab all

def value_iteration(env_info, gamma, num_iters):
    env_desc = env_info['desc']  # 2D array shows what each item means
    prob_idx = env_info['trans_prob_idx']
    nextstate_idx = env_info['nextstate_idx']
    reward_idx = env_info['reward_idx']
    num_states = env_info['num_states']
    num_actions = env_info['num_actions']
    mdp = env_info['mdp']

    V  = np.zeros((num_iters + 1, num_states))
    Q  = np.zeros((num_iters + 1, num_states, num_actions))
    pi = np.zeros((num_iters + 1, num_states))

    for k in range(1, num_iters + 1):
        for s in range(num_states):
            for a in range(num_actions):
                # Calculate \sum_{s'} p(s'\mid s,a) [r + \gamma v_k(s')]
                for pxrds in mdp[(s,a)]:
                    # mdp(s,a): [(p1,next1,r1,d1),(p2,next2,r2,d2),..]
                    pr = pxrds[prob_idx]  # p(s'\mid s,a)
                    nextstate = pxrds[nextstate_idx]  # Next state
                    reward = pxrds[reward_idx]  # Reward
                    Q[k,s,a] += pr * (reward + gamma * V[k - 1, nextstate])
            # Record max value and max action
            V[k,s] = np.max(Q[k,s,:])
            pi[k,s] = np.argmax(Q[k,s,:])
    d2l.show_value_function_progress(env_desc, V[:-1], pi[:-1])

value_iteration(env_info=env_info, gamma=gamma, num_iters=num_iters)
```

上の図は、方策（矢印は行動を示す）と価値関数（色の変化は、初期値としての濃い色から最適値としての明るい色へ、価値関数が時間とともにどのように変化するかを示す）を表している。見てわかるように、値反復は10回の反復後に最適価値関数を見つけ、H セルでない限り、どの状態から始めてもゴール状態（G）に到達できる。実装のもう1つの興味深い点は、最適価値関数を見つけるだけでなく、この価値関数に対応する最適方策 $\pi^*$ も自動的に得られることである。


## まとめ
値反復アルゴリズムの主な考え方は、動的計画法の原理を用いて、与えられた状態から得られる最適な平均リターンを求めることである。なお、値反復アルゴリズムを実装するには、マルコフ決定過程（MDP）、たとえば遷移関数や報酬関数を完全に知っている必要がある。


## 演習

1. グリッドサイズを $8 \times 8$ に増やしてみよ。$4 \times 4$ グリッドと比べて、最適価値関数を見つけるのに何回の反復が必要であるか？
1. 値反復アルゴリズムの計算量はどれくらいか？
1. $\gamma$（上のコードの "gamma"）を $0$、$0.5$、$1$ にしたときに、再度値反復アルゴリズムを実行し、その結果を分析せよ。
1. $\gamma$ の値は、値反復が収束するまでに要する反復回数にどのように影響するか？ $\gamma=1$ のときはどうなるか？
