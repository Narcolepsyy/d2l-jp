{.python .input}
%load_ext d2lbook.tab
tab.interact_select(["pytorch"])
#required_libs("setuptools==66", "wheel==0.38.4", "gym==0.21.0")
```

# Q学習
:label:`sec_qlearning`

前節では、遷移関数や報酬関数など、完全なマルコフ決定過程（MDP）へのアクセスを必要とする価値反復アルゴリズムについて議論しました。この節では、MDPを必ずしも知らなくても価値関数を学習できるアルゴリズムであるQ学習 :cite:`Watkins.Dayan.1992` を見ていきます。このアルゴリズムは強化学習の中心的な考え方を体現しています。すなわち、ロボットが自分自身のデータを取得できるようにすることです。
<!-- , instead of relying upon the expert. -->

## Q学習アルゴリズム

:ref:`sec_valueiter` における行動価値関数の価値反復は、次の更新に対応することを思い出してください。

$$Q_{k+1}(s, a) = r(s, a) + \gamma \sum_{s' \in \mathcal{S}} P(s' \mid s, a) \max_{a' \in \mathcal{A}} Q_k (s', a'); \ \textrm{for all } s \in \mathcal{S} \textrm{ and } a \in \mathcal{A}.$$

前述したように、このアルゴリズムを実装するにはMDP、特に遷移関数 $P(s' \mid s, a)$ を知っている必要があります。Q学習の鍵となる考え方は、上の式にあるすべての $s' \in \mathcal{S}$ にわたる和を、ロボットが訪れた状態にわたる和で置き換えることです。これにより、遷移関数を知る必要を回避できます。

## Q学習の背後にある最適化問題

ロボットが方策 $\pi_e(a \mid s)$ を使って行動すると想像しましょう。前章と同様に、$T$ タイムステップからなる $n$ 本の軌跡 $\{ (s_t^i, a_t^i)_{t=0,\ldots,T-1}\}_{i=1,\ldots, n}$ からなるデータセットを収集します。価値反復とは、実際には異なる状態と行動の行動価値 $Q^*(s, a)$ を互いに結び付ける制約の集合であることを思い出してください。ロボットが $\pi_e$ を用いて収集したデータを使えば、価値反復の近似版を次のように実装できます。

$$\hat{Q} = \min_Q \underbrace{\frac{1}{nT} \sum_{i=1}^n \sum_{t=0}^{T-1} (Q(s_t^i, a_t^i) - r(s_t^i, a_t^i) - \gamma \max_{a'} Q(s_{t+1}^i, a'))^2}_{\stackrel{\textrm{def}}{=} \ell(Q)}.$$
:eqlabel:`q_learning_optimization_problem`

まず、この式と上の価値反復との類似点と相違点を見てみましょう。もしロボットの方策 $\pi_e$ が最適方策 $\pi^*$ と等しく、かつ無限量のデータを収集できるなら、この最適化問題は価値反復の背後にある最適化問題と同一になります。しかし、価値反復では $P(s' \mid s, a)$ を知る必要があるのに対し、この最適化目的関数にはこの項がありません。これはごまかしではありません。ロボットが状態 $s_t^i$ で方策 $\pi_e$ を使って行動 $a_t^i$ を選ぶとき、次状態 $s_{t+1}^i$ は遷移関数から引かれたサンプルです。したがって、この最適化目的関数も遷移関数にアクセスしていますが、それはロボットが収集したデータという形で暗黙的に与えられているのです。

この最適化問題の変数は、すべての $s \in \mathcal{S}$ と $a \in \mathcal{A}$ に対する $Q(s, a)$ です。勾配降下法を用いて目的関数を最小化できます。データセット中の各組 $(s_t^i, a_t^i)$ について、次のように書けます。

$$\begin{aligned}Q(s_t^i, a_t^i) &\leftarrow Q(s_t^i, a_t^i) - \alpha \nabla_{Q(s_t^i,a_t^i)} \ell(Q) \\&=(1 - \alpha) Q(s_t^i,a_t^i) - \alpha \Big( r(s_t^i, a_t^i) + \gamma \max_{a'} Q(s_{t+1}^i, a') \Big),\end{aligned}$$
:eqlabel:`q_learning`

ここで $\alpha$ は学習率です。通常、実際の問題では、ロボットが目標地点に到達すると軌跡は終了します。このような終端状態の価値は0です。なぜなら、ロボットはこの状態以降、これ以上行動を取らないからです。このような状態を扱うために、更新式を次のように修正する必要があります。

$$Q(s_t^i, a_t^i) =(1 - \alpha) Q(s_t^i,a_t^i) - \alpha \Big( r(s_t^i, a_t^i) + \gamma (1 - \mathbb{1}_{s_{t+1}^i \textrm{ is terminal}} )\max_{a'} Q(s_{t+1}^i, a') \Big).$$

ここで $\mathbb{1}_{s_{t+1}^i \textrm{ is terminal}}$ は指示変数であり、$s_{t+1}^i$ が終端状態なら1、それ以外なら0です。データセットの一部ではない状態-行動の組 $(s, a)$ の値は $-\infty$ に設定されます。このアルゴリズムはQ学習として知られています。

これらの更新の解 $\hat{Q}$、すなわち最適価値関数 $Q^*$ の近似が得られれば、この価値関数に対応する最適な決定論的方策を次のように簡単に得られます。

$$\hat{\pi}(s) = \mathrm{argmax}_{a} \hat{Q}(s, a).$$

同じ最適価値関数に対応する決定論的方策が複数存在する場合があります。そのような同値は任意に解いて構いません。なぜなら、それらは同じ価値関数を持つからです。

## Q学習における探索

データ収集にロボットが用いる方策 $\pi_e$ は、Q学習がうまく機能するために重要です。結局のところ、私たちは遷移関数 $P(s' \mid s, a)$ による $s'$ に関する期待値を、ロボットが収集したデータで置き換えているのです。もし方策 $\pi_e$ が状態-行動空間の多様な部分に到達しないなら、推定値 $\hat{Q}$ が最適な $Q^*$ の不十分な近似になることは容易に想像できます。このような状況では、$\pi_e$ によって訪れた状態だけでなく、*すべての状態* $s \in \mathcal{S}$ における $Q^*$ の推定も悪くなることに注意することが重要です。これは、Q学習の目的関数（あるいは価値反復）が、すべての状態-行動対の価値を結び付ける制約だからです。したがって、データを収集するために適切な方策 $\pi_e$ を選ぶことが極めて重要です。

この懸念は、$\mathcal{A}$ から一様ランダムに行動をサンプルする完全にランダムな方策 $\pi_e$ を選ぶことで緩和できます。そのような方策はすべての状態を訪れますが、そうなるまでには大量の軌跡が必要になります。

こうしてQ学習における第2の重要な考え方、すなわち探索に至ります。Q学習の典型的な実装では、現在の $Q$ の推定値と方策 $\pi_e$ を結び付けて、次のように設定します。

$$\pi_e(a \mid s) = \begin{cases}\mathrm{argmax}_{a'} \hat{Q}(s, a') & \textrm{with prob. } 1-\epsilon \\ \textrm{uniform}(\mathcal{A}) & \textrm{with prob. } \epsilon,\end{cases}$$
:eqlabel:`epsilon_greedy`

ここで $\epsilon$ は「探索パラメータ」と呼ばれ、ユーザが選びます。方策 $\pi_e$ は探索方策と呼ばれます。この特定の $\pi_e$ は、現在の推定値 $\hat{Q}$ に基づく最適行動を確率 $1-\epsilon$ で選び、残りの確率 $\epsilon$ でランダムに探索するため、$\epsilon$-greedy 探索方策と呼ばれます。いわゆる softmax 探索方策も使えます。

$$\pi_e(a \mid s) = \frac{e^{\hat{Q}(s, a)/T}}{\sum_{a'} e^{\hat{Q}(s, a')/T}};$$

ここでハイパーパラメータ $T$ は温度と呼ばれます。$\epsilon$-greedy 方策における大きな $\epsilon$ は、softmax 方策における大きな温度 $T$ と同様に機能します。

現在の行動価値関数の推定値 $\hat{Q}$ に依存する探索を選ぶときには、最適化問題を定期的に解き直す必要があることに注意することが重要です。Q学習の典型的な実装では、$\pi_e$ を用いて毎回行動を取った後、収集したデータセット中のいくつかの状態-行動対（通常はロボットの直前のタイムステップで収集されたもの）を使って1回のミニバッチ更新を行います。

## Q学習の「自己修正」特性

Q学習中にロボットが収集するデータセットは、時間とともに増えていきます。探索方策 $\pi_e$ も推定値 $\hat{Q}$ も、ロボットがより多くのデータを収集するにつれて変化します。これにより、Q学習がうまく機能する理由について重要な洞察が得られます。状態 $s$ を考えましょう。ある行動 $a$ が現在の推定値 $\hat{Q}(s,a)$ の下で大きな値を持つなら、$\epsilon$-greedy 探索方策も softmax 探索方策も、この行動を選ぶ確率が高くなります。もしこの行動が実際には*理想的な行動ではない*なら、この行動から生じる将来の状態は低い報酬しか持ちません。したがって、次のQ学習目的関数の更新では $\hat{Q}(s,a)$ の値が下がり、次にロボットが状態 $s$ を訪れたときにこの行動を選ぶ確率も下がります。悪い行動、たとえば $\hat{Q}(s,a)$ において過大評価されている行動は、ロボットによって探索されますが、Q学習目的関数の次の更新でその値は修正されます。良い行動、たとえば $\hat{Q}(s, a)$ が大きい行動は、ロボットによってより頻繁に探索され、その結果として強化されます。この性質を用いると、Q学習はランダムな方策 $\pi_e$ から始めたとしても最適方策に収束できることを示せます :cite:`Watkins.Dayan.1992`。

新しいデータを収集するだけでなく、適切な種類のデータを収集する能力こそが強化学習アルゴリズムの中心的な特徴であり、これが教師あり学習との違いです。深層ニューラルネットワークを用いたQ学習（これは後のDQN章で見ます）は、強化学習の再興を支えました :cite:`mnih2013playing`。

## Q学習の実装

ここでは、[Open AI Gym](https://gym.openai.com) の FrozenLake に対してQ学習を実装する方法を示します。これは :ref:`sec_valueiter` の実験で考えたものと同じ設定です。

```{.python .input}
%%tab all

%matplotlib inline
import numpy as np
import random
from d2l import torch as d2l

seed = 0  # Random number generator seed
gamma = 0.95  # Discount factor
num_iters = 256  # Number of iterations
alpha   = 0.9  # Learing rate
epsilon = 0.9  # Epsilon in epsilion gready algorithm
random.seed(seed)  # Set the random seed
np.random.seed(seed)

# Now set up the environment
env_info = d2l.make_env('FrozenLake-v1', seed=seed)
```

FrozenLake環境では、ロボットは $4 \times 4$ のグリッド（これらが状態です）上を移動し、行動として「上」($\uparrow$)、「下」($\rightarrow$)、「左」($\leftarrow$)、「右」($\rightarrow$) を取ります。環境にはいくつかの穴（H）のセルと凍結した（F）セル、そして目標セル（G）が含まれていますが、これらはすべてロボットには未知です。問題を簡単にするため、ロボットの行動は確実である、すなわちすべての $s \in \mathcal{S}, a \in \mathcal{A}$ に対して $P(s' \mid s, a) = 1$ と仮定します。ロボットが目標に到達すると、試行は終了し、行動に関係なく報酬 $1$ を受け取ります。それ以外の状態での報酬は、すべての行動に対して $0$ です。ロボットの目的は、与えられた開始位置（S）（これは $s_0$ です）から目標位置（G）に到達する方策を学習し、*収益*を最大化することです。

まず、$\epsilon$-greedy 法を次のように実装します。

```{.python .input}
%%tab all

def e_greedy(env, Q, s, epsilon):
    if random.random() < epsilon:
        return env.action_space.sample()

    else:
        return np.argmax(Q[s,:])

```

これでQ学習を実装する準備が整いました。

```{.python .input}
%%tab all

def q_learning(env_info, gamma, num_iters, alpha, epsilon):
    env_desc = env_info['desc']  # 2D array specifying what each grid item means
    env = env_info['env']  # 2D array specifying what each grid item means
    num_states = env_info['num_states']
    num_actions = env_info['num_actions']

    Q  = np.zeros((num_states, num_actions))
    V  = np.zeros((num_iters + 1, num_states))
    pi = np.zeros((num_iters + 1, num_states))

    for k in range(1, num_iters + 1):
        # Reset environment
        state, done = env.reset(), False
        while not done:
            # Select an action for a given state and acts in env based on selected action
            action = e_greedy(env, Q, state, epsilon)
            next_state, reward, done, _ = env.step(action)

            # Q-update:
            y = reward + gamma * np.max(Q[next_state,:])
            Q[state, action] = Q[state, action] + alpha * (y - Q[state, action])

            # Move to the next state
            state = next_state
        # Record max value and max action for visualization purpose only
        for s in range(num_states):
            V[k,s]  = np.max(Q[s,:])
            pi[k,s] = np.argmax(Q[s,:])
    d2l.show_Q_function_progress(env_desc, V[:-1], pi[:-1])

q_learning(env_info=env_info, gamma=gamma, num_iters=num_iters, alpha=alpha, epsilon=epsilon)

```

この結果は、Q学習が約250回の反復の後にこの問題の最適解を見つけられることを示しています。しかし、この結果を価値反復アルゴリズムの結果（:ref:`subsec_valueitercode` を参照）と比較すると、価値反復アルゴリズムの方がこの問題の最適解を見つけるのにずっと少ない反復回数で済むことがわかります。これは、価値反復アルゴリズムが完全なMDPにアクセスできるのに対し、Q学習はそうではないためです。


## まとめ
Q学習は、最も基本的な強化学習アルゴリズムの1つです。近年の強化学習の成功、特にビデオゲームをプレイする学習において中心的な役割を果たしてきました :cite:`mnih2013playing`。Q学習の実装には、マルコフ決定過程（MDP）、たとえば遷移関数や報酬関数を完全に知っている必要はありません。

## 演習

1. グリッドサイズを $8 \times 8$ に増やしてみてください。$4 \times 4$ グリッドと比べて、最適価値関数を見つけるのに何回の反復が必要ですか？
1. Q学習アルゴリズムを、$\gamma$（つまり上のコードの "gamma"）を $0$、$0.5$、$1$ にした場合で再度実行し、その結果を分析してください。
1. Q学習アルゴリズムを、$\epsilon$（つまり上のコードの "epsilon"）を $0$、$0.5$、$1$ にした場合で再度実行し、その結果を分析してください。
