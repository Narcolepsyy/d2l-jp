{.python .input}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow', 'jax'])
```

# 畳み込みネットワークのアーキテクチャ設計
:label:`sec_cnn-design`

前の節では、コンピュータビジョンにおける現代的なネットワーク設計を一通り見てきました。そこで扱った研究に共通していたのは、科学者の直感に大きく依存していたことです。多くのアーキテクチャは人間の創造性に強く支えられており、深層ネットワークが提供する設計空間を体系的に探索することの比重はそれよりかなり小さいものでした。それでもなお、この *ネットワーク工学* 的アプローチは非常に成功してきました。 

AlexNet (:numref:`sec_alexnet`)
が ImageNet 上で従来のコンピュータビジョンモデルに勝って以来、
同じパターンに従って設計された畳み込みブロックを積み重ねることで、
非常に深いネットワークを構築することが一般的になりました。 
特に、$3 \times 3$ の畳み込みは
VGG ネットワーク (:numref:`sec_vgg`) によって広く普及しました。
NiN (:numref:`sec_nin`) は、局所的な非線形性を加えることで $1 \times 1$ の畳み込みでさえ
有益になりうることを示しました。 
さらに NiN は、ネットワークの最後で情報を集約する問題を、
全位置にわたって集約することで解決しました。 
GoogLeNet (:numref:`sec_googlenet`) は、異なる畳み込み幅を持つ複数の分岐を追加し、
Inception ブロックの中で VGG と NiN の利点を組み合わせました。 
ResNet (:numref:`sec_resnet`) 
は帰納バイアスを恒等写像（$f(x) = 0$ から）へと変えました。これにより非常に深いネットワークが可能になりました。ほぼ10年後の今でも、ResNet の設計は依然として人気があり、その設計の優秀さを物語っています。最後に、ResNeXt (:numref:`subsec_resnext`) はグループ畳み込みを追加し、パラメータ数と計算量の間でより良いトレードオフを提供しました。Vision Transformer の先駆けともいえる Squeeze-and-Excitation Networks（SENets）は、位置間で効率的に情報を伝達できるようにします
:cite:`Hu.Shen.Sun.2018`。これは、チャネルごとのグローバルな注意関数を計算することで実現されました。 

これまでのところ、*ニューラルアーキテクチャ探索*（NAS）によって得られるネットワークは扱ってきませんでした :cite:`zoph2016neural,liu2018darts`。その理由は、通常そのコストが非常に大きく、総当たり探索、遺伝的アルゴリズム、強化学習、あるいは他の何らかのハイパーパラメータ最適化に依存するからです。固定された探索空間が与えられたとき、
NAS は探索戦略を用いて、
返された性能推定値に基づいてアーキテクチャを自動的に選択します。
NAS の結果は
単一のネットワーク実体です。EfficientNet はこの探索の注目すべき成果です :cite:`tan2019efficientnet`。

以下では、*単一の最良ネットワーク* を探すこととはかなり異なる考え方を議論します。これは計算コストが比較的低く、その過程で科学的な洞察も得られ、しかも結果の品質という点でもかなり有効です。では、:citet:`Radosavovic.Kosaraju.Girshick.ea.2020` による *ネットワーク設計空間を設計する* 戦略を見ていきましょう。この戦略は手作業による設計と NAS の強みを組み合わせたものです。これは *ネットワークの分布* を対象にし、その分布を最適化してネットワークのファミリー全体に対して良い性能を得ることで実現されます。その成果が *RegNet*、具体的には RegNetX と RegNetY であり、さらに高性能な CNN を設計するための一連の指針です。

```{.python .input}
%%tab mxnet
from d2l import mxnet as d2l
from mxnet import np, npx, init
from mxnet.gluon import nn

npx.set_np()
```

```{.python .input}
%%tab pytorch
from d2l import torch as d2l
import torch
from torch import nn
from torch.nn import functional as F
```

```{.python .input}
%%tab tensorflow
import tensorflow as tf
from d2l import tensorflow as d2l
```

```{.python .input}
%%tab jax
from d2l import jax as d2l
from flax import linen as nn
```

## AnyNet の設計空間
:label:`subsec_the-anynet-design-space`

以下の説明は、:citet:`Radosavovic.Kosaraju.Girshick.ea.2020` の考え方にほぼ沿っていますが、本書の範囲に収めるためにいくつか省略しています。 
まず、探索対象となるネットワーク族のテンプレートが必要です。この章で扱う設計に共通する点の一つは、ネットワークが *stem*、*body*、*head* から構成されることです。stem は初期の画像処理を行い、しばしばより大きな窓サイズの畳み込みを通して処理します。body は複数のブロックからなり、生の画像から物体表現へ移るために必要な変換の大部分を担います。最後に head は、たとえば多クラス分類のための softmax 回帰器のように、これを所望の出力へ変換します。 
body 自体は、解像度を下げながら画像に作用する複数の stage から構成されます。実際、stem もその後の各 stage も空間解像度を 4 分の 1 にします。最後に、各 stage は 1 個以上のブロックからなります。このパターンは VGG から ResNeXt まで、すべてのネットワークに共通です。実際、一般的な AnyNet ネットワークを設計するために、:citet:`Radosavovic.Kosaraju.Girshick.ea.2020` は :numref:`fig_resnext_block` の ResNeXt ブロックを用いました。


![AnyNet の設計空間。各矢印に沿った数値 $(\mathit{c}, \mathit{r})$ は、その時点でのチャネル数 $c$ と画像の解像度 $\mathit{r} \times \mathit{r}$ を示す。左から右へ：stem、body、head からなる一般的なネットワーク構造；4つの stage からなる body；stage の詳細構造；ブロックの2つの代替構造、1つはダウンサンプリングなし、もう1つは各次元の解像度を半分にする。設計上の選択には、任意の stage $\mathit{i}$ に対する深さ $\mathit{d_i}$、出力チャネル数 $\mathit{c_i}$、グループ数 $\mathit{g_i}$、ボトルネック比 $\mathit{k_i}$ が含まれる。](../img/anynet.svg)
:label:`fig_anynet_full`

:numref:`fig_anynet_full` に示した構造を詳しく見ていきましょう。述べたように、AnyNet は stem、body、head から成ります。stem は RGB 画像（3チャネル）を入力とし、stride が $2$ の $3 \times 3$ 畳み込みの後に batch norm を適用して、解像度を $r \times r$ から $r/2 \times r/2$ に半減します。さらに、body への入力となる $c_0$ チャネルを生成します。 

このネットワークは $224 \times 224 \times 3$ 形状の ImageNet 画像でうまく動作するように設計されているため、body は 4つの stage を通してこれを $7 \times 7 \times c_4$ に縮小します（$224 / 2^{1+4} = 7$ を思い出してください）。各 stage は最終的に stride $2$ を持ちます。最後に、head は NiN (:numref:`sec_nin`) と同様に global average pooling を用い、その後に全結合層を置くという完全に標準的な設計で、$n$ クラス分類のための $n$ 次元ベクトルを出力します。 

関連する設計上の決定の大部分はネットワークの body に内在しています。body は stage ごとに進み、各 stage は :numref:`subsec_resnext` で議論したのと同じ種類の ResNeXt ブロックから構成されます。ここでの設計もまた完全に一般的です。まず、stride を $2$ にして解像度を半分にするブロックから始めます（:numref:`fig_anynet_full` の最右端）。これに合わせるため、ResNeXt ブロックの残差分岐は $1 \times 1$ 畳み込みを通る必要があります。このブロックの後には、解像度とチャネル数の両方を変えない追加の ResNeXt ブロックが可変個続きます。畳み込みブロックの設計では、わずかなボトルネックを加えるのが一般的な慣行であることに注意してください。 
そのため、ボトルネック比 $k_i \geq 1$ を用いて、stage $i$ の各ブロック内に $c_i/k_i$ 個のチャネルを割り当てます（実験が示すように、これは実際にはあまり有効ではなく、使わないほうがよいです）。最後に、ResNeXt ブロックを扱っているので、stage $i$ におけるグループ畳み込みのグループ数 $g_i$ も選ぶ必要があります。 

この一見一般的な設計空間にも、なお多くのパラメータがあります。すなわち、ブロック幅（チャネル数） $c_0, \ldots c_4$、各 stage の深さ（ブロック数） $d_1, \ldots d_4$、ボトルネック比 $k_1, \ldots k_4$、そしてグループ幅（グループ数） $g_1, \ldots g_4$ を設定できます。 
合計すると 17 個のパラメータになり、探索すべき構成の数は非常に大きくなります。この巨大な設計空間を効果的に縮小するための道具が必要です。ここで設計空間という概念の美しさが生きてきます。その前に、まず一般的な設計を実装しましょう。

```{.python .input}
%%tab mxnet
class AnyNet(d2l.Classifier):
    def stem(self, num_channels):
        net = nn.Sequential()
        net.add(nn.Conv2D(num_channels, kernel_size=3, padding=1, strides=2),
                nn.BatchNorm(), nn.Activation('relu'))
        return net
```

```{.python .input}
%%tab pytorch
class AnyNet(d2l.Classifier):
    def stem(self, num_channels):
        return nn.Sequential(
            nn.LazyConv2d(num_channels, kernel_size=3, stride=2, padding=1),
            nn.LazyBatchNorm2d(), nn.ReLU())
```

```{.python .input}
%%tab tensorflow
class AnyNet(d2l.Classifier):
    def stem(self, num_channels):
        return tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(num_channels, kernel_size=3, strides=2,
                                   padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation('relu')])
```

```{.python .input}
%%tab jax
class AnyNet(d2l.Classifier):
    arch: tuple
    stem_channels: int
    lr: float = 0.1
    num_classes: int = 10
    training: bool = True

    def setup(self):
        self.net = self.create_net()

    def stem(self, num_channels):
        return nn.Sequential([
            nn.Conv(num_channels, kernel_size=(3, 3), strides=(2, 2),
                    padding=(1, 1)),
            nn.BatchNorm(not self.training),
            nn.relu
        ])
```

各 stage は `depth` 個の ResNeXt ブロックからなり、
`num_channels` はブロック幅を指定します。
最初のブロックが入力画像の高さと幅を半分にすることに注意してください。

```{.python .input}
%%tab mxnet
@d2l.add_to_class(AnyNet)
def stage(self, depth, num_channels, groups, bot_mul):
    net = nn.Sequential()
    for i in range(depth):
        if i == 0:
            net.add(d2l.ResNeXtBlock(
                num_channels, groups, bot_mul, use_1x1conv=True, strides=2))
        else:
            net.add(d2l.ResNeXtBlock(
                num_channels, num_channels, groups, bot_mul))
    return net
```

```{.python .input}
%%tab pytorch
@d2l.add_to_class(AnyNet)
def stage(self, depth, num_channels, groups, bot_mul):
    blk = []
    for i in range(depth):
        if i == 0:
            blk.append(d2l.ResNeXtBlock(num_channels, groups, bot_mul,
                use_1x1conv=True, strides=2))
        else:
            blk.append(d2l.ResNeXtBlock(num_channels, groups, bot_mul))
    return nn.Sequential(*blk)
```

```{.python .input}
%%tab tensorflow
@d2l.add_to_class(AnyNet)
def stage(self, depth, num_channels, groups, bot_mul):
    net = tf.keras.models.Sequential()
    for i in range(depth):
        if i == 0:
            net.add(d2l.ResNeXtBlock(num_channels, groups, bot_mul,
                use_1x1conv=True, strides=2))
        else:
            net.add(d2l.ResNeXtBlock(num_channels, groups, bot_mul))
    return net
```

```{.python .input}
%%tab jax
@d2l.add_to_class(AnyNet)
def stage(self, depth, num_channels, groups, bot_mul):
    blk = []
    for i in range(depth):
        if i == 0:
            blk.append(d2l.ResNeXtBlock(num_channels, groups, bot_mul,
                use_1x1conv=True, strides=(2, 2), training=self.training))
        else:
            blk.append(d2l.ResNeXtBlock(num_channels, groups, bot_mul,
                                        training=self.training))
    return nn.Sequential(blk)
```

ネットワークの stem、body、head を組み合わせることで、
AnyNet の実装が完成します。

```{.python .input}
%%tab pytorch, mxnet, tensorflow
@d2l.add_to_class(AnyNet)
def __init__(self, arch, stem_channels, lr=0.1, num_classes=10):
    super(AnyNet, self).__init__()
    self.save_hyperparameters()
    if tab.selected('mxnet'):
        self.net = nn.Sequential()
        self.net.add(self.stem(stem_channels))
        for i, s in enumerate(arch):
            self.net.add(self.stage(*s))
        self.net.add(nn.GlobalAvgPool2D(), nn.Dense(num_classes))
        self.net.initialize(init.Xavier())
    if tab.selected('pytorch'):
        self.net = nn.Sequential(self.stem(stem_channels))
        for i, s in enumerate(arch):
            self.net.add_module(f'stage{i+1}', self.stage(*s))
        self.net.add_module('head', nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(),
            nn.LazyLinear(num_classes)))
        self.net.apply(d2l.init_cnn)
    if tab.selected('tensorflow'):
        self.net = tf.keras.models.Sequential(self.stem(stem_channels))
        for i, s in enumerate(arch):
            self.net.add(self.stage(*s))
        self.net.add(tf.keras.models.Sequential([
            tf.keras.layers.GlobalAvgPool2D(),
            tf.keras.layers.Dense(units=num_classes)]))
```

```{.python .input}
%%tab jax
@d2l.add_to_class(AnyNet)
def create_net(self):
    net = nn.Sequential([self.stem(self.stem_channels)])
    for i, s in enumerate(self.arch):
        net.layers.extend([self.stage(*s)])
    net.layers.extend([nn.Sequential([
        lambda x: nn.avg_pool(x, window_shape=x.shape[1:3],
                            strides=x.shape[1:3], padding='valid'),
        lambda x: x.reshape((x.shape[0], -1)),
        nn.Dense(self.num_classes)])])
    return net
```

## 設計空間の分布とパラメータ

:numref:`subsec_the-anynet-design-space` で述べたように、設計空間のパラメータは、その設計空間に属するネットワークのハイパーパラメータです。
AnyNet の設計空間で良いパラメータを見つける問題を考えてみましょう。与えられた計算量（たとえば FLOPs や計算時間）に対して、*単一の最良* のパラメータ選択を見つけようとすることはできます。各パラメータに対して仮にたった *2つ* の候補しか許さないとしても、最良解を見つけるには $2^{17} = 131072$ 通りの組合せを調べなければなりません。これは明らかに、あまりにも高コストで実行不可能です。さらに悪いことに、この作業からは、ネットワークをどう設計すべきかという観点ではほとんど何も学べません。次に X-stage や shift 演算などを追加するときには、また最初からやり直しになります。しかも、学習の確率性（丸め、シャッフル、ビット誤りなど）のため、2回の実行がまったく同じ結果を生む可能性は低いです。より良い戦略は、パラメータの選択がどのように関係すべきかについて一般的な指針を見つけることです。たとえば、ボトルネック比、チャネル数、ブロック数、グループ数、あるいは層間でのそれらの変化は、簡単な規則の集合によって支配されるのが理想です。:citet:`radosavovic2019network` のアプローチは、次の4つの仮定に基づいています。

1. 一般的な設計原理は実際に存在し、それらを満たす多くのネットワークが良い性能を示すと仮定します。したがって、ネットワークの *分布* を特定することは妥当な戦略です。言い換えると、干し草の山の中に良い針がたくさんあると仮定します。
1. ネットワークが良いかどうかを評価する前に、収束まで学習させる必要はありません。最終的な精度の信頼できる指針として、中間結果を使えば十分です。目的関数を最適化するために（近似的な）代理指標を用いることを multi-fidelity optimization と呼びます :cite:`forrester2007multi`。したがって、設計の最適化は、データセットを数回通しただけで得られる精度に基づいて行われ、コストを大幅に削減できます。 
1. 小規模で得られた結果（より小さなネットワーク）は、より大きなネットワークにも一般化します。したがって、最適化は構造的には似ているが、ブロック数やチャネル数などが少ないネットワークに対して行います。最後に、見つかったネットワークがスケールを大きくしても良い性能を示すことを確認すれば十分です。 
1. 設計の各要素は近似的に因子分解できるので、その結果の品質への影響をある程度独立に推定できます。言い換えると、最適化問題は中程度に容易です。

これらの仮定により、多くのネットワークを安価に試せます。特に、構成空間から一様に *サンプリング* して、その性能を評価できます。その後、そうして得られたネットワークで達成可能な誤差/精度の *分布* を調べることで、パラメータ選択の質を評価できます。ある設計空間から確率分布 $p$ に従って引いたネットワークが犯す誤差について、その累積分布関数（CDF）を $F(e)$ とします。すなわち、 

$$F(e, p) \stackrel{\textrm{def}}{=} P_{\textrm{net} \sim p} \{e(\textrm{net}) \leq e\}.$$

私たちの目標は、ほとんどのネットワークが非常に低い誤差率を持ち、かつ $p$ の支持が簡潔であるような、*ネットワーク* 上の分布 $p$ を見つけることです。もちろん、これを正確に行うのは計算上不可能です。そこで、分布 $p$ からネットワークの標本 $\mathcal{Z} \stackrel{\textrm{def}}{=} \{\textrm{net}_1, \ldots \textrm{net}_n\}$（それぞれの誤差を $e_1, \ldots, e_n$ とする）を取り、代わりに経験的 CDF $\hat{F}(e, \mathcal{Z})$ を用います。

$$\hat{F}(e, \mathcal{Z}) = \frac{1}{n}\sum_{i=1}^n \mathbf{1}(e_i \leq e).$$

ある選択集合の CDF が別の CDF を優越する（あるいは一致する）なら、そのパラメータ選択はより優れている（あるいは同等である）ことになります。これに従って
:citet:`Radosavovic.Kosaraju.Girshick.ea.2020` は、ネットワークのすべての stage に対してボトルネック比を共有する $k_i = k$ を試しました。これにより、ボトルネック比を支配する4つのパラメータのうち3つが不要になります。これが性能に悪影響を与えるかどうかを評価するには、制約付き分布と非制約分布からネットワークをサンプリングし、対応する CDF を比較すればよいです。すると、この制約はネットワーク分布の精度にまったく影響しないことがわかります。これは :numref:`fig_regnet-fig` の最初のパネルに示されています。 
同様に、ネットワークのさまざまな stage に現れるグループ幅を同じ $g_i = g$ にすることもできます。これもまた性能に影響しません。これは :numref:`fig_regnet-fig` の2番目のパネルに示されています。
この2つを合わせると、自由パラメータの数は6つ減ります。 

![設計空間の経験的誤差分布関数の比較。$\textrm{AnyNet}_\mathit{A}$ は元の設計空間、$\textrm{AnyNet}_\mathit{B}$ はボトルネック比を共有し、$\textrm{AnyNet}_\mathit{C}$ はさらにグループ幅も共有し、$\textrm{AnyNet}_\mathit{D}$ は stage 間でネットワーク深さを増加させる。左から右へ：(i) ボトルネック比を共有しても性能に影響はない；(ii) グループ幅を共有しても性能に影響はない；(iii) stage 間でネットワーク幅（チャネル数）を増やすと性能が向上する；(iv) stage 間でネットワーク深さを増やすと性能が向上する。図は :citet:`Radosavovic.Kosaraju.Girshick.ea.2020` より。](../img/regnet-fig.png)
:label:`fig_regnet-fig`

次に、stage の幅と深さに関する多数の候補を減らす方法を考えます。より深くなるにつれてチャネル数は増えるべき、すなわち $c_i \geq c_{i-1}$（:numref:`fig_regnet-fig` における彼らの記法では $w_{i+1} \geq w_i$）と仮定するのは妥当であり、これにより 
$\textrm{AnyNetX}_D$ が得られます。同様に、stage が進むにつれて深くなる、すなわち $d_i \geq d_{i-1}$ と仮定するのも同様に妥当であり、これにより $\textrm{AnyNetX}_E$ が得られます。これはそれぞれ :numref:`fig_regnet-fig` の3番目と4番目のパネルで実験的に確認できます。

## RegNet

こうして得られる $\textrm{AnyNetX}_E$ の設計空間は、理解しやすい設計原理に従う単純なネットワークから成ります。

* すべての stage $i$ でボトルネック比を共有する $k_i = k$；
* すべての stage $i$ でグループ幅を共有する $g_i = g$；
* stage 間でネットワーク幅を増やす：$c_{i} \leq c_{i+1}$；
* stage 間でネットワーク深さを増やす：$d_{i} \leq d_{i+1}$。

これで、最終的な選択肢が残ります。すなわち、最終的な $\textrm{AnyNetX}_E$ 設計空間における上記パラメータの具体的な値をどう選ぶかです。$\textrm{AnyNetX}_E$ の分布から最良性能を示すネットワークを調べると、次のことが観察できます。ネットワークの幅は、理想的にはネットワーク全体のブロックインデックスに対して線形に増加します。すなわち、$c_j \approx c_0 + c_a j$ であり、ここで $j$ はブロックインデックス、傾きは $c_a > 0$ です。stage ごとに異なるブロック幅しか選べないことを考えると、この依存関係に合わせて設計された区分的定数関数に行き着きます。さらに、実験はボトルネック比 $k = 1$ が最も良いことも示しており、つまりボトルネックをまったく使わないのが望ましいということです。 

異なる計算量に対する特定のネットワーク設計の詳細については、:citet:`Radosavovic.Kosaraju.Girshick.ea.2020` を参照することをお勧めします。たとえば、効果的な 32 層の RegNetX 変種は、$k = 1$（ボトルネックなし）、$g = 16$（グループ幅 16）、第1 stage と第2 stage のチャネル数がそれぞれ $c_1 = 32$ と $c_2 = 80$ で、深さがそれぞれ $d_1=4$ ブロック、$d_2=6$ ブロックとなるように選ばれます。この設計から得られる驚くべき洞察は、より大規模なネットワークを調べる場合でもなお成り立つということです。さらに良いことに、グローバルなチャネル活性化を持つ Squeeze-and-Excitation（SE）ネットワーク設計（RegNetY）にも当てはまります :cite:`Hu.Shen.Sun.2018`。

```{.python .input}
%%tab pytorch, mxnet, tensorflow
class RegNetX32(AnyNet):
    def __init__(self, lr=0.1, num_classes=10):
        stem_channels, groups, bot_mul = 32, 16, 1
        depths, channels = (4, 6), (32, 80)
        super().__init__(
            ((depths[0], channels[0], groups, bot_mul),
             (depths[1], channels[1], groups, bot_mul)),
            stem_channels, lr, num_classes)
```

```{.python .input}
%%tab jax
class RegNetX32(AnyNet):
    lr: float = 0.1
    num_classes: int = 10
    stem_channels: int = 32
    arch: tuple = ((4, 32, 16, 1), (6, 80, 16, 1))
```

各 RegNetX stage が、解像度を段階的に下げつつ出力チャネル数を増やしていくことがわかります。

```{.python .input}
%%tab mxnet, pytorch
RegNetX32().layer_summary((1, 1, 96, 96))
```

```{.python .input}
%%tab tensorflow
RegNetX32().layer_summary((1, 96, 96, 1))
```

```{.python .input}
%%tab jax
RegNetX32(training=False).layer_summary((1, 96, 96, 1))
```

## 学習

Fashion-MNIST データセットで 32 層の RegNetX を学習するのは、これまでと同じです。

```{.python .input}
%%tab mxnet, pytorch, jax
model = RegNetX32(lr=0.05)
trainer = d2l.Trainer(max_epochs=10, num_gpus=1)
data = d2l.FashionMNIST(batch_size=128, resize=(96, 96))
trainer.fit(model, data)
```

```{.python .input}
%%tab tensorflow
trainer = d2l.Trainer(max_epochs=10)
data = d2l.FashionMNIST(batch_size=128, resize=(96, 96))
with d2l.try_gpu():
    model = RegNetX32(lr=0.01)
    trainer.fit(model, data)
```

## 議論

局所性や平行移動不変性（:numref:`sec_why-conv`）のような、視覚に対する望ましい帰納バイアス（仮定や好み）を持つため、CNN はこの分野で支配的なアーキテクチャでした。これは LeNet から、Transformers (:numref:`sec_transformer`) :cite:`Dosovitskiy.Beyer.Kolesnikov.ea.2021,touvron2021training` が精度の面で CNN を上回り始めるまで続きました。最近の vision Transformer の進歩の多くは CNN に *移植可能* ですが :cite:`liu2022convnet`、それはより高い計算コストを伴う場合に限られます。同様に重要なのは、NVIDIA Ampere や Hopper といった最近のハードウェア最適化が、Transformer に有利な差をさらに広げていることです。 

Transformer は、CNN に比べて局所性や平行移動不変性に対する帰納バイアスの度合いがかなり低いことに注意する価値があります。学習された構造が優勢になったのは、最大50億枚の画像を含む LAION-400m や LAION-5B :cite:`schuhmann2022laion` のような大規模画像コレクションが利用可能になったことも大きな要因です。驚くべきことに、この文脈でより重要な研究の中には MLP さえ含まれています :cite:`tolstikhin2021mlp`。 

要するに、vision Transformer (:numref:`sec_vision-transformer`) は現在、大規模画像分類における最先端性能の面で先行しており、*スケーラビリティは帰納バイアスに勝る* ことを示しています :cite:`Dosovitskiy.Beyer.Kolesnikov.ea.2021`。
これには、multi-head self-attention (:numref:`sec_multihead-attention`) を用いた大規模 Transformer の事前学習 (:numref:`sec_large-pretraining-transformers`) も含まれます。これらの章をぜひ読み進めて、より詳細な議論に触れてください。

## 演習

1. stage の数を4に増やしてください。より深く、より良い性能を示す RegNetX を設計できますか。
1. ResNeXt ブロックを ResNet ブロックに置き換えて、RegNet を De-ResNeXt 化してください。新しいモデルの性能はどうなりますか。
1. RegNetX の設計原理を *破る* ことで、「VioNet」ファミリーの複数の実体を実装してください。それらの性能はどうなりますか。($d_i$, $c_i$, $g_i$, $b_i$) のうち、最も重要な要因はどれですか。
1. あなたの目標は「完璧な」MLP を設計することです。上で導入した設計原理を使って、良いアーキテクチャを見つけられますか。小規模ネットワークから大規模ネットワークへ外挿することは可能ですか。\n
