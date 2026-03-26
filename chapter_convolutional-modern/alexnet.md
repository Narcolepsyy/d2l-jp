{.python .input  n=1}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow', 'jax'])
```

# 畳み込み深層ニューラルネットワーク（AlexNet）
:label:`sec_alexnet`


CNNは、LeNet の導入 :cite:`LeCun.Jackel.Bottou.ea.1995` 以降、
コンピュータビジョンおよび機械学習のコミュニティではよく知られていましたが、
すぐにこの分野を席巻したわけではありません。
LeNet は初期の小規模データセットでは良好な結果を示しましたが、
より大規模で現実的なデータセットに対して CNN を訓練する性能と実現可能性は、まだ確立されていませんでした。
実際、1990年代初頭から2012年の画期的な結果 :cite:`Krizhevsky.Sutskever.Hinton.2012` までの長い期間の多くにおいて、
ニューラルネットワークは、カーネル法 :cite:`Scholkopf.Smola.2002`、アンサンブル法 :cite:`Freund.Schapire.ea.1996`、
構造化推定 :cite:`Taskar.Guestrin.Koller.2004` などの他の機械学習手法にしばしば劣っていました。

コンピュータビジョンに関しては、この比較は必ずしも正確ではありません。
つまり、畳み込みネットワークへの入力は
生の、あるいは（たとえば中心化のような）軽い前処理を施した画素値から成りますが、実務家が生の画素を従来のモデルにそのまま入力することはありません。
代わりに、典型的なコンピュータビジョンのパイプラインは、
SIFT :cite:`Lowe.2004`、SURF :cite:`Bay.Tuytelaars.Van-Gool.2006`、visual words の bag :cite:`Sivic.Zisserman.2003` などの特徴抽出パイプラインを手作業で設計するものでした。
特徴を *学習する* のではなく、特徴は *作り込まれて* いました。
進歩の大半は、一方では特徴抽出に関するより巧妙なアイデア、他方では幾何学 :cite:`Hartley.Zisserman.2000` に対する深い洞察によってもたらされました。学習アルゴリズムは、しばしば後回しにされていました。

1990年代にはいくつかのニューラルネットワーク用アクセラレータが利用可能でしたが、
深い多チャネル・多層の CNN を多数のパラメータで構成するには、まだ十分な性能がありませんでした。
たとえば、1999年の NVIDIA GeForce 256 は、
ゲーム以外の処理に対する意味のあるプログラミング基盤もないまま、
加算や乗算などの浮動小数点演算を1秒あたり最大4億8千万回（MFLOPS）しか処理できませんでした。
今日のアクセラレータは、1デバイスあたり1000 TFLOPs を超える性能を発揮できます。
さらに、データセットもまだ比較的小規模でした。60,000枚の低解像度 $28 \times 28$ ピクセル画像に対する OCR は、非常に難しい課題と見なされていました。
これらの障害に加えて、ニューラルネットワークを訓練するための重要な工夫、
すなわちパラメータ初期化のヒューリスティクス :cite:`Glorot.Bengio.2010`、
確率的勾配降下法の巧妙な変種 :cite:`Kingma.Ba.2014`、
非飽和活性化関数 :cite:`Nair.Hinton.2010`、
効果的な正則化手法 :cite:`Srivastava.Hinton.Krizhevsky.ea.2014` なども、まだ欠けていました。

したがって、*エンドツーエンド*（画素から分類まで）のシステムを訓練するのではなく、
古典的なパイプラインは次のようなものでした。

1. 興味深いデータセットを入手する。初期の頃は、これらのデータセットには高価なセンサーが必要でした。たとえば、1994年の [Apple QuickTake 100](https://en.wikipedia.org/wiki/Apple_QuickTake) は、わずか \$1000 で、最大8枚の画像を保存できる、驚くべき 0.3 メガピクセル（VGA）解像度を備えていました。
1. 光学、幾何学、その他の解析的手法、そして時には幸運な大学院生による偶然の発見に基づく手作りの特徴を用いてデータセットを前処理する。
1. SIFT（scale-invariant feature transform） :cite:`Lowe.2004`、SURF（speeded up robust features） :cite:`Bay.Tuytelaars.Van-Gool.2006`、あるいは他の多くの手調整されたパイプラインのような標準的な特徴抽出器にデータを通す。OpenCV は今日でも SIFT 抽出器を提供しています！
1. 得られた表現を、好みの分類器、たいていは線形モデルかカーネル法に投入して分類器を訓練する。

機械学習研究者に話を聞けば、
彼らは機械学習が重要で美しいものだと答えるでしょう。
さまざまな分類器の性質は洗練された理論によって証明され :cite:`boucheron2005theory`、凸最適化 :cite:`Boyd.Vandenberghe.2004` はそれらを得るための主流となっていました。
機械学習分野は活気に満ち、厳密で、きわめて有用でした。しかし、
コンピュータビジョン研究者に話を聞けば、
まったく異なる話が返ってきたでしょう。
彼らが語る画像認識の不都合な真実は、
新しい学習アルゴリズムではなく、特徴、幾何学 :cite:`Hartley.Zisserman.2000,hartley2009global`、そして工学が進歩を牽引していたということです。
コンピュータビジョン研究者は、
少し大きい、あるいは少しきれいなデータセット、
あるいは少し改善された特徴抽出パイプラインのほうが、
どんな学習アルゴリズムよりも最終的な精度にずっと大きく影響すると、もっともな理由で考えていました。

```{.python .input  n=2}
%%tab mxnet
from d2l import mxnet as d2l
from mxnet import np, init, npx
from mxnet.gluon import nn
npx.set_np()
```

```{.python .input  n=3}
%%tab pytorch
from d2l import torch as d2l
import torch
from torch import nn
```

```{.python .input  n=4}
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

## 表現学習

状況を別の言い方で表すなら、
パイプラインで最も重要な部分は表現でした。
そして2012年までは、その表現は主として機械的に計算されていました。
実際、新しい特徴関数の設計、結果の改善、手法の記述は、
論文の中でいずれも大きな位置を占めていました。
SIFT :cite:`Lowe.2004`、
SURF :cite:`Bay.Tuytelaars.Van-Gool.2006`、
HOG（histograms of oriented gradient） :cite:`Dalal.Triggs.2005`、
visual words の bag :cite:`Sivic.Zisserman.2003`、
および同様の特徴抽出器が主流でした。

Yann LeCun、Geoff Hinton、Yoshua Bengio、
Andrew Ng、Shun-ichi Amari、Juergen Schmidhuber などを含む別の研究者グループは、
異なる構想を持っていました。
彼らは、特徴そのものが学習されるべきだと考えていました。
さらに、特徴が十分に複雑であるためには、
複数の共同学習された層から階層的に構成され、
各層が学習可能なパラメータを持つべきだと考えていました。
画像の場合、最下層は動物の視覚系が入力を処理する仕組みに倣って、
エッジ、色、テクスチャを検出するようになるかもしれません。
特に、sparse coding :cite:`olshausen1996emergence` によって得られるような視覚特徴の自動設計は、
現代の CNN が登場するまで未解決の課題でした。
画像データから特徴を自動生成するというアイデアが大きな支持を得たのは、
:citet:`Dean.Corrado.Monga.ea.2012,le2013building` になってからでした。

最初の現代的な CNN :cite:`Krizhevsky.Sutskever.Hinton.2012` は、
その発明者の一人である Alex Krizhevsky にちなんで *AlexNet* と名付けられ、
LeNet に対する主として進化的な改良でした。
これは2012年の ImageNet コンペティションで優れた性能を達成しました。

![AlexNet の第1層で学習された画像フィルタ。再現図は :citet:`Krizhevsky.Sutskever.Hinton.2012` より。](../img/filters.png)
:width:`400px`
:label:`fig_filters`

興味深いことに、ネットワークの最下層では、
モデルは従来のフィルタに似た特徴抽出器を学習しました。
:numref:`fig_filters`
は低レベルの画像記述子を示しています。
ネットワークのより高い層は、これらの表現を基にして、
目、鼻、草の葉などのより大きな構造を表現するかもしれません。
さらに高い層では、人、飛行機、犬、フリスビーのような
物体全体を表現するかもしれません。
最終的には、最後の隠れ状態が画像の内容を要約したコンパクトな表現を学習し、
異なるカテゴリに属するデータを容易に分離できるようになります。

AlexNet（2012）とその前身である LeNet（1995）は、多くのアーキテクチャ要素を共有しています。では、なぜこれほど長い時間がかかったのでしょうか？
重要な違いは、過去20年間で利用可能なデータ量と計算能力が大幅に増加したことでした。そのため AlexNet ははるかに大規模であり、1995年に利用可能だった CPU と比べて、より多くのデータで、より高速な GPU 上で訓練されました。

### 欠けていた要素: データ

多数の層を持つ深層モデルは、
従来手法（たとえば線形法やカーネル法）に基づく凸最適化を大きく上回る段階に入るために、
大量のデータを必要とします。
しかし、コンピュータのストレージ容量の制約、
（画像）センサーの相対的な高コスト、
そして1990年代における研究予算の比較的厳しさのため、
多くの研究は小さなデータセットに依存していました。
多くの論文は UCI のデータセット集に依拠しており、
その多くは、低解像度で撮影され、しばしば人工的にきれいな背景を持つ、
数百枚、あるいは（せいぜい）数千枚の画像しか含んでいませんでした。

2009年、ImageNet データセットが公開され :cite:`Deng.Dong.Socher.ea.2009`、
1000の異なるカテゴリの物体からそれぞれ1000例、合計100万例からモデルを学習するという課題を研究者に突きつけました。
カテゴリ自体は WordNet :cite:`Miller.1995` における最も一般的な名詞ノードに基づいていました。
ImageNet チームは Google Image Search を使って各カテゴリの大規模な候補集合を事前にふるい分け、
Amazon Mechanical Turk のクラウドソーシング・パイプラインを用いて、
各画像が対応するカテゴリに属するかどうかを確認しました。
この規模は前例がなく、他のデータセットを1桁以上上回っていました
（たとえば CIFAR-100 は 60,000 枚の画像です）。
もう一つの特徴は、画像が
80 million-sized の TinyImages データセット :cite:`Torralba.Fergus.Freeman.2008` のような $32 \times 32$ ピクセルのサムネイルとは異なり、
比較的高解像度の $224 \times 224$ ピクセルであったことです。
これにより、より高次の特徴を形成できました。
これに対応するコンペティションである ImageNet Large Scale Visual Recognition
Challenge :cite:`russakovsky2015imagenet`
は、コンピュータビジョンと機械学習研究を前進させ、
研究者に、学術界がそれまで考えていたよりも大きな規模で、
どのモデルが最良の性能を示すのかを突き止めるよう挑みました。LAION-5B
:cite:`schuhmann2022laion` のような最大級の視覚データセットには、追加のメタデータを伴う数十億枚の画像が含まれています。

### 欠けていた要素: ハードウェア

深層学習モデルは計算資源を大量に消費します。
訓練には数百エポックかかることがあり、各反復では
計算コストの高い線形代数演算を多数の層に通してデータを流す必要があります。
これが、1990年代から2000年代初頭にかけて、
より効率的に最適化された凸目的関数に基づく単純なアルゴリズムが好まれた主な理由の一つです。

*グラフィックス処理装置*（GPU）は、深層学習を実用的にするうえで
ゲームチェンジャーであることが証明されました。
これらのチップはもともと、コンピュータゲーム向けのグラフィックス処理を高速化するために開発されていました。
特に、多くのコンピュータグラフィックス処理に必要な、高スループットの $4 \times 4$
行列—ベクトル積に最適化されていました。
幸いなことに、その数学は
畳み込み層の計算に必要なものと驚くほど似ています。
その頃、NVIDIA と ATI は GPU を一般計算向けに最適化し始めており :cite:`Fernando.2004`、
それらを *general-purpose GPUs*（GPGPUs）として売り出すまでになっていました。

直感を得るために、現代のマイクロプロセッサ（CPU）のコアを考えてみましょう。
各コアはかなり高性能で、高いクロック周波数で動作し、
大きなキャッシュ（L3 で数メガバイトに達することもある）を備えています。
各コアは、分岐予測器、深いパイプライン、専用実行ユニット、投機実行、
そしてその他多くの付加機能を備え、
幅広い命令を実行するのに適しています。
これにより、複雑な制御フローを持つ多種多様なプログラムを実行できます。
しかし、この見かけ上の強みは、同時に弱点でもあります。
汎用コアは構築コストが非常に高いのです。制御フローの多い汎用コードには優れています。
そのためには、実際に計算が行われる ALU（arithmetic logical unit）だけでなく、
前述の付加機能すべて、さらに
メモリインターフェース、コア間のキャッシュ制御、高速相互接続などにも
多くのチップ面積が必要になります。CPU は、
専用ハードウェアと比べると、単一のタスクに関しては相対的に不得手です。
現代のノートパソコンは 4--8 コアを備え、
ハイエンドサーバーでさえソケットあたり 64 コアを超えることはまれです。
単純に、費用対効果がよくないからです。

これに対して、GPU は数千の小さな処理要素から構成されることがあります（NIVIDA の最新の Ampere チップは最大 6912 CUDA コアを備えます）。
それらはしばしば、より大きなグループ（NVIDIA では warp と呼びます）にまとめられます。
詳細は NVIDIA、AMD、ARM、その他のチップベンダーで多少異なります。各コアは比較的弱く、
約 1GHz のクロック周波数で動作しますが、
そのようなコアの総数が、GPU を CPU より桁違いに高速にしています。
たとえば、NVIDIA の最近の Ampere A100 GPU は、特殊な16ビット精度（BFLOAT16）の行列—行列乗算で1チップあたり300 TFLOPs超を提供し、
より汎用的な浮動小数点演算（FP32）では最大20 TFLOPsを実現します。
一方、CPU の浮動小数点性能は 1 TFLOPs を超えることはめったにありません。たとえば、Amazon の Graviton 3 は16ビット精度演算でピーク 2 TFLOPs に達し、これは Apple の M1 プロセッサの GPU 性能に近い値です。

GPU が FLOPs の観点で CPU よりはるかに高速である理由は多くあります。
第一に、消費電力はクロック周波数に対して *二乗的* に増加する傾向があります。
したがって、4倍高速に動作する CPU コア1個の電力予算で（典型的な値です）、
速度を $\frac{1}{4}$ に落とした GPU コアを16個使えば、
$16 \times \frac{1}{4} = 4$ 倍の性能が得られます。
第二に、GPU コアははるかに単純であり
（実際、長い間、汎用コードを実行することすら *できませんでした*）、
そのためエネルギー効率が高いのです。たとえば、(i) 投機的実行を通常はサポートせず、(ii) 各処理要素を個別にプログラムすることは通常できず、(iii) コアごとのキャッシュははるかに小さい傾向があります。
最後に、深層学習の多くの演算は高いメモリ帯域幅を必要とします。
この点でも GPU は優れており、多くの CPU より少なくとも10倍広いバスを備えています。

2012年に戻りましょう。大きな突破口は、
Alex Krizhevsky と Ilya Sutskever が
GPU 上で動作する深い CNN を実装したときに訪れました。
彼らは、CNN における計算上のボトルネックである
畳み込みと行列乗算が、いずれもハードウェアで並列化できる演算であることに気づきました。
3GB のメモリを持つ 2枚の NVIDIA GTX 580 を用い、
それぞれが 1.5 TFLOPs を実行可能であったため（10年後でも多くの CPU にとっては依然として難題でした）、
彼らは高速な畳み込みを実装しました。
[cuda-convnet](https://code.google.com/archive/p/cuda-convnet/) のコードは十分に優れており、
その後数年間にわたって業界標準となり、
深層学習ブームの最初の数年間を支えました。

## AlexNet

8層 CNN を採用した AlexNet は、
ImageNet Large Scale Visual Recognition Challenge 2012 で
大差をつけて優勝しました :cite:`Russakovsky.Deng.Huang.ea.2013`。
このネットワークは、学習によって得られる特徴が手作業で設計された特徴を超えうることを、
コンピュータビジョンにおいて初めて示し、従来のパラダイムを打ち破りました。

AlexNet と LeNet のアーキテクチャは、 :numref:`fig_alexnet` が示すように、驚くほど似ています。
ここでは、2012年当時にモデルを2つの小さな GPU に収めるために必要だった
いくつかの設計上の癖を取り除いた、やや簡略化した AlexNet を示します。

![LeNet（左）から AlexNet（右）へ。](../img/alexnet.svg)
:label:`fig_alexnet`

AlexNet と LeNet の間には、重要な違いもあります。
第一に、AlexNet は比較的小さな LeNet-5 よりもはるかに深いです。
AlexNet は8層から成り、5つの畳み込み層、
2つの全結合隠れ層、1つの全結合出力層を含みます。
第二に、AlexNet は活性化関数として sigmoid ではなく ReLU を用いました。以下で詳細を見ていきましょう。

### アーキテクチャ

AlexNet の第1層では、畳み込みウィンドウの形状は $11\times11$ です。
ImageNet の画像は MNIST の画像よりも縦横ともに8倍大きいため、
ImageNet データ中の物体は、より多くの画素とより豊かな視覚的詳細を占める傾向があります。
したがって、物体を捉えるにはより大きな畳み込みウィンドウが必要です。
第2層の畳み込みウィンドウの形状は $5\times5$ に縮小され、その後に $3\times3$ が続きます。
さらに、第1、第2、第5の畳み込み層の後には、
ウィンドウ形状 $3\times3$、ストライド 2 の最大プーリング層が追加されます。
また、AlexNet の畳み込みチャネル数は LeNet の10倍です。

最後の畳み込み層の後には、4096出力を持つ非常に大きな全結合層が2つあります。
これらの層だけでほぼ1GBのモデルパラメータを必要とします。
初期の GPU はメモリが限られていたため、
元の AlexNet ではデュアルデータストリーム設計が採用され、
2つの GPU がそれぞれモデルの半分だけを保存・計算する役割を担っていました。
幸いなことに、現在は GPU メモリが比較的豊富なので、
最近ではモデルを GPU 間で分割する必要はほとんどありません
（この点で、ここでの AlexNet モデルは元論文と異なります）。

### 活性化関数

さらに、AlexNet は sigmoid 活性化関数を、より単純な ReLU 活性化関数に変更しました。第一に、ReLU 活性化関数の計算はより単純です。たとえば、sigmoid 活性化関数にある指数演算がありません。
第二に、ReLU 活性化関数は、異なるパラメータ初期化法を用いる場合にモデル訓練を容易にします。これは、sigmoid 活性化関数の出力が 0 または 1 に非常に近いとき、その領域の勾配はほぼ 0 となり、逆伝播がモデルパラメータの一部を更新し続けられなくなるためです。対照的に、ReLU 活性化関数の正の区間における勾配は常に 1 です（:numref:`subsec_activation-functions`）。したがって、モデルパラメータが適切に初期化されていない場合、sigmoid 関数では正の区間で勾配がほぼ 0 になり、モデルを効果的に訓練できない可能性があります。

### 容量制御と前処理

AlexNet は dropout（:numref:`sec_dropout`）によって
全結合層のモデル複雑度を制御する一方、
LeNet は重み減衰のみを使用します。
さらにデータを増強するために、AlexNet の訓練ループでは、
反転、切り抜き、色の変化など、
多くの画像拡張が追加されました。
これによりモデルはより頑健になり、実質的なサンプルサイズの増加によって過学習が抑えられます。
このような前処理手順の詳細なレビューについては :citet:`Buslaev.Iglovikov.Khvedchenya.ea.2020` を参照してください。

```{.python .input  n=5}
%%tab pytorch, mxnet, tensorflow
class AlexNet(d2l.Classifier):
    def __init__(self, lr=0.1, num_classes=10):
        super().__init__()
        self.save_hyperparameters()
        if tab.selected('mxnet'):
            self.net = nn.Sequential()
            self.net.add(
                nn.Conv2D(96, kernel_size=11, strides=4, activation='relu'),
                nn.MaxPool2D(pool_size=3, strides=2),
                nn.Conv2D(256, kernel_size=5, padding=2, activation='relu'),
                nn.MaxPool2D(pool_size=3, strides=2),
                nn.Conv2D(384, kernel_size=3, padding=1, activation='relu'),
                nn.Conv2D(384, kernel_size=3, padding=1, activation='relu'),
                nn.Conv2D(256, kernel_size=3, padding=1, activation='relu'),
                nn.MaxPool2D(pool_size=3, strides=2),
                nn.Dense(4096, activation='relu'), nn.Dropout(0.5),
                nn.Dense(4096, activation='relu'), nn.Dropout(0.5),
                nn.Dense(num_classes))
            self.net.initialize(init.Xavier())
        if tab.selected('pytorch'):
            self.net = nn.Sequential(
                nn.LazyConv2d(96, kernel_size=11, stride=4, padding=1),
                nn.ReLU(), nn.MaxPool2d(kernel_size=3, stride=2),
                nn.LazyConv2d(256, kernel_size=5, padding=2), nn.ReLU(),
                nn.MaxPool2d(kernel_size=3, stride=2),
                nn.LazyConv2d(384, kernel_size=3, padding=1), nn.ReLU(),
                nn.LazyConv2d(384, kernel_size=3, padding=1), nn.ReLU(),
                nn.LazyConv2d(256, kernel_size=3, padding=1), nn.ReLU(),
                nn.MaxPool2d(kernel_size=3, stride=2), nn.Flatten(),
                nn.LazyLinear(4096), nn.ReLU(), nn.Dropout(p=0.5),
                nn.LazyLinear(4096), nn.ReLU(),nn.Dropout(p=0.5),
                nn.LazyLinear(num_classes))
            self.net.apply(d2l.init_cnn)
        if tab.selected('tensorflow'):
            self.net = tf.keras.models.Sequential([
                tf.keras.layers.Conv2D(filters=96, kernel_size=11, strides=4,
                                       activation='relu'),
                tf.keras.layers.MaxPool2D(pool_size=3, strides=2),
                tf.keras.layers.Conv2D(filters=256, kernel_size=5, padding='same',
                                       activation='relu'),
                tf.keras.layers.MaxPool2D(pool_size=3, strides=2),
                tf.keras.layers.Conv2D(filters=384, kernel_size=3, padding='same',
                                       activation='relu'),
                tf.keras.layers.Conv2D(filters=384, kernel_size=3, padding='same',
                                       activation='relu'),
                tf.keras.layers.Conv2D(filters=256, kernel_size=3, padding='same',
                                       activation='relu'),
                tf.keras.layers.MaxPool2D(pool_size=3, strides=2),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(4096, activation='relu'),
                tf.keras.layers.Dropout(0.5),
                tf.keras.layers.Dense(4096, activation='relu'),
                tf.keras.layers.Dropout(0.5),
                tf.keras.layers.Dense(num_classes)])
```

```{.python .input}
%%tab jax
class AlexNet(d2l.Classifier):
    lr: float = 0.1
    num_classes: int = 10
    training: bool = True

    def setup(self):
        self.net = nn.Sequential([
            nn.Conv(features=96, kernel_size=(11, 11), strides=4, padding=1),
            nn.relu,
            lambda x: nn.max_pool(x, window_shape=(3, 3), strides=(2, 2)),
            nn.Conv(features=256, kernel_size=(5, 5)),
            nn.relu,
            lambda x: nn.max_pool(x, window_shape=(3, 3), strides=(2, 2)),
            nn.Conv(features=384, kernel_size=(3, 3)), nn.relu,
            nn.Conv(features=384, kernel_size=(3, 3)), nn.relu,
            nn.Conv(features=256, kernel_size=(3, 3)), nn.relu,
            lambda x: nn.max_pool(x, window_shape=(3, 3), strides=(2, 2)),
            lambda x: x.reshape((x.shape[0], -1)),  # flatten
            nn.Dense(features=4096),
            nn.relu,
            nn.Dropout(0.5, deterministic=not self.training),
            nn.Dense(features=4096),
            nn.relu,
            nn.Dropout(0.5, deterministic=not self.training),
            nn.Dense(features=self.num_classes)
        ])
```

高さと幅がともに224の[**単一チャネルのデータ例を構成**]し[**各層の出力形状を観察するため**]、 :numref:`fig_alexnet` の AlexNet アーキテクチャに対応させます。

```{.python .input  n=6}
%%tab pytorch, mxnet
AlexNet().layer_summary((1, 1, 224, 224))
```

```{.python .input  n=7}
%%tab tensorflow
AlexNet().layer_summary((1, 224, 224, 1))
```

```{.python .input}
%%tab jax
AlexNet(training=False).layer_summary((1, 224, 224, 1))
```

## 訓練

AlexNet は :citet:`Krizhevsky.Sutskever.Hinton.2012` で ImageNet 上で訓練されましたが、
ここでは Fashion-MNIST を使います。
ImageNet モデルを収束まで訓練するには、現代の GPU でも数時間から数日かかることがあるからです。
AlexNet を [**Fashion-MNIST**] に直接適用する際の問題の一つは、
その [**画像の解像度が低い**]（$28 \times 28$ ピクセル）
[**ImageNet の画像よりも。**] ことです。
これを動作させるために、[**$224 \times 224$ にアップサンプリングします**]。
これは、情報を増やさずに計算複雑度を単に増やすだけなので、一般には賢明な方法ではありません。それでも、AlexNet のアーキテクチャに忠実であるためにここではそうします。
このリサイズは `d2l.FashionMNIST` コンストラクタの `resize` 引数で行います。

これで、[**AlexNet の訓練を開始できます。**]
:numref:`sec_lenet` の LeNet と比べると、
ここでの主な変更点は、より小さな学習率の使用と、
より深く広いネットワーク、より高い画像解像度、そしてより高コストな畳み込みによる、はるかに遅い訓練です。

```{.python .input  n=8}
%%tab pytorch, mxnet, jax
model = AlexNet(lr=0.01)
data = d2l.FashionMNIST(batch_size=128, resize=(224, 224))
trainer = d2l.Trainer(max_epochs=10, num_gpus=1)
trainer.fit(model, data)
```

```{.python .input  n=9}
%%tab tensorflow
trainer = d2l.Trainer(max_epochs=10)
data = d2l.FashionMNIST(batch_size=128, resize=(224, 224))
with d2l.try_gpu():
    model = AlexNet(lr=0.01)
    trainer.fit(model, data)
```

## 議論

AlexNet の構造は LeNet と驚くほど似ていますが、精度向上（dropout）と訓練のしやすさ（ReLU）の両面でいくつかの重要な改良が加えられています。同様に驚くべきなのは、深層学習ツールの進歩の大きさです。2012年には数か月かかった作業が、今ではどの現代的フレームワークでも十数行のコードで実現できます。

アーキテクチャを見直すと、AlexNet には効率面で弱点があることがわかります。最後の2つの隠れ層は、それぞれサイズ $6400 \times 4096$ と $4096 \times 4096$ の行列を必要とします。これは 164 MB のメモリと 81 MFLOPs の計算に相当し、どちらも、特に携帯電話のような小型デバイスでは無視できない負担です。これが、AlexNet が後続の節で扱う、はるかに効果的なアーキテクチャに取って代わられた理由の一つです。それでも、これは現在使われている浅いネットワークから深いネットワークへの重要な一歩です。実験では、パラメータ数が訓練データ量をはるかに上回っているにもかかわらず（最後の2層だけで 4000万以上のパラメータがあり、6万枚の画像からなるデータセットで訓練しています）、過学習はほとんど見られません。訓練損失と検証損失は、訓練を通して事実上同一です。これは、現代の深層ネットワーク設計に本質的な dropout などの正則化が改善されたためです。

AlexNet の実装は LeNet より数行多いだけに見えますが、この概念的変化を受け入れ、その優れた実験結果を活用するまでに、学術界は長い年月を要しました。これも、効率的な計算ツールが欠けていたためです。当時は DistBelief :cite:`Dean.Corrado.Monga.ea.2012` も Caffe :cite:`Jia.Shelhamer.Donahue.ea.2014` も存在せず、Theano :cite:`Bergstra.Breuleux.Bastien.ea.2010` もまだ多くの特徴を欠いていました。状況を劇的に変えたのは、TensorFlow :cite:`Abadi.Barham.Chen.ea.2016` の登場でした。

## 演習

1. 上の議論を踏まえて、AlexNet の計算特性を分析しなさい。
    1. 畳み込み層と全結合層それぞれのメモリ使用量を計算しなさい。どちらが支配的か。
    1. 畳み込み層と全結合層の計算コストを計算しなさい。
    1. メモリ（読み書き帯域幅、レイテンシ、サイズ）は計算にどのように影響するか。訓練と推論でその影響に違いはあるか。
1. あなたはチップ設計者であり、計算とメモリ帯域幅のトレードオフを考えなければならない。たとえば、より高速なチップはより多くの電力を消費し、場合によってはより大きなチップ面積を必要とする。より多いメモリ帯域幅は、より多くのピンと制御ロジックを必要とし、やはり面積を増やす。どのように最適化するか。
1. なぜエンジニアはもはや AlexNet の性能ベンチマークを報告しないのか。
1. AlexNet の訓練でエポック数を増やしてみなさい。LeNet と比べて、結果はどう異なるか。なぜか。
1. AlexNet は Fashion-MNIST データセットには複雑すぎるかもしれない。特に初期画像の解像度が低いためである。
    1. 訓練を速くしつつ、精度が大きく低下しないようにモデルを単純化してみなさい。
    1. $28 \times 28$ 画像に直接適用できる、より良いモデルを設計しなさい。
1. バッチサイズを変更し、スループット（images/s）、精度、GPU メモリの変化を観察しなさい。
1. LeNet-5 に dropout と ReLU を適用しなさい。改善するか。画像に内在する不変性を利用するために前処理を行うことで、さらに改善できるか。
1. AlexNet を過学習させることはできるか。訓練を破綻させるには、どの特徴を取り除くか、あるいは変更する必要があるか。\n
