# 序文

ほんの数年前まで、大企業やスタートアップで知的な製品やサービスを開発する深層学習研究者の大軍勢は存在しませんでした。
私たちがこの分野に入った当時、機械学習は日刊紙の見出しを飾るようなものではありませんでした。
私たちの親は機械学習が何であるかをまったく知らず、ましてやなぜ医学や法学といった王道のキャリアではなく、あえてこの分野を選ぶのか理解していませんでした。
当時の機械学習は、将来性は期待されていたものの、主として学術的な分野に留まっており、産業界での重要性は音声認識やコンピュータビジョンなど、ごく限られた実世界への応用に限られていました。
さらに、これらの応用の多くは膨大なドメイン知識を必要としたため、機械学習は全体の一部にすぎず、（特定の応用領域ごとに）まったく別個の分野だとみなされることもしばしばでした。
また、本書で扱う深層学習手法の前身であるニューラルネットワークは、当時、一般に時代遅れだと考えられていました。


しかし、わずか数年のうちに、深層学習は世界を驚かせ、コンピュータビジョン、自然言語処理、自動音声認識、強化学習、生体医療情報学といった多岐にわたる分野で急速な進歩を牽引してきました。
さらに、実用上極めて重要な多くの課題で深層学習が成功したことは、理論機械学習や統計学の発展をも促しました。
こうした進歩を手にした今では、以前よりもはるかに高い自律性を備えた自動運転車（もっとも、いくつかの企業が主張するほどではありませんが）や、曖昧な点を質問してコードをデバッグする対話システム、そしてかつては何十年も先のことだと思われていた、囲碁のようなボードゲームで世界最高の人間プレイヤーを打ち負かすソフトウェアエージェントを構築できるようになりました。
また、これらのツールは産業や社会にますます広範な影響を及ぼしており、映画の制作方法や病気の診断方法を根本から変えつつあり、天体物理学から気候モデリング、気象予測、生物医学に至る基礎科学の分野でも重要な役割を果たしつつあります。



## 本書について

本書は、深層学習を身近なものにするための私たちの試みであり、*概念*、*文脈*、そして*コード*を教えることを目的としています。

### コード、数式、HTMLを一体化した媒体

あらゆる計算技術がその真価を発揮するためには、十分に理解され、十分に文書化され、成熟して保守の行き届いたツールによって支えられていなければなりません。
重要なアイデアは明確に要約され、新しい実践者が最新の知識に追いつくために必要な導入時間を最小限に抑えるべきです。
成熟したライブラリは一般的な作業を自動化し、模範的なコードは実践者が一般的な応用を自分のニーズに合わせて修正・適用・拡張しやすくするべきです。


例として、動的なWebアプリケーションを考えてみましょう。
1990年代には、Amazonのような多くの企業がデータベース駆動型のWebアプリケーションを成功裏に開発していましたが、この技術が創造的な起業家を支援する潜在能力を本格的に開花させたのは、強力で文書化の行き届いたフレームワークの発展にも後押しされた、ここ10年ほどのことでした。


深層学習の可能性を検証することには独特の難しさがあります。なぜなら、単一の応用であってもさまざまな分野の知識が結びつくからです。
深層学習を適用するには、同時に次のことを理解する必要があります。
(i) 問題を特定の形で定式化する動機
(ii) 与えられたモデルの数学的形式
(iii) モデルをデータに適合させるための最適化アルゴリズム
(iv) モデルが未知データに一般化すると期待できる条件を教えてくれる統計的原理と、実際に一般化したことを検証するための実践的手法
(v) モデルを効率よく学習させるために必要な工学的技法、すなわち数値計算の落とし穴を回避し、利用可能なハードウェアを最大限に活用する方法

問題を定式化するために必要な批判的思考力、それを解くための数学、そしてその解を実装するためのソフトウェアツールを、すべて一か所で教えることは、非常に大きな課題です。
本書の目標は、将来の実践者を迅速に実務レベルへと引き上げるための、統一的なリソースを提示することです。

本書の企画を始めたとき、次の条件を同時に満たすリソースは存在しませんでした。
(i) 常に最新であること
(ii) 現代の機械学習の実践の広がりを十分な技術的深さでカバーしていること
(iii) 教科書に期待される質の高い解説と、実践的チュートリアルに期待される洗練された実行可能コードを交互に提示していること

私たちは、特定の深層学習フレームワークの使い方を示すコード例（たとえば、TensorFlowで行列を使った基本的な計算を行う方法）や、特定の技法を実装するためのコード片（たとえば、LeNet、AlexNet、ResNet などのコードスニペット）が、さまざまなブログ記事やGitHubリポジトリに散在しているのを数多く見つけました。
しかし、これらの例は通常、ある手法を*どのように*実装するかに焦点を当てており、なぜそのようなアルゴリズム上の判断がなされたのか、という議論に欠けていました。
特定の話題を掘り下げるための、たとえばウェブサイト [Distill](http://distill.pub) に掲載された魅力的な記事や個人ブログのような対話的なリソースが散発的に現れることもありましたが、それらは深層学習の限られた話題しか扱っておらず、関連するコードが伴わないことも多くありました。
一方で、いくつかの深層学習の教科書――たとえば、深層学習の基礎を包括的に概説した :citet:`Goodfellow.Bengio.Courville.2016` など――も登場していましたが、これらのリソースは概念の説明とコードによる実装を結びつけておらず、読者がいざ実装しようとしたときに途方に暮れてしまうこともありました。
さらに、多くの優れたリソースが商業的な講座提供者の有料の壁（ペイウォール）の向こうに隠されています。

私たちは、次の条件を満たすリソースを作ろうとしました。
(i) 誰もが無料で利用できること
(ii) 実際に応用機械学習研究者となるための出発点として、十分な技術的深さを備えていること
(iii) 実行可能なコードを含み、実践で問題を*どのように*解くかを読者に示すこと
(iv) 私たち自身だけでなく、広いコミュニティを通じても迅速に更新できること
(v) 技術的詳細について対話的に議論し、質問に答えるための [フォーラム](https://discuss.d2l.ai/c/5) が併設されていること

これらの目標は、しばしば互いに衝突しました。
数式、定理、引用は LaTeX で管理・整形するのが最適です。
コードは Python で記述するのが最適です。
そしてWebページは HTML と JavaScript が本来の形式です。
さらに、読者がこれらの内容を、実行可能なコードとしても、紙の書籍としても、ダウンロード可能なPDFとしても、そしてインターネット上のWebサイトとしても利用できるようにしたいと考えました。
こうした要求をすべて満たす既存のワークフローは見当たらなかったため、私たちは独自にシステムを組み立てることにしました（:numref:`sec_how_to_contribute`）。
私たちは、ソースコードを共有しコミュニティの貢献を促進するために GitHub を採用し、コード・数式・テキストを混在させるために Jupyter notebook を採用し、レンダリングエンジンとして Sphinx を採用し、議論の場として Discourse を採用しました。
私たちのシステムは完璧ではありませんが、これらの選択は相反する要求の間で妥協点を見いだすものです。
私たちは、*Dive into Deep Learning* が、このような統合されたワークフローを用いて出版された最初の書籍かもしれないと自負しています。


### 実践しながら学ぶ

多くの教科書は、概念を順に提示し、それぞれを徹底的に詳述する構成をとります。
たとえば、:citet:`Bishop.2006` の優れた教科書は、各トピックを非常に丁寧に教えるため、線形回帰の章にたどり着くまでにかなりの労力を要します。
専門家はその徹底ぶりをまさに理由としてこの教科書を高く評価しますが、真の初心者にとっては、この性質が入門書としての参入障壁を高めることになります。

本書では、ほとんどの概念を*必要なときにその場で*教える手法をとります。
言い換えれば、何らかの実用的な目的を達成するために必要になった瞬間に、その概念を学ぶことになります。
冒頭では線形代数や確率といった基礎的な前提知識を学ぶために少し時間を割きますが、より難解な概念を気にする前に、まずは最初のモデルを学習させる満足感を読者にぜひ味わってほしいのです。

線形代数や確率の基礎を短期集中で学ぶためのいくつかの予備ノートブックを除けば、その後の各章は、適度な数の新しい概念を導入すると同時に、実データセットを用いた、自己完結した実行可能なコード例を提供します。
これは構成上の課題を生みました。
いくつかのモデルは、論理的には一つのノートブックにまとめることができます。
また、いくつかのアイデアは、複数のモデルを順番に実行しながら教えるのが最適かもしれません。
それに対して、私たちがとった*一つの動作例につき一つのノートブック* という方針には、大きな利点があります。
それは、読者が私たちのコードを活用して自分自身の研究プロジェクトを始めることを、できる限り簡単にしてくれるからです。
ノートブックをコピーして、自分の目的に合わせて修正を始めるだけでよいのです。

本書全体を通して、必要に応じて実行可能なコードと背景説明を交互に配置しています。
一般的に、私たちはツールを完全に説明する前に、まず使ってみて実感できる形にしています（より詳細な背景説明は後から補うことが多いです）。
たとえば、*確率的勾配降下法* を、その数学的な裏付けや直感的な仕組みを説明する前に実践で使うことがあります。
これは、実践者が問題を素早く解くための強力な武器をいち早く手に入れる助けになりますが、その代わりに、いくつかの学習上の「順序の逆転」を受け入れてもらう必要を生じさせます。

本書は深層学習の概念をゼロから教えます。
ときには、現代の深層学習フレームワークでは通常ユーザーから隠されているような、モデルの細部にまで踏み込みます。
これは特に基礎的なチュートリアルで顕著であり、それは読者に、ある層やオプティマイザの内部で何が起きているのかをすべて理解してほしいからです。
こうした場合、私たちはしばしばコード例を二つのバージョンで示します。
一つは、NumPy に似た機能と自動微分だけに頼り、すべてをゼロから実装するバージョンです。
もう一つは、深層学習フレームワークの高水準APIを使って簡潔なコードを書く、より実践的なバージョンです。
あるコンポーネントの動作原理をコードで説明した後は、その後のチュートリアルでは高水準APIによる実装に頼ることになります。


### 内容と構成

本書は、おおむね三つの部分に分けられます。
すなわち、前提知識、深層学習技法、そして実システムと応用に焦点を当てた高度な話題です（:numref:`fig_book_org`）。

![Book structure.](../img/book-org.svg)
:label:`fig_book_org`


* **第1部: 基礎と前提知識**。
:numref:`chap_introduction` は深層学習の入門です。
次に、 :numref:`chap_preliminaries` では、データの保存と操作の方法や、線形代数・微積分・確率の初歩的な概念に基づくさまざまな数値演算の適用方法など、実践的な深層学習に必要な前提知識を素早く身につけます。
:numref:`chap_regression` と :numref:`chap_perceptrons` では、回帰と分類、線形モデル、多層パーセプトロン、過学習と正則化を含む、深層学習における最も基本的な概念と技法を扱います。

* **第2部: 現代的な深層学習技法**。
:numref:`chap_computation` では、深層学習システムの主要な計算要素を説明し、より複雑なモデルを後続で実装するための基盤を築きます。
続いて、 :numref:`chap_cnn` と :numref:`chap_modern_cnn` では、現代のコンピュータビジョンシステムの大半の中核を成す強力なツールである畳み込みニューラルネットワーク（CNN）を紹介します。
同様に、 :numref:`chap_rnn` と :numref:`chap_modern_rnn` では、データの逐次的な（たとえば時間的な）構造を活用し、自然言語処理や時系列予測で一般的に用いられる再帰型ニューラルネットワーク（RNN）を紹介します。
:numref:`chap_attention-and-transformers` では、いわゆる*注意機構*に基づく比較的新しいモデル群を説明します。これは、ほとんどの自然言語処理タスクにおいてRNNに取って代わった支配的なアーキテクチャです。
これらの節を通じて、深層学習実践者が広く用いている、最も強力で汎用的なツールを身につけることができます。

* **第3部: スケーラビリティ、効率性、応用**（[オンライン](https://d2l.ai)で利用可能）。
第12章では、深層学習モデルの学習に用いられるいくつかの一般的な最適化アルゴリズムを議論します。
次に、第13章では、深層学習コードの計算性能に影響を与えるいくつかの重要な要因を検討します。
続いて、第14章では、コンピュータビジョンにおける深層学習の主要な応用を示します。
最後に、第15章と第16章では、言語表現モデルを事前学習し、それを自然言語処理タスクに適用する方法を示します。


### コード
:label:`sec_code`

本書の大部分の節には実行可能なコードが含まれています。
私たちは、試行錯誤を通じてコードを少しずつ調整し、その結果を観察することで、より良い直感が育まれると考えています。
理想的には、洗練された数学理論が、望ましい結果を得るためにコードをどのように調整すべきかを正確に教えてくれるはずです。
しかし、今日の深層学習実践者は、確かな理論による指針がない領域を手探りで進まざるを得ないことがしばしばあります。
私たちが最善を尽くしても、さまざまな技法がなぜ有効なのかという形式的な説明は、依然として多くの理由から不足しています。それらの理由は、これらのモデルを特徴づける数学が非常に難しいこと、説明が現在まだ明確に定義されていないデータの性質に依存している可能性があること、そしてこの分野への本格的な研究がようやく本格化したばかりであること、などです。
深層学習理論が進歩するにつれて、本書の将来の版では、現在利用可能な知識を凌駕するような深い洞察を提供できることを期待しています。

不要な繰り返しを避けるため、私たちは最も頻繁にインポートして使用する関数やクラスのいくつかを `d2l` パッケージにまとめています。
全体を通して、コードブロック（関数、クラス、あるいは import 文の集合など）には `#@save` を付け、それらが後で `d2l` パッケージ経由で参照されることを示します。
これらのクラスと関数の詳細な概要は :numref:`sec_d2l` で説明します。
`d2l` パッケージは軽量で、以下の依存関係だけを必要とします。

```{.python .input}
#@tab all
#@save
import inspect
import collections
from collections import defaultdict
from IPython import display
import math
from matplotlib import pyplot as plt
from matplotlib_inline import backend_inline
import os
import pandas as pd
import random
import re
import shutil
import sys
import tarfile
import time
import requests
import zipfile
import hashlib
d2l = sys.modules[__name__]
```

:begin_tab:`mxnet`
本書のコードの大部分は Apache MXNet を基盤にしています。Apache MXNet はオープンソースの深層学習フレームワークであり、AWS（Amazon Web Services）や多くの大学・企業で好んで使われています。
本書のすべてのコードは、最新の MXNet バージョンでテストを通過しています。
しかし、深層学習の急速な発展により、*印刷版* の一部のコードは将来の MXNet バージョンでは正しく動作しない可能性があります。
オンライン版は最新の状態に保つ予定です。
問題が発生した場合は、:ref:`chap_installation` を参照して、コードと実行環境を更新してください。
以下に MXNet 実装での依存関係を示します。
:end_tab:

:begin_tab:`pytorch`
本書のコードの大部分は PyTorch を基盤にしています。PyTorch は、深層学習研究コミュニティに熱心に受け入れられている人気のオープンソースフレームワークです。
本書のすべてのコードは、最新の安定版 PyTorch でテストを通過しています。
しかし、深層学習の急速な発展により、*印刷版* の一部のコードは将来の PyTorch バージョンでは正しく動作しない可能性があります。
オンライン版は最新の状態に保つ予定です。
問題が発生した場合は、:ref:`chap_installation` を参照して、コードと実行環境を更新してください。
以下に PyTorch 実装での依存関係を示します。
:end_tab:

:begin_tab:`tensorflow`
本書のコードの大部分は TensorFlow を基盤にしています。TensorFlow は、産業界で広く採用され、研究者の間でも人気のあるオープンソースの深層学習フレームワークです。
本書のすべてのコードは、最新の安定版 TensorFlow でテストを通過しています。
しかし、深層学習の急速な発展により、*印刷版* の一部のコードは将来の TensorFlow バージョンでは正しく動作しない可能性があります。
オンライン版は最新の状態に保つ予定です。
問題が発生した場合は、:ref:`chap_installation` を参照して、コードと実行環境を更新してください。
以下に TensorFlow 実装での依存関係を示します。
:end_tab:

:begin_tab:`jax`
本書のコードの大部分は Jax を基盤にしています。Jax は、任意の Python および NumPy 関数の微分や JIT コンパイル、ベクトル化など、合成可能な関数変換を可能にするオープンソースフレームワークです。機械学習研究の分野で人気が高まりつつあり、学びやすい NumPy 風のAPIを備えています。実際、JAX は NumPy との 1:1 の互換性を目指しているため、コードの切り替えは import 文を一つ変えるだけで済むかもしれません！
しかし、深層学習の急速な発展により、*印刷版* の一部のコードは将来の Jax バージョンでは正しく動作しない可能性があります。
オンライン版は最新の状態に保つ予定です。
問題が発生した場合は、:ref:`chap_installation` を参照して、コードと実行環境を更新してください。
以下に JAX 実装での依存関係を示します。
:end_tab:

```{.python .input}
#@tab mxnet
#@save
from mxnet import autograd, context, gluon, image, init, np, npx
from mxnet.gluon import nn, rnn
```

```{.python .input}
#@tab pytorch
#@save
import numpy as np
import torch
import torchvision
from torch import nn
from torch.nn import functional as F
from torchvision import transforms
from PIL import Image
from scipy.spatial import distance_matrix
```

```{.python .input}
#@tab tensorflow
#@save
import numpy as np
import tensorflow as tf
```

```{.python .input}
#@tab jax
#@save
from dataclasses import field
from functools import partial
import flax
from flax import linen as nn
from flax.training import train_state
import jax
from jax import numpy as jnp
from jax import grad, vmap
import numpy as np
import optax
import tensorflow as tf
import tensorflow_datasets as tfds
from types import FunctionType
from typing import Any
```

### 対象読者

本書は、深層学習の実践的技法をしっかり理解したい学生（学部生・大学院生）、エンジニア、研究者を対象としています。
すべての概念をゼロから説明するため、読者に深層学習や機械学習の事前知識は求めません。
しかし、深層学習の手法について十分に説明するには、ある程度の数学とプログラミングの知識が必要です。そのため本書では、線形代数、微積分、確率論、および Python プログラミングの基礎的な知識があることだけを前提とします。
万一何か忘れてしまった場合でも、本書に登場する数学の大半については、[オンライン付録](https://d2l.ai/chapter_appendix-mathematics-for-deep-learning/index.html) が復習用の解説を提供しています。
また、私たちは一部の例外を除き、数学的な厳密さよりも直感とアイデアを優先しています。
さらに深く数学的知識を広げたい読者のために、本書以外にも優れた参考資料をおすすめしています。
:citet:`Bollobas.1999` の *Linear Analysis* は、線形代数と関数解析を非常に深く扱っています。
*All of Statistics* :cite:`Wasserman.2013` は、統計学への素晴らしい入門書です。
Joe Blitzstein 氏による確率と推論に関する [書籍](https://www.amazon.com/Introduction-Probability-Chapman-Statistical-Science/dp/1138369918) と [講義](https://projects.iq.harvard.edu/stat110/home) は、教育的価値の高い素晴らしいリソースです。
また、これまで Python を使ったことがない読者には、こちらの [Python チュートリアル](http://learnpython.org/) を一読することをお勧めします。


### ノートブック、Webサイト、GitHub、フォーラム

すべてのノートブックは [D2L.ai のWebサイト](https://d2l.ai) と [GitHub](https://github.com/d2l-ai/d2l-en) からダウンロードできます。
本書に関連して、[discuss.d2l.ai](https://discuss.d2l.ai/c/5) に議論フォーラムを開設しました。
本書のどの節についても質問があるときは、各ノートブックの末尾にある関連議論ページへのリンクをたどることができます。



## 謝辞

英語版と中国語版の草稿の両方に対して、何百人もの貢献者に深く感謝します。
彼らは内容の改善に協力し、貴重なフィードバックを提供してくれました。
本書は当初、MXNet を主要フレームワークとして実装されました。
以前の MXNet コードの大部分を、それぞれ PyTorch と TensorFlow の実装へ適応させてくれた Anirudh Dagar と Yuan Tang に感謝します。
2021年7月以降、私たちは本書を PyTorch、MXNet、TensorFlow で再設計・再実装し、PyTorch を主要フレームワークとして採用しました。
より最近の PyTorch コードの大部分を JAX 実装へ適応させてくれた Anirudh Dagar に感謝します。
中国語版草稿において、より最近の PyTorch コードの大部分を PaddlePaddle 実装へ適応させてくれた Baidu の Gaosheng Wu、Liujun Hu、Ge Zhang、Jiehang Xie に感謝します。
PDF のビルドにおいて、出版社の LaTeX スタイルを統合してくれた Shuai Zhang に感謝します。

GitHub では、この英語版草稿をより良いものにしてくれたすべての貢献者に感謝します。
彼らの GitHub ID または名前は（順不同）次のとおりです。
alxnorden, avinashingit, bowen0701, brettkoonce, Chaitanya Prakash Bapat,
cryptonaut, Davide Fiocco, edgarroman, gkutiel, John Mitro, Liang Pu,
Rahul Agarwal, Mohamed Ali Jamaoui, Michael (Stu) Stewart, Mike Müller,
NRauschmayr, Prakhar Srivastav, sad-, sfermigier, Sheng Zha, sundeepteki,
topecongiro, tpdi, vermicelli, Vishaal Kapoor, Vishwesh Ravi Shrimali, YaYaB, Yuhong Chen,
Evgeniy Smirnov, lgov, Simon Corston-Oliver, Igor Dzreyev, Ha Nguyen, pmuens,
Andrei Lukovenko, senorcinco, vfdev-5, dsweet, Mohammad Mahdi Rahimi, Abhishek Gupta,
uwsd, DomKM, Lisa Oakley, Bowen Li, Aarush Ahuja, Prasanth Buddareddygari, brianhendee,
mani2106, mtn, lkevinzc, caojilin, Lakshya, Fiete Lüer, Surbhi Vijayvargeeya,
Muhyun Kim, dennismalmgren, adursun, Anirudh Dagar, liqingnz, Pedro Larroy,
lgov, ati-ozgur, Jun Wu, Matthias Blume, Lin Yuan, geogunow, Josh Gardner,
Maximilian Böther, Rakib Islam, Leonard Lausen, Abhinav Upadhyay, rongruosong,
Steve Sedlmeyer, Ruslan Baratov, Rafael Schlatter, liusy182, Giannis Pappas,
ati-ozgur, qbaza, dchoi77, Adam Gerson, Phuc Le, Mark Atwood, christabella, vn09,
Haibin Lin, jjangga0214, RichyChen, noelo, hansent, Giel Dops, dvincent1337, WhiteD3vil,
Peter Kulits, codypenta, joseppinilla, ahmaurya, karolszk, heytitle, Peter Goetz, rigtorp,
Tiep Vu, sfilip, mlxd, Kale-ab Tessera, Sanjar Adilov, MatteoFerrara, hsneto,
Katarzyna Biesialska, Gregory Bruss, Duy–Thanh Doan, paulaurel, graytowne, Duc Pham,
sl7423, Jaedong Hwang, Yida Wang, cys4, clhm, Jean Kaddour, austinmw, trebeljahr, tbaums,
Cuong V. Nguyen, pavelkomarov, vzlamal, NotAnotherSystem, J-Arun-Mani, jancio, eldarkurtic,
the-great-shazbot, doctorcolossus, gducharme, cclauss, Daniel-Mietchen, hoonose, biagiom,
abhinavsp0730, jonathanhrandall, ysraell, Nodar Okroshiashvili, UgurKap, Jiyang Kang,
StevenJokes, Tomer Kaftan, liweiwp, netyster, ypandya, NishantTharani, heiligerl, SportsTHU,
Hoa Nguyen, manuel-arno-korfmann-webentwicklung, aterzis-personal, nxby, Xiaoting He, Josiah Yoder,
mathresearch, mzz2017, jroberayalas, iluu, ghejc, BSharmi, vkramdev, simonwardjones, LakshKD,
TalNeoran, djliden, Nikhil95, Oren Barkan, guoweis, haozhu233, pratikhack, Yue Ying, tayfununal,
steinsag, charleybeller, Andrew Lumsdaine, Jiekui Zhang, Deepak Pathak, Florian Donhauser, Tim Gates,
Adriaan Tijsseling, Ron Medina, Gaurav Saha, Murat Semerci, Lei Mao, Levi McClenny, Joshua Broyde,
jake221, jonbally, zyhazwraith, Brian Pulfer, Nick Tomasino, Lefan Zhang, Hongshen Yang, Vinney Cavallo,
yuntai, Yuanxiang Zhu, amarazov, pasricha, Ben Greenawald, Shivam Upadhyay, Quanshangze Du, Biswajit Sahoo,
Parthe Pandit, Ishan Kumar, HomunculusK, Lane Schwartz, varadgunjal, Jason Wiener, Armin Gholampoor,
Shreshtha13, eigen-arnav, Hyeonggyu Kim, EmilyOng, Bálint Mucsányi, Chase DuBois, Juntian Tao,
Wenxiang Xu, Lifu Huang, filevich, quake2005, nils-werner, Yiming Li, Marsel Khisamutdinov,
Francesco "Fuma" Fumagalli, Peilin Sun, Vincent Gurgul, qingfengtommy, Janmey Shukla, Mo Shan,
Kaan Sancak, regob, AlexSauer, Gopalakrishna Ramachandra, Tobias Uelwer, Chao Wang, Tian Cao,
Nicolas Corthorn, akash5474, kxxt, zxydi1992, Jacob Britton, Shuangchi He, zhmou, krahets, Jie-Han Chen,
Atishay Garg, Marcel Flygare, adtygan, Nik Vaessen, bolded, Louis Schlessinger, Balaji Varatharajan,
atgctg, Kaixin Li, Victor Barbaros, Riccardo Musto, Elizabeth Ho, azimjonn, Guilherme Miotto, Alessandro Finamore,
Joji Joseph, Anthony Biel, Zeming Zhao, shjustinbaek, gab-chen, nantekoto, Yutaro Nishiyama, Oren Amsalem,
Tian-MaoMao, Amin Allahyar, Gijs van Tulder, Mikhail Berkov, iamorphen, Matthew Caseres, Andrew Walsh,
pggPL, RohanKarthikeyan, Ryan Choi, and Likun Lei.

本書の執筆にあたり、Amazon Web Services、特に Wen-Ming Ye、George Karypis、Swami Sivasubramanian、Peter DeSantis、Adam Selipsky、
そして Andrew Jassy の寛大な支援に感謝します。
十分な時間、リソース、同僚との議論、そして彼らからの絶え間ない励ましがなければ、この本は実現しなかったでしょう。
出版準備の過程では、Cambridge University Press が素晴らしい支援を提供してくれました。
私たちは、委嘱編集者 David Tranah の助力とプロフェッショナリズムに深く感謝します。


## 要約

深層学習はパターン認識に革命をもたらし、コンピュータビジョン、自然言語処理、自動音声認識といった多様な分野で、現在さまざまな技術を支える基盤となっています。
深層学習をうまく適用するには、問題をどのように定式化するか、モデリングの基本数学、モデルをデータに適合させるアルゴリズム、そしてそれらすべてを実装する工学的技法を理解する必要があります。
本書は、文章、図、数式、コードをすべて一か所にまとめた包括的なリソースを提供します。



## 演習

1. 本書の議論フォーラム [discuss.d2l.ai](https://discuss.d2l.ai/) にアカウントを登録してください。
1. 自分のコンピュータに Python をインストールしてください。
1. 節の下部にあるリンクからフォーラムへ移動し、著者やより広いコミュニティと交流しながら、助けを求めたり、本書について議論したり、質問への答えを見つけたりしてください。

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/18)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/20)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/186)
:end_tab:

:begin_tab:`jax`
[Discussions](https://discuss.d2l.ai/t/17963)
:end_tab:\n