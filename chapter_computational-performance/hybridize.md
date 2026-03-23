# コンパイラとインタプリタ
:label:`sec_hybridize`

これまで本書では、`print`、`+`、`if` などの文を用いてプログラムの状態を変更する命令型プログラミングに焦点を当ててきました。以下に、単純な命令型プログラムの例を示します。

```{.python .input}
#@tab all
def add(a, b):
    return a + b

def fancy_func(a, b, c, d):
    e = add(a, b)
    f = add(c, d)
    g = add(e, f)
    return g

print(fancy_func(1, 2, 3, 4))
```

Python は *インタプリタ型言語* です。上の `fancy_func` 関数を評価するとき、関数本体を構成する操作を *順番に* 実行します。つまり、`e = add(a, b)` を評価して結果を変数 `e` に保存し、それによってプログラムの状態を変更します。続く 2 つの文 `f = add(c, d)` と `g = add(e, f)` も同様に実行され、加算が行われて結果が変数として保存されます。:numref:`fig_compute_graph` はデータの流れを示しています。

![命令型プログラムにおけるデータフロー。](../img/computegraph.svg)
:label:`fig_compute_graph`

命令型プログラミングは便利ですが、非効率な場合があります。ひとつには、`fancy_func` 全体で `add` 関数が繰り返し呼び出されても、Python は 3 回の関数呼び出しを個別に実行するからです。これらが、たとえば GPU 上（あるいは複数 GPU 上）で実行される場合、Python インタプリタに起因するオーバーヘッドが非常に大きくなることがあります。さらに、`fancy_func` 内のすべての文が実行されるまで、変数 `e` と `f` の値を保存しておく必要があります。これは、文 `e = add(a, b)` と `f = add(c, d)` が実行された後に、変数 `e` と `f` がプログラムの他の部分で使われるかどうかが分からないためです。

## シンボリックプログラミング

代替案として *シンボリックプログラミング* を考えましょう。これは通常、処理が完全に定義されてから計算を行う方式です。この戦略は、Theano や TensorFlow（後者は命令型拡張を取り込んでいます）を含む複数の深層学習フレームワークで使われています。通常、次の手順を含みます。

1. 実行する操作を定義する。
1. 操作を実行可能なプログラムにコンパイルする。
1. 必要な入力を与え、コンパイル済みプログラムを呼び出して実行する。

これにより、大幅な最適化が可能になります。まず、多くの場合 Python インタプリタを省略できるため、単一の Python スレッドが CPU 上で動作しながら複数の高速 GPU を使うような状況で顕著になりうる性能ボトルネックを取り除けます。  
第二に、コンパイラは上のコードを `print((1 + 2) + (3 + 4))`、あるいは `print(10)` にまで最適化して書き換えられるかもしれません。これは、コンパイラが機械命令に変換する前にコード全体を見渡せるから可能です。たとえば、ある変数が不要になった時点でメモリを解放したり（あるいは最初から確保しなかったり）できます。また、コード全体を等価な別のコードへ変換することもできます。  
よりよく理解するために、以下の命令型プログラミングのシミュレーション（結局のところ Python です）を見てみましょう。

```{.python .input}
#@tab all
def add_():
    return '''
def add(a, b):
    return a + b
'''

def fancy_func_():
    return '''
def fancy_func(a, b, c, d):
    e = add(a, b)
    f = add(c, d)
    g = add(e, f)
    return g
'''

def evoke_():
    return add_() + fancy_func_() + 'print(fancy_func(1, 2, 3, 4))'

prog = evoke_()
print(prog)
y = compile(prog, '', 'exec')
exec(y)
```

命令型（インタプリタ型）プログラミングとシンボリックプログラミングの違いは次のとおりです。

* 命令型プログラミングは簡単です。Python で命令型プログラミングを使う場合、コードの大部分は素直で書きやすいものです。また、命令型プログラミングのコードはデバッグしやすいです。これは、関連する中間変数の値をすべて取得して表示したり、Python に組み込まれたデバッグツールを使ったりしやすいためです。
* シンボリックプログラミングはより効率的で、移植もしやすいです。シンボリックプログラミングでは、コンパイル中にコードを最適化しやすいだけでなく、Python に依存しない形式へプログラムを移植することもできます。これにより、Python 以外の環境でもプログラムを実行でき、Python インタプリタに関連する潜在的な性能問題を回避できます。


## ハイブリッドプログラミング

歴史的に、多くの深層学習フレームワークは命令型かシンボリック型のどちらかを選んできました。たとえば、Theano、TensorFlow（前者に触発されました）、Keras、CNTK はモデルをシンボリックに記述します。逆に、Chainer と PyTorch は命令型アプローチを採用しています。TensorFlow 2.0 と Keras には、後の改訂で命令型モードが追加されました。

:begin_tab:`mxnet`
Gluon を設計する際、開発者たちは両方のプログラミングパラダイムの利点を組み合わせられるかどうかを検討しました。その結果、ユーザーは純粋な命令型プログラミングで開発とデバッグを行いながら、製品レベルの計算性能やデプロイが必要なときには、ほとんどのプログラムをシンボリックプログラムへ変換して実行できるハイブリッドモデルが生まれました。

実際には、`HybridBlock` または `HybridSequential` クラスを使ってモデルを構築します。デフォルトでは、どちらも命令型プログラミングにおける `Block` や `Sequential` クラスと同じように実行されます。  
`HybridSequential` クラスは `HybridBlock` のサブクラスです（`Sequential` が `Block` のサブクラスであるのと同じです）。`hybridize` 関数が呼ばれると、Gluon はモデルをシンボリックプログラミングで使われる形式にコンパイルします。これにより、モデルの実装方法を犠牲にすることなく、計算集約的な部分を最適化できます。以下では、その利点を順次モデルとブロックに焦点を当てて示します。
:end_tab:

:begin_tab:`pytorch`
上で述べたように、PyTorch は命令型プログラミングに基づいており、動的計算グラフを使用します。シンボリックプログラミングの移植性と効率性を活用するため、開発者たちは両方のプログラミングパラダイムの利点を組み合わせられるかどうかを検討しました。その結果、ユーザーは純粋な命令型プログラミングで開発とデバッグを行いながら、製品レベルの計算性能やデプロイが必要なときには、ほとんどのプログラムをシンボリックプログラムへ変換して実行できる torchscript が生まれました。
:end_tab:

:begin_tab:`tensorflow`
命令型プログラミングパラダイムは現在、Tensorflow 2 のデフォルトであり、初学者にとって歓迎すべき変更です。しかし、同じシンボリックプログラミング技術と、それに続く計算グラフは TensorFlow にも依然として存在しており、使いやすい `tf.function` デコレータからアクセスできます。これにより、命令型プログラミングパラダイムが TensorFlow に導入され、より直感的な関数を定義し、それをラップして TensorFlow チームが [autograph](https://www.tensorflow.org/api_docs/python/tf/autograph) と呼ぶ機能を使って自動的に計算グラフへコンパイルできるようになりました。
:end_tab:

## `Sequential` クラスのハイブリッド化

ハイブリッド化の仕組みを理解する最も簡単な方法は、複数層を持つ深いネットワークを考えることです。従来は、Python インタプリタがすべての層のコードを実行して、CPU や GPU に転送できる命令を生成する必要がありました。単一の（高速な）計算デバイスであれば、これは大きな問題にはなりません。一方、AWS P3dn.24xlarge インスタンスのような高度な 8 GPU サーバーを使う場合、Python はすべての GPU を忙しく保つのに苦労します。ここでは単一スレッドの Python インタプリタがボトルネックになります。`Sequential` を `HybridSequential` に置き換えることで、コードのかなりの部分についてこの問題にどう対処できるか見てみましょう。まず、単純な MLP を定義します。

```{.python .input}
#@tab mxnet
from d2l import mxnet as d2l
from mxnet import np, npx
from mxnet.gluon import nn
npx.set_np()

# Factory for networks
def get_net():
    net = nn.HybridSequential()  
    net.add(nn.Dense(256, activation='relu'),
            nn.Dense(128, activation='relu'),
            nn.Dense(2))
    net.initialize()
    return net

x = np.random.normal(size=(1, 512))
net = get_net()
net(x)
```

```{.python .input}
#@tab pytorch
from d2l import torch as d2l
import torch
from torch import nn

# Factory for networks
def get_net():
    net = nn.Sequential(nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 2))
    return net

x = torch.randn(size=(1, 512))
net = get_net()
net(x)
```

```{.python .input}
#@tab tensorflow
from d2l import tensorflow as d2l
import tensorflow as tf
from tensorflow.keras.layers import Dense

# Factory for networks
def get_net():
    net = tf.keras.Sequential()
    net.add(Dense(256, input_shape = (512,), activation = "relu"))
    net.add(Dense(128, activation = "relu"))
    net.add(Dense(2, activation = "linear"))
    return net

x = tf.random.normal([1,512])
net = get_net()
net(x)
```

:begin_tab:`mxnet`
`hybridize` 関数を呼び出すことで、MLP 内の計算をコンパイルして最適化できます。モデルの計算結果は変わりません。
:end_tab:

:begin_tab:`pytorch`
`torch.jit.script` 関数を使ってモデルを変換することで、MLP 内の計算をコンパイルして最適化できます。モデルの計算結果は変わりません。
:end_tab:

:begin_tab:`tensorflow`
以前は、TensorFlow で構築されたすべての関数は計算グラフとして作成され、そのためデフォルトで JIT コンパイルされていました。しかし、TensorFlow 2.X と EagerTensor の登場により、これはもはやデフォルトの動作ではありません。  
この機能は tf.function で再び有効にできます。tf.function は関数デコレータとして使われることが多いですが、以下に示すように通常の Python 関数として直接呼び出すこともできます。モデルの計算結果は変わりません。
:end_tab:

```{.python .input}
#@tab mxnet
net.hybridize()
net(x)
```

```{.python .input}
#@tab pytorch
net = torch.jit.script(net)
net(x)
```

```{.python .input}
#@tab tensorflow
net = tf.function(net)
net(x)
```

:begin_tab:`mxnet`
これは、あまりにも都合がよすぎるように見えるかもしれません。単にブロックを `HybridSequential` にし、以前と同じコードを書いて `hybridize` を呼び出すだけです。こうするとネットワークは最適化されます（性能は後でベンチマークします）。残念ながら、これがすべての層に魔法のように効くわけではありません。つまり、ある層が `HybridBlock` クラスではなく `Block` クラスを継承している場合、その層は最適化されません。
:end_tab:

:begin_tab:`pytorch`
これは、あまりにも都合がよすぎるように見えるかもしれません。以前と同じコードを書き、単に `torch.jit.script` を使ってモデルを変換するだけです。こうするとネットワークは最適化されます（性能は後でベンチマークします）。
:end_tab:

:begin_tab:`tensorflow`
これは、あまりにも都合がよすぎるように見えるかもしれません。以前と同じコードを書き、単に `tf.function` を使ってモデルを変換するだけです。こうするとネットワークは TensorFlow の MLIR 中間表現として計算グラフに構築され、コンパイラレベルで大幅に最適化され、高速に実行されます（性能は後でベンチマークします）。
`tf.function()` 呼び出しに `jit_compile = True` フラグを明示的に追加すると、TensorFlow で XLA（Accelerated Linear Algebra）機能が有効になります。XLA は、特定の状況では JIT コンパイルされたコードをさらに最適化できます。グラフモード実行はこの明示的な指定がなくても有効ですが、XLA により、特に GPU 環境では、深層学習アプリケーションで見られるような大規模な線形代数演算をさらに高速化できることがあります。
:end_tab:

### ハイブリッド化による高速化

コンパイルによって得られる性能向上を示すために、ハイブリッド化の前後で `net(x)` の評価に必要な時間を比較します。まず、この時間を測定するクラスを定義しましょう。これは、性能を測定し（そして改善し）ようとする本章全体で役立ちます。

```{.python .input}
#@save
class Benchmark:
    """For measuring running time."""
    def __init__(self, description='Done'):
        self.description = description

    def __enter__(self):
        self.timer = d2l.Timer()
        return self

    def __exit__(self, *args):
        print(f'{self.description}: {self.timer.stop():.4f} sec')
```

:begin_tab:`mxnet`
では、ネットワークを 2 回呼び出してみましょう。1 回はハイブリッド化なし、もう 1 回はハイブリッド化ありです。
:end_tab:

:begin_tab:`pytorch`
では、ネットワークを 2 回呼び出してみましょう。1 回は torchscript なし、もう 1 回は torchscript ありです。
:end_tab:

:begin_tab:`tensorflow`
では、ネットワークを 3 回呼び出してみましょう。1 回は eager 実行、1 回はグラフモード実行、そしてもう 1 回は JIT コンパイルされた XLA を使います。
:end_tab:

```{.python .input}
#@tab mxnet
net = get_net()
with Benchmark('Without hybridization'):
    for i in range(1000): net(x)
    npx.waitall()

net.hybridize()
with Benchmark('With hybridization'):
    for i in range(1000): net(x)
    npx.waitall()
```

```{.python .input}
#@tab pytorch
net = get_net()
with Benchmark('Without torchscript'):
    for i in range(1000): net(x)

net = torch.jit.script(net)
with Benchmark('With torchscript'):
    for i in range(1000): net(x)
```

```{.python .input}
#@tab tensorflow
net = get_net()
with Benchmark('Eager Mode'):
    for i in range(1000): net(x)

net = tf.function(net)
with Benchmark('Graph Mode'):
    for i in range(1000): net(x)
```

:begin_tab:`mxnet`
上の結果から分かるように、`HybridSequential` のインスタンスが `hybridize` 関数を呼び出した後は、シンボリックプログラミングの利用によって計算性能が向上します。
:end_tab:

:begin_tab:`pytorch`
上の結果から分かるように、`nn.Sequential` のインスタンスを `torch.jit.script` 関数でスクリプト化した後は、シンボリックプログラミングの利用によって計算性能が向上します。
:end_tab:

:begin_tab:`tensorflow`
上の結果から分かるように、`tf.keras.Sequential` のインスタンスを `tf.function` 関数でスクリプト化した後は、TensorFlow のグラフモード実行によるシンボリックプログラミングの利用によって計算性能が向上します。 
:end_tab:

### シリアライズ

:begin_tab:`mxnet`
モデルをコンパイルする利点のひとつは、モデルとそのパラメータをディスクにシリアライズ（保存）できることです。これにより、選択したフロントエンド言語に依存しない形でモデルを保存できます。したがって、学習済みモデルを他のデバイスへデプロイしたり、他のフロントエンドプログラミング言語を簡単に使ったりできます。同時に、コードは命令型プログラミングで達成できるものよりもしばしば高速です。`export` 関数を見てみましょう。
:end_tab:

:begin_tab:`pytorch`
モデルをコンパイルする利点のひとつは、モデルとそのパラメータをディスクにシリアライズ（保存）できることです。これにより、選択したフロントエンド言語に依存しない形でモデルを保存できます。したがって、学習済みモデルを他のデバイスへデプロイしたり、他のフロントエンドプログラミング言語を簡単に使ったりできます。同時に、コードは命令型プログラミングで達成できるものよりもしばしば高速です。`save` 関数を見てみましょう。
:end_tab:

:begin_tab:`tensorflow`
モデルをコンパイルする利点のひとつは、モデルとそのパラメータをディスクにシリアライズ（保存）できることです。これにより、選択したフロントエンド言語に依存しない形でモデルを保存できます。したがって、学習済みモデルを他のデバイスへデプロイしたり、他のフロントエンドプログラミング言語を簡単に使ったり、あるいは学習済みモデルをサーバー上で実行したりできます。同時に、コードは命令型プログラミングで達成できるものよりもしばしば高速です。  
TensorFlow で保存を可能にする低レベル API は `tf.saved_model` です。  
`saved_model` インスタンスを見てみましょう。
:end_tab:

```{.python .input}
#@tab mxnet
net.export('my_mlp')
!ls -lh my_mlp*
```

```{.python .input}
#@tab pytorch
net.save('my_mlp')
!ls -lh my_mlp*
```

```{.python .input}
#@tab tensorflow
net = get_net()
tf.saved_model.save(net, 'my_mlp')
!ls -lh my_mlp*
```

:begin_tab:`mxnet`
モデルは、大きなバイナリのパラメータファイルと、モデル計算の実行に必要なプログラムの JSON 記述に分解されます。これらのファイルは、C++、R、Scala、Perl など、Python や MXNet がサポートする他のフロントエンド言語から読み込めます。モデル記述の最初の数行を見てみましょう。
:end_tab:

```{.python .input}
#@tab mxnet
!head my_mlp-symbol.json
```

:begin_tab:`mxnet`
先ほど、`hybridize` 関数を呼び出した後は、モデルが優れた計算性能と移植性を達成できることを示しました。ただし、ハイブリッド化はモデルの柔軟性、特に制御フローの面に影響を与える可能性があることに注意してください。  

また、`Block` インスタンスでは `forward` 関数を使う必要があるのに対し、`HybridBlock` インスタンスでは `hybrid_forward` 関数を使う必要があります。
:end_tab:

```{.python .input}
#@tab mxnet
class HybridNet(nn.HybridBlock):
    def __init__(self, **kwargs):
        super(HybridNet, self).__init__(**kwargs)
        self.hidden = nn.Dense(4)
        self.output = nn.Dense(2)

    def hybrid_forward(self, F, x):
        print('module F: ', F)
        print('value  x: ', x)
        x = F.npx.relu(self.hidden(x))
        print('result  : ', x)
        return self.output(x)
```

:begin_tab:`mxnet`
上のコードは、4 個の隠れユニットと 2 個の出力を持つ単純なネットワークを実装しています。`hybrid_forward` 関数は追加の引数 `F` を取ります。これは、コードがハイブリッド化されているかどうかによって、処理に使うライブラリが少し異なる（`ndarray` か `symbol`）ために必要です。両者は非常によく似た機能を持っており、MXNet が自動的に引数を決定します。何が起きているかを理解するために、関数呼び出しの一部として引数を表示しています。
:end_tab:

```{.python .input}
#@tab mxnet
net = HybridNet()
net.initialize()
x = np.random.normal(size=(1, 3))
net(x)
```

:begin_tab:`mxnet`
順伝播計算を繰り返しても同じ出力になります（詳細は省略します）。では、`hybridize` 関数を呼び出すと何が起こるか見てみましょう。
:end_tab:

```{.python .input}
#@tab mxnet
net.hybridize()
net(x)
```

:begin_tab:`mxnet`
`ndarray` の代わりに、`F` として `symbol` モジュールを使うようになっています。さらに、入力は `ndarray` 型であるにもかかわらず、ネットワークを流れるデータはコンパイル過程の一部として `symbol` 型に変換されます。関数呼び出しを繰り返すと、驚くべき結果になります。
:end_tab:

```{.python .input}
#@tab mxnet
net(x)
```

:begin_tab:`mxnet` 
これは先ほど見たものとはかなり異なります。`hybrid_forward` で定義したすべての `print` 文が省略されています。実際、ハイブリッド化後は `net(x)` の実行に Python インタプリタはもはや関与しません。つまり、`print` 文のような余計な Python コードは省かれ、より洗練された実行と高い性能が得られます。代わりに、MXNet は C++ バックエンドを直接呼び出します。また、`symbol` モジュールではサポートされない関数（たとえば `asnumpy`）があること、さらに `a += b` や `a[:] = a + b` のようなインプレース演算は `a = a + b` と書き換える必要があることにも注意してください。それでも、速度が重要な場合にはモデルのコンパイルは十分に価値があります。利点は、モデルの複雑さ、CPU の速度、GPU の速度と数に応じて、数パーセントから 2 倍以上の高速化まで幅があります。
:end_tab:

## まとめ


* 命令型プログラミングでは、制御フローを使え、Python の豊富なソフトウェアエコシステムを活用できるため、新しいモデルを設計しやすいです。
* シンボリックプログラミングでは、実行前にプログラムを指定してコンパイルする必要があります。その利点は性能向上です。

:begin_tab:`mxnet` 
* MXNet は必要に応じて、両方のアプローチの利点を組み合わせられます。
* `HybridSequential` と `HybridBlock` クラスで構築したモデルは、`hybridize` 関数を呼び出すことで命令型プログラムをシンボリックプログラムへ変換できます。
:end_tab:


## 演習


:begin_tab:`mxnet` 
1. この節の `HybridNet` クラスの `hybrid_forward` 関数の最初の行に `x.asnumpy()` を追加してください。コードを実行し、発生するエラーを観察してください。なぜそのようなことが起こるのでしょうか。
1. 制御フロー、すなわち `hybrid_forward` 関数内に Python の `if` と `for` 文を追加するとどうなりますか。
1. 以前の章で興味を持ったモデルを見直してください。再実装することで計算性能を改善できますか。
:end_tab:

:begin_tab:`pytorch,tensorflow` 
1. 以前の章で興味を持ったモデルを見直してください。再実装することで計算性能を改善できますか。
:end_tab:




:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/360)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/2490)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/2492)
:end_tab:\n