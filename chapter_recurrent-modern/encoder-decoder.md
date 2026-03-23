{.python .input  n=1}
%load_ext d2lbook.tab
tab.interact_select('mxnet', 'pytorch', 'tensorflow', 'jax')
```

# エンコーダ--デコーダアーキテクチャ
:label:`sec_encoder-decoder`

一般に、機械翻訳
(:numref:`sec_machine_translation`)
のようなシーケンス・ツー・シーケンス問題では、
入力と出力は長さがさまざまで、
互いに対応が取れていません。
この種のデータを扱う標準的な方法は、
2つの主要な構成要素からなる
*エンコーダ--デコーダ*アーキテクチャ (:numref:`fig_encoder_decoder`)
を設計することです。
すなわち、可変長の系列を入力として受け取る
*エンコーダ*と、
エンコードされた入力と
ターゲット系列の左側の文脈を受け取り、
ターゲット系列における次のトークンを予測する
条件付き言語モデルとして働く
*デコーダ*です。


![エンコーダ--デコーダアーキテクチャ。](../img/encoder-decoder.svg)
:label:`fig_encoder_decoder`

英仏機械翻訳を例に考えてみましょう。
英語の入力系列
"They", "are", "watching", "."
が与えられると、
このエンコーダ--デコーダアーキテクチャは
まず可変長の入力を状態にエンコードし、
その後、その状態をデコードして
翻訳された系列を
トークンごとに出力として生成します。
"Ils", "regardent", "."。
エンコーダ--デコーダアーキテクチャは
後続の節でさまざまなシーケンス・ツー・シーケンスモデルの基盤となるため、
この節ではこのアーキテクチャを
後で実装するためのインターフェースに落とし込みます。

```{.python .input}
%%tab mxnet
from d2l import mxnet as d2l
from mxnet.gluon import nn
```

```{.python .input}
%%tab pytorch
from d2l import torch as d2l
from torch import nn
```

```{.python .input}
%%tab tensorflow
from d2l import tensorflow as d2l
import tensorflow as tf
```

```{.python .input}
%%tab jax
from d2l import jax as d2l
from flax import linen as nn
```

## [**エンコーダ**]

エンコーダのインターフェースでは、
エンコーダが可変長系列を入力 `X` として受け取ることだけを指定します。
実装は、この基底 `Encoder` クラスを継承する任意のモデルによって提供されます。

```{.python .input}
%%tab mxnet
class Encoder(nn.Block):  #@save
    """The base encoder interface for the encoder--decoder architecture."""
    def __init__(self):
        super().__init__()

    # Later there can be additional arguments (e.g., length excluding padding)
    def forward(self, X, *args):
        raise NotImplementedError
```

```{.python .input}
%%tab pytorch
class Encoder(nn.Module):  #@save
    """The base encoder interface for the encoder--decoder architecture."""
    def __init__(self):
        super().__init__()

    # Later there can be additional arguments (e.g., length excluding padding)
    def forward(self, X, *args):
        raise NotImplementedError
```

```{.python .input}
%%tab tensorflow
class Encoder(tf.keras.layers.Layer):  #@save
    """The base encoder interface for the encoder--decoder architecture."""
    def __init__(self):
        super().__init__()

    # Later there can be additional arguments (e.g., length excluding padding)
    def call(self, X, *args):
        raise NotImplementedError
```

```{.python .input}
%%tab jax
class Encoder(nn.Module):  #@save
    """The base encoder interface for the encoder--decoder architecture."""
    def setup(self):
        raise NotImplementedError

    # Later there can be additional arguments (e.g., length excluding padding)
    def __call__(self, X, *args):
        raise NotImplementedError
```

## [**デコーダ**]

以下のデコーダのインターフェースでは、
エンコーダの出力 (`enc_all_outputs`) を
エンコードされた状態に変換するための
追加の `init_state` メソッドを加えます。
このステップでは、
入力の有効長など、
追加の入力が必要になる場合があることに注意してください。
これは
:numref:`sec_machine_translation`
で説明しました。
可変長系列をトークンごとに生成するために、
デコーダは毎回、
入力
（たとえば、1つ前の時刻ステップで生成されたトークン）
と
エンコードされた状態を
現在の時刻ステップにおける出力トークンへ写像します。

```{.python .input}
%%tab mxnet
class Decoder(nn.Block):  #@save
    """The base decoder interface for the encoder--decoder architecture."""
    def __init__(self):
        super().__init__()

    # Later there can be additional arguments (e.g., length excluding padding)
    def init_state(self, enc_all_outputs, *args):
        raise NotImplementedError

    def forward(self, X, state):
        raise NotImplementedError
```

```{.python .input}
%%tab pytorch
class Decoder(nn.Module):  #@save
    """The base decoder interface for the encoder--decoder architecture."""
    def __init__(self):
        super().__init__()

    # Later there can be additional arguments (e.g., length excluding padding)
    def init_state(self, enc_all_outputs, *args):
        raise NotImplementedError

    def forward(self, X, state):
        raise NotImplementedError
```

```{.python .input}
%%tab tensorflow
class Decoder(tf.keras.layers.Layer):  #@save
    """The base decoder interface for the encoder--decoder architecture."""
    def __init__(self):
        super().__init__()

    # Later there can be additional arguments (e.g., length excluding padding)
    def init_state(self, enc_all_outputs, *args):
        raise NotImplementedError

    def call(self, X, state):
        raise NotImplementedError
```

```{.python .input}
%%tab jax
class Decoder(nn.Module):  #@save
    """The base decoder interface for the encoder--decoder architecture."""
    def setup(self):
        raise NotImplementedError

    # Later there can be additional arguments (e.g., length excluding padding)
    def init_state(self, enc_all_outputs, *args):
        raise NotImplementedError

    def __call__(self, X, state):
        raise NotImplementedError
```

## [**エンコーダとデコーダを組み合わせる**]

順伝播では、
エンコーダの出力が
エンコードされた状態を生成するために使われ、
この状態は
デコーダによってその入力の1つとして
さらに利用されます。

```{.python .input}
%%tab mxnet, pytorch
class EncoderDecoder(d2l.Classifier):  #@save
    """The base class for the encoder--decoder architecture."""
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, enc_X, dec_X, *args):
        enc_all_outputs = self.encoder(enc_X, *args)
        dec_state = self.decoder.init_state(enc_all_outputs, *args)
        # Return decoder output only
        return self.decoder(dec_X, dec_state)[0]
```

```{.python .input}
%%tab tensorflow
class EncoderDecoder(d2l.Classifier):  #@save
    """The base class for the encoder--decoder architecture."""
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def call(self, enc_X, dec_X, *args):
        enc_all_outputs = self.encoder(enc_X, *args, training=True)
        dec_state = self.decoder.init_state(enc_all_outputs, *args)
        # Return decoder output only
        return self.decoder(dec_X, dec_state, training=True)[0]
```

```{.python .input}
%%tab jax
class EncoderDecoder(d2l.Classifier):  #@save
    """The base class for the encoder--decoder architecture."""
    encoder: nn.Module
    decoder: nn.Module
    training: bool

    def __call__(self, enc_X, dec_X, *args):
        enc_all_outputs = self.encoder(enc_X, *args, training=self.training)
        dec_state = self.decoder.init_state(enc_all_outputs, *args)
        # Return decoder output only
        return self.decoder(dec_X, dec_state, training=self.training)[0]
```

次の節では、
このエンコーダ--デコーダアーキテクチャに基づいて
RNNを用いて
シーケンス・ツー・シーケンスモデルを設計する方法を見ます。


## まとめ

エンコーダ--デコーダアーキテクチャは、
入力と出力の両方が
可変長系列からなる場合に対応できるため、
機械翻訳のような
シーケンス・ツー・シーケンス問題に適しています。
エンコーダは可変長系列を入力として受け取り、
それを固定形状の状態へ変換します。
デコーダは、その固定形状のエンコードされた状態を
可変長系列へ写像します。


## 演習

1. ニューラルネットワークを用いてエンコーダ--デコーダアーキテクチャを実装すると仮定します。エンコーダとデコーダは同じ種類のニューラルネットワークでなければならないでしょうか。
1. 機械翻訳以外に、エンコーダ--デコーダアーキテクチャを適用できる別の応用例を考えられますか。

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/341)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1061)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/3864)
:end_tab:

:begin_tab:`jax`
[Discussions](https://discuss.d2l.ai/t/18021)
:end_tab:\n