{.python .input}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow', 'jax'])
```

# ドキュメント
:begin_tab:`mxnet`
すべての MXNet の関数やクラスをここで紹介することは到底できませんし、
それらの情報はすぐに古くなってしまう可能性もあります。
しかし、[API ドキュメント](https://mxnet.apache.org/versions/1.8.0/api) や
追加の [チュートリアル](https://mxnet.apache.org/versions/1.8.0/api/python/docs/tutorials/) と例が、
最新かつ詳細な情報を提供しています。
この節では、MXNet の API を効率的に調べるためのヒントを紹介します。
:end_tab:

:begin_tab:`pytorch`
すべての PyTorch の関数やクラスをここで紹介することは到底できませんし、
その情報はすぐに古くなってしまうかもしれませんが、
[API ドキュメント](https://pytorch.org/docs/stable/index.html) や追加の [チュートリアル](https://pytorch.org/tutorials/beginner/basics/intro.html) と例が
そのような文書を提供しています。
この節では、PyTorch の API を調べるための指針を示します。
:end_tab:

:begin_tab:`tensorflow`
すべての TensorFlow の関数やクラスをここで紹介することは到底できませんし、
その情報はすぐに古くなってしまうかもしれませんが、
[API ドキュメント](https://www.tensorflow.org/api_docs) や追加の [チュートリアル](https://www.tensorflow.org/tutorials) と例が
そのような文書を提供しています。
この節では、TensorFlow の API を調べるための指針を示します。
:end_tab:

```{.python .input}
%%tab mxnet
from mxnet import np
```

```{.python .input}
%%tab pytorch
import torch
```

```{.python .input}
%%tab tensorflow
import tensorflow as tf
```

```{.python .input}
%%tab jax
import jax
```

## モジュール内の関数とクラス

モジュール内でどの関数やクラスを呼び出せるかを知るには、
`dir` 関数を使います。たとえば、
[**乱数生成のためのモジュール内のすべての属性を問い合わせる**]ことができます。

```{.python .input  n=1}
%%tab mxnet
print(dir(np.random))
```

```{.python .input  n=1}
%%tab pytorch
print(dir(torch.distributions))
```

```{.python .input  n=1}
%%tab tensorflow
print(dir(tf.random))
```

```{.python .input}
%%tab jax
print(dir(jax.random))
```

一般に、`__` で始まり `__` で終わる関数（Python の特別なオブジェクト）や、
単一の `_` で始まる関数（通常は内部関数）は無視して構いません。
残った関数名や属性名から判断すると、
このモジュールは一様分布（`uniform`）、
正規分布（`normal`）、
多項分布（`multinomial`）からのサンプリングを含む、
さまざまな乱数生成メソッドを提供していると推測できます。

## 特定の関数とクラス

特定の関数やクラスの使い方について詳しく知るには、
`help` 関数を呼び出します。例として、
[**テンソルの `ones` 関数の使い方を調べてみましょう**]。

```{.python .input}
%%tab mxnet
help(np.ones)
```

```{.python .input}
%%tab pytorch
help(torch.ones)
```

```{.python .input}
%%tab tensorflow
help(tf.ones)
```

```{.python .input}
%%tab jax
help(jax.numpy.ones)
```

ドキュメントから、`ones` 関数は
指定された形状をもつ新しいテンソルを作成し、
すべての要素を 1 に設定することがわかります。
可能な限り、解釈が正しいことを確認するために
[**簡単なテストを実行する**]べきです。

```{.python .input}
%%tab mxnet
np.ones(4)
```

```{.python .input}
%%tab pytorch
torch.ones(4)
```

```{.python .input}
%%tab tensorflow
tf.ones(4)
```

```{.python .input}
%%tab jax
jax.numpy.ones(4)
```

Jupyter ノートブックでは、`?` を使ってドキュメントを別ウィンドウに表示できます。
たとえば、`list?` は `help(list)` とほぼ同じ内容を生成し、
新しいブラウザウィンドウに表示します。
さらに、`list??` のように疑問符を 2 つ使うと、
その関数を実装している Python コードも表示されます。

公式ドキュメントには、この本の範囲を超える豊富な説明と実例が掲載されています。
私たちは、単なる網羅的なリストを作るよりも、実際に問題を解決するために役立つ重要な使用例を重視して紹介していきます。
また、ライブラリのソースコードを調べて、
本番コードの高品質な実装例を見ることも勧めます。
そうすることで、より優れた科学者になるだけでなく、
より優れたエンジニアにもなれるでしょう。\n
