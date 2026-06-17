{.python .input}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow', 'jax'])
```

# ドキュメント
:begin_tab:`mxnet`
MXNet のすべての関数やクラスをここで網羅的に紹介することは不可能であり、
そのような情報はすぐに古くなるおそれもある。
しかし、[API ドキュメント](https://mxnet.apache.org/versions/1.8.0/api) や
追加の [チュートリアル](https://mxnet.apache.org/versions/1.8.0/api/python/docs/tutorials/)、
および各種の例が、最新かつ詳細な情報を提供している。
この節では、MXNet の API を効率よく調べるための手がかりを示す。
:end_tab:

:begin_tab:`pytorch`
PyTorch のすべての関数やクラスをここで網羅的に紹介することは不可能であり、
そのような情報はすぐに古くなるおそれもある。
しかし、[API ドキュメント](https://pytorch.org/docs/stable/index.html) や
追加の [チュートリアル](https://pytorch.org/tutorials/beginner/basics/intro.html)、
および各種の例が、必要な情報を提供している。
この節では、PyTorch の API を調べるための指針を示す。
:end_tab:

:begin_tab:`tensorflow`
TensorFlow のすべての関数やクラスをここで網羅的に紹介することは不可能であり、
そのような情報はすぐに古くなるおそれもある。
しかし、[API ドキュメント](https://www.tensorflow.org/api_docs) や
追加の [チュートリアル](https://www.tensorflow.org/tutorials)、
および各種の例が、必要な情報を提供している。
この節では、TensorFlow の API を調べるための指針を示す。
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

モジュール内で利用できる関数やクラスを知るには、
`dir` 関数を使う。たとえば、
[**乱数生成用モジュールに含まれるすべての属性を調べられる**]。

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

一般に、`__` で始まり `__` で終わる関数（Python の特殊オブジェクト）や、
単一の `_` で始まる関数（通常は内部用の関数）は無視してよい。
残りの関数名や属性名を見ると、
このモジュールは一様分布（`uniform`）、
正規分布（`normal`）、
多項分布（`multinomial`）からのサンプリングを含む、
さまざまな乱数生成メソッドを提供していると推測できる。

## 特定の関数とクラス

特定の関数やクラスの使い方を詳しく知るには、
`help` 関数を呼び出す。たとえば、
[**テンソルの `ones` 関数の使い方を調べる**]。

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

ドキュメントを見ると、`ones` 関数は
指定した形状の新しいテンソルを作成し、
すべての要素を 1 に設定することがわかる。
可能であれば、解釈が正しいことを確かめるために
[**簡単なテストを実行すべきである**]。

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

Jupyter ノートブックでは、`?` を使ってドキュメントを別ウィンドウに表示できる。
たとえば、`list?` は `help(list)` とほぼ同じ内容を生成し、
新しいブラウザウィンドウに表示する。
さらに、`list??` のように疑問符を 2 つ付けると、
その関数を実装している Python コードも表示される。

公式ドキュメントには、この本の範囲を超える豊富な説明と実例が掲載されている。
ここでは、単なる網羅的な一覧を示すのではなく、実際の問題解決に役立つ重要な使用例を重視する。
また、ライブラリのソースコードを調べて、
実運用向けコードの高品質な実装例を見ることも勧める。
そうすれば、より優れた科学者になるだけでなく、
より優れたエンジニアにもなれるだろう。