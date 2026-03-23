# 分散型キー・バリュー・ストア
:label:`sec_key_value`

KVStore はデータ共有のための場所です。これを、異なるデバイス（GPU やコンピュータ）間で共有される単一のオブジェクトだと考えてください。各デバイスはそこにデータを push し、そこからデータを pull できます。

## 初期化
簡単な例として、(int, NDArray) のペアをストアに初期化し、その値を取り出してみましょう。

```{.python .input  n=1}
#@tab mxnet
from mxnet import np, npx, kv
npx.set_np()
```

```{.python .input  n=2}
#@tab mxnet
np.ones((2,3))
```

```{.python .input  n=11}
#@tab mxnet
help(kv)
```

```{.python .input  n=3}
#@tab mxnet
kv = kv.create('local')  # ローカルな kv ストアを作成する。
shape = (2,3)
kv.init(3, np.ones(shape) * 2)
a = np.zeros(shape)
kv.pull(3, out = a)
print(a)
```

## Push、集約、更新

初期化済みの任意のキーに対して、同じ形状の新しい値をそのキーへ push できます。

```{.python .input  n=4}
#@tab mxnet
kv.push(3, np.ones(shape)*8)
kv.pull(3, out = a)  # 値を取り出す
print(a.asnumpy())
```

push するデータは、どのデバイス上にあってもかまいません。さらに、同じキーに複数の値を push することもでき、その場合 KVStore はまずそれらの値をすべて加算し、その後で集約された値を push します。ここでは CPU 上の値のリストを push する例だけを示します。なお、加算が行われるのは値のリストが 2 個以上ある場合のみです。

```{.python .input  n=5}
#@tab mxnet
devices = [npx.cpu(i) for i in range(4)]
b = [np.ones(shape, ctx=device) for device in devices]
kv.push(3, b)
kv.pull(3, out = a)
print(a)
```

各 push に対して、KVStore は updater を用いて push された値とストア内の値を結合します。デフォルトの updater は ASSIGN です。データのマージ方法を制御するために、デフォルトを置き換えることができます。

```{.python .input  n=6}
#@tab mxnet
def update(key, input, stored):
    print(f'key: {key} の更新')
    stored += input * 2
kv._set_updater(update)
kv.pull(3, out=a)
print(a)
```

```{.python .input  n=7}
#@tab mxnet
kv.push(3, np.ones(shape))
kv.pull(3, out=a)
print(a)
```

## Pull

単一のキー・バリュー・ペアを pull する方法はすでに見ました。同様に、push と同じように、1 回の呼び出しで複数のデバイスへ値を pull することもできます。

```{.python .input  n=8}
#@tab mxnet
b = [np.ones(shape, ctx=device) for device in devices]
kv.pull(3, out = b)
print(b[1])
```

## キー・バリュー・ペアのリストを扱う

これまでに紹介した操作はすべて単一のキーを対象としていました。KVStore は、キー・バリュー・ペアのリストに対するインターフェースも提供します。

単一デバイスの場合:

```{.python .input  n=9}
#@tab mxnet
keys = [5, 7, 9]
kv.init(keys, [np.ones(shape)]*len(keys))
kv.push(keys, [np.ones(shape)]*len(keys))
b = [np.zeros(shape)]*len(keys)
kv.pull(keys, out = b)
print(b[1])
```

複数デバイスの場合:

```{.python .input  n=10}
#@tab mxnet
b = [[np.ones(shape, ctx=device) for device in devices]] * len(keys)
kv.push(keys, b)
kv.pull(keys, out = b)
print(b[1][1])
```\n