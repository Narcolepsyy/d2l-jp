{.python .input}
%load_ext d2lbook.tab
tab.interact_select(['pytorch', 'jax'])
```

# 画像向けTransformer
:label:`sec_vision-transformer`

Transformerアーキテクチャは当初、
機械翻訳に焦点を当てた
系列変換学習のために提案された。
その後、Transformerは
さまざまな自然言語処理タスクにおける第一選択のモデルとして台頭した :cite:`Radford.Narasimhan.Salimans.ea.2018,Radford.Wu.Child.ea.2019,brown2020language,Devlin.Chang.Lee.ea.2018,raffel2020exploring`。
しかし、コンピュータビジョンの分野では、
支配的なアーキテクチャは依然として
CNNのままであった（:numref:`chap_modern_cnn`）。
当然ながら、研究者たちは
Transformerモデルを画像データに適用することで
より良い性能が得られるのではないかと考え始めた。
この問いは、コンピュータビジョンコミュニティに
大きな関心を呼び起こした。
最近では、:citet:`ramachandran2019stand` が
畳み込みを自己注意で置き換える方式を提案した。
しかし、注意機構に特殊なパターンを用いるため、
ハードウェアアクセラレータ上でモデルを大規模化しにくい。
その後、:citet:`cordonnier2020relationship` は理論的に、
自己注意が畳み込みと同様に振る舞うよう学習できることを証明した。
実証的には、画像から $2 \times 2$ のパッチを入力として取り出したが、
パッチサイズが小さいため、このモデルは
低解像度の画像データにしか適用できない。

パッチサイズに特別な制約を設けずに、
*vision Transformers*（ViT）は
画像からパッチを抽出し、
それらをTransformerエンコーダに入力して
グローバルな表現を得る。
そして最終的に、その表現を分類用に変換する :cite:`Dosovitskiy.Beyer.Kolesnikov.ea.2021`。
特筆すべきは、TransformerはCNNよりもスケーラビリティに優れていることである。
より大きなデータセットでより大規模なモデルを学習すると、
vision TransformerはResNetを大きく上回る。
自然言語処理におけるネットワークアーキテクチャ設計の潮流と同様に、
Transformerはコンピュータビジョンにおいてもゲームチェンジャーとなった。

```{.python .input}
%%tab pytorch
from d2l import torch as d2l
import torch
from torch import nn
```

```{.python .input}
%%tab jax
from d2l import jax as d2l
from flax import linen as nn
import jax
from jax import numpy as jnp
```

## モデル

:numref:`fig_vit` は
vision Transformerのモデルアーキテクチャを示している。
このアーキテクチャは、
画像をパッチ化するstem、
多層Transformerエンコーダに基づくbody、
そしてグローバル表現を
出力ラベルへ変換するheadから構成される。

![The vision Transformer architecture. In this example, an image is split into nine patches. A special “&lt;cls&gt;” token and the nine flattened image patches are transformed via patch embedding and $\mathit{n}$ Transformer encoder blocks into ten representations, respectively. The “&lt;cls&gt;” representation is further transformed into the output label.](../img/vit.svg)
:label:`fig_vit`

高さ $h$、幅 $w$、
チャネル数 $c$ をもつ入力画像を考える。
パッチの高さと幅をともに $p$ とすると、
画像は $m = hw/p^2$ 個のパッチ列に分割され、
各パッチは長さ $cp^2$ のベクトルに平坦化される。
このようにして、画像パッチはTransformerエンコーダによって
テキスト系列中のトークンと同様に扱うことができる。
特別な “&lt;cls&gt;”（class）トークンと
$m$ 個の平坦化された画像パッチは線形射影されて
$m+1$ 個のベクトル列となり、
学習可能な位置埋め込みが加算される。
多層Transformerエンコーダは
$m+1$ 個の入力ベクトルを
同じ長さの $m+1$ 個の出力ベクトル表現へ変換する。
これは :numref:`fig_transformer` における元のTransformerエンコーダと
まったく同じように動作し、
正規化の位置だけが異なる。
“&lt;cls&gt;” トークンは自己注意を通じて
すべての画像パッチに注意を向けるため（:numref:`fig_cnn-rnn-self-attention` を参照）、
Transformerエンコーダ出力におけるその表現は
さらに出力ラベルへ変換される。

## パッチ埋め込み

vision Transformerを実装するには、まず
:numref:`fig_vit` のパッチ埋め込みから始めよう。
画像をパッチに分割し、
それらの平坦化されたパッチを線形射影する操作は、
カーネルサイズとストライドサイズの両方をパッチサイズに設定した
1つの畳み込み演算として簡略化できる。

```{.python .input}
%%tab pytorch
class PatchEmbedding(nn.Module):
    def __init__(self, img_size=96, patch_size=16, num_hiddens=512):
        super().__init__()
        def _make_tuple(x):
            if not isinstance(x, (list, tuple)):
                return (x, x)
            return x
        img_size, patch_size = _make_tuple(img_size), _make_tuple(patch_size)
        self.num_patches = (img_size[0] // patch_size[0]) * (
            img_size[1] // patch_size[1])
        self.conv = nn.LazyConv2d(num_hiddens, kernel_size=patch_size,
                                  stride=patch_size)

    def forward(self, X):
        # Output shape: (batch size, no. of patches, no. of channels)
        return self.conv(X).flatten(2).transpose(1, 2)
```

```{.python .input}
%%tab jax
class PatchEmbedding(nn.Module):
    img_size: int = 96
    patch_size: int = 16
    num_hiddens: int = 512

    def setup(self):
        def _make_tuple(x):
            if not isinstance(x, (list, tuple)):
                return (x, x)
            return x
        img_size, patch_size = _make_tuple(self.img_size), _make_tuple(self.patch_size)
        self.num_patches = (img_size[0] // patch_size[0]) * (
            img_size[1] // patch_size[1])
        self.conv = nn.Conv(self.num_hiddens, kernel_size=patch_size,
                            strides=patch_size, padding='SAME')

    def __call__(self, X):
        # Output shape: (batch size, no. of patches, no. of channels)
        X = self.conv(X)
        return X.reshape((X.shape[0], -1, X.shape[3]))
```

次の例では、高さと幅が `img_size` の画像を入力として、
パッチ埋め込みは `(img_size//patch_size)**2` 個のパッチを出力し、
それらは長さ `num_hiddens` のベクトルへ線形射影される。

```{.python .input}
%%tab pytorch
img_size, patch_size, num_hiddens, batch_size = 96, 16, 512, 4
patch_emb = PatchEmbedding(img_size, patch_size, num_hiddens)
X = d2l.zeros(batch_size, 3, img_size, img_size)
d2l.check_shape(patch_emb(X),
                (batch_size, (img_size//patch_size)**2, num_hiddens))
```

```{.python .input}
%%tab jax
img_size, patch_size, num_hiddens, batch_size = 96, 16, 512, 4
patch_emb = PatchEmbedding(img_size, patch_size, num_hiddens)
X = d2l.zeros((batch_size, img_size, img_size, 3))
output, _ = patch_emb.init_with_output(d2l.get_key(), X)
d2l.check_shape(output, (batch_size, (img_size//patch_size)**2, num_hiddens))
```

## Vision Transformerエンコーダ
:label:`subsec_vit-encoder`

vision TransformerエンコーダのMLPは、
元のTransformerエンコーダの位置ごとのFFNとは少し異なる（:numref:`subsec_positionwise-ffn` を参照）。
第一に、ここでは活性化関数としてガウス誤差線形ユニット（GELU）を用いる。
これはReLUのより滑らかな版とみなせる :cite:`Hendrycks.Gimpel.2016`。
第二に、正則化のために、MLP内の各全結合層の出力にドロップアウトを適用する。

```{.python .input}
%%tab pytorch
class ViTMLP(nn.Module):
    def __init__(self, mlp_num_hiddens, mlp_num_outputs, dropout=0.5):
        super().__init__()
        self.dense1 = nn.LazyLinear(mlp_num_hiddens)
        self.gelu = nn.GELU()
        self.dropout1 = nn.Dropout(dropout)
        self.dense2 = nn.LazyLinear(mlp_num_outputs)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout2(self.dense2(self.dropout1(self.gelu(
            self.dense1(x)))))
```

```{.python .input}
%%tab jax
class ViTMLP(nn.Module):
    mlp_num_hiddens: int
    mlp_num_outputs: int
    dropout: float = 0.5

    @nn.compact
    def __call__(self, x, training=False):
        x = nn.Dense(self.mlp_num_hiddens)(x)
        x = nn.gelu(x)
        x = nn.Dropout(self.dropout, deterministic=not training)(x)
        x = nn.Dense(self.mlp_num_outputs)(x)
        x = nn.Dropout(self.dropout, deterministic=not training)(x)
        return x
```

vision Transformerエンコーダブロックの実装は、
:numref:`fig_vit` における事前正規化の設計に従っている。
ここでは、正規化はマルチヘッド注意またはMLPの直前に適用される。
:numref:`fig_transformer` の「add & norm」のような事後正規化では、
正規化は残差接続の直後に置かれるのに対し、
事前正規化はTransformerの学習をより効果的または効率的にする :cite:`baevski2018adaptive,wang2019learning,xiong2020layer`。

```{.python .input}
%%tab pytorch
class ViTBlock(nn.Module):
    def __init__(self, num_hiddens, norm_shape, mlp_num_hiddens,
                 num_heads, dropout, use_bias=False):
        super().__init__()
        self.ln1 = nn.LayerNorm(norm_shape)
        self.attention = d2l.MultiHeadAttention(num_hiddens, num_heads,
                                                dropout, use_bias)
        self.ln2 = nn.LayerNorm(norm_shape)
        self.mlp = ViTMLP(mlp_num_hiddens, num_hiddens, dropout)

    def forward(self, X, valid_lens=None):
        X = X + self.attention(*([self.ln1(X)] * 3), valid_lens)
        return X + self.mlp(self.ln2(X))
```

```{.python .input}
%%tab jax
class ViTBlock(nn.Module):
    num_hiddens: int
    mlp_num_hiddens: int
    num_heads: int
    dropout: float
    use_bias: bool = False

    def setup(self):
        self.attention = d2l.MultiHeadAttention(self.num_hiddens, self.num_heads,
                                                self.dropout, self.use_bias)
        self.mlp = ViTMLP(self.mlp_num_hiddens, self.num_hiddens, self.dropout)

    @nn.compact
    def __call__(self, X, valid_lens=None, training=False):
        X = X + self.attention(*([nn.LayerNorm()(X)] * 3),
                               valid_lens, training=training)[0]
        return X + self.mlp(nn.LayerNorm()(X), training=training)
```

:numref:`subsec_transformer-encoder` と同様に、
vision Transformerエンコーダブロックは入力の形状を変えない。

```{.python .input}
%%tab pytorch
X = d2l.ones((2, 100, 24))
encoder_blk = ViTBlock(24, 24, 48, 8, 0.5)
encoder_blk.eval()
d2l.check_shape(encoder_blk(X), X.shape)
```

```{.python .input}
%%tab jax
X = d2l.ones((2, 100, 24))
encoder_blk = ViTBlock(24, 48, 8, 0.5)
d2l.check_shape(encoder_blk.init_with_output(d2l.get_key(), X)[0], X.shape)
```

## 全体をまとめる

以下のvision Transformerの順伝播は単純である。
まず、入力画像は `PatchEmbedding` インスタンスに与えられ、
その出力は “&lt;cls&gt;” トークン埋め込みと連結される。
それらはドロップアウトの前に、学習可能な位置埋め込みと加算される。
次に、その出力は `ViTBlock` クラスのインスタンスを `num_blks` 個積み重ねたTransformerエンコーダに入力される。
最後に、“&lt;cls&gt;” トークンの表現がネットワークのheadによって射影される。

```{.python .input}
%%tab pytorch
class ViT(d2l.Classifier):
    """Vision Transformer."""
    def __init__(self, img_size, patch_size, num_hiddens, mlp_num_hiddens,
                 num_heads, num_blks, emb_dropout, blk_dropout, lr=0.1,
                 use_bias=False, num_classes=10):
        super().__init__()
        self.save_hyperparameters()
        self.patch_embedding = PatchEmbedding(
            img_size, patch_size, num_hiddens)
        self.cls_token = nn.Parameter(d2l.zeros(1, 1, num_hiddens))
        num_steps = self.patch_embedding.num_patches + 1  # Add the cls token
        # Positional embeddings are learnable
        self.pos_embedding = nn.Parameter(
            torch.randn(1, num_steps, num_hiddens))
        self.dropout = nn.Dropout(emb_dropout)
        self.blks = nn.Sequential()
        for i in range(num_blks):
            self.blks.add_module(f"{i}", ViTBlock(
                num_hiddens, num_hiddens, mlp_num_hiddens,
                num_heads, blk_dropout, use_bias))
        self.head = nn.Sequential(nn.LayerNorm(num_hiddens),
                                  nn.Linear(num_hiddens, num_classes))

    def forward(self, X):
        X = self.patch_embedding(X)
        X = d2l.concat((self.cls_token.expand(X.shape[0], -1, -1), X), 1)
        X = self.dropout(X + self.pos_embedding)
        for blk in self.blks:
            X = blk(X)
        return self.head(X[:, 0])
```

```{.python .input}
%%tab jax
class ViT(d2l.Classifier):
    """Vision Transformer."""
    img_size: int
    patch_size: int
    num_hiddens: int
    mlp_num_hiddens: int
    num_heads: int
    num_blks: int
    emb_dropout: float
    blk_dropout: float
    lr: float = 0.1
    use_bias: bool = False
    num_classes: int = 10
    training: bool = False

    def setup(self):
        self.patch_embedding = PatchEmbedding(self.img_size, self.patch_size,
                                              self.num_hiddens)
        self.cls_token = self.param('cls_token', nn.initializers.zeros,
                                    (1, 1, self.num_hiddens))
        num_steps = self.patch_embedding.num_patches + 1  # Add the cls token
        # Positional embeddings are learnable
        self.pos_embedding = self.param('pos_embed', nn.initializers.normal(),
                                        (1, num_steps, self.num_hiddens))
        self.blks = [ViTBlock(self.num_hiddens, self.mlp_num_hiddens,
                              self.num_heads, self.blk_dropout, self.use_bias)
                    for _ in range(self.num_blks)]
        self.head = nn.Sequential([nn.LayerNorm(), nn.Dense(self.num_classes)])

    @nn.compact
    def __call__(self, X):
        X = self.patch_embedding(X)
        X = d2l.concat((jnp.tile(self.cls_token, (X.shape[0], 1, 1)), X), 1)
        X = nn.Dropout(emb_dropout, deterministic=not self.training)(X + self.pos_embedding)
        for blk in self.blks:
            X = blk(X, training=self.training)
        return self.head(X[:, 0])
```

## 学習

Fashion-MNISTデータセットでvision Transformerを学習するのは、
:numref:`chap_modern_cnn` でCNNを学習したときと同じである。

```{.python .input}
%%tab all
img_size, patch_size = 96, 16
num_hiddens, mlp_num_hiddens, num_heads, num_blks = 512, 2048, 8, 2
emb_dropout, blk_dropout, lr = 0.1, 0.1, 0.1
model = ViT(img_size, patch_size, num_hiddens, mlp_num_hiddens, num_heads,
            num_blks, emb_dropout, blk_dropout, lr)
trainer = d2l.Trainer(max_epochs=10, num_gpus=1)
data = d2l.FashionMNIST(batch_size=128, resize=(img_size, img_size))
trainer.fit(model, data)
```

## 要約と考察

Fashion-MNISTのような小規模データセットでは、
実装したvision Transformerが
:numref:`sec_resnet` のResNetを上回らないことに
気づいたかもしれない。
同様の観察は、ImageNetデータセット（120万枚の画像）でも成り立つ。
これは、Transformerには
畳み込みにおける有用な帰納バイアス、
たとえば平行移動不変性や局所性（:numref:`sec_why-conv`）が
*欠けている* ためである。
しかし、より大きなデータセット（たとえば3億枚の画像）で
より大規模なモデルを学習すると状況は変わり、
その場合、vision Transformerは画像分類でResNetを大きく上回り、
スケーラビリティにおけるTransformerの本質的な優位性を示している :cite:`Dosovitskiy.Beyer.Kolesnikov.ea.2021`。
vision Transformerの導入は、
画像データをモデル化するためのネットワーク設計の潮流を変えた。
その後すぐに、DeiTのデータ効率の高い学習戦略によって
ImageNetデータセットで有効であることが示された :cite:`touvron2021training`。
しかし、自己注意の二次計算量
（:numref:`sec_self-attention-and-positional-encoding`）
のため、Transformerアーキテクチャは
高解像度画像にはあまり適していない。
コンピュータビジョンにおける汎用バックボーンネットワークを目指して、
Swin Transformerは画像サイズに対する二次的な計算複雑性を
（:numref:`subsec_cnn-rnn-self-attention`）
解消し、畳み込みに似た事前知識を再導入した。
その結果、Transformerの適用範囲は
画像分類を超えたさまざまなコンピュータビジョンタスクへと広がり、
最先端の結果を達成している :cite:`liu2021swin`。

## 演習

1. `img_size` の値は学習時間にどのように影響するか。
1. “&lt;cls&gt;” トークン表現を出力へ射影する代わりに、平均化したパッチ表現をどのように射影するか。これを実装し、精度への影響を調べよ。
1. ハイパーパラメータを調整して、vision Transformerの精度を改善できるか。
