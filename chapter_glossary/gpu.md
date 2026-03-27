# グラフィックス処理装置 (GPU)
:label:`sec_glossary_gpu`

## 定義 (Definition)

*グラフィックス処理装置*（GPU）は、深層学習を実用的にするうえで
ゲームチェンジャーであることが証明されました。
これらのチップはもともと、コンピュータゲーム向けのグラフィックス処理を高速化するために開発されていました。
特に、多くのコンピュータグラフィックス処理に必要な、高スループットの $4 \times 4$
行列—ベクトル積に最適化されていました。
幸いなことに、その数学は
畳み込み層の計算に必要なものと驚くほど似ています。
その頃、NVIDIA と ATI は GPU を一般計算向けに最適化し始めており :cite:`Fernando.2004`、
それらを *general-purpose GPUs*（GPGPUs）として売り出すまでになっていました。

## 参照 (Reference)

この用語の詳細な文脈については Dive into Deep Learning の対応する章を参照してください:
- [元章で読む](../chapter_convolutional-modern/alexnet.md)
