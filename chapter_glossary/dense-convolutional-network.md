# DenseNet（密結合畳み込みネットワーク）
:label:`sec_glossary_dense-convolutional-network`

## 定義 (Definition)

ResNet は、深いネットワークにおける関数をどのようにパラメータ化するかという見方を大きく変えた。*DenseNet*（dense convolutional network）は、ある意味でその論理的な拡張である :cite:`Huang.Liu.Van-Der-Maaten.ea.2017`。
DenseNet の特徴は、各層がそれ以前のすべての層と接続される接続パターンと、ResNet の加算演算子ではなく連結演算を用いて、以前の層からの特徴を保持し再利用する点にある。
これをどのように導くかを理解するために、少し数学に寄り道しよう。

## 参照 (Reference)

この用語の詳細な文脈については Dive into Deep Learning の対応する章を参照してほしい:
- [元章で読む](../chapter_convolutional-modern/densenet.md)
