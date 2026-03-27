# vision Transformers (ViT)
:label:`sec_glossary_vit`

## 定義 (Definition)

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

## 参照 (Reference)

この用語の詳細な文脈については Dive into Deep Learning の対応する章を参照してください:
- [元章で読む](../chapter_attention-mechanisms-and-transformers/vision-transformer.md)
