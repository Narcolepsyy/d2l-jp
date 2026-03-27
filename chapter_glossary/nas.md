# ニューラルアーキテクチャ探索 (NAS)
:label:`sec_glossary_nas`

## 定義 (Definition)

これまでのところ、*ニューラルアーキテクチャ探索*（NAS）によって得られるネットワークは扱ってきませんでした :cite:`zoph2016neural,liu2018darts`。その理由は、通常そのコストが非常に大きく、総当たり探索、遺伝的アルゴリズム、強化学習、あるいは他の何らかのハイパーパラメータ最適化に依存するからです。固定された探索空間が与えられたとき、
NAS は探索戦略を用いて、
返された性能推定値に基づいてアーキテクチャを自動的に選択します。
NAS の結果は
単一のネットワーク実体です。EfficientNet はこの探索の注目すべき成果です :cite:`tan2019efficientnet`。

## 参照 (Reference)

この用語の詳細な文脈については Dive into Deep Learning の対応する章を参照してください:
- [元章で読む](../chapter_convolutional-modern/cnn-design.md)
