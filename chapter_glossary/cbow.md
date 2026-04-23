# 連続的なBag of Words（CBOW）
:label:`sec_glossary_cbow`

## 定義 (Definition)

上記の問題に対処するために [word2vec](https://code.google.com/archive/p/word2vec/) ツールが提案された。
各単語を固定長ベクトルに写像し、これらのベクトルは異なる単語間の類似性や類推関係をよりよく表現できる。
word2vec ツールには2つのモデル、すなわち *skip-gram* :cite:`Mikolov.Sutskever.Chen.ea.2013` と *continuous bag of words*（CBOW） :cite:`Mikolov.Chen.Corrado.ea.2013` がある。
意味的に有意味な表現を得るために、
その学習は
条件付き確率に依存しており、
コーパス中の
周囲のいくつかの単語を使って
いくつかの単語を予測するものとみなせる。
教師信号がラベルなしデータから得られるため、
skip-gram と continuous bag of words の両方は
自己教師ありモデルである。

## 参照 (Reference)

この用語の詳細な文脈については Dive into Deep Learning の対応する章を参照してほしい:
- [元章で読む](../chapter_natural-language-processing-pretraining/word2vec.md)
