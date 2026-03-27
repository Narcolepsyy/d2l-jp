# 二乗誤差 (squared error)
:label:`sec_glossary_squared-error`

## 定義 (Definition)

数値を予測しようとする場合、最も一般的な損失関数は*二乗誤差*（squared error）である。すなわち、予測と正解ターゲット（ground truth target）の差の二乗である。分類において最も一般的な目的は、誤り率（error rate）、つまり私たちの予測が正解と一致しないデータ例の割合を最小化することである。一部の目的（例えば、二乗誤差）は簡単に最適化できるが、他の目的（例えば、誤り率）は、微分不可能であるか、または他の複雑な理由から直接最適化することが困難である。このような場合、代わりに*代替目的*（surrogate objective）を最適化することが一般的である。

## 参照 (Reference)

この用語の詳細な文脈については Dive into Deep Learning の対応する章を参照してください:
- [元章で読む](../chapter_introduction/index.md)
