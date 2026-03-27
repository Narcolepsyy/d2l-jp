# 非最大抑制 (NMS)
:label:`sec_glossary_nms`

## 定義 (Definition)

アンカーボックスが多数あると、
同じ物体を囲む非常に似た（大きく重なった）予測バウンディングボックスが
多数出力される可能性があります。
出力を簡潔にするために、
*非最大抑制*（NMS）を用いて、同じ物体に属する似た予測バウンディングボックスをまとめることができます。

## 参照 (Reference)

この用語の詳細な文脈については Dive into Deep Learning の対応する章を参照してください:
- [元章で読む](../chapter_computer-vision/anchor.md)
