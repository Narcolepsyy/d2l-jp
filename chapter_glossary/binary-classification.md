# 2値分類 (binary classification)
:label:`sec_glossary_binary-classification`

## 定義 (Definition)

*分類*では、モデルが特徴量（例えば画像のピクセル値）を見て、ある離散的な選択肢の集合の中で、データ例がどの*カテゴリ*（category、あるいは*クラス* class とも呼ばれる）に属するかを予測することを求める。手書き数字の場合、0から9までの数字に対応する10個のクラスがあるかもしれない。分類の最も単純な形態は、2つのクラスしかない場合であり、この問題を私たちは*2値分類*（binary classification）と呼ぶ。例えば、データセットが動物の画像で構成され、ラベルが $\textrm{\{cat, dog\}}$ というクラスである場合などだ。回帰では数値をを出力する回帰器（regressor）を求めたが、分類では出力を予測されるクラスへの割り当てとする分類器（classifier）を求める。

## 参照 (Reference)

この用語の詳細な文脈については Dive into Deep Learning の対応する章を参照してください:
- [元章で読む](../chapter_introduction/index.md)
