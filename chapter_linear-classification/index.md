# 分類のための線形ニューラルネットワーク
:label:`chap_classification`

ここまでで、仕組みのすべてを学んできたので、
学んだスキルをより広い種類のタスクに適用する準備が整った。
分類へと話を移しても、
大部分の土台は同じままである。
すなわち、データを読み込み、それをモデルに通し、
出力を生成し、損失を計算し、
重みに関する勾配を求め、
モデルを更新する。
ただし、ターゲットの正確な形式、
出力層のパラメータ化、
そして損失関数の選択は、
*分類* の設定に合わせて調整される。

```toc
:maxdepth: 2

softmax-regression
image-classification-dataset
classification
softmax-regression-scratch
softmax-regression-concise
generalization-classification
environment-and-distribution-shift
```