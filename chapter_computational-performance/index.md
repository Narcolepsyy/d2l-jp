# 計算性能
:label:`chap_performance`

深層学習では、  
データセットやモデルは通常大規模であり、  
そのため重い計算を伴います。  
したがって、計算性能は非常に重要です。  
この章では、計算性能に影響を与える主な要因、すなわち、命令型プログラミング、記号型プログラミング、非同期計算、自動並列化、およびマルチGPU計算に焦点を当てます。  
この章を学ぶことで、前の章で実装したこれらのモデルの計算性能をさらに向上させることができます。  
たとえば、精度を損なうことなく学習時間を短縮する、といった改善が可能です。

```toc
:maxdepth: 2

hybridize
async-computation
auto-parallelism
hardware
multiple-gpus
multiple-gpus-concise
parameterserver
```\n