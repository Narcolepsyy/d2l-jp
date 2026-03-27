# 特徴豊富な推薦システム

インタラクションデータは、ユーザーの嗜好や関心を示す最も基本的な指標である。これは、前に紹介したモデルにおいて重要な役割を果たす。しかし、インタラクションデータは通常きわめて疎であり、時にはノイズを含む。この問題に対処するために、アイテムの特徴、ユーザーのプロファイル、さらにはそのインタラクションがどのような文脈で発生したかといったサイド情報を推薦モデルに統合できる。これらの特徴を活用することは、特にインタラクションデータが不足している場合に、ユーザーの関心を効果的に予測できるため、推薦に有用である。そのため、推薦モデルにはこれらの特徴を扱う能力も備わっており、モデルにコンテンツ／文脈認識を持たせることが重要である。この種の推薦モデルを示すために、オンライン広告推薦におけるクリック率（CTR）について別のタスクを導入し :cite:`McMahan.Holt.Sculley.ea.2013`、匿名化された広告データセットを紹介する。ターゲット広告サービスは広く注目を集めており、しばしば推薦エンジンとして扱われる。ユーザーの個人的な嗜好や関心に合った広告を推薦することは、クリック率の向上にとって重要である。


デジタルマーケターは、オンライン広告を用いて顧客に広告を表示す。クリック率は、広告主が広告に対して得たクリック数を表示回数で割った指標であり、次の式で計算される割合として表される。 

$$ \textrm{CTR} = \frac{\#\textrm{Clicks}} {\#\textrm{Impressions}} \times 100 \% .$$

クリック率は、予測アルゴリズムの有効性を示す重要なシグナルである。クリック率予測は、ウェブサイト上の何かがクリックされる可能性を予測するタスクである。CTR予測のモデルは、ターゲット広告システムだけでなく、一般的なアイテム（たとえば映画、ニュース、製品）の推薦システム、メールキャンペーン、さらには検索エンジンにも利用できる。また、ユーザー満足度やコンバージョン率とも密接に関連しており、現実的な期待値を設定するのに役立つため、キャンペーン目標の設定にも有用である。

```{.python .input}
#@tab mxnet
from collections import defaultdict
from d2l import mxnet as d2l
from mxnet import gluon, np
import os
```

## オンライン広告データセット

インターネットとモバイル技術の大きな進歩に伴い、オンライン広告は重要な収益源となり、インターネット産業の収益の大部分を生み出している。無関係な広告ではなく、ユーザーの関心を引く広告を表示することは重要であり、そうすることで、偶然訪れた訪問者を有料顧客へと転換できる。ここで紹介するデータセットはオンライン広告データセットである。これは34個のフィールドからなり、最初の列は広告がクリックされたかどうかを示す目的変数（1）または（0）を表す。残りの列はすべてカテゴリ特徴である。各列は、広告ID、サイトまたはアプリケーションID、デバイスID、時刻、ユーザープロファイルなどを表している可能性がある。匿名化とプライバシー保護のため、特徴の実際の意味は公開されていない。

以下のコードは、データセットをサーバーからダウンロードし、ローカルのデータフォルダに保存する。

```{.python .input  n=15}
#@tab mxnet
#@save
d2l.DATA_HUB['ctr'] = (d2l.DATA_URL + 'ctr.zip',
                       'e18327c48c8e8e5c23da714dd614e390d369843f')

data_dir = d2l.download_extract('ctr')
```

訓練セットとテストセットがあり、それぞれ15000件と3000件のサンプル／行から構成されている。

## データセットラッパー

データ読み込みを便利にするために、CSVファイルから広告データセットを読み込み、`DataLoader` で使用できる `CTRDataset` を実装する。

```{.python .input  n=13}
#@tab mxnet
#@save
class CTRDataset(gluon.data.Dataset):
    def __init__(self, data_path, feat_mapper=None, defaults=None,
                 min_threshold=4, num_feat=34):
        self.NUM_FEATS, self.count, self.data = num_feat, 0, {}
        feat_cnts = defaultdict(lambda: defaultdict(int))
        self.feat_mapper, self.defaults = feat_mapper, defaults
        self.field_dims = np.zeros(self.NUM_FEATS, dtype=np.int64)
        with open(data_path) as f:
            for line in f:
                instance = {}
                values = line.rstrip('\n').split('\t')
                if len(values) != self.NUM_FEATS + 1:
                    continue
                label = np.float32([0, 0])
                label[int(values[0])] = 1
                instance['y'] = [np.float32(values[0])]
                for i in range(1, self.NUM_FEATS + 1):
                    feat_cnts[i][values[i]] += 1
                    instance.setdefault('x', []).append(values[i])
                self.data[self.count] = instance
                self.count = self.count + 1
        if self.feat_mapper is None and self.defaults is None:
            feat_mapper = {i: {feat for feat, c in cnt.items() if c >=
                               min_threshold} for i, cnt in feat_cnts.items()}
            self.feat_mapper = {i: {feat_v: idx for idx, feat_v in enumerate(feat_values)}
                                for i, feat_values in feat_mapper.items()}
            self.defaults = {i: len(feat_values) for i, feat_values in feat_mapper.items()}
        for i, fm in self.feat_mapper.items():
            self.field_dims[i - 1] = len(fm) + 1
        self.offsets = np.array((0, *np.cumsum(self.field_dims).asnumpy()
                                 [:-1]))
        
    def __len__(self):
        return self.count
    
    def __getitem__(self, idx):
        feat = np.array([self.feat_mapper[i + 1].get(v, self.defaults[i + 1])
                         for i, v in enumerate(self.data[idx]['x'])])
        return feat + self.offsets, self.data[idx]['y']
```

次の例では、訓練データを読み込み、最初のレコードを表示す。

```{.python .input  n=16}
#@tab mxnet
train_data = CTRDataset(os.path.join(data_dir, 'train.csv'))
train_data[0]
```

見てわかるように、34個のフィールドはすべてカテゴリ特徴である。各値は、対応するエントリの one-hot インデックスを表す。ラベル $0$ はクリックされていないことを意味する。この `CTRDataset` は、Criteo display advertising challenge [dataset](https://labs.criteo.com/2014/02/kaggle-display-advertising-challenge-dataset/) や Avazu click-through rate prediction [dataset](https://www.kaggle.com/c/avazu-ctr-prediction) など、他のデータセットの読み込みにも使用できる。  

## まとめ 
* クリック率は、広告システムや推薦システムの有効性を測るために用いられる重要な指標である。
* クリック率予測は通常、二値分類問題に変換される。目的は、与えられた特徴に基づいて、広告／アイテムがクリックされるかどうかを予測することである。

## 演習

* 提供された `CTRDataset` を使って、Criteo と Avazu のデータセットを読み込めますか。なお、Criteo データセットには実数値特徴が含まれているため、コードを少し修正する必要があるかもしれない。
