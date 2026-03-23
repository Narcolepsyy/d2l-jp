# ビルド

## 開発者向けインストール

```
wget https://repo.anaconda.com/miniconda/Miniconda3-py39_4.12.0-Linux-x86_64.sh  # For py3.8, wget  https://repo.anaconda.com/miniconda/Miniconda3-py38_4.12.0-Linux-x86_64.sh
sh Miniconda3-py39_4.12.0-Linux-x86_64.sh -b  # For py3.8: sh Miniconda3-py38_4.12.0-Linux-x86_64.sh -b
~/miniconda3/bin/conda init
. ~/.bashrc
conda create --name d2l python=3.9 -y  # For py3.8: conda create --name d2l python=3.8 -y
conda activate d2l
pip install torch torchvision
pip install d2lbook
git clone https://github.com/d2l-ai/d2l-en.git
jupyter notebook --generate-config
echo "c.NotebookApp.contents_manager_class = 'notedown.NotedownContentsManager'" >> ~/.jupyter/jupyter_notebook_config.py
cd d2l-en
pip install -e .  # Install the d2l library from source
jupyter notebook
```

任意: `jupyter_contrib_nbextensions` を使用する場合

```
pip install jupyter_contrib_nbextensions
jupyter contrib nbextension install --user
# jupyter nbextension enable execute_time/ExecuteTime
```



## 評価なしでビルドする

`config.ini` の `eval_notebook = True` を `eval_notebook = False` に変更します。


## PDF のビルド

```
# Install d2lbook
pip install git+https://github.com/d2l-ai/d2l-book

sudo apt-get install texlive-full
sudo apt-get install librsvg2-bin
sudo apt-get install pandoc  # If not working, conda install pandoc

# To import d2l
cd d2l-en
pip install -e .

# Build PDF
d2lbook build pdf
```

### PDF 用フォント

```
wget https://raw.githubusercontent.com/d2l-ai/utils/master/install_fonts.sh
sudo bash install_fonts.sh
```


## HTML のビルド

```
bash static/build_html.sh
```

## フォントのインストール

```
wget -O source-serif-pro.zip https://www.fontsquirrel.com/fonts/download/source-serif-pro
unzip source-serif-pro -d source-serif-pro
sudo mv source-serif-pro /usr/share/fonts/opentype/

wget -O source-sans-pro.zip https://www.fontsquirrel.com/fonts/download/source-sans-pro
unzip source-sans-pro -d source-sans-pro
sudo mv source-sans-pro /usr/share/fonts/opentype/

wget -O source-code-pro.zip https://www.fontsquirrel.com/fonts/download/source-code-pro
unzip source-code-pro -d source-code-pro
sudo mv source-code-pro /usr/share/fonts/opentype/

wget -O Inconsolata.zip https://www.fontsquirrel.com/fonts/download/Inconsolata
unzip Inconsolata -d Inconsolata
sudo mv Inconsolata /usr/share/fonts/opentype/

sudo fc-cache -f -v

```

## リリースチェックリスト

### d2l-en

- d2lbook をリリースする
- [任意、ハードコピー版の書籍またはパートナー製品のみ]
    - [setup.py](http://setup.py) → requirements と static/build.yml（d2lbook を含む）でライブラリのバージョンを固定する
    - 再評価する
    - インストール時に d2l のバージョン（下の pypi に表示されるもの）を固定する
- d2l.xxx の docstring を追加する
- フロントページの告知を更新する
- （メジャーリリースのみ）wa 0.8.0 を見て、本文で修正が必要な箇所がないか確認する
- d2lbook build lib
- ランダムな colab をテストする
- http://ci.d2l.ai/computer/d2l-worker/script

```python
"rm -rf /home/d2l-worker/workspace/d2l-en-release".execute().text
"rm -rf /home/d2l-worker/workspace/d2l-en-release@2".execute().text
"rm -rf /home/d2l-worker/workspace/d2l-en-release@tmp".execute().text
"rm -rf /home/d2l-worker/workspace/d2l-en-release@2@tmp".execute().text
"ls /home/d2l-worker/workspace/".execute().text
```

- リリース PR を評価する
- badahnau と transformer における attention の乱数性が固定されていることを確認する
- config.ini と build.yml の間でライブラリ（たとえば sagemaker 配下）のバージョンが一致していることを確認する
- config.ini と d2l/__init__.py のバージョン番号、および installation.md の d2l バージョンを変更する
- master を release に、個々のコミットを保持したままマージする（マージコミットを作成する）
- git checkout master
- rr -rf d2l.egg-info dist
- d2l を pypi にアップロードする（チームアカウント）
- colab と d2l を再テストする
- release ブランチで git tag を打つ
- git checkout master
- README の最新バージョンをブランチで更新し、その後 squash してマージして元に戻す
- [任意] CloudFront キャッシュを無効化する
- [任意、ハードコピー版の書籍のみ]
    - config.ini: other_file_s3urls
- [任意、ハードコピー版の書籍またはパートナー製品のみ]
    - [setup.py](http://setup.py) → requirements のライブラリのバージョンを元に戻す
 
### d2l-zh

- フロントページの告知を更新する
- （必要かどうか？）d2lbook build lib
- ランダムな colab をテストする
- static/build.yml を d2l-en のものに更新する
- [http://ci.d2l.ai/computer/(master)/script](http://ci.d2l.ai/computer/(master)/script)
- http://ci.d2l.ai/computer/d2l-worker/script

```python
"rm -rf /home/d2l-worker/workspace/d2l-zh-release".execute().text
"rm -rf /home/d2l-worker/workspace/d2l-zh-release@2".execute().text
"rm -rf /home/d2l-worker/workspace/d2l-zh-release@tmp".execute().text
"rm -rf /home/d2l-worker/workspace/d2l-zh-release@2@tmp".execute().text
"ls /home/d2l-worker/workspace/".execute().text
```

- リリース PR を評価する（badahnau と transformer の attention の乱数性を修正する）
- config.ini と build.yml の間でライブラリ（たとえば sagemaker）version が一致していることを確認する
- config.ini と d2l/__init__.py のバージョン番号を変更する
- master を release に、個々のコミットを保持したままマージする（マージコミットを作成する）
- colab を再テストする
- release ブランチで git tag を打つ
- git checkout master
- README の最新バージョンをブランチで更新し、その後 squash してマージして元に戻す
- 2.0.0 リリース追加作業
    - s3 コンソールで
        - [zh-v2.d2l.ai](http://zh-v2.d2l.ai) bucket/d2l-zh.zip を d2l-webdata bucket/d2l-zh.zip にコピーする
        - d2l-webdata bucket/d2l-zh.zip を d2l-webdata bucket/d2l-zh-2.0.0.zip にリネームする
        - d2l-zh/release の CI を実行して config 内の other_file_s3urls をトリガーする
        - インストールをテストするために cloudfront キャッシュを無効化する
    - インストールをテストする\n