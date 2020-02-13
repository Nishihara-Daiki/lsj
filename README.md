# Lexical Simplification Japanese

[English README](README-EN.md)

## 公開物
大きく分けて以下の３つがあります。

+ 単語難易度辞書
+ 平易な言い換え辞書
+ 語彙平易化ツールキット

## 単語難易度辞書
単語難易度辞書 ([data/word2complexity.tsv](data/word2complexity.tsv)) は、各単語に、初級/中級/上級のラベルがタブ区切りで付与されています。
ラベルごとの収録単語数は以下の通りです。

| ラベル | 収録単語数 |
| -- | --: |
| 初級 | 749 |
| 中級 | 7,808 |
| 上級 | 32,048 |
| 合計 | 40,605 |

## 平易な言い換え辞書
手法 (pointwise / pairwise) の違いにより、2つの言い換え辞書があります。

+ pointwise [data/ss.pointwise.tsv](data/ss.pointwise.tsv)
+ pairwise [data/ss.pairwise.ours-B.tsv](data/ss.pairwise.ours-B.tsv)

pointwiseの形式は、以下通りタブ区切りで提供されます。難易度は、初級:0/中級:1/上級:2です。cos(単語1,単語2)は単語1と単語2の分散表現の余弦類似度です。
```
単語1 単語2 P(単語2|単語1)  cos(単語1,単語2)    単語1の難易度 単語2の難易度
```
pairwiseの形式は、pointwiseの1〜4列目までと同じです。手法の特性上、各単語の難易度は、推定されないため提供されません。

## 語彙平易化ツールキット

文中の指定された単語（対象単語）を平易な表現に言い換えます。
大きく2つのステップで言い換えを行います。

1. **言い換え候補取得**  対象単語の言い換え候補を取得します。
2. **ランキング**  その言い換え候補から最適なものを選ぶため、ランキングを行います。


### 環境構築（クイックスタート）

+ Python 3.7.2
+ Mecab (IPADIC 2.7.0)

必要パッケージと評価用データセットのインストール
```sh
pip3 install -r requirements.txt
git clone https://github.com/KodairaTomonori/EvaluationDataset
```

動作確認
```sh
python3 scripts/lexical_simplification.py \
    --candidate gold \
    --ranking none \
    --data ../EvaluationDataset \
    --output output/tmp.out
```
出力例
```
[log] word2vec vocab size is 0
2010it [00:00, 7789.41it/s]
acc/prec/changed = 70.50    70.50   100.00
candidate potential/prec/recall = 76.99 31.46   62.67
```

### 引数

* `--candidate, -C` 言い換え候補取得の手法を選択
    - `glavas` Light-LS (Glavas and Stajner 2015)
    - `synonym` 単語同義語辞書を使う
    - `bert` BERT-LS (Qiang et al. 2019)
    - `all` glavas, synonym, bert のいずれかで得られる言い換え候補を使う
    - `gold` 評価データセットを用いて正解の言い換え候補を使う
* `--ranking, -R` ランキング手法を選択
    - `glavas` Light-LS
    - `language-model` 言語モデルのスコアでランキングを行う
    - `bert` BERTの予測スコアでランキングを行う ※`--candidate=bert` を同時に指定しなければならない。
    - `none` ランキングを行わない
* `--output, -o` 言い換えの出力ファイルを指定する。`stdout`を指定すると、標準出力に言い換えを出力する。
* `--log, -g` ログの出力ファイルを指定する。`stdout`を指定すると、標準出力にログを出力する。
* `--data, -d` [小平らの評価用データセット](https://github.com/KodairaTomonori/EvaluationDataset)のパス
* `--embedding, -e` 単語分散表現のバイナリファイル
* `--language-model, -m` KenLMによる訓練済み言語モデル
* `--most-similar, -n` 言い換え候補取得時のtop-n
* `--word-to-freq, -f` 単語頻度辞書（単語,頻度のtsvファイル）
* `--synonym-dict, -p` 同義語辞書（単語1,単語2,スコアのtsvファイル）
* `--pretraind-bert, -b` 訓練済みBERTモデル
* `--word-to-complexity, -l` 単語難易度辞書（単語,スコアのtsvファイル）
* `--cos-threshold, -c` 言い換え候補取得時の閾値
* `--device, -u` 使用するGPU





## 実験の再現

### ダウンロードとインストール

必要なパッケージをインストールします。MeCabは、日本語対応できるか確認してください。
```sh
apt install -y mecab libmecab-dev mecab-ipadic-utf8 libboost-all-dev
pip3 install -r requirements.txt
```

使用するスクリプトに合わせて、データを用意します。
単語難易度辞書の作成（`word_complexity.py`）や平易な言い換え辞書の作成（`simple_synonym.py`）では、**分散表現**、**単語頻度**、**文字頻度**、**品詞辞書**を用意します。

語彙平易化ツールキット（`lexical_simplification.py`）では、**評価用データセット**に加え、下記の通り使用するオプションによって必要なデータが異なります。

|                                | 言語モデル | 分散表現 | 単語頻度 | 言い換え辞書 |
|--------------------------------|:-------:|:-------:|:-------:|:-------:|
| `--candidate = glavas`         |         | &check; |         |         |
| `--candidate = synonym`        |         | &check; |         | &check; |
| `--candidate = glavas+synonym` |         | &check; |         | &check; |
| `--simplicity = glavas`        |         | &check; | &check; |         |
| `--simplicity = point-wise`    |         | &check; |         |         |
| `--simplicity = pari-wise`     |         | &check; |         |         |
| `--ranking = glavas`           | &check; | &check; | &check; |         |
| `--ranking = language-model`   | &check; | &check; |         |         |
| `--ranking = ours`             | &check; | &check; | &check; | &check; |



#### 評価用データセット
```sh
git clone https://github.com/KodairaTomonori/EvaluationDataset
```

#### 言語モデル

日本語Wikipediaをダウンロードします。
```sh
wget https://dumps.wikimedia.org/jawiki/20190801/jawiki-20190801-pages-articles-multistream.xml.bz2
```

Wikipediaから本文を抽出します。
```sh
wget https://raw.githubusercontent.com/attardi/wikiextractor/3162bb6c3c9ebd2d15be507aa11d6fa818a454ac/WikiExtractor.py -P scripts/
python3 scripts/WikiExtractor.py -b 5G -o extracted jawiki-20190801-pages-articles-multistream.xml.bz2 -q
mv extracted/AA/wiki_00 .
rmdir extracted/AA extracted
```

日本語Wikipediaをトークナイズします。
```sh
cat wiki_00 | grep -v -e "<doc" -e "</doc>" -e '^\s*$' | mecab -O wakati > wiki.tok
```

KenLMをダウンロードおよびビルドします。
```sh
git clone https://github.com/kpu/kenlm
mkdir -p kenlm/build
cd kenlm/build
cmake ..
make -j 4
```

言語モデルを訓練します。
```sh
mkdir -p tmp
${KENLM}/bin/lmplz -o 5 -S 80% -T tmp < wiki.tok > wiki.arpa
${KENLM}/bin/build_binary -i wiki.arpa wiki.arpa.bin
```


#### 分散表現

[朝日新聞単語ベクトル](https://cl.asahi.com/api_data/wordembedding.html)を入手してください。
Pythonで、テキスト形式の分散表現をバイナリに変換します。
```python
from gensim.models import KeyedVectors
model = KeyedVectors.load_word2vec_format('skipram.txt')
model.save('skipgram.bin')
```

#### 単語頻度

`word_complexity.py`, `simple_synonym.py` を使用する時や、`lexical_simplificaton.py` の `--ranking` が `glavas`/`bert` の時に必要です。

筑波Webコーパスの頻度表を入手します。エクセルファイルで配布しているので、tsvに変換します。
```sh
wget http://nlt.tsukuba.lagoinst.info/site_media/xlsx/NLT1.30_freq_list.xlsx
pip3 install xlsx2csv
python3 -m xlsx2csv -n "NLT 1.30頻度リスト" -d tab NLT1.30_freq_list.xlsx NLT1.30_freq_list.tsv
```
現代日本語書き言葉均衡コーパスの頻度表を入手します。
```sh
wget https://pj.ninjal.ac.jp/corpus_center/bccwj/data-files/frequency-list/BCCWJ-main_goihyo.zip
unzip -j BCCWJ-main_goihyo.zip 2_BCCWJ/BCCWJ.txt
```
PythonでWikipediaの単語頻度を数え、筑波Webコーパス、現代日本語書き言葉均衡コーパスの単語頻度表とともに1つにまとめます。
```python
from collections import Counter
with open('wiki.tok') as f:
    wiki = Counter(word for line in f for word in line.strip().split())
with open('NLT1.30_freq_list.tsv') as f:
    tsukuba = {word:freq for line in f for i,(word,_,_,freq) in enumerate(line.strip().split()) if i != 0}
with open('BCCWJ.txt') as f:
    bccwj = {l[3], sum(int(i) for i in l[9:15]) for line in f for i,l in enumerate(line.strip().split()) if i != 0}
with open('word2freq.tsv', 'w') as f:
for word in set(wiki) & set(tsukuba) & set(bccwj):
    f.write('{}\t{}\t{}\t{}'.format(word, wiki[word], tsukuba[word], bccwj[word]))
```

#### 文字頻度

PythonでWikipediaの文字頻度を数えます。
```python
import collections
with open('wiki.tok') as f:
    c = collections.Counter(c for line in f for word in line.strip().split() for c in word)
with open('char2freq.tsv', 'w') as f:
    for k,v in c.items():
        f.write('{}\t{}'.format(k,v))
```

#### 品詞辞書

Pythonで現代日本語書き言葉均衡コーパスから品詞辞書を作ります。
```python
with open('BCCWJ.txt') as f:
    bccwj = {l[3]:l[6] for line in f for i,l in enumerate(line.strip().split()) if i != 0}
with open('word2pos.tsv', 'w') as f:
    for k,v in bccwj.items():
        f.write('{}\t{}'.format(k, v))
```

#### 言い換え辞書

PPDB:Japanese (10best) をダウンロードします。
```sh
wget https://ahcweb01.naist.jp/old/resource/jppdb/data/10best.gz
gzip -d 10best.gz
```

PythonでPPDB:Japaneseを整形します。
```python
with open("10best") as inputf, open("ppdb-10best.tsv", 'w') as outf:
    for line in inputf:
        word1, word2, probs,_,_ = line.rstrip().split(' ||| ')
        if ' ' in word1 or ' ' in word2: # 単語以外は飛ばす
            continue
        prob12, prob21 = [float(a) for a in probs.split()]
        if word1 != word2:
            outf.write('{}\t{}\t{}\n'.format(word1, word2, prob12))
            outf.write('{}\t{}\t{}\n'.format(word2, word1, prob21))
```


### 実行

全ての実験を実行
```sh
./experiments.sh
```

語彙平易化のみを実行する場合
```sh
./experiments.sh simplification
```
