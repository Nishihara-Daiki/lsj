# Lexical Simplification Japanese

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


## 実験の再現
### 環境
+ Python 3.7.2
+ Mecab (IPADIC 2.7.0)

### ダウンロードとインストール
必要なパッケージをインストールします。MeCabは、日本語対応できるか確認してください。
```sh
apt install -y mecab libmecab-dev mecab-ipadic-utf8 libboost-all-dev
pip3 install -r requirements.txt
```

評価用データセットをダウンロードします。
```sh
git clone https://github.com/KodairaTomonori/EvaluationDataset
```

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

[朝日新聞単語ベクトル](https://cl.asahi.com/api_data/wordembedding.html)を入手してください。
Pythonで、テキスト形式の分散表現をバイナリに変換します。
```python
from gensim.models import KeyedVectors
model = KeyedVectors.load_word2vec_format('skipram.txt')
model.save('skipgram.bin')
```

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

PythonでWikipediaの文字頻度を数えます。
```python
import collections
with open('wiki.tok') as f:
    c = collections.Counter(c for line in f for word in line.strip().split() for c in word)
with open('char2freq.tsv', 'w') as f:
    for k,v in c.items():
        f.write('{}\t{}'.format(k,v))
```

Pythonで現代日本語書き言葉均衡コーパスから品詞辞書を作ります。
```python
with open('BCCWJ.txt') as f:
    bccwj = {l[3]:l[6] for line in f for i,l in enumerate(line.strip().split()) if i != 0}
with open('word2pos.tsv', 'w') as f:
    for k,v in bccwj.items():
        f.write('{}\t{}'.format(k, v))
```

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
```sh
./experiments.sh simplification
```
実験の全てを実行したい場合は引数無しに実行
```sh
./experiments.sh
```

