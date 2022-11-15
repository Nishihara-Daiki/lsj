# Lexical Simplification Japanese

## Publication

+ a large-scale word complexity lexicon
+ a simplified synonym lexicon from complex words into simpler ones
+ a toolkit for development and benchmarking lexical simplification system


## Word complexity lexicon
Word complexity lexicon ([data/word2complexity.tsv](data/word2complexity.tsv)) has an easy/medium/difficult label for each word, separated by tabs.
The number of words included in each label is as follows:

| label | #words |
| -- | --: |
| easy | 749 |
| medium | 7,808 |
| difficult | 32,048 |
| total | 40,605 |

## A simplified synonym lexicon
There are 2 lexicon by pointwise or pairwise methods.

+ pointwise [data/ss.pointwise.tsv](data/ss.pointwise.tsv)
+ pairwise [data/ss.pairwise.ours-B.tsv](data/ss.pairwise.ours-B.tsv)

pointwise is separated by tabs as follows:
```
word1 word2 P(word2|word1)  cos(word1,word2)    complexity_of_word1 complexity_of_word2
```
where complexity is 0(easy), 1(medium) or 2(difficult) and cos(word1,word2) is cosine similarity of word embeddings of word1 and word2.
Pairwise is also separated by tabs as similar as pointwise, but it has only first 4 columns. Pairwise method does not estimate the complexity score.


## A toolkit for lexical simplification

Rewrite a target word in the sentence into simpler one.
It consists 2 parts:
1. **Aqcuire candidates**  aquire some paraphrase candidates.
2. **rank**  Rank the paraphrase and output the best candidate.

### Environment

+ Python 3.7.2
+ Mecab (IPADIC 2.7.0)

Install packages and download the evaluation dataset:
```sh
pip3 install -r requirements.txt
git clone https://github.com/KodairaTomonori/EvaluationDataset
```

Check:
```sh
python3 scripts/lexical_simplification.py \
    --candidate gold \
    --ranking none \
    --data ../EvaluationDataset \
    --output output/tmp.out
```
Output:
```
[log] word2vec vocab size is 0
2010it [00:00, 7789.41it/s]
acc/prec/changed = 70.50    70.50   100.00
candidate potential/prec/recall = 76.99 31.46   62.67
```

### Arguments

* `--candidate, -C` How to acuire paraphrase candidates.
    - `glavas` Light-LS (Glavas and Stajner 2015).
    - `synonym` Use synonym dictionary.
    - `glavas+synonym` BERT-LS (Qiang et al. 2019).
* `--simplicity, -S` How to determine if the candidate word is simpler than the original one.
    - `none` Regard all candidate words are simple.
    - `glavas` Light-LS.
    - `point-wise` Use word-to-complexity dictionary.
    - `pair-wise` Use simple paraphrase dictionlay.
* `--ranking, -R` How to rank the candidates.
    - `glavas` Light-LS
    - `language-model` Use language model's score.
    - `ours` Similar to `glavas` but also use score of synonym dictionaly.
* `--output, -o` output filename of paraphrase. Default is `stdout` (starndard output).
* `--log, -g` output log filename. Default is `stdout`.
* `--data, -d` Path to [Kodaira's evaluation dataset](https://github.com/KodairaTomonori/EvaluationDataset)
* `--embedding, -e` Binary file of word embedding.
* `--language-model, -m` Pretrained language model by KenLM.
* `--most-similar, -n` Acuire top-n of paraphrase candidates.
* `--word-to-freq, -f` A word to word frequency dictionary .tsv file.
* `--synonym-dict, -p` A synonym dictionlary .tsv file. It consits 3 colomun of word 1, word 2 and score.
* `--word-to-complexity, -l` A word to word complexity dictionary .tsv file.
* `--cos-threshold, -c` Remove paraphrase candidates below cosine threshold.


## Experiment

### Install and download

Install packages. Check your MeCab supports Japanese.
```sh
apt install -y mecab libmecab-dev mecab-ipadic-utf8 libboost-all-dev
pip3 install -r requirements.txt
```

word complexity lexicon tool（`word_complexity.py`）or a simplified synonym lexicon tool（`simple_synonym.py`）need **word embedding**、**word freq**、**char freq**、**word to pos**.


A toolkit for lexical simplification.（`lexical_simplification.py`）needs **Kodaira's evaluation dataset**

|                                | language-model | embedding | word2freq | word2freq |
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



#### Evaluation dataset
```sh
git clone https://github.com/KodairaTomonori/EvaluationDataset
```

#### Language model

Down load Japanese Wikipedia:
```sh
wget https://dumps.wikimedia.org/jawiki/20190801/jawiki-20190801-pages-articles-multistream.xml.bz2
```

Get and extranct Wikipedia text:
```sh
wget https://raw.githubusercontent.com/attardi/wikiextractor/3162bb6c3c9ebd2d15be507aa11d6fa818a454ac/WikiExtractor.py -P scripts/
python3 scripts/WikiExtractor.py -b 5G -o extracted jawiki-20190801-pages-articles-multistream.xml.bz2 -q
mv extracted/AA/wiki_00 .
rmdir extracted/AA extracted
```

Toknize Japanese:
```sh
cat wiki_00 | grep -v -e "<doc" -e "</doc>" -e '^\s*$' | mecab -O wakati > wiki.tok
```

Download KenLM:
```sh
git clone https://github.com/kpu/kenlm
mkdir -p kenlm/build
cd kenlm/build
cmake ..
make -j 4
```

Train a language model:
```sh
mkdir -p tmp
${KENLM}/bin/lmplz -o 5 -S 80% -T tmp < wiki.tok > wiki.arpa
${KENLM}/bin/build_binary -i wiki.arpa wiki.arpa.bin
```


#### Train embedding

Get [Asahi Shinbun word vector](https://cl.asahi.com/api_data/wordembedding.html).
Convert text to binary by Python.
```python
from gensim.models import KeyedVectors
model = KeyedVectors.load_word2vec_format('skipram.txt')
model.save('skipgram.bin')
```

#### Word frequency

Get Tsukuba Web Corpus and convert from .xlsx to .tsv.
```sh
wget http://nlt.tsukuba.lagoinst.info/site_media/xlsx/NLT1.30_freq_list.xlsx
pip3 install xlsx2csv
python3 -m xlsx2csv -n "NLT 1.30頻度リスト" -d tab NLT1.30_freq_list.xlsx NLT1.30_freq_list.tsv
```
Get BCCWJ word frequency:
```sh
wget https://pj.ninjal.ac.jp/corpus_center/bccwj/data-files/frequency-list/BCCWJ-main_goihyo.zip
unzip -j BCCWJ-main_goihyo.zip 2_BCCWJ/BCCWJ.txt
```
In Python, count a word in Wikipedia and combine three word frequency file into one.
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

#### Charactor frequency

Count Wikipedia charactor frequency by Python:

```python
import collections
with open('wiki.tok') as f:
    c = collections.Counter(c for line in f for word in line.strip().split() for c in word)
with open('char2freq.tsv', 'w') as f:
    for k,v in c.items():
        f.write('{}\t{}'.format(k,v))
```

#### A word to part of speech dictionary

Create a word to part of speech dictionary from BCCWJ corpus by Python:
```python
with open('BCCWJ.txt') as f:
    bccwj = {l[3]:l[6] for line in f for i,l in enumerate(line.strip().split()) if i != 0}
with open('word2pos.tsv', 'w') as f:
    for k,v in bccwj.items():
        f.write('{}\t{}'.format(k, v))
```

#### Paraphrase dictionary

Download PPDB:Japanese (10best):
```sh
wget https://ahcweb01.naist.jp/old/resource/jppdb/data/10best.gz
gzip -d 10best.gz
```

Make a synonym dictionary from PPDB:Japanese by Python:
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


### Run

Run all:
```sh
./experiments.sh
```

Run lexical simplification only:
```sh
./experiments.sh simplification
```


## License

Language resources in this repository follows [Creative Commons Attribution-ShareAlike 4.0 (CC BY-SA 4.0)](https://creativecommons.org/licenses/by-sa/4.0/) license.
Source codes in this repository follows [Apache License 2.0](https://licenses.opensource.jp/Apache-2.0/Apache-2.0.html).

In this experiments, we use the following resources:

* [BCCWJ frequency list](https://clrd.ninjal.ac.jp/bccwj/freq-list.html)
* [Japanese Education Vocabulary List](http://jhlee.sakura.ne.jp/JEV/)
* [Asahi Shimbun word vector](https://cl.asahi.com/api_data/wordembedding.html)
