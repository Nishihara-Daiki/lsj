#!/bin/bash

# ================================================
#   Prepare
#   These files need to be prepared.
# ================================================
JEV=data/jev.csv
EVALUATION_DATASET=../EvaluationDataset
WORD_EMBEDDING=data/skipgram.bin
WORD_2_FREQ=data/word2freq.tsv
CHAR_2_FREQ=data/char2freq.tsv
WORD_2_POS=data/word2pos.tsv
WORD_PAIR_2_PROB=data/ppdb-10best.tsv
WORD_PAIR_2_PROB_BY_GIZA=data/giza.tsv
LANGUAGE_MODEL=data/wiki.arpa.bin

# The number of GPU device for bert
DEVICE=0


# ================================================
#   Word Complexity Estimation
# ================================================
mkdir -p log

if [ $# -eq 0 ]; then
	# Comparison: part of speech (POS)
	name=wc.pos
	python3 scripts/word_complexity.py --data ${JEV} --split-list data/split-list.tsv --word-to-pos ${WORD_2_POS} --freq-scaling log > log/${name}.log

	# Comparison: character frequency (CF)
	name=wc.cf
	python3 scripts/word_complexity.py --data ${JEV} --split-list data/split-list.tsv --char-to-freq ${CHAR_2_FREQ} --freq-scaling log > log/${name}.log

	# Comparison: word frequency (WF)
	name=wc.wf
	python3 scripts/word_complexity.py --data ${JEV} --split-list data/split-list.tsv --word-to-freq ${WORD_2_FREQ} --freq-scaling log > log/${name}.log

	# Comparison: word embedding (WE)
	name=wc.we
	python3 scripts/word_complexity.py --data ${JEV} --split-list data/split-list.tsv --embedding ${WORD_EMBEDDING} --freq-scaling log > log/${name}.log

	# Ours: POS+WF+CF+WE
	name=wc.ours
	python3 scripts/word_complexity.py --data ${JEV} --split-list data/split-list.tsv --word-to-pos ${WORD_2_POS} --char-to-freq ${CHAR_2_FREQ} --word-to-freq ${WORD_2_FREQ} --embedding ${WORD_EMBEDDING} --freq-scaling log --output data/word2complexity.tsv > log/${name}.log

	# Ours w/o POS
	name=wc.ours.wo-pos
	python3 scripts/word_complexity.py --data ${JEV} --split-list data/split-list.tsv --char-to-freq ${CHAR_2_FREQ} --word-to-freq ${WORD_2_FREQ} --embedding ${WORD_EMBEDDING} --freq-scaling log > log/${name}.log

	# Ours w/o CF
	name=wc.ours.wo-cf
	python3 scripts/word_complexity.py --data ${JEV} --split-list data/split-list.tsv --word-to-pos ${WORD_2_POS} --word-to-freq ${WORD_2_FREQ} --embedding ${WORD_EMBEDDING} --freq-scaling log > log/${name}.log

	# Ours w/o WF
	name=wc.ours.wo-wf
	python3 scripts/word_complexity.py --data ${JEV} --split-list data/split-list.tsv --word-to-pos ${WORD_2_POS} --char-to-freq ${CHAR_2_FREQ} --embedding ${WORD_EMBEDDING} --freq-scaling log > log/${name}.log

	# Ours w/o WE
	name=wc.ours.wo-we
	python3 scripts/word_complexity.py --data ${JEV} --split-list data/split-list.tsv --word-to-pos ${WORD_2_POS} --char-to-freq ${CHAR_2_FREQ} --word-to-freq ${WORD_2_FREQ} --freq-scaling log > log/${name}.log
fi


# ================================================
#   Word Pair Complexity Estimation
# ================================================

if [ $# -eq 0 ]; then
	# Ours A: pointwise
	name=ss.pointwise
	python3 scripts/simple_synonym.py --data ${JEV} --split-list data/split-list.tsv --embedding ${WORD_EMBEDDING} --word-to-pos ${WORD_2_POS} --char-to-freq ${CHAR_2_FREQ} --word-to-freq ${WORD_2_FREQ} --freq-scaling log --synonym-dict ${WORD_PAIR_2_PROB} --word-to-complexity data/word2complexity.tsv --type point-wise --output data/${name}.tsv > log/${name}.log

	# Comparison: pairwise (POS)
	name=ss.pairwise.pos
	python3 scripts/simple_synonym.py --without charfreq wordfreq-single wordfreq-comb embedding-single embedding-comb --data ${JEV} --split-list data/split-list.tsv --embedding ${WORD_EMBEDDING} --word-to-pos ${WORD_2_POS} --char-to-freq ${CHAR_2_FREQ} --word-to-freq ${WORD_2_FREQ} --freq-scaling log --synonym-dict ${WORD_PAIR_2_PROB} --word-to-complexity data/word2complexity.tsv --type pair-wise > log/${name}.log

	# Comparison: pairwise (CF)
	name=ss.pairwise.cf
	python3 scripts/simple_synonym.py --without pos wordfreq-single wordfreq-comb embedding-single embedding-comb --data ${JEV} --split-list data/split-list.tsv --embedding ${WORD_EMBEDDING} --word-to-pos ${WORD_2_POS} --char-to-freq ${CHAR_2_FREQ} --word-to-freq ${WORD_2_FREQ} --freq-scaling log --synonym-dict ${WORD_PAIR_2_PROB} --word-to-complexity data/word2complexity.tsv --type pair-wise > log/${name}.log

	# Comparison: pairwise (WF)
	name=ss.pairwise.wf
	python3 scripts/simple_synonym.py --without pos charfreq wordfreq-comb embedding-single embedding-comb --data ${JEV} --split-list data/split-list.tsv --embedding ${WORD_EMBEDDING} --word-to-pos ${WORD_2_POS} --char-to-freq ${CHAR_2_FREQ} --word-to-freq ${WORD_2_FREQ} --freq-scaling log --synonym-dict ${WORD_PAIR_2_PROB} --word-to-complexity data/word2complexity.tsv --type pair-wise > log/${name}.log

	# Comparison: pairwise (WE)
	name=ss.pairwise.we
	python3 scripts/simple_synonym.py --without pos charfreq wordfreq-single wordfreq-comb embedding-comb --data ${JEV} --split-list data/split-list.tsv --embedding ${WORD_EMBEDDING} --word-to-pos ${WORD_2_POS} --char-to-freq ${CHAR_2_FREQ} --word-to-freq ${WORD_2_FREQ} --freq-scaling log --synonym-dict ${WORD_PAIR_2_PROB} --word-to-complexity data/word2complexity.tsv --type pair-wise > log/${name}.log

	# Ours B: pairwise (POS+CF+WF+WE)
	name=ss.pairwise.ours-B
	python3 scripts/simple_synonym.py --without wordfreq-comb embedding-comb --data ${JEV} --split-list data/split-list.tsv --embedding ${WORD_EMBEDDING} --word-to-pos ${WORD_2_POS} --char-to-freq ${CHAR_2_FREQ} --word-to-freq ${WORD_2_FREQ} --freq-scaling log --synonym-dict ${WORD_PAIR_2_PROB} --word-to-complexity data/word2complexity.tsv --type pair-wise --output data/${name}.tsv --model data/${name}.pkl > log/${name}.log

	# Ours B w/o POS
	name=ss.pairwise.ours-B.wo-pos
	python3 scripts/simple_synonym.py --without pos wordfreq-comb embedding-comb --data ${JEV} --split-list data/split-list.tsv --embedding ${WORD_EMBEDDING} --word-to-pos ${WORD_2_POS} --char-to-freq ${CHAR_2_FREQ} --word-to-freq ${WORD_2_FREQ} --freq-scaling log --synonym-dict ${WORD_PAIR_2_PROB} --word-to-complexity data/word2complexity.tsv --type pair-wise > log/${name}.log

	# Ours B w/o CF
	name=ss.pairwise.ours-B.wo-charfreq
	python3 scripts/simple_synonym.py --without charfreq wordfreq-comb embedding-comb --data ${JEV} --split-list data/split-list.tsv --embedding ${WORD_EMBEDDING} --word-to-pos ${WORD_2_POS} --char-to-freq ${CHAR_2_FREQ} --word-to-freq ${WORD_2_FREQ} --freq-scaling log --synonym-dict ${WORD_PAIR_2_PROB} --word-to-complexity data/word2complexity.tsv --type pair-wise > log/${name}.log

	# Ours B w/o WF
	name=ss.pairwise.ours-B.wo-wordfreq
	python3 scripts/simple_synonym.py --without wordfreq-single wordfreq-comb embedding-comb --data ${JEV} --split-list data/split-list.tsv --embedding ${WORD_EMBEDDING} --word-to-pos ${WORD_2_POS} --char-to-freq ${CHAR_2_FREQ} --word-to-freq ${WORD_2_FREQ} --freq-scaling log --synonym-dict ${WORD_PAIR_2_PROB} --word-to-complexity data/word2complexity.tsv --type pair-wise > log/${name}.log

	# Ours B w/o WE
	name=ss.pairwise.ours-B.wo-embedding
	python3 scripts/simple_synonym.py --without embedding-single wordfreq-comb embedding-comb --data ${JEV} --split-list data/split-list.tsv --embedding ${WORD_EMBEDDING} --word-to-pos ${WORD_2_POS} --char-to-freq ${CHAR_2_FREQ} --word-to-freq ${WORD_2_FREQ} --freq-scaling log --synonym-dict ${WORD_PAIR_2_PROB} --word-to-complexity data/word2complexity.tsv --type pair-wise > log/${name}.log

	# Ours C: Ours B with difference features
	name=ss.pairwise.ours-B.w-diff
	python3 scripts/simple_synonym.py --data ${JEV} --split-list data/split-list.tsv --embedding ${WORD_EMBEDDING} --word-to-pos ${WORD_2_POS} --char-to-freq ${CHAR_2_FREQ} --word-to-freq ${WORD_2_FREQ} --freq-scaling log --synonym-dict ${WORD_PAIR_2_PROB} --word-to-complexity data/word2complexity.tsv --type pair-wise > log/${name}.log
fi


# ================================================
#   Lexical Simplification
# ================================================

mkdir -p output

if [ $# -eq 0 ]; then

	# candidate method / ranking method

	# Light-LS / Light-LS
	name=lsj.Light-LS.Light-LS
	python3 scripts/lexical_simplification.py --candidate glavas --ranking glavas --data ${EVALUATION_DATASET} --output output/${name}.out --embedding ${WORD_EMBEDDING} --language-model ${LANGUAGE_MODEL} --word-to-freq ${WORD_2_FREQ} --synonym-dict ${WORD_PAIR_2_PROB} --pretrained-bert bert-base-japanese-whole-word-masking --word-to-complexity data/word2complexity.tsv > log/${name}.log

	# Light-LS / 5-gram language model
	name=lsj.Light-LS.Light-LS
	python3 scripts/lexical_simplification.py --candidate glavas --ranking language-model --data ${EVALUATION_DATASET} --output output/${name}.out --embedding ${WORD_EMBEDDING} --language-model ${LANGUAGE_MODEL} --word-to-freq ${WORD_2_FREQ} --synonym-dict ${WORD_PAIR_2_PROB} --pretrained-bert bert-base-japanese-whole-word-masking --word-to-complexity data/word2complexity.tsv > log/${name}.log


	# PPDB / 5-gram language model
	name=lsj.ppdb.language-model
	python3 scripts/lexical_simplification.py --candidate sysnonym --ranking language-model --data ${EVALUATION_DATASET} --output output/${name}.out --embedding ${WORD_EMBEDDING} --language-model ${LANGUAGE_MODEL} --word-to-freq ${WORD_2_FREQ} --synonym-dict ${WORD_PAIR_2_PROB} --pretrained-bert bert-base-japanese-whole-word-masking --word-to-complexity data/word2complexity.tsv > log/${name}.log

	# PPDB / Light-LS
	name=lsj.ppdb.Light-LS
	python3 scripts/lexical_simplification.py --candidate synonym --ranking glavas --data ${EVALUATION_DATASET} --output output/${name}.out --embedding ${WORD_EMBEDDING} --language-model ${LANGUAGE_MODEL} --word-to-freq ${WORD_2_FREQ} --synonym-dict ${WORD_PAIR_2_PROB} --pretrained-bert bert-base-japanese-whole-word-masking --word-to-complexity data/word2complexity.tsv > log/${name}.log


	# GIZA / 5-gram language model
	name=lsj.giza.language-model
	python3 scripts/lexical_simplification.py --candidate synonym --ranking language-model --data ${EVALUATION_DATASET} --output output/${name}.out --embedding ${WORD_EMBEDDING} --language-model ${LANGUAGE_MODEL} --word-to-freq ${WORD_2_FREQ} --synonym-dict ${WORD_PAIR_2_PROB_BY_GIZA} --pretrained-bert bert-base-japanese-whole-word-masking --word-to-complexity data/word2complexity.tsv > log/${name}.log

	# GIZA / Light-LS
	name=lsj.giza.Light-LS
	python3 scripts/lexical_simplification.py --candidate synonym --ranking glavas --data ${EVALUATION_DATASET} --output output/${name}.out --embedding ${WORD_EMBEDDING} --language-model ${LANGUAGE_MODEL} --word-to-freq ${WORD_2_FREQ} --synonym-dict ${WORD_PAIR_2_PROB_BY_GIZA} --pretrained-bert bert-base-japanese-whole-word-masking --word-to-complexity data/word2complexity.tsv > log/${name}.log


	# PPDB:point-wise / 5-gram language model
	name=lsj.point-wise.language-model
	python3 scripts/lexical_simplification.py --candidate synonym --ranking language-model --data ${EVALUATION_DATASET} --output output/${name}.out --embedding ${WORD_EMBEDDING} --language-model ${LANGUAGE_MODEL} --word-to-freq ${WORD_2_FREQ} --synonym-dict data/ss.pointwise.tsv --pretrained-bert bert-base-japanese-whole-word-masking --word-to-complexity data/word2complexity.tsv > log/${name}.log

	# PPDB:point-wise / Light-LS
	name=lsj.point-wise.language-model
	python3 scripts/lexical_simplification.py --candidate synonym --ranking glavas --data ${EVALUATION_DATASET} --output output/${name}.out --embedding ${WORD_EMBEDDING} --language-model ${LANGUAGE_MODEL} --word-to-freq ${WORD_2_FREQ} --synonym-dict data/ss.pointwise.tsv --pretrained-bert bert-base-japanese-whole-word-masking --word-to-complexity data/word2complexity.tsv > log/${name}.log


	# bert / bert
	name=lsj.bert.bert
	python3 scripts/lexical_simplification.py --candidate bert --ranking bert --data ${EVALUATION_DATASET} --output output/${name}.out --embedding ${WORD_EMBEDDING} --language-model ${LANGUAGE_MODEL} --word-to-freq ${WORD_2_FREQ} --synonym-dict ${WORD_PAIR_2_PROB} --pretrained-bert bert-base-japanese-whole-word-masking --device ${DEVICE} --word-to-complexity data/word2complexity.tsv > log/${name}.log


	### 言い換え候補選択の評価
	name=lsj.all.none
	python3 scripts/lexical_simplification.py --candidate all --ranking none --data ${EVALUATION_DATASET} --output output/${name}.out --embedding ${WORD_EMBEDDING} --language-model ${LANGUAGE_MODEL} --word-to-freq ${WORD_2_FREQ} --synonym-dict ${WORD_PAIR_2_PROB} --pretrained-bert bert-base-japanese-whole-word-masking --device ${DEVICE} --word-to-complexity data/word2complexity.tsv > log/${name}.log

	### ランキングの評価
	# gold / 5-gram language model
	name=lsj.gold.language-model
	python3 scripts/lexical_simplification.py --candidate gold --ranking language-model --data ${EVALUATION_DATASET} --output output/${name}.out --embedding ${WORD_EMBEDDING} --language-model ${LANGUAGE_MODEL} --word-to-freq ${WORD_2_FREQ} --synonym-dict ${WORD_PAIR_2_PROB} --pretrained-bert bert-base-japanese-whole-word-masking --word-to-complexity data/word2complexity.tsv > log/${name}.log

	# gold / Light-LS
	name=lsj.gold.glavas
	python3 scripts/lexical_simplification.py --candidate gold --ranking glavas --data ${EVALUATION_DATASET} --output output/${name}.out --embedding ${WORD_EMBEDDING} --language-model ${LANGUAGE_MODEL} --word-to-freq ${WORD_2_FREQ} --synonym-dict ${WORD_PAIR_2_PROB} --pretrained-bert bert-base-japanese-whole-word-masking --word-to-complexity data/word2complexity.tsv > log/${name}.log
fi
