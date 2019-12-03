import sys
import argparse
import math
import pickle
from gensim.models import KeyedVectors
import MeCab
import kenlm
from word_complexity import load_word2vec, load_freqs
from simple_synonym import load_word2level, load_synonym
from collections import defaultdict

from pprint import pprint


def main():
	args = parse()

	input_sentence_file = args.data + '/Sentence/sentences.txt'
	target_word_file = args.data + '/BCCWJ_target_location/location.txt'
	reference_file = args.data + '/substitutes/mle_rank.csv'

	word2vec = load_word2vec(args.embedding)
	w2v_vocab = set(word2vec.vocab.keys())
	language_model = kenlm.Model(args.language_model)
	mecab = MeCab.Tagger("" if not args.mecab_dict else "-d " + args.mecab_dict)
	word2level = load_word2level(args.word_to_complexity)
	word2synonym = load_word2synonym(args.synonym_dict, word2level)
	word2freq = load_freqs(args.word_to_freq, 'none')[0]

	freq_total = sum(word2freq.values())
	simple_synonym = load_simple_synonym(args.simple_synonym)

	using_vocab = w2v_vocab

	output_log(args.log, 'vocab size is {}'.format(len(w2v_vocab)))

	results = list()

	with open(input_sentence_file, "r") as input_f, open(target_word_file, "r") as target_f:
		for i,(line,target) in enumerate(zip(input_f, target_f)):
			line = line.strip()
			sentence = morphological_analysis(line, mecab)
			_,_,_,phrase,word,stem,_ = target.split(",")
			target = phrase if phrase in using_vocab else word if word in using_vocab else stem if stem in using_vocab else ""
			if target == "":
				rst = phrase
			else:
				attached_words = tuple(phrase.replace(target if target in phrase else word, '\t').split('\t'))
				attached_words = attached_words if len(attached_words) == 2 else ('','')
				candidates = pick_candidates(target, args.most_similar, word2vec, w2v_vocab, word2synonym, args.candidate, args.cos_threshold)
				candidates = pick_simple_before_ranking(target, candidates, word2freq, freq_total, word2level, simple_synonym, args.simplicity)
				candidatelist = ranking(target, candidates, sentence, word2vec, w2v_vocab, word2freq, freq_total, language_model, attached_words, mecab, word2synonym, args.ranking)
				candidatelist = pick_simple(candidatelist, args.simplicity, target, word2level, word2freq, freq_total, simple_synonym)
				left,right = attached_words
				rst = ",".join([" ".join([left + c[1] + right for c in rank]) for rank in candidatelist])
			results.append(rst)

	output_results(args.output, results)
	evaluate(target_word_file, results, reference_file)



# 候補をとってくる
def pick_candidates(target, most_similar, word2vec, w2v_vocab, word2synonym, candidate_type, cos_threshold):
	candidates = list()
	if candidate_type == 'glavas' or candidate_type == 'glavas+synonym':
		candidates += [c for c,_ in word2vec.most_similar(target, topn=most_similar)]
	if candidate_type == 'synonym' or candidate_type == 'glavas+synonym':
		candidates += [c for c,_ in word2synonym[target] if c in w2v_vocab]
	elif candidate_type == 'oracle':
		pass # 未実装
	candidates = [c for c in candidates if word2vec.similarity(c, target) >= cos_threshold]
	return candidates


# 提案手法では、ランキングの前に難易度判定を行う
def pick_simple_before_ranking(target, candidates, word2freq, freq_total, word2level, simple_synonym, simplicity_type):
	# if simplicity_type == 'glavas':
	# 	candidates = [c for c in candidates if information_contents(word2freq, freq_total, c) < information_contents(word2freq, freq_total, target)]
	if simplicity_type == "point-wise":
		candidates = [c for c in candidates if point_wise(c, target, word2level) == -1]
	if simplicity_type == 'pair-wise':
		candidates = [c for c in candidates if (target, c) in simple_synonym]
	return candidates


def load_word2synonym(filename, word2level):
	synonym = load_synonym(filename, word2level)
	word2synonym = defaultdict(list)
	for w1, w2, prob in synonym:
		word2synonym[w1].append((w2, prob))
	return word2synonym


def load_simple_synonym(filename):
	simple_synonym = {}
	with open(filename, 'r') as f:
		for line in f:
			line = line.strip().split('\t')
			comp, simp = line[:2]
			simple_synonym[(comp, simp)] = line[2:]
	return simple_synonym


# ランキング
def ranking(target, candidates, sentence, word2vec, w2v_vocab, word2freq, freq_total, language_model, attached_words, mecab, word2synonym, ranking_type):
	def context_sim(sentence, candidate):
		sentence_tk = [(token["surface"], token["pos"]) for token in sentence]
		similarity_list = [word2vec.similarity(candidate, context) for context, pos in sentence_tk if is_content(pos) and context != candidate and context in w2v_vocab]
		similarity = sum(similarity_list) / len(similarity_list) if similarity_list else 0
		return similarity

	def language_model_score(sentence, target, candidate, attached_words):
		left,right = attached_words
		sentence_text = "".join([w["surface"] for w in sentence])
		c_sentence = morphological_analysis(sentence_text.replace(target, candidate), mecab)
		c_sentence_text = " ".join([w["surface"] for w in c_sentence])
		return language_model.score(c_sentence_text)

	def get_p_score(word2synonym, target, c):
		prob = 0
		for w,p in word2synonym[c]:
			if w == target:
				prob = p
		return prob

	# スコアの大きいものから順位づけ
	def make_ranking(lst):
		sorted_with_idx = sorted([(i,l) for i,l in enumerate(lst)], key=lambda x: x[1], reverse=True)
		ranking = sorted([(i,*a) for i,a in enumerate(sorted_with_idx)], key=lambda x: x[1])
		return [a[0] for a in ranking]

	def sorting_considering_same_order(lst, key=lambda x: x, reverse=False):
		ranks = sorted(set(key(x) for x in lst), reverse=reverse)
		result = [list() for _ in range(len(ranks))]
		for x in lst:
			result[ranks.index(key(x))].append(x)
		return result

	ranktable = list()
	if ranking_type in {'glavas', 'ours'}:
		ranktable.append( make_ranking([word2vec.similarity(target, c) for c in candidates]) )
		ranktable.append( make_ranking([context_sim(sentence, c) for c in candidates]) )
		ranktable.append( make_ranking([information_contents(word2freq, freq_total, target) - information_contents(word2freq, freq_total, c) for c in candidates]) )
	if ranking_type in {'glavas', 'language-model', 'ours'}:
		ranktable.append( make_ranking([language_model_score(sentence, target, c, attached_words) for c in candidates]) )

	if ranking_type in {'ours'}:
		ranktable.append( make_ranking([get_p_score(word2synonym, target, c) for c in candidates]) )


	sum_rank = [sum(f) for f in zip(*ranktable)]
	candidatelist = sorting_considering_same_order(list(zip(sum_rank, candidates)), key=lambda x:x[0], reverse=False)
	return candidatelist


def point_wise(word1, word2, word2level):
	if word1 not in word2level or word2 not in word2level:
		return None
	word1_lv, word2_lv = word2level[word1], word2level[word2]
	if word1_lv	< word2_lv:
		return -1
	if word1_lv	== word2_lv:
		return 0
	if word1_lv	> word2_lv:
		return 1


# 平易化されているか判断
def pick_simple(candidatelist, simplicity_type, target, word2level, word2freq, freq_total, simple_synonym):
	if len(candidatelist) == 0:
		return [[(0, target)]]
	best_candidate = candidatelist[0][0][1]
	if simplicity_type == 'glavas' and information_contents(word2freq, freq_total, best_candidate) < information_contents(word2freq, freq_total, target):
		return candidatelist
	elif simplicity_type == 'point-wise' and point_wise(best_candidate, target, word2level) == -1:
		return candidatelist
	elif simplicity_type == 'pair-wise' and (target, best_candidate) in simple_synonym:
		return candidatelist
	else:
		return [[(0, target)]]


def morphological_analysis(text, mecab):
	r = list()
	node = mecab.parseToNode(text)
	while node:
		r.append({"surface": node.surface, "pos": node.feature.split(",")[0]})
		node = node.next
	return r[1:-1]


def information_contents(word2freq, freq_total, w):
	return - math.log((word2freq[w] + 1) / (freq_total + 1))


def is_content(pos):
	return pos in ["名詞", "動詞", "形容詞", "副詞"]


def output_results(output_path, results):
	if output_path == "stdout":
		out_f = sys.stdout
	elif output_path:
		out_f = open(output_path, "w", 1)
	for rst in results:
		out_f.write(rst + "\n")
	if output_path != "stdout":
		out_f.close()


def output_log(output_path, log):
	output_results(output_path, ["[log] " + log])


def evaluate(input_file, output, reference_file):
	in_f = open(input_file, "r")
	out_f = output
	ref_f = open(reference_file, "r")

	accuracy_list = []
	changed_list = []
	precision_list = []
	tochange_list = []

	for in_line, out_line, ref_line in zip(in_f, out_f, ref_f):
		_, _, _, in_phrase, _, _, _ = in_line.strip().split(",")
		out_phrase = out_line.strip().split(",")[0].split(" ")[0]

		# 評価データセットの、入力とリファレンスのフォーマットが統一されtない問題を対処する：ここから
		# '漂って'だけは評価データの制作ミス（？）なので例外処理
		if in_phrase == 'が漂って':
			ref_phrases = ['が出て', 'がして']
		else:
			refs = ref_line.strip().replace(',', ' ').split(' ')[1:]
			rlist = [r for r in refs if r in in_phrase]
			input_in_ref_phrase = sorted(rlist, key=len, reverse=True)[0]
			left, right = in_phrase.replace(input_in_ref_phrase, ' ').split(' ')
		# ここまで

			ref_phrases_list = ref_line.strip().split(",")[1:]
			ref_phrases = []
			for r in ref_phrases_list:
				if input_in_ref_phrase in r.split():
					break
				else:
					ref_phrases.extend([left + p + right for p in r.split()])

			# 正解が存在しない場合は評価の対象外
			if not ref_phrases:
				continue

			# left と right が含まれているリファレンスが１つ以上ある場合に限定する
			if sum(left == r[:len(left)] and right == r[-len(right):] for r in ref_phrases) == 0:
				continue
		accuracy_list.append(out_phrase in ref_phrases and in_phrase not in ref_phrases)
		tochange_list.append(in_phrase not in ref_phrases)
		precision_list.append(out_phrase in ref_phrases and out_phrase != in_phrase)
		changed_list.append(out_phrase != in_phrase)

	accuracy = sum(accuracy_list) / sum(tochange_list)
	precision = sum(precision_list) / sum(changed_list)
	changed = sum(changed_list) / len(changed_list)
	print("acc/prec/changed = {:>5.2f}\t{:>5.2f}\t{:>5.2f}".format(accuracy*100, precision*100, changed*100))
	print("acc/prec/changed = {}/{}\t{}/{}\t{}/{}".format(
		sum(accuracy_list), sum(tochange_list), sum(precision_list), sum(changed_list), sum(changed_list), len(changed_list)
	))

	in_f.close()
	ref_f.close()


def parse():
	parser = argparse.ArgumentParser(description='')
	parser.add_argument('--candidate',          '-C', help='言い換え集合の取得手法', choices=['glavas', 'synonym', 'glavas+synonym', 'oracle'])
	parser.add_argument('--simplicity',         '-S', help='難易度判定手法', choices=['none', 'glavas', 'point-wise', 'pair-wise', 'oracle'])
	parser.add_argument('--ranking',            '-R', help='ランキング手法', choices=['glavas', 'language-model', 'ours', 'oracle'])
	parser.add_argument('--output',             '-o', help='output file; default is "stdout"', default='stdout')
	parser.add_argument('--log',                '-g', help='log file; default is "stdout"', default='stdout')
	parser.add_argument('--data',               '-d', help='Kodaira\'s dataset e.g. /path/to/EvaluationDataset')
	parser.add_argument('--embedding',          '-e', help='binary embedding file')
	parser.add_argument('--language-model',     '-m', help='kenlm language model path')
	parser.add_argument('--most-similar',       '-n', help='select n most sililar candidate words', default=10, type=int)
	parser.add_argument('--word-to-freq',       '-f', help='単語 \t 頻度 の tsv。頻度が複数列あっても1列目しか見ない')
	parser.add_argument('--synonym-dict',       '-p', help='--candidate=synonym の時に使う同義語辞書')
	parser.add_argument('--simple-synonym',     '-i', help='--simplicity=pair-wiseの時に使う平易化辞書')
	parser.add_argument('--word-to-complexity', '-l', help='--simplicity=point-wise の時に使う単語難易度辞書')
	parser.add_argument('--cos-threshold',      '-c', help='言い換え集合を取得する際に、この値未満の単語は含まない', default=0.0, type=float)
	# オプション
	parser.add_argument('--mecab-dict',         '-D', help='mecab dictionary path', default='')
	args = parser.parse_args()
	return args


if __name__ == '__main__':
	main()
