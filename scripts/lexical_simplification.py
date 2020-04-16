import sys
import argparse
import math
import pickle
from gensim.models import KeyedVectors
import MeCab
import kenlm
from word_complexity import load_word2vec, load_freqs
from simple_synonym import load_word2level
from collections import defaultdict
import torch
from transformers.tokenization_bert_japanese import BertJapaneseTokenizer
from transformers.modeling_bert import BertForMaskedLM
import subprocess
from tqdm import tqdm

from pprint import pprint


def main():
	args = parse()

	input_sentence_file = args.data + '/Sentence/sentences.txt'
	target_word_file = args.data + '/BCCWJ_target_location/location.txt'
	reference_file = args.data + '/substitutes/mle_rank.csv'

	word2vec, w2v_vocab, language_model, mecab, word2level, word2synonym, word2freq, bert, freq_total, using_vocab = load(args)

	output_log(args.log, 'word2vec vocab size is {}'.format(len(w2v_vocab)))

	results = list()
	candidates_list = list()

	with open(input_sentence_file, "r") as input_f, open(target_word_file, "r") as target_f, open(reference_file, "r") as ref_f:
		for line,target,ref in tqdm(zip(input_f, target_f, ref_f)):
			line = line.strip()
			sentence = morphological_analysis(line, mecab)
			_,_,_,phrase,word,stem,_ = target.split(",")
			target = word if word in using_vocab or not using_vocab else stem if stem in using_vocab else ""
			if target == "":
				rst = phrase
			else:
				attached_words = tuple(phrase.replace(target if target in phrase else word, '\t').split('\t'))
				attached_words = attached_words if len(attached_words) == 2 else ('','')
				candidates, scores = pick_candidates(target, args.most_similar, word2vec, w2v_vocab, word2synonym, bert, sentence, args.candidate, args.cos_threshold, ref, mecab, args.device)
				candidates_list.append(candidates)
				candidatelist = ranking(target, candidates, scores, sentence, word2vec, w2v_vocab, word2freq, freq_total, language_model, attached_words, mecab, bert, args.ranking)
				candidatelist = pick_simple(candidatelist, target, word2freq, freq_total, args.candidate, args.ranking)
				left,right = attached_words
				rst = ",".join([" ".join([left + c[1] + right for c in rank]) for rank in candidatelist])
			results.append(rst)

	output_results(args.output, results)
	evaluate(target_word_file, results, candidates_list, reference_file)


def load(args):
	word2vec = load_word2vec(args.embedding)
	w2v_vocab = set(word2vec.vocab.keys()) if word2vec else {}
	language_model = load_language_model(args.language_model)
	mecab = MeCab.Tagger("" if not args.mecab_dict else "-d " + args.mecab_dict)
	word2level = load_word2level(args.word_to_complexity)
	word2synonym = load_word2synonym(args.synonym_dict)
	word2freq = load_freqs(args.word_to_freq, 'none')
	word2freq = word2freq[0] if word2freq else None
	freq_total = sum(word2freq.values()) if word2freq else 0
	bert = load_bertmodel(args.pretrained_bert)
	using_vocab = w2v_vocab
	return word2vec, w2v_vocab, language_model, mecab, word2level, word2synonym, word2freq, bert, freq_total, using_vocab


# 候補をとってくる
def pick_candidates(target, most_similar, word2vec, w2v_vocab, word2synonym, bert, sentence, candidate_type, cos_threshold, ref, mecab, device):
	candidates = list()
	word2vec_score, bert_score = list(), list()

	if candidate_type in {'glavas', 'all'}:
		for c,v in word2vec.most_similar(target, topn=most_similar):
			candidates.append(c)
			word2vec_score.append(c)

	if candidate_type in {'synonym', 'all'}:
		candidates += [c for c,_ in word2synonym[target] if c in w2v_vocab]

	if candidate_type in {'bert', 'all'}:
		for c,v in zip(*word_to_bertsynonym(device, bert, target, sentence, most_similar)):
			if c in w2v_vocab:
				candidates.append(c)
				bert_score.append(v)

	if candidate_type == 'gold':
		candidates = [l[0] for l in [[w["surface"] for w in morphological_analysis(c, mecab) if is_content(w["pos"]) and (w["surface"] in w2v_vocab or not w2v_vocab)] for c in ref.rstrip().replace(',', ' ').split()[1:]] if len(l) > 0]

	if cos_threshold > 0:
		if word2vec_score:
			candidates = [c for c,v in zip(candidates, word2vec_score) if v >= cos_threshold]
		else:
			candidates = [c for c in candidates if word2vec.similarity(c, target) >= cos_threshold]

	scores = {'word2vec': word2vec_score, 'bert': bert_score}

	return candidates, scores


def load_language_model(filename):
	return kenlm.Model(filename) if filename else None


def load_word2synonym(filename):
	word2synonym = defaultdict(list)
	if filename:
		with open(filename) as f:
			for line in f:
				w1, w2, prob = line.rstrip().split('\t')[:3]
				word2synonym[w1].append((w2, prob))
	return word2synonym


def load_bertmodel(modelname):
	if modelname:
		tokenizer = BertJapaneseTokenizer.from_pretrained(modelname)
		model = BertForMaskedLM.from_pretrained(modelname)
	else:
		tokenizer, model = None, None
	return tokenizer, model


def word_to_bertsynonym(device, bert, target, sentence, topk):
	tokenizer, model = bert
	sentence = ''.join([w['surface'] for w in sentence])
	line = sentence.split(target)
	input_string = sentence + tokenizer.sep_token + line[0] + tokenizer.mask_token + ''.join(line[1:])
	input_ids = tokenizer.encode(input_string, return_tensors='pt')
	if device != -1:
		cuda = 'cuda:' + str(device)
		input_ids = input_ids.to(cuda)
		model.to(cuda)
	masked_index = torch.where(input_ids == tokenizer.mask_token_id)[1].tolist()[0]
	result = model(input_ids)
	pred = result[0][:, masked_index].topk(topk)
	pred_ids, pred_values = pred.indices.tolist()[0], pred.values.tolist()[0]
	target_id = tokenizer.convert_tokens_to_ids(target)
	candidates = tokenizer.convert_ids_to_tokens([i for i in pred_ids if i != target_id])
	return candidates, pred_values



def language_model_score(sentence, target, candidate, attached_words, language_model, mecab):
	left,right = attached_words
	sentence_text = "".join([w["surface"] for w in sentence])
	c_sentence = morphological_analysis(sentence_text.replace(target, candidate), mecab)
	c_sentence_text = " ".join([w["surface"] for w in c_sentence])
	return language_model.score(c_sentence_text)


# ランキング
def ranking(target, candidates, scores, sentence, word2vec, w2v_vocab, word2freq, freq_total, language_model, attached_words, mecab, bert, ranking_type):
	def context_sim(sentence, candidate):
		sentence_tk = [(token["surface"], token["pos"]) for token in sentence]
		similarity_list = [word2vec.similarity(candidate, context) for context, pos in sentence_tk if is_content(pos) and context != candidate and context in w2v_vocab]
		similarity = sum(similarity_list) / len(similarity_list) if similarity_list else 0
		return similarity

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
	if ranking_type in {'glavas', 'bert'}:                   ranktable.append( make_ranking([word2vec.similarity(target, c) for c in candidates]) )
	if ranking_type in {'glavas'}:                           ranktable.append( make_ranking([context_sim(sentence, c) for c in candidates]) )
	if ranking_type in {'glavas', 'bert'}:                   ranktable.append( make_ranking([word2freq[c] for c in candidates]) )
	if ranking_type in {'glavas', 'language-model', 'bert'}: ranktable.append( make_ranking([language_model_score(sentence, target, c, attached_words, language_model, mecab) for c in candidates]) )
	if ranking_type in {'bert'}:                             ranktable.append( make_ranking(scores['bert']) )
	if ranking_type == 'none':                               ranktable.append( [0 for c in candidates] )


	sum_rank = [sum(f) for f in zip(*ranktable)]
	candidatelist = sorting_considering_same_order(list(zip(sum_rank, candidates)), key=lambda x:x[0], reverse=False)
	return candidatelist


# 平易化されているか判断
def pick_simple(candidatelist, target, word2freq, freq_total, candidate_type, ranking_type):
	if len(candidatelist) == 0:
		return [[(0, target)]]
	if candidate_type == 'glavas' and ranking_type == 'glavas':
		best_candidate = candidatelist[0][0][1]
		if information_contents(word2freq, freq_total, best_candidate) < information_contents(word2freq, freq_total, target):
			return candidatelist
		else:
			return [[(0, target)]]
	return candidatelist



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


def evaluate(input_file, output, candidates_list, reference_file):
	in_f = open(input_file, "r")
	out_f = output
	ref_f = open(reference_file, "r")

	accuracy_list = []
	changed_list = []
	precision_list = []
	tochange_list = []

	c_potential_list, c_precision_list, c_recall_list = list(), list(), list()

	for in_line, out_line, ref_line, candidates in zip(in_f, out_f, ref_f, candidates_list):
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

		candidates = {left + c + right for c in candidates}
		ref_phrases = set(ref_phrases)

		c_potential_list.append(len(ref_phrases & candidates) > 0)
		c_precision_list.append(len(ref_phrases & candidates) / len(candidates) if len(candidates) != 0 else 0)
		c_recall_list.append(len(ref_phrases & candidates) / len(ref_phrases) if len(candidates) != 0 else 0)

	accuracy = sum(accuracy_list) / sum(tochange_list)
	precision = sum(precision_list) / sum(changed_list)
	changed = sum(changed_list) / len(changed_list)
	print("acc/prec/changed = {:>5.2f}\t{:>5.2f}\t{:>5.2f}".format(accuracy*100, precision*100, changed*100))

	c_potential = sum(c_potential_list) / len(c_potential_list)
	c_precision = sum(c_precision_list) / len(c_precision_list)
	c_recall = sum(c_recall_list) / len(c_recall_list)

	print("candidate potential/prec/recall = {:>5.2f}\t{:>5.2f}\t{:>5.2f}".format(c_potential*100, c_precision*100, c_recall*100))

	in_f.close()
	ref_f.close()


def parse():
	parser = argparse.ArgumentParser(description='')
	parser.add_argument('--candidate',          '-C', help='言い換え集合の取得手法。allはglavas, synonym, bertの3手法のうちいずれかで得られる言い換え集合。goldはリファレンスの言い換え集合をとる。', choices=['glavas', 'synonym', 'bert', 'all', 'gold'])
	parser.add_argument('--ranking',            '-R', help='ランキング手法', choices=['glavas', 'language-model', 'bert', 'none'])
	parser.add_argument('--output',             '-o', help='output file; default is "stdout"', default='stdout')
	parser.add_argument('--log',                '-g', help='log file; default is "stdout"', default='stdout')
	parser.add_argument('--data',               '-d', help='Kodaira\'s dataset directory, e.g. /path/to/EvaluationDataset')
	parser.add_argument('--embedding',          '-e', help='binary embedding file')
	parser.add_argument('--language-model',     '-m', help='kenlm language model path')
	parser.add_argument('--most-similar',       '-n', help='select n most sililar candidate words', default=10, type=int)
	parser.add_argument('--word-to-freq',       '-f', help='単語 \t 頻度 の tsv。頻度が複数列あっても1列目しか見ない')
	parser.add_argument('--synonym-dict',       '-p', help='--candidate=synonym の時に使う同義語辞書')
	parser.add_argument('--pretrained-bert',    '-b', help='学習済みの日本語BERTモデル　デフォルト：%(default)s', default='bert-base-japanese-whole-word-masking')
	parser.add_argument('--word-to-complexity', '-l', help='--simplicity=point-wise の時に使う単語難易度辞書')
	parser.add_argument('--cos-threshold',      '-c', help='言い換え集合を取得する際に、この値未満の単語は含まない', default=0.0, type=float)
	parser.add_argument('--device',             '-u', help='使用するGPU. -1で不使用. デフォルト：%(default)s', default=-1, type=int)
	# オプション
	parser.add_argument('--mecab-dict',         '-D', help='mecab dictionary path', default='')
	args = parser.parse_args()
	return args


if __name__ == '__main__':
	main()
