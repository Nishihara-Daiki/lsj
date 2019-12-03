import argparse
import pickle
import random
from itertools import combinations
from word_complexity import load_data, load_pos, load_freqs, load_word2vec, get_charfreq, train, dev, test
from sklearn.metrics import accuracy_score

lv_map = {
	'初級': 0,
	'中級': 1,
	'上級': 2
}


def main():
	random.seed(0)
	args = parse()

	word2vec = load_word2vec(args.embedding)
	word2level = load_word2level(args.word_to_complexity)
	synonym_dict = load_synonym(args.synonym_dict, word2level)

	if args.data:
		pair_data = load_pair_data(args.data, args.split_list, args.data_size)

	if args.type == 'point-wise':
		rst = pointwise(pair_data, synonym_dict, word2vec, word2level)

	if args.type == 'pair-wise':
		word2pos, id2pos = load_pos(args.word_to_pos)
		char2freqs = load_freqs(args.char_to_freq, args.freq_scaling)
		word2freqs = load_freqs(args.word_to_freq, args.freq_scaling)

		rst, parameters = pairwise(pair_data, synonym_dict, word2pos, char2freqs, word2freqs, word2vec, args.without)
		if args.model:
			save_pkl(args.model, parameters)

	if args.output:
		with open(args.output, 'w') as f:
			for line in rst:
				f.write('\t'.join(line) + '\n')


def pointwise(pair_data, synonym_dict, word2vec, word2level):
	if pair_data:
		_, dev_data, test_data = pair_data
		preds, labels = list(), list()
		for i, (word1, word2, label) in enumerate(test_data):
			lv1, lv2 = word2level[word1], word2level[word2]
			pred = 0 if lv1 > lv2 else 1 if lv1 == lv2 else 2
			preds.append(pred)
			labels.append(int(label))
		score = accuracy_score(labels, preds)
		print("Test-Accuracy = {:1.3f}".format(score))

	rst = list()
	for i, (word1, word2, prob) in enumerate(synonym_dict):
		lv1, lv2 = word2level[word1], word2level[word2]
		if lv1 > lv2:
			sim = word2vec.similarity(word1, word2)
			rst.append([word1, word2, str(prob), str(sim), str(lv1), str(lv2)])
	return rst


def pairwise(pair_data, synonym_dict, word2pos, char2freqs, word2freqs, word2vec, without):
	train_data, dev_data, test_data = pair_data
	train_data = load_features(train_data, word2pos, char2freqs, word2freqs, word2vec, without)
	dev_data   = load_features(dev_data,   word2pos, char2freqs, word2freqs, word2vec, without)
	test_data  = load_features(test_data,  word2pos, char2freqs, word2freqs, word2vec, without)

	print("feature-length = {}".format(len(train_data[0][0])), flush=True)

	classifiers = train(train_data)
	parameters = dev(dev_data, classifiers)
	test(test_data, parameters)

	synonym_data = [(a,b,0) for a,b,_ in synonym_dict]
	synonym_data = load_features(synonym_data, word2pos, char2freqs, word2freqs, word2vec, without)
	preds = test(synonym_data, parameters, without_scoring=True)
	rst = list()
	for (word1, word2, prob), pred in zip(synonym_dict, preds):
		if pred == 0:
			if word2vec == None:
				sim = 0
			else:
				sim = word2vec.similarity(word1, word2) if word1 in word2vec.vocab and word2 in word2vec.vocab else 0.0
			rst.append([word1, word2, str(prob), str(sim)])

	return rst, parameters


# ラベル情報 単語ペア(A, B)に対し
# 0: Aが難しい
# 1: 同じ
# 2: Bが難しい
def load_pair_data(data, split_list, data_size):
	if not data:
		return None
	train_data, dev_data, test_data = load_data(data, split_list)

	data = [train_data, dev_data, test_data]
	datasizes = [data_size[0], data_size[1], data_size[2]]
	rst = list()

	for wordlist, size in zip(data, datasizes):
		# random.seed(0)
		diff, same = [], []
		for v in combinations(wordlist, 2):
			leveldiff = int(v[0][1]) - int(v[1][1])
			if leveldiff > 0:
				diff.append((v[0][0], v[1][0], 0))
			elif leveldiff == 0:
				same.append((v[0][0], v[1][0], 1))
			elif leveldiff < 0:
				diff.append((v[1][0], v[0][0], 0))

		mid_idx = size // 3
		diff = random.sample(diff, mid_idx * 2)
		diff = diff[:mid_idx] + [(b,a,2) for a,b,_ in diff[mid_idx:mid_idx*2]]
		same = random.sample(same, size - mid_idx * 2)
		wordlist = diff + same
		random.shuffle(wordlist)
		rst.append(wordlist)

	return rst


def load_features(data, word2pos, char2freqs, word2freqs, word2vec, without):
	features, labels = list(), list()
	for w1, w2, label in data:
		feature = []
		# add word2pos feature
		if word2pos and 'pos' not in without:
			feature += word2pos[w1] + word2pos[w2]
		# add char2freq features
		if char2freqs and 'charfreq' not in without:
			for c2f in char2freqs:
				feature += get_charfreq(w1, c2f) + get_charfreq(w2, c2f)
		# add word2freq features
		if word2freqs:
			if 'wordfreq-single' not in without:
				for w2f in word2freqs:
					feature += [w2f[w1]]
				for w2f in word2freqs:
					feature += [w2f[w2]]
			if 'wordfreq-comb' not in without:
				for w2f in word2freqs:
					feature += [w2f[w1] - w2f[w2]]
		# add word2vec feature
		if word2vec:
			if 'embedding-single' not in without:
				feature += list(word2vec[w1]) + list(word2vec[w2])
			if 'embedding-comb' not in  without:
				feature += list(word2vec[w1] - word2vec[w2])
		features.append(feature)
		labels.append(int(label))
	return features, labels



def load_synonym(synonymdict_file, word2level):
	data = list()
	with open(synonymdict_file) as f:
		for line in f:
			if len(line.rstrip().split('\t')) != 3:
				print(line)
				exit()
			word1, word2, prob = line.rstrip().split('\t')
			if word1 not in word2level or word2 not in word2level:
				continue
			data.append((word1, word2, prob))
	return data



def load_word2level(word2level_file):
	word2level = {}
	if word2level_file:
		with open(word2level_file, 'r') as f:
			for line in f:
				word, level = line.strip().split('\t')
				word2level[word] = int(lv_map[level])
	return word2level


def save_pkl(model_file, model):
	with open(model_file, 'wb') as f:
		pickle.dump(model, f)


def parse():
	parser = argparse.ArgumentParser(description='')

	###### Common #################
	# input
	parser.add_argument('--data',         '-d', help='jev.csv;　type=pair-wiseの時は必須')
	parser.add_argument('--split-list',   '-u', help='jev.csvの単語を、train/dev/test/noneのどれに使うかが書かれたリストのファイル')
	parser.add_argument('--embedding',    '-e', help='embedding bin file')
	parser.add_argument('--synonym-dict', '-r', help='同義語辞書')
	parser.add_argument('--word-to-complexity','-l', help='単語難易度辞書')
	# setting
	parser.add_argument('--type',         '-t', help='', choices=['point-wise', 'pair-wise'])
	parser.add_argument('--data-size',    '-n', help='the number of train, dev and test. default=%(default)s', nargs=3, type=int, default=[10000, 1000, 1000])
	parser.add_argument('--without',            help='無視する属性（複数可）: pos, charfreq, wordfreq-single, wordfreq-comb, embedding-single, embedding-comb', nargs='*', default=[])
	# output
	parser.add_argument('--output',       '-o', help='平易化に限定した同義語辞書')
	parser.add_argument('--model',        '-m', help='モデル保存先')

	###### pair-wise ###########
	# input
	parser.add_argument('--word-to-pos',  '-p', help='')
	parser.add_argument('--char-to-freq', '-c', help='')
	parser.add_argument('--word-to-freq', '-f', help='')
	# setting
	parser.add_argument('--freq-scaling', '-s', choices=['none', 'normalize', 'log'], default='log')

	args = parser.parse_args()
	return args


if __name__ == '__main__':
	main()


