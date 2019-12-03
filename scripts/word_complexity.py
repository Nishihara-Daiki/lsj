from collections import defaultdict
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import argparse
import math
from gensim.models import KeyedVectors

level_map = {
	'1.初級前半': 0,
	'2.初級後半': 0,
	'3.中級前半': 1,
	'4.中級後半': 1,
	'5.上級前半': 2,
	'6.上級後半': 2,
}

id2leveltext = {
	0: '初級',
	1: '中級',
	2: '上級'
}

def main():
	args = parse()
	word2pos, id2pos = load_pos(args.word_to_pos)
	char2freqs = load_freqs(args.char_to_freq, args.freq_scaling)
	word2freqs = load_freqs(args.word_to_freq, args.freq_scaling)
	word2vec = load_word2vec(args.embedding)

	train_data, dev_data, test_data = load_data(args.data, args.split_list)
	train_data = load_features(train_data, word2pos, char2freqs, word2freqs, word2vec)
	dev_data   = load_features(dev_data,   word2pos, char2freqs, word2freqs, word2vec)
	test_data  = load_features(test_data,  word2pos, char2freqs, word2freqs, word2vec)

	classifiers = train(train_data)
	parameters = dev(dev_data, classifiers)
	test(test_data, parameters)

	# 単語難易度辞書の作成
	if args.output:
		all_word = get_all_word(word2pos, char2freqs, word2freqs, word2vec)
		all_word_data = load_features(all_word, word2pos, char2freqs, word2freqs, word2vec)
		preds = test(all_word_data, parameters, without_scoring=True)
		write_word2complexity(args.output, all_word, preds)


def load_data(data_file, splitlist_file):
	train, dev, test = [], [], []
	with open(data_file, 'r') as fd, open(splitlist_file, 'r') as fs:
		for dline, sline in zip(fd, fs):
			_,word,_,level,_,_,_ = dline.strip().split(',')
			level = level_map[level]
			splittype, num = sline.strip().split('\t')
			num = int(num)
			if splittype == 'train':
				train.append((word, level, num))
			elif splittype == 'dev':
				dev.append((word, level, num))
			elif splittype == 'test':
				test.append((word, level, num))
	train = [(w, l) for w,l,n in sorted(train, key=lambda x: x[2])]
	dev   = [(w, l) for w,l,n in sorted(dev,   key=lambda x: x[2])]
	test  = [(w, l) for w,l,n in sorted(test,  key=lambda x: x[2])]
	return train, dev, test


def load_pos(word2pos_file):
	def onehot(hotid, num_of_poses):
		onehotvec = [0] * num_of_poses
		onehotvec[hotid] = 1
		return onehotvec
	word2pos, id2pos = None, None
	if word2pos_file:
		with open(word2pos_file, "r") as f:
			w2p = dict()
			for line in f:
				word, pos = line.strip().split('\t')
				w2p[word] = pos
		poses = sorted(set(w2p.values()))
		pos2id = {pos:i for i,pos in enumerate(poses)}
		id2pos = {i:pos for i,pos in enumerate(poses)}
		num_of_poses = len(poses)
		word2pos = defaultdict(None)
		for word,pos in w2p.items():
			word2pos[word] = onehot(pos2id[pos], num_of_poses)
	return word2pos, id2pos


def load_freqs(freq_file, scaletype):
	label2freqs = None
	if freq_file:
		# load file
		with open(freq_file, "r") as f:
			for i, line in enumerate(f):
				line = line.strip().split('\t')
				word, freqs = line[0], line[1:]
				if i == 0:
					label2freqs = [defaultdict(int) for _ in freqs]
				for j,freq in enumerate(freqs):
					label2freqs[j][word] = int(freq)
		# scaling
		for i in range(len(label2freqs)):
			if scaletype == 'log':
				for k in label2freqs[i]:
					label2freqs[i][k] = math.log(label2freqs[i][k])
			elif scaletype == 'normalize':
				total = sum(label2freq.values())
				for k in label2freqs[i]:
					label2freqs[i][k] = label2freqs[i][k] / total
	return label2freqs


def load_word2vec(word2vec_file):
	word2vec = None
	if word2vec_file:
		word2vec = KeyedVectors.load(word2vec_file)
	return word2vec


def get_charfreq(word, charfreq):
	freqlist = [charfreq[c] for c in word]
	return [min(freqlist), max(freqlist)]


def load_features(data, word2pos, char2freqs, word2freqs, word2vec):
	features, labels = list(), list()
	for word,label in data:
		feature = []
		# add word2pos feature
		if word2pos:
			feature += word2pos[word]
		# add char2freq features
		if char2freqs:
			for c2f in char2freqs:
				feature += get_charfreq(word, c2f)
		# add word2freq features
		if word2freqs:
			for w2f in word2freqs:
				feature += [w2f[word]]
		# add word2vec feature
		if word2vec:
			feature += list(word2vec[word])
		features.append(feature)
		labels.append(int(label))
	return features, labels



def train(train_data):
	features, labels = train_data
	classifiers = list()
	i = 1
	for c in [0.001, 0.01, 0.1, 1.0, 10.0]:
		for gamma in [0.001, 0.01, 0.1, 1.0, 10.0]:
			classifier = SVC(C=c, gamma=gamma, kernel="rbf", random_state=0)
			classifier.fit(features, labels)
			classifiers.append((c, gamma, classifier))
			print('[log] trained {}/25 model(s)'.format(i), flush=True)
			i += 1
	return classifiers


def dev(dev_data, classifiers):
	features, labels = dev_data
	scores = list()
	for c, gamma, classifier in classifiers:
		preds = classifier.predict(features)
		score = accuracy_score(labels, preds)
		scores.append((c, gamma, classifier, score))
	parameters = sorted(scores, key=lambda x: x[-1], reverse=True)[0]
	return parameters


def test(test_data, parameters, without_scoring=False):
	c, gamma, classifier, dev_maxf1 = parameters
	features, labels = test_data
	preds = classifier.predict(features)
	if not without_scoring:
		print("Dev-MaxAccuracy = %1.3f\tC = %s\tgamma = %s\tTest-Accuracy = %1.3f" % (dev_maxf1, str(c), str(gamma), accuracy_score(labels, preds)))
	return preds


def get_all_word(word2pos, char2freqs, word2freqs, word2vec):
	vocabs = set()
	for vs in [set(word2pos), *[set(a) for a in word2freqs], set(word2vec.vocab)]:
		if vs:
			if vocabs:
				vocabs &= vs
			else:
				vocabs = vs

	all_word = sorted([(vocab, -1) for vocab in vocabs]) # -1 is dummy
	return all_word


def write_word2complexity(outputfile, all_word, preds):
	with open(outputfile, 'w') as f:
		for word, pred in zip(all_word, preds):
			level = id2leveltext[pred]
			f.write('{}\t{}\n'.format(word[0], level))


def parse():
	parser = argparse.ArgumentParser(description='')
	parser.add_argument('--data',         '-d', help='jev.csv')
	parser.add_argument('--split-list',   '-u', help='jev.csvの単語を、train/dev/test/noneのどれに使うかが書かれたリストのファイル')
	parser.add_argument('--word-to-pos',  '-p', help='')
	parser.add_argument('--char-to-freq', '-c', help='')
	parser.add_argument('--word-to-freq', '-f', help='')
	parser.add_argument('--embedding',    '-e', help='embedding bin file')
	parser.add_argument('--freq-scaling', '-s', choices=['none', 'normalize', 'log'], default='log')
	parser.add_argument('--output',       '-o', help='単語難易度辞書の出力先')

	args = parser.parse_args()
	return args


if __name__ == '__main__':
	main()
