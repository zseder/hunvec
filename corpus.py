from collections import defaultdict

import numpy

from pylearn2.datasets.dense_design_matrix import DenseDesignMatrix


class Corpus(object):
    def __init__(self, sentences, vocab):
        self.corpus = sentences
        self.vocab = vocab

    @staticmethod
    def read_corpus(path):
        f = open(path)

        corpus = []
        vocab = {}
        for l in f:
            l = l.decode("utf-8")
            s = l.split()
            for w in s:
                if w not in vocab:
                    vocab[w] = len(vocab)
            s = [vocab[w] for w in s]
            corpus.append(s)
        return Corpus(corpus, vocab)

    def filter_freq(self, n=10000):
        freq = defaultdict(int)
        for s in self.corpus:
            for w in s:
                freq[w] += 1
        needed = sorted(freq.iteritems(), key=lambda x: x[1], reverse=True)[:n]
        needed = set(k for k, v in needed)
        words = [w for w, i in
                 sorted(self.vocab.iteritems(), key=lambda x: x[1])]
        vocab = {}
        for s in self.corpus:
            for i in xrange(len(s)):
                if s[i] not in needed:
                    s[i] = n
                else:
                    w_str = words[s[i]]
                    if w_str not in vocab:
                        vocab[w_str] = len(vocab)
                    s[i] = vocab[w_str]
        vocab[n] = "RARE"
        self.vocab = vocab

    def iterate_ngram_training(self, n=3):
        for s in self.corpus:
            for i in xrange(len(s) - n):
                ngr = s[i:i+n]
                y = s[i+n]
                yield ngr, y

    def get_matrices(self, n=3, ratios=[.7, .15, .15]):
        X = []
        y = []
        for ngr, w in self.iterate_ngram_training(n):
            X.append(ngr)
            y.append([w])
        X = numpy.array(X)
        y = numpy.array(y)
        total = len(y)
        training = round(total * ratios[0])
        valid = training + round(total * ratios[1])
        #test = total - training - valid
        labels = len(self.vocab)
        training_data = DenseDesignMatrix(X=X[:training, :], y=y[:training],
                                          X_labels=labels, y_labels=labels)
        valid_data = DenseDesignMatrix(X=X[training:valid, :],
                                       y=y[training:valid], X_labels=labels,
                                       y_labels=labels)
        test_data = DenseDesignMatrix(X=X[valid:, :], y=y[valid:],
                                      X_labels=labels, y_labels=labels)
        return training_data, valid_data, test_data
