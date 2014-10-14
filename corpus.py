from collections import defaultdict
from random import shuffle

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

    def filter_freq(self, lower_n=0, n=10000):
        freq = defaultdict(int)
        for s in self.corpus:
            for w in s:
                freq[w] += 1
        needed = sorted(freq.iteritems(), key=lambda x: x[1], reverse=True)
        needed = needed[lower_n:lower_n+n]
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
        num_labels = len(self.vocab)
        rare = num_labels - 1
        for ngr, w in self.iterate_ngram_training(n):
            # skip rare words in labels
            if w == rare:
                continue
            # skip ngrams with multiple (rare) words
            if len(set(ngr)) < len(ngr):
                continue
            X.append(ngr)
            y.append([w])
        X = numpy.array(X)
        y = numpy.array(y)
        total = len(y)
        indices = range(total)
        shuffle(indices)
        training = int(round(total * ratios[0]))
        valid = int(round(total * ratios[1]))
        training_indices = indices[:training]
        valid_indices = indices[training:training + valid]
        #test = total - training - valid
        training_data = DenseDesignMatrix(X=X[training_indices, :],
                                          y=y[training_indices],
                                          X_labels=num_labels,
                                          y_labels=num_labels)
        valid_data = DenseDesignMatrix(X=X[valid_indices, :],
                                       y=y[valid_indices],
                                       X_labels=num_labels,
                                       y_labels=num_labels)
        test_data = DenseDesignMatrix(X=X[valid:, :], y=y[valid:],
                                      X_labels=num_labels,
                                      y_labels=num_labels)
        return training_data, valid_data, test_data
