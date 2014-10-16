from collections import defaultdict
from random import shuffle

import numpy

from pylearn2.datasets.dense_design_matrix import DenseDesignMatrix


class Corpus(object):
    def __init__(self, fn, batch_size=100000, window_size=3, top_n=10000):
        self.vocab = {}
        self.bs = batch_size
        self.ws = window_size
        self.top_n = 10000
        self.needed = self.get_freq_words(fn)
        self.f = open(fn)
        self.eof = False

    def get_freq_words(self, fn):
        v = {}
        for l in open(fn):
            l = l.decode("utf-8")
            s = l.split()
            for w in s:
                v[w] = v.get(w, 0) + 1
        needed = sorted(v.iteritems(), key=lambda x: -x[1])[:self.top_n]
        return set(w for w, _ in needed)

    def read_batch(self):
        if self.eof:
            return
        c = 0
        X, Y = [], []
        for l in self.f:
            l = l.decode("utf-8")
            s = l.split()
            for w in s:
                if w not in self.vocab:
                    if w in self.needed:
                        self.vocab[w] = len(self.vocab)
                    else:
                        self.vocab[w] = -1
            s = [self.vocab[w] for w in s]
            for ngr, y in self.sentence_ngrams(s):
                if y not in self.needed:
                    continue
                if len(set(ngr)) < self.ws:
                    continue
                X.append(ngr)
                Y.append(y)
                c += 1
            if c >= self.bs:
                break
        if len(c) < self.bs:
            # end of file
            self.eof = True

        return X, Y

    def sentence_ngrams(self, s):
        n = self.ws
        for i in xrange(len(s) - n):
            ngr = s[i:i+n]
            y = s[i+n]
            yield ngr, y

    def create_batch_matrices(self, ratios=[.7, .15, .15]):
        res = self.read_batch()
        if res is None:
            return None
        X, y = res
        num_labels = len(self.needed) + 1  # for filtered words
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
