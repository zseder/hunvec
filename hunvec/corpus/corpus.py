import logging
from random import shuffle

import numpy

from pylearn2.datasets.dense_design_matrix import DenseDesignMatrix

from hunvec.utils.binary_tree import create_binary_tree, vector_encoder


class Corpus(object):
    def __init__(self, fn, batch_size=100000, window_size=3, top_n=10000,
                 hs=False):
        self.bs = batch_size
        self.ws = window_size
        self.top_n = top_n
        self.compute_needed_words(fn)
        self.hs = hs
        if hs:
            self.ht = create_binary_tree(self.needed)
            self.v_enc = vector_encoder(self.ht)
        self.f = open(fn)
        self.eof = False
        self.skip_str = "__FILTERED__"

    def compute_needed_words(self, fn):
        v = {}
        for l in open(fn):
            l = l.decode("utf-8")
            s = l.split()
            for w in s:
                v[w] = v.get(w, 0) + 1
        sorted_v = sorted(v.iteritems(), key=lambda x: -x[1])
        needed = sorted_v[:self.top_n]
        self.vocab = dict((k, i) for i, (k, _) in enumerate(needed))
        needed = dict((self.vocab[w], f) for w, f in needed)
        needed[-1] = sum(v for _, v in sorted_v[self.top_n:])
        self.needed = needed

    def read_batch(self):
        if self.eof:
            return
        c = 0
        X, Y = [], []
        for l in self.f:
            l = l.decode("utf-8")
            s = l.split()
            s = [(self.vocab[w] if w in self.vocab else -1) for w in s]
            for ngr, y in self.sentence_ngrams(s):
                if y == -1:
                    continue
                if len(set(ngr)) < self.ws:
                    continue
                X.append(ngr)
                if self.hs:
                    y = self.ht[y]
                else:
                    y = [y]
                Y.append(y)
                c += 1
                print X, y
                quit()
            if c >= self.bs:
                break
        if c < self.bs:
            # end of file
            self.eof = True
            logging.info("End of file")

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
