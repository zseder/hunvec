import logging
from random import shuffle

import numpy

from pylearn2.datasets.dense_design_matrix import DenseDesignMatrix

from hunvec.utils.binary_tree import BinaryTreeEncoder


class Corpus(object):
    def __init__(self, fn, batch_size=100000, window_size=3, top_n=10000,
                 hs=False, max_corpus_epoch=2, future=False):
        self.bs = batch_size
        self.ws = window_size
        self.top_n = top_n
        self.compute_needed_words(fn)
        self.hs = hs
        self.future = future
        if hs:
            self.w_enc = BinaryTreeEncoder(self.needed).word_encoder
        self.fn = fn
        self.f = open(fn)
        self.max_corpus_epoch = max_corpus_epoch
        self.epoch_count = 0
        self.skip_str = "__FILTERED__"
        logging.info("Corpus initialized")

    def compute_needed_words(self, fn):
        v = {}
        for l in open(fn):
            l = l.decode("utf-8")
            s = l.split()
            for w in s:
                v[w] = v.get(w, 0) + 1
        sorted_v = sorted(v.iteritems(), key=lambda x: -x[1])
        needed = sorted_v[:self.top_n]
        self.index2word = dict((i, w) for i, (w, f) in enumerate(needed))
        self.index2word[-1] = '<unk>'
        self.vocab = dict((k, i) for i, (k, _) in enumerate(needed))
        needed = dict((self.vocab[w], f) for w, f in needed)
        needed[-1] = sum(v for _, v in sorted_v[self.top_n:])
        self.needed = needed

    def read_batch(self):
        if self.epoch_count == self.max_corpus_epoch:
            return
        c = 0
        X = []
        if self.hs:
            Y = numpy.zeros((self.bs, len(self.vocab)), dtype=numpy.int8)
        else:
            Y = []
        for l in self.f:
            l = l.decode("utf-8")
            s = l.split()
            s = [(self.vocab[w] if w in self.vocab else -1) for w in s]
            for ngr, y in self.sentence_to_examples(s):
                if y == -1:
                    continue
                X.append(ngr)
                if self.hs:
                    y = self.w_enc(y)
                    Y[c] = y
                else:
                    Y.append([y])
                c += 1
                if c >= self.bs:
                    break
            if c >= self.bs:
                logging.info("Batch read.")
                break
        if c < self.bs:
            self.epoch_count += 1
            logging.info("epoch #{}.finished".format(self.epoch_count))
            if self.epoch_count < self.max_corpus_epoch:
                self.f = open(self.fn)

        logging.info("Batch data ready.")
        return X[:c], Y[:c]

    def sentence_to_examples(self, s):
        n = self.ws
        end = (len(s) - n if not self.future else
               len(s) - 2 * n)
        for i in xrange(end):
            if self.future:
                context = s[i:i+n] + s[i+n+1:i+n+1+n]
            else:
                context = s[i:i+n]
            y = s[i+n]
            yield context, y

    def create_batch_matrices(self, ratios=[.7, .15, .15]):
        res = self.read_batch()
        if res is None:
            return None
        X, y = res
        x_labels = len(self.needed)  # for filtered words
        y_labels = len(self.needed)  # for filtered words
        if self.hs:
            y_labels = None
        else:
            # in hierarchical softmax, we automatically create arrays
            y = numpy.array(y)
        X = numpy.array(X)
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
                                          X_labels=x_labels,
                                          y_labels=y_labels)
        valid_data = DenseDesignMatrix(X=X[valid_indices, :],
                                       y=y[valid_indices],
                                       X_labels=x_labels,
                                       y_labels=y_labels)
        test_data = DenseDesignMatrix(X=X[valid:, :], y=y[valid:],
                                      X_labels=x_labels,
                                      y_labels=y_labels)
        return training_data, valid_data, test_data
