import sys

import theano
import numpy

from pylearn2.utils import serial

from corpus import Corpus


def manual_test():
    inputs = numpy.array([[i, j] for i in xrange(500) for j in xrange(500)])

    model_path = sys.argv[1]
    model = serial.load(model_path)
    results = model.fprop(theano.shared(inputs, name='inputs')).eval()
    for i in xrange(len(results)):
        input_ = inputs[i, :]
        res = results[i]
        #print input_
        sorted_indices = res.argsort()[-5:]
        if sorted_indices.tolist() != [11, 23, 8, 10, 3]:
            print "{0}\t{1}".format(input_, sorted_indices)
        #print sorted_indices


def corpus_test(model, corpus):
    tr, v, test = corpus.get_matrices(3)
    good, bad = 0, 0
    targets = test.get_targets()
    b = 1000
    c = 0
    for inputs in test.iterator(mode='sequential', batch_size=b):
        outputs = targets[c*1000:(c+1)*1000]

        results = model.fprop(theano.shared(inputs, name='inputs')).eval()
        for i in xrange(len(results)):
            res = results[i]
            sorted_best_5 = res.argsort()[-5:]
            gold = outputs[i]
            if gold in sorted_best_5:
                good += 1
            else:
                bad += 1
        print "Batch done", good, bad
    print good, bad


def main():
    m = serial.load(sys.argv[1])
    c = Corpus.read_corpus(sys.argv[2])
    c.filter_freq(n=5000)
    corpus_test(m, c)

if __name__ == "__main__":
    main()
