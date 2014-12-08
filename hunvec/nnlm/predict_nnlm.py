import sys

import theano
import numpy

from pylearn2.utils import serial

from hunvec.corpus.corpus import Corpus


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


def corpus_test(model, corpus, hs=False):
    good, bad = 0, 0
    X = model.get_input_space().make_theano_batch()
    Y = model.fprop(X)
    f = theano.function([X], Y)
    if hs:
        v = set(corpus.vocab.itervalues())
        v = [corpus.w_enc(w) for w in v]
        d = {}
        for w, i in corpus.vocab.iteritems():
            code = corpus.w_enc(i)
            d[tuple(code)] = w

    dataset, v_size = corpus.read_dataset()
    test = dataset['test']

    while True:
        X, targets = test.get_batch_design(batch_size=1000,
                                           include_labels=True)
        Y = f(X)

        #targets = batch.get_targets()
        for i in xrange(targets.shape[0]):
            tgt = targets[i]
            needed = tgt >= 0
            y = Y[i]
            if i == 100:
                quit()
            if hs:
                probs = numpy.array([numpy.log(y[w == 0]).sum()
                                     + numpy.log(1.0 - y[w == 1]).sum()
                                     for w in v])
                closests = set([tuple(v[w][needed])
                                for w in probs.argsort()[-5:]])
                for w in probs.argsort()[-5:]:
                    ww = tuple(v[w])
                    print d[ww]
                if tuple(tgt[needed]) in closests:
                    good += 1
                else:
                    bad += 1
            else:
                closests = y.argsort()[-5:]
                print tgt
                print closests

            print good, bad

    print "Batch done", good, bad
    print good, bad


def main():
    m = serial.load(sys.argv[1])
    c = Corpus(dump_path=sys.argv[2])
    corpus_test(m, c)

if __name__ == "__main__":
    main()
