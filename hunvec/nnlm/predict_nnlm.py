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


def corpus_test(model, corpus):
    good, bad = 0, 0
    c = 0
    X = model.get_input_space().make_theano_batch()
    Y = model.fprop(X)
    f = theano.function([X], Y)
    v = set(corpus.vocab.itervalues())
    v = [corpus.w_enc(w) for w in v]
    while True:
        dataset = corpus.create_batch_matrices()
        if dataset is None:
            break
        test = dataset[2]
        Y = f(test.X)

        targets = test.get_targets()
        outputs = targets[c*1000:(c+1)*1000]
        for i in xrange(outputs.shape[0]):
            tgt = outputs[i]
            needed = tgt >= 0
            y = Y[c*1000 + i]
            #distances = numpy.array([cosine(y[needed], w[needed]) for w in v])
            probs = numpy.array([y[w == 0].prod() * (1.0 - y[w == 1]).prod()
                                 for w in v])
            closests = set([tuple(v[w][needed])
                            for w in probs.argsort()[-5:]])
            if tuple(tgt[needed]) in closests:
                good += 1
            else:
                bad += 1
            print good, bad

        c += 1

        print "Batch done", good, bad
    print good, bad


def main():
    m = serial.load(sys.argv[1])
    c = Corpus(sys.argv[2], batch_size=1000, window_size=5, hs=True,
               max_corpus_epoch=1)
    corpus_test(m, c)

if __name__ == "__main__":
    main()
