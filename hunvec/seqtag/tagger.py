import sys
import argparse

import numpy

from pylearn2.utils import serial


def create_argparser():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('model')
    argparser.add_argument('--input', dest='input_')
    return argparser.parse_args()


def get_sen(f):
    sen = []
    for l in f:
        l = l.strip().decode('utf-8')
        if len(l) > 0:
            sen.append(l)
        else:
            yield sen
            sen = []


def tag(args):
    wt = serial.load(args.model)
    # read input from stdin sentence by sentence
    input_ = (open(args.input_) if args.input_ is not None else sys.stdin)
    for sen in get_sen(input_):
        # TODO data should be converted to arrays with dataset somehow
        features = numpy.array([wt.featurizer.featurize(w) for w in sen])
        words = numpy.array([(wt.w2i[w] if w in wt.w2i else -1) for w in sen])
        tagged = wt.tag_seq(words, features)
        print sen, list(tagged)
        quit()
    # call tagging
    # printout result


def main():
    args = create_argparser()
    tag(args)


if __name__ == '__main__':
    main()
