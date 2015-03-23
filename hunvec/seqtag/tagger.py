import sys
import argparse
from itertools import izip

from pylearn2.utils import serial

from hunvec.datasets.word_tagger_dataset import WordTaggerDataset


def create_argparser():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('model')
    argparser.add_argument('--input', dest='input_')
    argparser.add_argument('--output', dest='output')
    return argparser.parse_args()


def get_sens(f):
    sen = []
    for l in f:
        l = l.strip().decode('utf-8')
        if len(l) > 0:
            sen.append(l)
        else:
            yield sen
            sen = []


def process_sentences(wt, sentences):
    for sen in sentences:
        sen = [(w, wt.featurizer.featurize(w)) for w in sen]
        words_, feats_ = [list(l) for l in zip(*sen)]
        iwords = [wt.w2i.get(w, -1) for w in words_]
        words, features = WordTaggerDataset.process_sentence(
            iwords, feats_, wt.window_size, wt.featurizer)
        yield words, features, words_


def tag(args):
    wt = serial.load(args.model)
    # read input from stdin sentence by sentence
    input_ = (open(args.input_) if args.input_ is not None else sys.stdin)
    output = (open(args.output, 'w') if args.output is not None
              else sys.stdout)
    sens = process_sentences(wt, get_sens(input_))
    i2t = [t for t, i in sorted(wt.t2i.items(), key=lambda x: x[1])]
    for words, feats, orig_words in sens:
        tags = wt.tag_sen(words, feats)
        for w, t in izip(orig_words, tags):
            t = i2t[t]
            output.write(u'{0}\t{1}\n'.format(w, t).encode('utf-8'))
        output.write('\n')


def main():
    args = create_argparser()
    tag(args)


if __name__ == '__main__':
    main()
