import sys
import argparse
from itertools import izip

from numpy import argsort, sort
from pylearn2.utils import serial

from hunvec.datasets.word_tagger_dataset import WordTaggerDataset


def create_argparser():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('model')
    argparser.add_argument('--input', dest='input_')
    argparser.add_argument('--output', dest='output')
    argparser.add_argument('--debug', action='store_true')
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
        iwords = [wt.w2i.get(w.lower(), -1) for w in words_]
        words, features = WordTaggerDataset.process_sentence(
            iwords, feats_, wt.window_size, wt.featurizer)
        yield words, features, words_


def tag(args):
    wt = serial.load(args.model)
    i2w = {}
    if args.debug:
        vectors = wt.tagger.layers[0].layers[0].get_params()[0].get_value()
        i2w = dict([(v, k) for k, v in wt.w2i.iteritems()])
    # read input from stdin sentence by sentence
    input_ = (open(args.input_) if args.input_ is not None else sys.stdin)
    output = (open(args.output, 'w') if args.output is not None
              else sys.stdout)
    sens = process_sentences(wt, get_sens(input_))
    i2t = [t for t, i in sorted(wt.t2i.items(), key=lambda x: x[1])]
    for words, feats, orig_words in sens:
        result = wt.tag_sen(words, feats, args.debug)
        
        if not args.debug:
            tags = result
            for w, t in izip(orig_words, tags):
                t = i2t[t]
                output.write(u'{0}\t{1}\n'.format(w, t).encode('utf-8'))
        else:
            tags, tagger_out, close_wds = result
            for w, t, to, cl in izip(orig_words, tags, tagger_out, close_wds):
                t = i2t[t]
                output.write(u'{0}\t{1}\n'.format(w, t).encode('utf-8'))
                tags = [i2t[i] for i in argsort(-to)][:5]
                probs = sorted(to, reverse = True)[:5]
                output.write('\t'.join([' '.join([tag.lower(), 
                    str(round(prob, 4))]) 
                    for tag, prob in zip(tags, probs)]))
                output.write('\n')
                close = [i2w[i] for i in cl]
                output.write('\t'.join(close))
                output.write('\n')
        output.write('\n')


def main():
    args = create_argparser()
    tag(args)


if __name__ == '__main__':
    main()
