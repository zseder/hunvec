import sys
import argparse
from itertools import izip

from numpy import argsort, sort
from pylearn2.utils import serial

from hunvec.datasets.word_tagger_dataset import WordTaggerDataset
from hunvec.corpus.tagged_corpus import RawCorpus

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

def tag(args):
    wt = serial.load(args.model)
    # read input from stdin sentence by sentence
    rc = RawCorpus(args.input_, wt.featurizer, w2i=wt.w2i, t2i=wt.t2i, 
            use_unknown=True)
    output = (open(args.output, 'w') if args.output is not None
              else sys.stdout)
    i2t = [t for t, i in sorted(wt.t2i.items(), key=lambda x: x[1])]
    for sen in rc.read():
        w, f, orig_words = [list(t) for t in zip(*sen)]
        window_words, window_feats = WordTaggerDataset.process_sentence(
                w, f, wt.window_size, wt.featurizer)
        if args.debug:
            tags, tagger_out = wt.tag_sen(window_words, window_feats, args.debug)
        else:
            tags = wt.tag_sen(window_words, window_feats, args.debug)
        for w, t, to in izip(orig_words, tags, tagger_out):
            t = i2t[t]
            output.write('{0}\t{1}\n'.format(w, t))
            if args.debug:
                tags = [i2t[i] for i in argsort(-to)][:5]
                probs = sorted(to, reverse = True)[:5]
                
                output.write('\t'.join([' '.join([tag.lower(), 
                    str(round(prob, 4))]) 
                    for tag, prob in zip(tags, probs)]))
                output.write('\n')
        output.write('\n')


def main():
    args = create_argparser()
    tag(args)


if __name__ == '__main__':
    main()
