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
    tc = RawCorpus(args.input_, wt.featurizer, w2i=wt.w2i, t2i=wt.t2i)
    print tc
    output = (open(args.output, 'w') if args.output is not None
              else sys.stdout)
    i2t = [t for t, i in sorted(wt.t2i.items(), key=lambda x: x[1])]
    for sen in tc.read():
        
        words, feats, orig_words = [list(t) for t in zip(*sen)]
        print words, feats, orig_words
        tags, tagger_out = wt.tag_sen(words, feats, args.debug)
        for w, t, to in izip(orig_words, tags, tagger_out):
            t = i2t[t]
            output.write(u'{0}\t{1}\n'.format(w, t).encode('utf-8'))
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
