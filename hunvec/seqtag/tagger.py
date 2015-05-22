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
    i2w = {}
    if args.debug:
        vectors = wt.tagger.layers[0].layers[0].get_params()[0].get_value()
        i2w = dict([(v, k) for k, v in wt.w2i.iteritems()])
        #range_ = 
    # read input from stdin sentence by sentence
    rc = RawCorpus(args.input_, wt.featurizer, w2i=wt.w2i,
            use_unknown=True)
    output = (open(args.output, 'w') if args.output is not None
              else sys.stdout)
    i2t = [t for t, i in sorted(wt.t2i.items(), key=lambda x: x[1])]
    for sen in rc.read():
        w, f, orig_words = [list(t) for t in zip(*sen)]
        window_words, window_feats = WordTaggerDataset.process_sentence(
                w, f, wt.window_size, wt.featurizer)
        result = wt.tag_sen(window_words, window_feats, args.debug)
        if args.debug:
            tags, tagger_out, close_wds, f_closests = result
            print close_wds
            for w, t, to, cl, f_cl in zip(orig_words, tags, tagger_out, close_wds, f_closests):
                t = i2t[t]
                output.write(u'{0}\t{1}\n'.format(w, t).encode('utf-8'))
                tags = [i2t[i] for i in argsort(-to)][:5]
                probs = sorted(to, reverse = True)[:5]
                output.write(u'\t'.join([' '.join([tag.lower(),
                    str(round(prob, 4))]) 
                    for tag, prob in zip(tags, probs)]).encode('utf-8'))
                output.write('\n')
                close = [i2w[i] for i in list(cl)]
                print close
                quit()
                output.write(u'\t'.join(close).encode('utf-8'))
                output.write('\n')
                string_f_cl = {}
                for k in f_cl:
                    shift = k/wt.featurizer.total 
                    if shift != wt.window_size:
                        continue
                    index = k % wt.featurizer.total
                    string = wt.featurizer.i2f[index]
                    string_f_cl[string] = []
                    for i in f_cl[k][:3]:
                        shift2 = i/wt.featurizer.total
                        if shift != wt.window_size:
                            continue
                        index2 = i % wt.featurizer.total
                        string2 = wt.featurizer.i2f[index2]
                        string_f_cl[string].append(string2)
                output.write(repr(string_f_cl))        
                output.write('\n')
        else:
            tags = result
            for w, t in izip(orig_words, tags):
                t = i2t[t]
                output.write('{0}\t{1}\n'.format(w, t))
        output.write('\n')


def main():
    args = create_argparser()
    tag(args)


if __name__ == '__main__':
    main()
