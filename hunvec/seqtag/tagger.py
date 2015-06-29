import sys
import argparse
from itertools import izip

from numpy import argsort, sort
from pylearn2.utils import serial

from hunvec.datasets.word_tagger_dataset import WordTaggerDataset
from hunvec.corpus.tagged_corpus import RawCorpus, TaggedCorpus

def create_argparser():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('model')
    argparser.add_argument('--input', dest='input_')
    argparser.add_argument('--output', dest='output')
    argparser.add_argument('--gold', action='store_true')
    return argparser.parse_args()

class Tagger:

    def __init__(self, args):
        
        self.wt = serial.load(args.model)
        self.i2t = [t for t, i in sorted(self.wt.t2i.items(), 
            key=lambda x: x[1])] 
        self.input_ = args.input_ 
        self.output = (open(args.output, 'w') if args.output is not None
                else sys.stdout)
        self.debug = False

    def tag(self):

        for sen_data in self.generate_sen_data():
            w, f, to_print = sen_data
            window_words, window_feats = WordTaggerDataset.process_sentence(
                    w, f, self.wt.window_size, self.wt.featurizer)
            result = self.wt.tag_sen(window_words, window_feats, debug=self.debug)
            self.write_result(sen_data, result)
        
    def generate_sen_data(self):
       
        rc = RawCorpus(self.input_, self.wt.featurizer, w2i=self.wt.w2i,
                use_unknown=True)
        for sen in rc.read():
            w, f, orig_words = [list(t) for t in zip(*sen)]
            yield [w, f, orig_words]
    
    def write_result(self, sen_data, result):
        for r in result:
            sen_data.append([list(i) for i in r])
        for item in izip(*(sen_data)):
            self.write_sen_result(item)
        self.output.write('\n')

    def write_sen_result(self, item):
        w, f, tp, res = item
        i = int(res[0])
        tag = self.i2t[i]
        self.output.write(u'{0}\t{1}\n'.format(tp, tag).encode('utf-8'))
        

class GoldLabeledTagger(Tagger):

    def generate_sen_data(self):
        
        tc = TaggedCorpus(self.input_, self.wt.featurizer, w2i=self.wt.w2i,
                t2i=self.wt.t2i, use_unknown=True)
        for sen in tc.read():
            w, orig_t, f, orig_words = [list(t) for t in zip(*sen)]
            to_print = map(lambda x: '{0}\t{1}'.format(x[0], self.i2t[x[1]]), 
                    izip(orig_words, orig_t))
            yield [w, f, to_print]


def main():
    args = create_argparser()
    if not args.gold:
        a = Tagger(args)
    else:
        a = GoldLabeledTagger(args)
    a.tag()


if __name__ == '__main__':
    main()
