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
    argparser.add_argument('--debug', action='store_true')
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


    def tag(self):
        #if args.debug:
        #    vectors = wt.tagger.layers[0].layers[0].get_params()[0].get_value()
        #    i2w = [w for w, i in sorted(wt.w2i.iteritems(), key=lambda x: x[1])]
        #    i2w.append("UNK")

        # read input from stdin sentence by sentence
        for sen_data in self.generate_sen_data():
            w, f, to_print = sen_data
            window_words, window_feats = WordTaggerDataset.process_sentence(
                    w, f, self.wt.window_size, self.wt.featurizer)
            result = self.wt.tag_sen(window_words, window_feats)
            self.write_result(sen_data, [result])
        
    def generate_sen_data(self):
       
        rc = RawCorpus(self.input_, self.wt.featurizer, w2i=self.wt.w2i,
                use_unknown=True)
        for sen in rc.read():
            w, f, orig_words = [list(t) for t in zip(*sen)]
            yield [w, f, orig_words]
    
    def write_result(self, sen_data, result):
        for r in result:
            sen_data.append(r)
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

#        i2t = [t for t, i in sorted(wt.t2i.items(), key=lambda x: x[1])]
#        for sen in rc.read():
#            w, f, orig_words = [list(t) for t in zip(*sen)]
#            window_words, window_feats = WordTaggerDataset.process_sentence(
#                    w, f, wt.window_size, wt.featurizer)
#            result = wt.tag_sen(window_words, window_feats, args.debug)
#
#            if args.debug:
#                tags, tagger_out, close_wds, f_closests = result
#                for w, t, to, cl, f_cl in zip(
#                        orig_words, tags, tagger_out, close_wds, f_closests):
#                    t = i2t[t]
#                    output.write(u'{0}\t{1}\n'.format(w, t).encode('utf-8'))
#                    for string in debug_data(wt, to, cl, f_cl, i2t, i2w):
#                        output.write(string.encode('utf-8'))
#                    output.write('\n')
#
#            else:
#                tags = result
#                for w, t in izip(orig_words, tags):
#                    t = i2t[t]
#                    output.write(u'{0}\t{1}\n'.format(w, t).encode('utf-8'))
#                output.write('\n')    

def debug_data(wt, to, cl, f_cl, i2t, i2w):
    # tag probabilities based only classifier(no transition scores)
    tags = [i2t[i] for i in argsort(-to)][:5]
    probs = sorted(to, reverse = True)[:5]
    yield u'{0}\n'.format(u'\t'.join([' '.join([tag.lower(), str(round(prob, 4))]) 
                    for tag, prob in zip(tags, probs)]))
    # close words in vector space
    close = [i2w[i] for i in list(cl)]
    yield u'{0}\n'.format(u'\t'.join(close))
    
    # close feats in vectors space
    string_f_cl = turn_feat_dict2string(wt, f_cl)
    yield u'{0}\n'.format(repr(string_f_cl))


def turn_feat_dict2string(wt, f_cl):
    string_f_cl = {}
    for k in f_cl:
        
        shift = k/wt.featurizer.total 
        #print k, shift, wt.featurizer.total
        if shift != wt.window_size:
            continue
        index = k % wt.featurizer.total
        
        string = wt.featurizer.i2f[index]
        string_f_cl[string] = []
        for i in f_cl[k][:10]:
            shift2 = i/wt.featurizer.total
            if shift2 != wt.window_size:
                continue
            index2 = i % wt.featurizer.total
            string2 = wt.featurizer.i2f[index2]
            string_f_cl[string].append(string2)
    return string_f_cl        

def main():
    args = create_argparser()
    if not args.gold:
        a = Tagger(args)
    else:
        a = GoldLabeledTagger(args)
    a.tag()


if __name__ == '__main__':
    main()
