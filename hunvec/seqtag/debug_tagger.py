from itertools import izip

import numpy
from numpy import argsort
from scipy.spatial.distance import cdist

from hunvec.seqtag.tagger import Tagger
from hunvec.corpus.tagged_corpus import TaggedCorpus
from hunvec.datasets.word_tagger_dataset import WordTaggerDataset
from hunvec.seqtag.tagger import create_argparser

class DebugTagger(Tagger):
    
    def __init__(self, args):
        Tagger.__init__(self, args)
        self.close_cache = {}
        self.close_feat_cache = {}
        self.i2w = [w for w, i in sorted(self.wt.w2i.iteritems(),
            key=lambda x: x[1])]
        self.i2w.append("PADDING")
        self.i2w.append("UNKNOWN")
        self.word_vectors = self.wt.tagger.layers[0].layers[0]\
                .get_params()[0].get_value()
        self.feat_vectors = self.wt.tagger.layers[0].layers[1]\
                .get_params()[0].get_value()
        self.shifted_i2f = {}        
    
    def tag_sen(self, sen_data):
        w, f = sen_data[:2]
        window_words, window_feats = WordTaggerDataset.process_sentence(
            w, f, self.wt.window_size, self.wt.featurizer)
        return self.wt.tag_sen(window_words, window_feats, debug=True) 

    def write_sen_result(self, item):
        w, f, tp, res, t_out = item
        tag = self.i2t[int(res)]
        self.output.write(u'{0}\t{1}\n'.format(tp, tag).encode('utf-8'))
        
        tags = [self.i2t[i] for i in argsort([-t for t in t_out])][:5]
        probs = sorted(t_out, reverse = True)[:5]
        self.output.write(u'\t'.join([' '.join([tag.lower(),
            str(round(prob, 4))])
            for tag, prob in zip(tags, probs)]).encode('utf-8'))
        self.output.write('\n')
        
        close_wds = self.get_close(self.close_cache, w, 
                self.word_vectors)
        self.output.write(u'{0}\n'.format('\t'\
                .join([self.i2w[c] for c in close_wds])).encode('utf-8'))

        close_f_dict = self.get_close_f_dict(f)
        self.output.write(u'{0}\n'.format(close_f_dict).encode('utf-8'))

    def update_sen_data(self, sen_data, res):
        result, tagger_out = res
        sen_data.append(list(result.flatten()))
        sen_data.append(tagger_out)

    def get_close_f_dict(self, f):
        
        close_f_dict = {}
        for f_ in f:
            f_str = self.get_fstring(f_)
            close_feats = self.get_close(self.close_feat_cache, f_,
                    self.feat_vectors, shift=True)
            
            cf_strings = []
            for i in close_feats:
                r = self.get_fstring(i, shift=True)
                if r != None:
                    cf_strings.append(r)
            close_f_dict[f_str] = cf_strings
        return close_f_dict    

    def get_close(self, cache, item, vec, shift=False):
        
        if shift:
            item += self.wt.featurizer.total*self.wt.window_size
        if item in cache:
            close = cache[item]
        else:
            close_i = cdist(numpy.array([vec[item]]), vec)
            close = numpy.argsort(close_i[0])[:10]
            cache[item] = close
        return close    

    def get_fstring(self, f, shift=False):
        if not shift:
            return self.wt.featurizer.i2f[f]
        if f in self.shifted_i2f:
            return self.shifted_i2f[f]
        shift = f/self.wt.featurizer.total
        if shift != self.wt.window_size:
            return None
        index = f % self.wt.featurizer.total
        self.shifted_i2f[f] = self.wt.featurizer.i2f[index]
        return repr(self.wt.featurizer.i2f[index])


class DebugGoldLabeledTagger(DebugTagger):
    
    def generate_sen_data(self):
        
        tc = TaggedCorpus(self.input_, self.wt.featurizer, w2i=self.wt.w2i,
                t2i=self.wt.t2i, use_unknown=True, num=self.wt.num)
        for sen in tc.read():
            w, orig_t, f, orig_words = [list(t) for t in zip(*sen)]
            to_print = map(lambda x: u'{0}\t{1}'.format(x[0], self.i2t[x[1]]), 

                    izip(orig_words, orig_t))
            yield [w, f, to_print]


def main():

    args = create_argparser()
    if not args.gold:
        a = DebugTagger(args)
    else:
        a = DebugGoldLabeledTagger(args)
    a.tag()

if __name__ == "__main__":
    main()
