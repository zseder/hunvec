import numpy

import theano

from pylearn2.space import CompositeSpace, IndexSequenceSpace
from pylearn2.space import VectorSequenceSpace

from hunvec.datasets.word_tagger_dataset import WordTaggerDataset

class ExtendedWordTaggerDataset(WordTaggerDataset):
    def __init__(self, tagger, other):
        self.tagger = tagger
        # HACK with feat nums, since WordTaggerDataset transforms them
        ws = other.window_size * 2 + 1
        tf = other.total_feats / ws
        fn = other.feat_num / ws
        super(ExtendedWordTaggerDataset, self).__init__(
            X=(other.X1, other.X2), y=other.y, vocab_size=other.vocab_size,
            window_size=other.window_size, total_feats=tf, feat_num=fn,
            n_classes=other.n_classes)
        tagged_sens = [tagger.tag_sen(self.X1[i], self.X2[i], return_probs=True)
                   for i in xrange(len(self.X1))]
        self.create_windowed_tagged_vectors(tagged_sens)

    @staticmethod
    def windowize_tagged_vectors(tsen, ws, n_classes):
        pad = numpy.array(
            [numpy.zeros(n_classes)] * ws, dtype=theano.config.floatX)
        ptsen = numpy.concatenate([pad, tsen, pad])
        l = []
        for i in xrange(ws, len(ptsen) - ws):
            window = ptsen[i - ws: i + ws + 1]
            window = window.flatten()
            l.append(window)
        return numpy.array(l)

    def create_windowed_tagged_vectors(self, tagged_sens):
        self.X3 = []
        ws = self.window_size
        n_classes = self.tagger.n_classes
        for tsen in tagged_sens:
            self.X3.append(ExtendedWordTaggerDataset.windowize_tagged_vectors(
                tsen, ws, n_classes))
            #print self.X3[-1].shape
            #quit()

    def _create_data_specs(self):
        ws = (self.window_size * 2 + 1)
        space = CompositeSpace((
            IndexSequenceSpace(max_labels=self.vocab_size, dim=ws),
            IndexSequenceSpace(max_labels=self.total_feats,
                               dim=self.feat_num),
            VectorSequenceSpace(dim=self.tagger.n_classes * ws),
            VectorSequenceSpace(dim=self.n_classes)
        ))
        source = ('words', 'features', 'tags', 'targets')
        self.data_specs = (space, source)

    def get(self, source, next_index):
        # HACK since there are iterators that are 'fancy', and others are
        # not, we have to be prepared for numbered and sliced indexing
        if type(next_index) is slice:
            return (self.X1[next_index][0],
                    self.X2[next_index][0],
                    self.X3[next_index][0],
                    self.y[next_index][0])
        else:
            return (self.X1[next_index], self.X2[next_index],
                    self.X3[next_index], self.y[next_index])
