import numpy
import functools

import theano

from pylearn2.models.model import Model
from pylearn2.space import CompositeSpace, IndexSequenceSpace
from pylearn2.utils import sharedX

from word_tagger import WordTaggerNetwork


class SequenceTaggerNetwork(Model):
    def __init__(self, vocab_size, window_size, feat_num, hdim, edim,
                 n_classes):

        self.input_space = CompositeSpace([
            IndexSequenceSpace(max_labels=vocab_size, dim=window_size),
            IndexSequenceSpace(max_labels=feat_num, dim=window_size)
        ])
        self.output_space = IndexSequenceSpace(max_labels=n_classes, dim=1)

        self.input_source = ('words', 'features')
        self.target_source = 'targets'

        self.tagger = WordTaggerNetwork(vocab_size, window_size, feat_num,
                                        hdim, edim, n_classes)

        # ^ and $, that's why +1 at dimensions
        A_value = numpy.random.uniform(low=-.1, high=.1,
                                       size=(self.n_classes + 1,
                                             self.n_classes + 1))
        self.A = sharedX(A_value, name='A')

    def compute_trans_prob(self):
        # probably a simple matrix-vector multiplication
        pass

    @functools.wraps(Model.fprop)
    def fprop(self, data):
        tagger_out = self.tagger.fprop(data)
        # TODO compute initial tag probability (can ^ be a "."?)
        # probably a simple multiplication with first row/column of A

        fn = lambda : self.compute_trans_prob()
        #((f, h, out), updates) = theano.scan(fn=fn,
        #    sequences=[features, phones],
        #    outputs_info=[init_in,
        #    dict(initial=init_h,
        #    taps=[-1]),
        #    init_out])

    @functools.wraps(Model.get_params)
    def get_params(self):
        return self.tagger.get_params() + [self.A]

    @functools.wraps(Model.get_monitoring_channels)
    def get_monitoring_channels(self):
        rval = Model.get_monitoring_channels(self)
        # TODO add own channels
        return rval
