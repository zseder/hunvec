from math import sqrt
from functools import wraps

import numpy

import theano

from pylearn2.models.mlp import CompositeLayer, Linear
from pylearn2.sandbox.nlp.models.mlp import ProjectionLayer
from pylearn2.space import CompositeSpace, IndexSpace, VectorSpace

from hunvec.seqtag.word_tagger import WordTaggerNetwork


class ExtendedHiddenNetwork(WordTaggerNetwork):
    def __init__(self, extender_dim, *args, **kwargs):
        self.extender_dim = extender_dim
        super(ExtendedHiddenNetwork, self).__init__(*args, **kwargs)
        embed = self.layers[0].layers[2]
        # HACK do not learn this, only set values to zero
        embed.set_param_vector(numpy.ones(embed.get_param_vector().shape,
                                          dtype=theano.config.floatX))

    def create_input_source(self):
        return ('words', 'features', 'tagger_out')

    def create_input_space(self):
        ws = (self.ws * 2 + 1)
        return CompositeSpace([
            IndexSpace(max_labels=self.vocab_size, dim=ws),
            IndexSpace(max_labels=self.total_feats, dim=self.feat_num),
            VectorSpace(dim=self.extender_dim * ws)
        ])

    def create_input_layer(self):
        size = self.extender_dim * (self.ws * 2 + 1)
        embed = Linear(layer_name='lin_embed', dim=size,
                       istdev=1. / sqrt(size))

        return CompositeLayer(
            layer_name='input',
            layers=[
                ProjectionLayer(layer_name='ext_words', dim=self.edim, irange=.1),
                ProjectionLayer(layer_name='ext_feats', dim=self.fedim, irange=.1),
                embed,
            ],
            inputs_to_layers={0: [0], 1: [1], 2: [2]}
        )

    @wraps(WordTaggerNetwork._modify_updates)
    def _modify_updates(self, updates):
        updates = super(WordTaggerNetwork, self)._modify_updates(updates)
        wW = self.layers[0].layers[2].W
        if wW in updates:
            updates[wW] = 0.
        return updates
