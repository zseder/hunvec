from math import sqrt
from functools import wraps


from pylearn2.models.mlp import MLP, CompositeLayer, Tanh, Linear
from pylearn2.space import CompositeSpace, IndexSpace
from pylearn2.sandbox.nlp.models.mlp import ProjectionLayer


class WordTaggerNetwork(MLP):
    def __init__(self, vocab_size, window_size, total_feats, feat_num,
                 hdims, edim, fedim, n_classes):
        self.vocab_size = vocab_size
        self.ws = window_size
        self.total_feats = total_feats
        self.feat_num = feat_num
        self.hdims = hdims
        self.edim = edim
        self.fedim = fedim
        self.n_classes = n_classes
        layers, input_space = self.create_network()
        input_source = self.create_input_source()
        super(WordTaggerNetwork, self).__init__(layers=layers,
                                                input_space=input_space,
                                                input_source=input_source)

    def create_network(self):
        input_space = self.create_input_space()
        input_ = self.create_input_layer()
        hiddens = self.create_hidden_layers()
        output = self.create_output_layer()
        return [input_] + hiddens + [output], input_space

    def create_input_source(self):
        return ('words', 'features')

    def create_input_space(self):
        ws = (self.ws * 2 + 1)
        return CompositeSpace([
            IndexSpace(max_labels=self.vocab_size, dim=ws),
            IndexSpace(max_labels=self.total_feats, dim=self.feat_num)
        ])

    def create_input_layer(self):
        return CompositeLayer(
            layer_name='input',
            layers=[
                ProjectionLayer(layer_name='words', dim=self.edim, irange=.1),
                ProjectionLayer(layer_name='feats', dim=self.fedim, irange=.1),
            ],
            inputs_to_layers={0: [0], 1: [1]}
        )

    def create_hidden_layers(self):
        # for parameter settings, see Remark 7 (Tricks) in NLP from scratch
        hiddens = []
        for i, hdim in enumerate(self.hdims):
            sc = 1. / hdim
            h = Tanh(layer_name='h{}'.format(i), dim=hdim,
                     istdev=1./sqrt(hdim), W_lr_scale=sc, b_lr_scale=sc)
            hiddens.append(h)
        return hiddens

    def create_output_layer(self):
        sc = 1. / self.n_classes
        output = Linear(layer_name='tagger_out',
                        istdev=1. / sqrt(self.n_classes),
                        dim=self.n_classes, W_lr_scale=sc, b_lr_scale=sc)
        return output

    @wraps(MLP._modify_updates)
    def _modify_updates(self, updates):
        wW = self.layers[0].layers[0].W
        if wW in updates:
            updates[wW][self.not_seen['words']] = 0.

        fW = self.layers[0].layers[1].W
        if fW in updates:
            updates[fW][self.not_seen['feats']] = 0.

