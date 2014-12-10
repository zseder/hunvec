from pylearn2.models.mlp import MLP, CompositeLayer, Tanh
from pylearn2.space import CompositeSpace, IndexSpace
from pylearn2.sandbox.nlp.models.mlp import ProjectionLayer, Softmax


class WordTagger(MLP):
    def __init__(self):
        layers, input_space = self.create_network()
        MLP.__init__(self, layers=layers, input_space=input_space)

    def create_network(self):
        # words and features
        input_space = CompositeSpace([
            IndexSpace(max_labels=self.vocab_size, dim=self.ws),
            IndexSpace(max_labels=self.feat_num, dim=self.ws)
        ])

        input_ = CompositeLayer(
            layer_name='input',
            layers=[
                ProjectionLayer(layer_name='words', dim=self.edim, irange=.1),
                ProjectionLayer(layer_name='feats', dim=self.edim, irange=.1),
            ],
            inputs_to_layers={0: 0, 1: 1}
        )

        h0 = Tanh(layer_name='h0', dim=self.hdim, irange=.1)

        output = Softmax(layer_name='softmax', binary_target_dim=1,
                         n_classes=self.n_classes, irange=0.1)

        return [input_, h0, output], input_space
