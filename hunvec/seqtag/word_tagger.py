from pylearn2.models.mlp import MLP, CompositeLayer, Tanh
from pylearn2.space import CompositeSpace, IndexSpace
from pylearn2.sandbox.nlp.models.mlp import ProjectionLayer, Softmax
from pylearn2.training_algorithms.sgd import SGD
from pylearn2.train import Train


class WordTaggerNetwork(MLP):
    def __init__(self, vocab_size, window_size, feat_num, hdim, edim,
                 n_classes):
        self.vocab_size = vocab_size
        self.ws = window_size
        self.feat_num = feat_num
        self.hdim = hdim
        self.edim = edim
        self.n_classes = n_classes
        layers, input_space = self.create_network()
        input_source = ('words', 'features')
        super(WordTaggerNetwork, self).__init__(layers=layers,
                                                input_space=input_space,
                                                input_source=input_source)

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
            inputs_to_layers={0: [0], 1: [1]}
        )

        h0 = Tanh(layer_name='h0', dim=self.hdim, irange=.1)

        output = Softmax(layer_name='softmax', binary_target_dim=1,
                         n_classes=self.n_classes, irange=0.1)

        return [input_, h0, output], input_space


class WordTagger(object):
    def __init__(self, **kwargs):
        self.net = WordTaggerNetwork(**kwargs)

    def create_algorithm(self, data):
        algorithm = SGD(batch_size=1, learning_rate=.1,
                        #monitoring_dataset=self.dataset['valid'],
                        train_iteration_mode='sequential')
        self.trainer = Train(dataset=data, model=self.net,
                             algorithm=algorithm)


def test_data():
    params = {
        "vocab_size": 10,
        "window_size": 3,
        "feat_num": 2,
        "hdim": 10,
        "edim": 10,
        "n_classes": 2
    }
    X = [[0, 1, 2] + [0, 1, 0], [3, 4, 5] + [1, 1, 1]]
    y = [0, 0]
    return (X, y), params


def test():
    data, params = test_data()
    wt = WordTagger(**params)
    wt.create_algorithm(data)


if __name__ == "__main__":
    test()
