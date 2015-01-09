import sys

import numpy

from pylearn2.models.mlp import MLP, CompositeLayer, Tanh, Softmax, Linear
from pylearn2.space import CompositeSpace, IndexSpace
from pylearn2.sandbox.nlp.models.mlp import ProjectionLayer
from pylearn2.training_algorithms.sgd import SGD, LinearDecay
from pylearn2.train import Train
from pylearn2.train_extensions.best_params import MonitorBasedSaveBest
from pylearn2.termination_criteria import MonitorBased, And, EpochCounter
from pylearn2.costs.cost import SumOfCosts
from pylearn2.costs.mlp import Default, WeightDecay
from pylearn2.training_algorithms import learning_rule

from word_tagger_dataset import WordTaggerDataset
from hunvec.corpus.tagged_corpus import TaggedCorpus


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

        #output = Softmax(layer_name='softmax', binary_target_dim=1,
        #                 n_classes=self.n_classes, irange=0.1)
        output = Linear(layer_name='tagger_out', irange=.1, dim=self.n_classes)

        return [input_, h0, output], input_space


class WordTagger(object):
    def __init__(self, **kwargs):
        self.net = WordTaggerNetwork(**kwargs)
        self.optimize_for = 'valid_softmax_misclass'
        self.max_epochs = 100

    def create_adjustors(self):
        initial_momentum = .9
        final_momentum = .99
        start = 1
        saturate = self.max_epochs
        self.momentum_adjustor = learning_rule.MomentumAdjustor(
            final_momentum, start, saturate)
        self.momentum_rule = learning_rule.Momentum(initial_momentum,
                                                    nesterov_momentum=True)

        decay_factor = .1
        self.learning_rate_adjustor = LinearDecay(
            start, saturate * 100, decay_factor)

    def get_monitoring_data_specs(self):
        return self.dataset['train'].get_data_specs()

    def create_algorithm(self, data, save_best_path):
        self.dataset = data
        epoch_cnt_crit = EpochCounter(max_epochs=self.max_epochs)
        cost_crit = MonitorBased(channel_name=self.optimize_for,
                                 prop_decrease=0., N=10)
        term = And(criteria=[cost_crit, epoch_cnt_crit])

        weightdecay = WeightDecay(coeffs=[5e-5, 5e-5, 5e-5])
        cost = SumOfCosts(costs=[Default(), weightdecay])

        self.create_adjustors()

        mbsb = MonitorBasedSaveBest(channel_name=self.optimize_for,
                                    save_path=save_best_path)
        algorithm = SGD(batch_size=32, learning_rate=.1,
                        #cost=cost,
                        #termination_criterion=term,
                        termination_criterion=epoch_cnt_crit,
                        monitoring_dataset=data['valid'],
                        learning_rule=self.momentum_rule,
                        update_callbacks=[self.learning_rate_adjustor],
                        )
        self.trainer = Train(dataset=data['train'], model=self.net,
                             algorithm=algorithm, extensions=[mbsb])


def test_data():
    params = {
        "vocab_size": 10,
        "window_size": 3,
        "feat_num": 2,
        "hdim": 10,
        "edim": 10,
        "n_classes": 2
    }
    X = [numpy.array([[0, 1, 2], [3, 5, 4]]),
         numpy.array([[0, 1, 0], [1, 1, 1]])]
    y = numpy.array([[0], [0]])
    d = WordTaggerDataset(X, y)
    return d, params


def test():
    data, params = test_data()
    wt = WordTagger(**params)
    wt.create_algorithm(data)
    wt.trainer.main_loop()


def train_brown_pos():
    fn = sys.argv[1]
    c = TaggedCorpus(fn)
    d = WordTaggerDataset.create_from_tagged_corpus(c)
    wt = WordTagger(vocab_size=d['train'].vocab_size,
                    window_size=d['train'].window_size,
                    feat_num=d['train'].feat_num,
                    n_classes=d['train'].n_classes,
                    edim=200, hdim=400)
    wt.create_algorithm(d, sys.argv[2])
    wt.trainer.main_loop()


if __name__ == "__main__":
    train_brown_pos()
