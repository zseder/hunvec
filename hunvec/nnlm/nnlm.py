import logging
import argparse

from pylearn2.space import IndexSpace
from pylearn2.models.mlp import MLP, Tanh
from pylearn2.sandbox.nlp.models.mlp import ProjectionLayer, Softmax
from pylearn2.training_algorithms.sgd import SGD, LinearDecay
from pylearn2.training_algorithms import learning_rule
from pylearn2.termination_criteria import MonitorBased, And, EpochCounter
from pylearn2.train_extensions.best_params import MonitorBasedSaveBest
from pylearn2.costs.cost import SumOfCosts
from pylearn2.costs.mlp import Default, WeightDecay

from hunvec.corpus.corpus import Corpus
from hunvec.layers.hs import HierarchicalSoftmax as HS
from hunvec.layers.cbow_projection import CBowProjectionLayer as CBP


class NNLM(object):
    def __init__(self, hidden_dim=20, window_size=3, embedding_dim=10,
                 optimize_for='valid_softmax_ppl', max_epochs=20, hs=False,
                 save_best_path='best_model_file', cbow=False, vectors_fn=None):
        self.hdim = hidden_dim
        self.window_size = window_size
        self.edim = embedding_dim
        self.optimize_for = optimize_for
        self.max_epochs = max_epochs
        self.hs = hs
        self.save_best_path = save_best_path
        self.cbow = cbow
        self.vectors_fn = vectors_fn

    def add_corpus(self, corpus):
        self.corpus = corpus
        self.dataset, self.vocab_size = self.corpus.read_dataset()

    def create_model(self):
        if not self.cbow:
            input_ = ProjectionLayer(layer_name='X', dim=self.edim, irange=.5)
        else:
            input_ = CBP(layer_name='X', dim=self.edim, irange=.5)
        h0 = Tanh(layer_name='h0', dim=self.hdim, irange=.5)
        if not self.hs:
            output = Softmax(layer_name='softmax', binary_target_dim=1,
                             n_classes=self.vocab_size, irange=0.5)
        else:
            output = HS(self.vocab_size, layer_name='hs', irange=0.01)

        ws = self.window_size
        if self.cbow:
            ws *= 2
        input_space = IndexSpace(max_labels=self.vocab_size, dim=ws)
        model = MLP(layers=[input_, h0, output], input_space=input_space)
        self.model = model

    def create_adjustors(self):
        initial_momentum = .5
        final_momentum = .99
        start = 1
        saturate = 100
        self.momentum_adjustor = learning_rule.MomentumAdjustor(
            final_momentum, start, saturate)
        self.momentum_rule = learning_rule.Momentum(initial_momentum)

        decay_factor = .1
        self.learning_rate_adjustor = LinearDecay(
            start, saturate * 1000, decay_factor)

    def create_algorithm(self):
        epoch_cnt_crit = EpochCounter(max_epochs=self.max_epochs)
        cost_crit = MonitorBased(channel_name=self.optimize_for,
                                 prop_decrease=0., N=3)
        term = And(criteria=[cost_crit, epoch_cnt_crit])

        weightdecay = WeightDecay(coeffs=[0., 5e-5, 0.])
        cost = SumOfCosts(costs=[Default(), weightdecay])

        self.create_adjustors()
        self.algorithm = SGD(batch_size=256, learning_rate=.1,
                             termination_criterion=term,
                             update_callbacks=[self.learning_rate_adjustor],
                             learning_rule=self.momentum_rule,
                             cost=cost,
                             train_iteration_mode='sequential')
        self.mbsb = MonitorBasedSaveBest(channel_name=self.optimize_for,
                                         save_path=self.save_best_path)

    def train(self):
        self.algorithm.monitoring_dataset = self.dataset
        self.algorithm.setup(self.model, self.dataset['train'])
        self.algorithm.train(dataset=self.dataset['train'])
        self.write_embedding()

    def write_embedding(self):
        fn = self.vectors_fn
        if fn is None:
            return

        embedding = self.model.get_params()[0].get_value()
        with open(fn, mode='w') as outfile:
            outfile.write('{} {}\n'.format(*embedding.shape))
            for i in xrange(-1, embedding.shape[0]-1):
                word = self.corpus.index2word[i].encode('utf8')
                vector = ' '.join(['{0:.4}'.format(coord)
                                   for coord in embedding[i].tolist()])
                outfile.write('{} {}\n'.format(word, vector))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('corpus')
    parser.add_argument('model')
    parser.add_argument(
        '--hidden-dim', default=200, type=int, dest='hdim')
    parser.add_argument(
        '--vector-dim', default=100, type=int, dest='vdim')
    parser.add_argument(
        '-e', '--epoch', default=20, type=int, dest='epoch')
    parser.add_argument(
        '--window', default=5, type=int)
    parser.add_argument(
        '--no-hierarchical-softmax', action='store_false', dest='hs')
    parser.add_argument(
        '--cost', default='valid_hs_ppl')
    parser.add_argument(
        '--vocab-size', default=50000, type=int, dest='vsize')
    parser.add_argument(
        '--vectors', default='vectors.txt')
    parser.add_argument(
        '--cbow', action='store_true', dest='cbow')
    return parser.parse_args()


def main():
    logging.basicConfig(level=logging.DEBUG,
                        format="%(asctime)s : %(module)s (%(lineno)s) - %(levelname)s - %(message)s")  # nopep8
    args = parse_args()
    logging.debug(args.cost)
    nnlm = NNLM(
        hidden_dim=args.hdim, embedding_dim=args.vdim, max_epochs=args.epoch,
        window_size=args.window, hs=args.hs, optimize_for=args.cost,
        save_best_path=args.model, cbow=args.cbow, vectors_fn=args.vectors)
    corpus = Corpus(
        hdf5_path=args.corpus, window_size=args.window,
        top_n=args.vsize, hs=args.hs, future=args.cbow)
    nnlm.add_corpus(corpus)
    nnlm.create_model()
    nnlm.create_algorithm()
    nnlm.train()


if __name__ == "__main__":
    main()
