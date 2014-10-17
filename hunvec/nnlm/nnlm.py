import sys
import logging

from pylearn2.space import IndexSpace
from pylearn2.models.mlp import MLP, Tanh
from pylearn2.sandbox.nlp.models.mlp import ProjectionLayer, Softmax
from pylearn2.training_algorithms.sgd import SGD
from pylearn2.termination_criteria import MonitorBased, And, EpochCounter
from pylearn2.train import Train
from pylearn2.train_extensions.best_params import MonitorBasedSaveBest
from pylearn2.costs.cost import SumOfCosts
from pylearn2.costs.mlp import Default, WeightDecay

from hunvec.corpus.corpus import Corpus
from hunvec.layers.hs import HierarchicalSoftmax as HS


class NNLM(object):
    def __init__(self, hidden_dim=20, window_size=3, embedding_dim=10,
                 optimize_for='valid_softmax_ppl', max_epochs=10000, hs=False):
        self.hdim = hidden_dim
        self.window_size = window_size
        self.edim = embedding_dim
        self.optimize_for = optimize_for
        self.max_epochs = max_epochs
        self.hs = hs

    def add_corpus(self, corpus):
        self.corpus = corpus
        self.vocab_size = len(corpus.needed) + 1  # for filtered words

    def create_model(self):

        input_ = ProjectionLayer(layer_name='X', dim=self.edim, irange=0.1)
        h0 = Tanh(layer_name='h0', dim=self.hdim, irange=.1)
        if not self.hs:
            output = Softmax(layer_name='softmax', binary_target_dim=1,
                             n_classes=self.vocab_size, irange=0.1)
        else:
            output = HS(self.vocab_size, layer_name='hs', irange=0.1)

        input_space = IndexSpace(max_labels=self.vocab_size,
                                 dim=self.window_size)
        model = MLP(layers=[input_, h0, output],
                    input_space=input_space)
        self.model = model

    def create_algorithm(self, dataset):
        cost_crit = MonitorBased(channel_name=self.optimize_for,
                                 prop_decrease=0., N=2)
        epoch_cnt_crit = EpochCounter(max_epochs=self.max_epochs)
        term = And(criteria=[cost_crit, epoch_cnt_crit])
        # TODO: weightdecay with projection layer?
        #weightdecay = WeightDecay(coeffs=[None, 5e-5, 5e-5, 5e-5])
        #cost = SumOfCosts(costs=[Default(), weightdecay])
        self.algorithm = SGD(batch_size=64, learning_rate=.1,
                             monitoring_dataset=dataset,
                             termination_criterion=term)

    def create_training_problem(self, dataset, save_best_path):
        ext1 = MonitorBasedSaveBest(channel_name=self.optimize_for,
                                    save_path=save_best_path)
        trainer = Train(dataset=dataset['train'], model=self.model,
                        algorithm=self.algorithm, extensions=[ext1])
        self.trainer = trainer

    def create_batch_trainer(self, save_best_path):
        dataset = self.corpus.create_batch_matrices()
        if dataset is None:
            if hasattr(self, "trainer"):
                del self.trainer
            return None
        if hasattr(self.model, 'monitor'):
            del self.model.monitor
            del self.trainer
            del self.algorithm
            del self.model.tag["MonitorBasedSaveBest"]
        d = {'train': dataset[0], 'valid': dataset[1], 'test': dataset[2]}
        self.create_algorithm(d)
        self.create_training_problem(d, save_best_path)


def main():
    logging.basicConfig(level=logging.INFO)
    nnlm = NNLM(hidden_dim=128, embedding_dim=64, max_epochs=20, window_size=3, hs=True, optimize_for='valid_hs_kl')
    corpus = Corpus(sys.argv[1], batch_size=100000, window_size=3, top_n=1000, hs=True)
    nnlm.add_corpus(corpus)
    nnlm.create_model()
    c = 1
    while True:
        logging.info("{0}. batch started".format(c))
        nnlm.create_batch_trainer(sys.argv[2])
        if not hasattr(nnlm, 'trainer'):
            break
        nnlm.trainer.main_loop()
        c += 1


if __name__ == "__main__":
    main()
