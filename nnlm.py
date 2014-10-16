import sys

from pylearn2.space import IndexSpace
from pylearn2.models.mlp import MLP, Tanh, RectifiedLinear
from pylearn2.sandbox.nlp.models.mlp import ProjectionLayer, Softmax
from pylearn2.training_algorithms.sgd import SGD
from pylearn2.termination_criteria import MonitorBased, And, EpochCounter
from pylearn2.train import Train
from pylearn2.train_extensions.best_params import MonitorBasedSaveBest
from pylearn2.costs.cost import SumOfCosts
from pylearn2.costs.mlp import Default, WeightDecay

from corpus import Corpus


class NNLM(object):
    def __init__(self, hidden_dim=20, window_size=3, embedding_dim=10,
                 optimize_for='valid_softmax_ppl', max_epochs=10000):
        self.hdim = hidden_dim
        self.window_size = window_size
        self.edim = embedding_dim
        self.optimize_for = optimize_for
        self.max_epochs = max_epochs

    def add_corpus(self, corpus):
        self.corpus = corpus
        self.vocab_size = len(corpus.needed) + 1  # for filtered words

    def create_model(self):

        input_ = ProjectionLayer(layer_name='X', dim=self.edim, irange=0.1)
        h0 = Tanh(layer_name='h0', dim=self.hdim, irange=.1)
        h1 = RectifiedLinear(layer_name='h1', dim=self.edim, irange=.1)
        output = Softmax(layer_name='softmax', binary_target_dim=1,
                         n_classes=self.vocab_size, irange=0.1)

        input_space = IndexSpace(max_labels=self.vocab_size,
                                 dim=self.window_size)
        model = MLP(layers=[input_, h0, h1, output],
                    input_space=input_space)
        self.model = model

    def create_algorithm(self, monitoring_dataset):
        cost_crit = MonitorBased(channel_name=self.optimize_for,
                                 prop_decrease=0., N=2)
        epoch_cnt_crit = EpochCounter(max_epochs=self.max_epochs)
        term = And(criteria=[cost_crit, epoch_cnt_crit])
        weightdecay = WeightDecay(coeffs=[5e-5, 5e-5, 5e-5])
        cost = SumOfCosts(costs=[Default(), weightdecay])
        self.algorithm = SGD(batch_size=64, learning_rate=.1,
                             monitoring_dataset=monitoring_dataset,
                             termination_criterion=term, cost=cost)

    def create_training_problem(self, training_dataset, save_best_path):
        ext1 = MonitorBasedSaveBest(channel_name=self.optimize_for,
                                    save_path=save_best_path)
        trainer = Train(dataset=training_dataset, model=self.model,
                        algorithm=self.algorithm, extensions=[ext1])
        self.trainer = trainer

    def create_batch_trainer(self, save_best_path):
        del self.trainer, self.algorithm
        self.create_model()
        dataset = self.corpus.create_batch_matrices()
        if dataset is None:
            return None
        self.create_algorithm(dataset[1])
        self.create_training_problem(dataset[0], save_best_path)


def main():
    nnlm = NNLM(hidden_dim=200, embedding_dim=100, max_epochs=100, window_size=5)
    corpus = Corpus(sys.argv[1])
    nnlm.add_corpus(corpus)
    while True:
        nnlm.create_batch_trainer(sys.argv[2])
        if not hasattr(nnlm, 'trainer'):
            break
        nnlm.trainer.main_loop()


if __name__ == "__main__":
    main()
