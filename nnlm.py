import sys

from pylearn2.space import IndexSpace
from pylearn2.models.mlp import MLP, Tanh
from pylearn2.sandbox.nlp.models.mlp import ProjectionLayer, Softmax
from pylearn2.training_algorithms.sgd import SGD
from pylearn2.termination_criteria import MonitorBased, And, EpochCounter
from pylearn2.train import Train
from pylearn2.train_extensions.best_params import MonitorBasedSaveBest

from corpus import Corpus


class NNLM(object):
    def __init__(self, hidden_dim=20, window_size=3, embedding_dim=10,
                 optimize_for='valid_softmax_nll', max_epochs=10000):
        self.hdim = hidden_dim
        self.window_size = window_size
        self.edim = embedding_dim
        self.optimize_for = optimize_for
        self.max_epochs = max_epochs

    def add_corpus(self, corpus):
        self.vocab_size = len(corpus.vocab)
        d = corpus.get_matrices(self.window_size)
        self.alg_datasets = {'train': d[0], 'valid': d[1], 'test': d[2]}

    def create_model(self):

        # input will be projected. ProjectionLayer? MatrixMul? IndexSpace?
        input_ = ProjectionLayer(layer_name='X', dim=self.edim, irange=0.)

        # sparse_init=15?
        h0 = Tanh(layer_name='h0', dim=self.hdim, irange=.01)
        output = Softmax(layer_name='softmax', binary_target_dim=1,
                         n_classes=self.vocab_size, irange=0.)

        input_space = IndexSpace(max_labels=self.vocab_size,
                                 dim=self.window_size)
        model = MLP(layers=[input_, h0, output],
                    input_space=input_space)
        self.model = model

    def create_algorithm(self):
        cost_crit = MonitorBased(channel_name=self.optimize_for,
                                 prop_decrease=0., N=10)
        epoch_cnt_crit = EpochCounter(max_epochs=self.max_epochs)
        term = And(criteria=[cost_crit, epoch_cnt_crit])
        self.algorithm = SGD(batch_size=256, learning_rate=.1,
                             monitoring_dataset=self.alg_datasets,
                             termination_criterion=term)

    def create_training_problem(self, save_best_path):
        ext1 = MonitorBasedSaveBest(channel_name=self.optimize_for,
                                    save_path=save_best_path)
        trainer = Train(dataset=self.alg_datasets['train'], model=self.model,
                        algorithm=self.algorithm, extensions=[ext1])
        self.trainer = trainer


def main():
    nnlm = NNLM(hidden_dim=40, embedding_dim=20, max_epochs=100, window_size=3)
    corpus = Corpus.read_corpus(sys.argv[1])
    corpus.filter_freq(n=2000)
    nnlm.add_corpus(corpus)
    nnlm.create_model()
    nnlm.create_algorithm()
    nnlm.create_training_problem(sys.argv[2])
    nnlm.trainer.main_loop()


if __name__ == "__main__":
    main()
