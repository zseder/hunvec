import sys

from pylearn2.space import IndexSpace
from pylearn2.models.mlp import MLP, Tanh
from pylearn2.sandbox.nlp.models.mlp import ProjectionLayer, Softmax
from pylearn2.training_algorithms.sgd import SGD
from pylearn2.termination_criteria import MonitorBased
from pylearn2.train import Train

from corpus import Corpus


class NNLM(object):
    def __init__(self, hidden_dim=20, window_size=3, embedding_dim=10):
        self.hdim = hidden_dim
        self.window_size = window_size
        self.edim = embedding_dim

    def add_corpus(self, corpus):
        self.vocab_size = len(corpus.vocab)
        d = corpus.get_matrices(self.window_size)
        self.alg_datasets = {'train': d[0], 'valid': d[1], 'test': d[2]}

    def create_model(self):

        # input will be projected. ProjectionLayer? MatrixMul? IndexSpace?
        input_ = ProjectionLayer(layer_name='X', dim=self.edim, irange=0.)

        # sparse_init=15?
        h0 = Tanh(layer_name='h0', dim=self.hdim, irange=.01)
        output = Softmax(layer_name='softmax',
                         n_classes=self.vocab_size, irange=0.,
                         binary_target_dim=1)

        input_space = IndexSpace(max_labels=self.vocab_size,
                                 dim=self.window_size)
        model = MLP(layers=[input_, h0, output],
                    input_space=input_space)
        self.model = model

    def create_algorithm(self):
        term = MonitorBased(channel_name='valid_softmax_nll', prop_decrease=0.,
                            N=10)
        self.algorithm = SGD(batch_size=100, learning_rate=.01,
                             monitoring_dataset=self.alg_datasets,
                             termination_criterion=term)

    def create_training_problem(self):
        trainer = Train(dataset=self.alg_datasets['train'], model=self.model,
                        algorithm=self.algorithm, extensions=[])
        self.trainer = trainer


def main():
    nnlm = NNLM()
    corpus = Corpus.read_corpus(sys.argv[1])
    corpus.filter_freq(1000)
    nnlm.add_corpus(corpus)
    nnlm.create_model()
    nnlm.create_algorithm()
    nnlm.create_training_problem()
    nnlm.trainer.main_loop()


if __name__ == "__main__":
    main()
