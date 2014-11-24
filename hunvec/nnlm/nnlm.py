import logging
import argparse

from pylearn2.space import IndexSpace
from pylearn2.models.mlp import MLP, Tanh
from pylearn2.sandbox.nlp.models.mlp import ProjectionLayer, Softmax
from pylearn2.training_algorithms.sgd import SGD
from pylearn2.termination_criteria import MonitorBased, And, EpochCounter
from pylearn2.train import Train
from pylearn2.train_extensions.best_params import MonitorBasedSaveBest
#from pylearn2.costs.cost import SumOfCosts
#from pylearn2.costs.mlp import Default, WeightDecay

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
        self.vocab_size = len(corpus.needed)  # for filtered words

    def create_model(self):

        input_ = ProjectionLayer(layer_name='X', dim=self.edim, irange=0.1)
        h0 = Tanh(layer_name='h0', dim=self.hdim, irange=.1)
        if not self.hs:
            output = Softmax(layer_name='softmax', binary_target_dim=1,
                             n_classes=self.vocab_size, irange=0.1)
        else:
            output = HS(self.vocab_size - 1, layer_name='hs', irange=0.1)

        input_space = IndexSpace(max_labels=self.vocab_size,
                                 dim=self.window_size)
        model = MLP(layers=[input_, h0, output], input_space=input_space)
        model.index2word = self.corpus.index2word
        self.model = model

    def create_algorithm(self, dataset):
        cost_crit = MonitorBased(channel_name=self.optimize_for,
                                 prop_decrease=0., N=2)
        epoch_cnt_crit = EpochCounter(max_epochs=self.max_epochs)
        term = And(criteria=[cost_crit, epoch_cnt_crit])
        # TODO: weightdecay with projection layer?
        #weightdecay = WeightDecay(coeffs=[None, 5e-5, 5e-5, 5e-5])
        #cost = SumOfCosts(costs=[Default(), weightdecay])
        self.algorithm = SGD(batch_size=32, learning_rate=.1,
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


def write_embedding(corpus, nnlm, filen):
    embedding = nnlm.model.get_params()[0].get_value()
    with open(filen, mode='w') as outfile:
        outfile.write('{} {}\n'.format(*embedding.shape))
        for i in xrange(-1, embedding.shape[0]-1):
            word = corpus.index2word[i].encode('utf8')
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
        '--corpus-epoch', default=2, type=int, dest='cepoch')
    parser.add_argument(
        '--batch-epoch', default=20, type=int, dest='bepoch')
    parser.add_argument(
        '--window', default=5, type=int)
    parser.add_argument(
        '--no-hierarchical-softmax', action='store_false', dest='hs')
    parser.add_argument(
        '--cost', default='valid_hs_ppl')
    parser.add_argument(
        '--batch-size', default=20000, type=int, dest='bsize')
    parser.add_argument(
        '--vocab-size', default=50000, type=int, dest='vsize')
    parser.add_argument(
        '--vectors', default='vectors.txt')
    return parser.parse_args()


def main():
    logging.basicConfig(level=logging.DEBUG,
                        format="%(asctime)s : %(module)s (%(lineno)s) - %(levelname)s - %(message)s")  # nopep8
    args = parse_args()
    logging.debug(args.cost)
    nnlm = NNLM(
        hidden_dim=args.hdim, embedding_dim=args.vdim, max_epochs=args.bepoch,
        window_size=args.window, hs=args.hs, optimize_for=args.cost)
    corpus = Corpus(
        args.corpus, batch_size=args.bsize, window_size=args.window,
        top_n=args.vsize, hs=args.hs, max_corpus_epoch=args.cepoch)
    nnlm.add_corpus(corpus)
    nnlm.create_model()
    c = 1
    while True:
        logging.info("{0}. batch started".format(c))
        nnlm.create_batch_trainer(args.model)
        if not hasattr(nnlm, 'trainer'):
            break
        logging.info("Training started.")
        nnlm.trainer.main_loop()
        c += 1
        write_embedding(corpus, nnlm, args.vectors)


if __name__ == "__main__":
    main()
