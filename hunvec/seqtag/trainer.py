import os
import argparse

from pylearn2.utils import serial

# the WordTaggerDataset import is needed because of pickle load
from hunvec.datasets.word_tagger_dataset import load_dataset, WordTaggerDataset  # nopep8
from hunvec.seqtag.sequence_tagger import SequenceTaggerNetwork


class CSL2L(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, [float(_) for _ in values.split(',')])


def create_argparser():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('dataset_file')
    argparser.add_argument('model_path')
    argparser.add_argument('--hidden', default='100',
                           help='comma separated integers, number of units' +
                           'of hidden layers',
                           action=CSL2L)
    argparser.add_argument('--embedding', default=50, type=int)
    argparser.add_argument('--feat_embedding', default=5, type=int)
    argparser.add_argument('--epochs', default=50, type=int)
    argparser.add_argument('--regularization', default=.0, type=float,
                           help='typical values are 1e-5, 1e-4')
    argparser.add_argument('--use_momentum', action='store_true'),
    argparser.add_argument('--lr', default=.01, type=float,
                           help='learning rate')
    argparser.add_argument('--lr_decay', default=.1, type=float,
                           help='decrease ratio over time on learning rate')
    argparser.add_argument('--valid_stop', type=bool, default=True,
                           help='don\'t use valid data to decide when to stop')
    argparser.add_argument('--dropout_params', action=CSL2L,
                           help='use dropout on inner network' +
                           'include probs per layer')
    argparser.add_argument('--dropout', action='store_true',
                           help='use dropout on inner network')
    argparser.add_argument('--embedding_init', help='embedding weights for ' +
                           'initialization, in word2vec format')
    return argparser.parse_args()


def init_network(args, dataset, corpus):
    if os.path.exists(args.model_path):
        # loading model instead of creating a new one
        wt = serial.load(args.model_path)
        wt.max_epochs = args.epochs
        wt.use_momentum = args.use_momentum
        wt.lr = args.lr
        wt.lr_decay = args.lr_decay
        wt.valid_stop = args.valid_stop
        wt.reg_factors = args.regularization
        wt.dropout_params = args.dropout_params
        wt.dropout = args.dropout or args.dropout_params is not None
        return wt

    wt = SequenceTaggerNetwork(edim=args.embedding, fedim=args.feat_embedding,
                               hdims=args.hidden,
                               dataset=dataset,
                               w2i=corpus.w2i, t2i=corpus.t2i,
                               featurizer=corpus.featurizer,
                               max_epochs=args.epochs,
                               use_momentum=args.use_momentum,
                               lr=args.lr, lr_decay=args.lr_decay,
                               valid_stop=args.valid_stop,
                               reg_factors=args.regularization,
                               dropout=args.dropout,
                               dropout_params=args.dropout_params,
                               embedding_init=args.embedding_init
                               )
    return wt


def init_network_corpus(args):
    d, train_c = load_dataset(args.dataset_file)
    wt = init_network(args, d['train'], train_c)
    return d, wt


def main():
    args = create_argparser()
    res = init_network_corpus(args)
    d = res[0]
    wt = res[1]
    wt.create_algorithm(d, args.model_path)
    wt.train()
    load_dataset()


if __name__ == "__main__":
    main()
