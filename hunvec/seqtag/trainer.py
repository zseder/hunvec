import os
import logging
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
    argparser.add_argument('--hidden',
                           help='comma separated integers, number of units' +
                           'of hidden layers, default=100',
                           action=CSL2L)
    argparser.add_argument('--embedding', type=int)
    argparser.add_argument('--feat_embedding', type=int)
    argparser.add_argument('--epochs', type=int)
    argparser.add_argument('--regularization', type=float,
                           help='typical values are 1e-5, 1e-4')
    argparser.add_argument('--use_momentum', action='store_true'),
    argparser.add_argument('--lr', type=float, help='learning rate')
    argparser.add_argument('--lr_lin_decay', type=float,
                           help='decrease ratio over time on learning rate')
    argparser.add_argument('--lr_scale', action='store_true',
                           help='decrease per-layer learning rate' +
                           ' based on its sizes')
    argparser.add_argument('--lr_monitor_decay', action='store_true',
                           help='decrease lr when no improvement on training' +
                           ' (used only when no lr_lin_decay)')
    argparser.add_argument('--valid_stop', action='store_true',
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
        if args.embedding or args.feat_embedding or args.hidden:
            msg = "Dimensions and architecture cannot be changed "
            msg += "when loading a model for further training"
            logging.warning(msg)
    else:
        wt = SequenceTaggerNetwork(
            dataset=dataset, w2i=corpus.w2i, t2i=corpus.t2i,
            featurizer=corpus.featurizer,
            edim=args.embedding, fedim=args.feat_embedding, hdims=args.hidden,
            embedding_init=args.embedding_init)

    if args.epochs:
        wt.max_epochs = args.epochs
    if args.use_momentum:
        wt.use_momentum = args.use_momentum
    if args.lr:
        wt.lr = args.lr
    if args.lr_lin_decay:
        wt.lr_lin_decay = args.lr_lin_decay
    if args.lr_monitor_decay:
        wt.lr_monitor_decay = args.lr_monitor_decay
    if args.lr_scale:
        wt.lr_scale = args.lr_scale
    if args.valid_stop:
        wt.valid_stop = args.valid_stop
    if args.regularization:
        wt.reg_factors = args.regularization
    if args.dropout_params:
        wt.dropout_params = args.dropout_params
        wt.dropout = True
    elif args.dropout:
        wt.dropout = args.dropout

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


if __name__ == "__main__":
    main()
