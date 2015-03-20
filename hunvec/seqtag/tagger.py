import argparse

from pylearn2.utils import serial

from hunvec.utils.fscore import FScCounter
from hunvec.datasets.word_tagger_dataset import load_dataset, WordTaggerDataset  # nopep8


class CSL2L(argparse.Action):
    """ convert Comma Separated List into (2) single List"""
    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, values.split(','))


def create_argparser():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('dataset')
    argparser.add_argument('model')
    argparser.add_argument('--fscore', action='store_true',
                           help='if given, don\'t tag, only compute f1 score')
    argparser.add_argument('--sets', action=CSL2L, default=['train'],
                           help='any subset of train, test and valid, csv')
    return argparser.parse_args()


def load_and_score(args):
    wt = serial.load(args.model)
    d, c = load_dataset(args.dataset)
    wt.prepare_tagging()
    wt.f1c = FScCounter(c.i2t)
    for ds_name in args.sets:
        print list(wt.get_score(d[ds_name], 'f1'))


def main():
    args = create_argparser()
    if args.fscore:
        load_and_score(args)
    else:
        raise Exception("Tagging is not implemented yet")


if __name__ == '__main__':
    main()
