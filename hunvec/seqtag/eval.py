import argparse

from pylearn2.utils import serial

from hunvec.datasets.prepare import load_dataset


class CSL2L(argparse.Action):
    """ convert Comma Separated List into (2) single List"""
    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, values.split(','))


def create_argparser():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('dataset')
    argparser.add_argument('model')
    argparser.add_argument('--fscore', action='store_true',
                           help='if given, compute f1 score')
    argparser.add_argument('--precision', action='store_true',
                           help='if given, compute per word precision')
    argparser.add_argument('--sets', action=CSL2L, default=['test'],
                           help='any subset of train, test and valid, csv')
    return argparser.parse_args()


def load_and_score(args):
    wt = serial.load(args.model)
    d, c = load_dataset(args.dataset)
    if not (args.fscore or args.precision):
        print 'needed one of the arguments: fscore, precision'
    if args.fscore:
        for ds_name in args.sets:
            print list(wt.get_score(d[ds_name], 'f1'))
    if args.precision:
        for ds_name in args.sets:
            print list(wt.get_score(d[ds_name], 'pwp'))


def main():
    args = create_argparser()
    load_and_score(args)


if __name__ == '__main__':
    main()
