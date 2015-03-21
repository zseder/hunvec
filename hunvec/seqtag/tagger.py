import argparse

from pylearn2.utils import serial


def create_argparser():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('model')
    return argparser.parse_args()


def tag(args):
    wt = serial.load(args.model)
    wt.prepare_tagging()


def main():
    args = create_argparser()
    tag(args)


if __name__ == '__main__':
    main()
