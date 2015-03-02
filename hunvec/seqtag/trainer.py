import argparse

from hunvec.seqtag.word_tagger_dataset import WordTaggerDataset
from hunvec.seqtag.word_tagger_dataset import create_splitted_datasets
from hunvec.corpus.tagged_corpus import TaggedCorpus
from hunvec.feature.featurizer import Featurizer
from hunvec.seqtag.sequence_tagger import SequenceTaggerNetwork


def create_argparser():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('train_file')
    argparser.add_argument('model_path')
    argparser.add_argument('--test_file')
    argparser.add_argument('--valid_file')
    argparser.add_argument('--train_split', default=[0.8, 0.1, 0.1],
                           help='train/test/valid ratios, used when only' +
                           ' training file is given')
    argparser.add_argument('-w', '--window', default=5, type=int,
                           dest='window')
    argparser.add_argument('--hidden', default=100, type=int)
    argparser.add_argument('--embedding', default=50, type=int)
    argparser.add_argument('--epochs', default=50, type=int)
    argparser.add_argument('--regularization', default=.0, type=float,
                           help='typical values are 1e-5, 1e-4')
    argparser.add_argument('--use_momentum', action='store_true'),
    argparser.add_argument('--lr_decay', default=1.0, type=float,
                           help='decrease ratio over time on learning rate')
    argparser.add_argument('--valid_stop', type=bool, default=True,
                           help='don\'t use valid data to decide when to stop')
    argparser.add_argument('--dropout', action='store_true',
                           help='use dropout on inner network')
    return argparser.parse_args()


def init_network(args, dataset, corpus):
    wt = SequenceTaggerNetwork(vocab_size=dataset.vocab_size,
                               window_size=dataset.window_size,
                               total_feats=dataset.total_feats,
                               feat_num=dataset.feat_num,
                               n_classes=dataset.n_classes,
                               edim=args.embedding, hdim=args.hidden,
                               dataset=dataset,
                               w2i=corpus.w2i, t2i=corpus.t2i,
                               featurizer=corpus.featurizer,
                               max_epochs=args.epochs,
                               use_momentum=args.use_momentum,
                               lr_decay=args.lr_decay,
                               valid_stop=args.valid_stop,
                               reg_factors=args.regularization,
                               dropout=args.dropout
                               )
    return wt


def init_network_corpus(args):
    ws = args.window + 1
    featurizer = Featurizer()
    c = TaggedCorpus(args.train_file, featurizer)
    res = WordTaggerDataset.create_from_tagged_corpus(c, window_size=ws)
    words, feats, y, vocab, classes = res
    n_words, n_classes = len(vocab), len(classes)
    d = create_splitted_datasets(words, feats, y, args.train_split, n_words,
                                 ws, featurizer.total, featurizer.feat_num,
                                 n_classes)
    wt = init_network(args, d['train'], c)
    return c, d, wt


def init_network_presplitted_corpus(args):
    train_fn = args.train_file
    valid_fn = args.valid_file
    test_fn = args.test_file
    ws = 6
    featurizer = Featurizer()
    train_c = TaggedCorpus(train_fn, featurizer)
    valid_c = TaggedCorpus(valid_fn, featurizer, w2i=train_c.w2i,
                           t2i=train_c.t2i)
    test_c = TaggedCorpus(test_fn, featurizer, w2i=valid_c.w2i,
                          t2i=valid_c.t2i)
    train_res = WordTaggerDataset.create_from_tagged_corpus(
        train_c, window_size=ws)
    valid_res = WordTaggerDataset.create_from_tagged_corpus(
        valid_c, window_size=ws)
    test_res = WordTaggerDataset.create_from_tagged_corpus(
        test_c, window_size=ws)
    words, feats, y, _, _ = train_res
    n_words = len(train_res[3] | test_res[3] | valid_res[3])
    n_classes = len(train_res[4] | test_res[4] | valid_res[4])
    train_ds = WordTaggerDataset((words, feats), y, n_words, ws,
                                 featurizer.total, featurizer.feat_num,
                                 n_classes)
    words, feats, y, _, _ = valid_res
    valid_ds = WordTaggerDataset((words, feats), y, n_words, ws,
                                 featurizer.total, featurizer.feat_num,
                                 n_classes)
    words, feats, y, _, _ = test_res
    test_ds = WordTaggerDataset((words, feats), y, n_words, ws,
                                featurizer.total, featurizer.feat_num,
                                n_classes)
    d = {'train': train_ds, 'valid': valid_ds, 'test': test_ds}
    wt = init_network(args, d['train'], train_c)
    return d, wt, train_c, valid_c, test_c


def main():
    args = create_argparser()
    if args.test_file and args.valid_file:
        res = init_network_presplitted_corpus(args)
    else:
        # use train_split option
        res = init_network_corpus(args)
    d = res[0]
    wt = res[1]
    wt.create_algorithm(d, args.model_path)
    wt.train()


if __name__ == "__main__":
    main()
