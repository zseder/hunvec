import sys
import functools
from itertools import izip

import numpy

import theano
import theano.tensor as T

from pylearn2.models.model import Model
from pylearn2.space import CompositeSpace
from pylearn2.utils import sharedX, serial
from pylearn2.termination_criteria import MonitorBased, And, EpochCounter
from pylearn2.train import Train
from pylearn2.train_extensions.best_params import MonitorBasedSaveBest
from pylearn2.training_algorithms import learning_rule
from pylearn2.training_algorithms.sgd import SGD, LinearDecay

from hunvec.seqtag.word_tagger import WordTaggerNetwork
from hunvec.seqtag.word_tagger_dataset import WordTaggerDataset
from hunvec.seqtag.word_tagger_dataset import create_splitted_datasets
from hunvec.corpus.tagged_corpus import TaggedCorpus
from hunvec.feature.featurizer import Featurizer
from hunvec.cost.seq_tagger_cost import SeqTaggerCost
from hunvec.utils.viterbi import viterbi
from hunvec.utils.fscore import FScCounter


class SequenceTaggerNetwork(Model):
    def __init__(self, vocab_size, window_size, total_feats, feat_num,
                 hdim, edim, n_classes, dataset, max_epochs=100):

        super(SequenceTaggerNetwork, self).__init__()

        self.vocab_size = vocab_size
        self.window_size = window_size
        self.total_feats = total_feats
        self.feat_num = feat_num
        self.n_classes = n_classes
        self.max_epochs = max_epochs

        self.input_space = CompositeSpace([
            dataset.data_specs[0].components[0],
            dataset.data_specs[0].components[1],
        ])
        self.output_space = dataset.data_specs[0].components[2]

        self.input_source = ('words', 'features')
        self.target_source = 'targets'

        self.tagger = WordTaggerNetwork(vocab_size, window_size,
                                        self.total_feats, self.feat_num,
                                        hdim, edim, n_classes)

        A_value = numpy.random.uniform(low=-.1, high=.1,
                                       size=(self.n_classes + 2,
                                             self.n_classes))
        self.A = sharedX(A_value, name='A')
        self.prepare_tagging()

    def __getstate__(self):
        d = {}
        d['vocab_size'] = self.vocab_size
        d['window_size'] = self.window_size
        d['feat_num'] = self.feat_num
        d['n_classes'] = self.n_classes
        d['max_epochs'] = self.max_epochs
        d['input_space'] = self.input_space
        d['output_space'] = self.output_space
        d['input_source'] = self.input_source
        d['target_source'] = self.target_source
        d['A'] = self.A
        d['tagger'] = self.tagger
        return d

    def fprop(self, data):
        tagger_out = self.tagger.fprop(data)
        probs = T.concatenate([self.A, tagger_out])
        return probs

    @functools.wraps(Model.get_lr_scalers)
    def get_lr_scalers(self):
        d = self.tagger.get_lr_scalers()
        return d

    @functools.wraps(Model.get_params)
    def get_params(self):
        return self.tagger.get_params() + [self.A]

#    @functools.wraps(Model.get_monitoring_channels)
#    def get_monitoring_channels(self, data):
#        rval = Model.get_monitoring_channels(self, data)
#        return rval

    def create_adjustors(self):
        initial_momentum = .5
        final_momentum = .99
        start = 1
        saturate = self.max_epochs / 3
        self.momentum_adjustor = learning_rule.MomentumAdjustor(
            final_momentum, start, saturate)
        self.momentum_rule = learning_rule.Momentum(initial_momentum,
                                                    nesterov_momentum=True)

        decay_factor = .1
        self.learning_rate_adjustor = LinearDecay(
            start, saturate * 10000, decay_factor)

    def create_algorithm(self, data, save_best_path=None):
        self.dataset = data
        self.create_adjustors()
        epoch_cnt_crit = EpochCounter(max_epochs=self.max_epochs)
        cost_crit = MonitorBased(channel_name='valid_Prec',
                                 prop_decrease=0., N=10)
        term = And(criteria=[cost_crit, epoch_cnt_crit])

        #(layers, A_weight_decay)
        coeffs = ([[1e-4, 1e-4], 1e-4, 1e-4], 1e-4)
        coeffs = None
        cost = SeqTaggerCost(coeffs)
        self.mbsb = MonitorBasedSaveBest(channel_name='valid_objective',
                                         save_path=save_best_path)
        self.algorithm = SGD(batch_size=1, learning_rate=0.05,
                             termination_criterion=epoch_cnt_crit,
                             monitoring_dataset=data,
                             cost=cost,
                             learning_rule=self.momentum_rule,
                             update_callbacks=[self.learning_rate_adjustor],
                             )
        self.trainer = Train(dataset=data['train'], model=self,
                             algorithm=self.algorithm)
        self.algorithm.setup(self, self.dataset['train'])

    def train(self):
        while True:
            self.algorithm.train(dataset=self.dataset['train'])
            self.monitor.report_epoch()
            self.monitor()
            self.mbsb.on_monitor(self, self.dataset['valid'], self.algorithm)
            if not self.algorithm.continue_learning(self):
                break
            self.momentum_adjustor.on_monitor(self, self.dataset['valid'],
                                              self.algorithm)
            print self.get_f1(self.dataset['valid'])

    def prepare_tagging(self):
        X = self.get_input_space().make_theano_batch()
        Y = self.fprop(X)
        self.f = theano.function([X[0], X[1]], Y)

    def tag_seq(self, words, features):
        start = self.A.get_value()[0]
        end = self.A.get_value()[1]
        A = self.A.get_value()[2:]

        for words_, feats_ in izip(words, features):
            y = self.f(words_, feats_)
            tagger_out = y[2 + self.n_classes:]
            _, best_path = viterbi(start, A, end, tagger_out, self.n_classes)
            yield best_path

    def get_f1(self, dataset):
        tagged = self.tag_seq(dataset.X1, dataset.X2)
        gold = dataset.y
        good, bad = 0., 0.
        for t, g in izip(tagged, gold):
            t_m = numpy.array(t)
            g_m = g.argmax(axis=1)
            good += sum(t_m == g_m)
            bad += sum(t_m != g_m)
        return good / (good + bad)


def init_brown():
    fn = sys.argv[1]
    ws = 6
    featurizer = Featurizer()
    c = TaggedCorpus(fn, featurizer)
    res = WordTaggerDataset.create_from_tagged_corpus(c, window_size=ws)
    words, feats, y, vocab, classes = res
    n_words, n_classes = len(vocab), len(classes)
    d = create_splitted_datasets(words, feats, y, [.8, .1, .1], n_words,
                                 ws, featurizer.total, featurizer.feat_num,
                                 n_classes)
    wt = SequenceTaggerNetwork(vocab_size=d['train'].vocab_size,
                               window_size=d['train'].window_size,
                               total_feats=d['train'].total_feats,
                               feat_num=d['train'].feat_num,
                               n_classes=d['train'].n_classes,
                               edim=50, hdim=300, dataset=d['train'],
                               max_epochs=300)
    return c, d, wt


def train_brown_pos():
    #c, d, wt = init_brown()
    #wt.create_algorithm(d, sys.argv[2])
    d, wt, _, _, _ = init_eng_ner()
    wt.create_algorithm(d, sys.argv[4])
    wt.train()


def init_eng_ner():
    train_fn = sys.argv[1]
    valid_fn = sys.argv[2]
    test_fn = sys.argv[3]
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
    wt = SequenceTaggerNetwork(vocab_size=d['train'].vocab_size,
                               window_size=d['train'].window_size,
                               total_feats=d['train'].total_feats,
                               feat_num=d['train'].feat_num,
                               n_classes=d['train'].n_classes,
                               edim=50, hdim=300, dataset=d['train'],
                               max_epochs=300)
    return d, wt, train_c, valid_c, test_c


def train_ner():
    d, wt, _, _, _ = init_eng_ner()
    wt.create_algorithm(d, sys.argv[4])
    wt.train()


def load_and_predict_pos():
    d, _, train_c, _, _ = init_eng_ner()
    d = d['train']
    wt = serial.load(sys.argv[4])
    print d.y[0].argmax(axis=1)
    wt.prepare_tagging()
    print list(wt.tag_seq(d.X1[:1], d.X2[:1]))


def load_and_predict():
    d, _, train_c, _, _ = init_eng_ner()
    d = d['train']
    wt = serial.load(sys.argv[4])
    fsc = FScCounter(train_c.i2t)
    golds = d.y
    #print d.y[0].argmax(axis=1)
    wt.prepare_tagging()
    #print list(wt.tag_seq(d.X1[:1], d.X2[:1]))
    for sc in fsc.count_score(golds, wt.tag_seq(d.X1, d.X2)):
        print sc


if __name__ == "__main__":
    #predict_test()
    train_brown_pos()
    #load_and_predict_pos()
    #train_ner()
