import sys
import functools

import numpy

import theano
import theano.tensor as T

from pylearn2.models.model import Model
from pylearn2.space import CompositeSpace
from pylearn2.utils import sharedX
from pylearn2.termination_criteria import MonitorBased, And, EpochCounter
from pylearn2.train import Train
from pylearn2.train_extensions.best_params import MonitorBasedSaveBest
from pylearn2.utils import serial
from pylearn2.training_algorithms import learning_rule
from pylearn2.training_algorithms.sgd import SGD, LinearDecay

from hunvec.seqtag.word_tagger import WordTaggerNetwork
from hunvec.seqtag.word_tagger_dataset import WordTaggerDataset
from hunvec.seqtag.word_tagger_dataset import create_splitted_datasets
from hunvec.corpus.tagged_corpus import TaggedCorpus
from hunvec.feature.featurizer import Featurizer
from hunvec.cost.seq_tagger_cost import SeqTaggerCost


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

    @functools.wraps(Model.get_monitoring_channels)
    def get_monitoring_channels(self, data):
        rval = Model.get_monitoring_channels(self, data)
        rval['A_min'] = self.A[2:].min()
        rval['A_max'] = self.A[2:].max()
        rval['A_mean'] = self.A[2:].mean()
        rval['start_min'] = self.A[0].min()
        rval['start_max'] = self.A[0].max()
        rval['start_mean'] = self.A[0].mean()
        rval['end_min'] = self.A[1].min()
        rval['end_max'] = self.A[1].max()
        rval['end_mean'] = self.A[1].mean()
        rval['tagger_min'] = self.tagger.layers[2].get_params()[0].min()
        rval['tagger_max'] = self.tagger.layers[2].get_params()[0].max()
        return rval

    def create_adjustors(self):
        initial_momentum = .5
        final_momentum = .99
        start = 1
        saturate = self.max_epochs
        self.momentum_adjustor = learning_rule.MomentumAdjustor(
            final_momentum, start, saturate)
        self.momentum_rule = learning_rule.Momentum(initial_momentum,
                                                    nesterov_momentum=True)

        decay_factor = .1
        self.learning_rate_adjustor = LinearDecay(
            start, saturate * 1000, decay_factor)

    def create_algorithm(self, data, save_best_path=None):
        self.dataset = data
        self.create_adjustors()
        epoch_cnt_crit = EpochCounter(max_epochs=self.max_epochs)
        cost_crit = MonitorBased(channel_name='valid_Prec',
                                 prop_decrease=0., N=10)
        term = And(criteria=[cost_crit, epoch_cnt_crit])

        #(layers, A_weight_decay)
        coeffs = ([[5e-4, 5e-4], 5e-4, 5e-4], 5e-4)
        coeffs = None
        cost = SeqTaggerCost(coeffs)
        self.mbsb = MonitorBasedSaveBest(channel_name='valid_objective',
                                         save_path=save_best_path)
        self.algorithm = SGD(batch_size=1, learning_rate=0.01,
                             termination_criterion=epoch_cnt_crit,
                             monitoring_dataset=data,
                             cost=cost,
                             #learning_rule=self.momentum_rule,
                             #update_callbacks=[self.learning_rate_adjustor],
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
            #self.momentum_adjustor.on_monitor(self, self.dataset['valid'],
            #                                  self.algorithm)


def test_data():
    params = {
        "vocab_size": 10,
        "window_size": 3,
        "feat_num": 2,
        "hdim": 10,
        "edim": 10,
        "n_classes": 2
    }

    # two sentences, with 2 and 3 lengths
    X = ([
         [[0, 1, 2], [3, 5, 4]],
         [[0, 1, 2], [3, 5, 4], [5, 4, 0]]
         ],
         [
             [[0, 1, 0], [1, 1, 1]],
             [[0, 1, 0], [1, 1, 1], [1, 0, 1]]
         ])
    y = [
        [[0, 0]],
        [[0, 1, 1]]
    ]
    d = WordTaggerDataset(X, y,
                          vocab_size=params['vocab_size'],
                          window_size=params['window_size'],
                          feat_num=params['feat_num'],
                          n_classes=params['n_classes'])
    return d, params


def test():
    data, params = test_data()
    st = SequenceTaggerNetwork(**params)
    st.create_algorithm(data)


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
                               edim=50, hdim=200, dataset=d['train'],
                               max_epochs=300)
    return c, d, wt


def train_brown_pos():
    c, d, wt = init_brown()
    wt.create_algorithm(d, sys.argv[2])
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
    print n_words, n_classes
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
                               edim=50, hdim=100, dataset=d['train'],
                               max_epochs=300)
    return d, wt


def train_ner():
    d, wt = init_eng_ner()
    wt.create_algorithm(d, sys.argv[4])
    wt.train()


def load_and_predict():
    c, d, _ = init_brown()
    wt = serial.load(sys.argv[2])
    print d['train'].y[0].argmax(axis=1)
    cost = SeqTaggerCost()
    words = T.matrix('words', dtype='int64')
    features = T.matrix('features', dtype='int64')
    targets = T.matrix('targets', dtype='float32')
    cost_expression = cost.compute_costs(wt, ((words, features), targets))
    fn = theano.function(inputs=[words, features, targets],
                         outputs=cost_expression)
    res = fn(d['train'].X1[0], d['train'].X2[0], d['train'].y[0])
    print res[1].argmax(), res[2].argmax(axis=1), res[3].argmax()


def predict_test():
    c, d, wt = init_brown()
    #print d['train'].y[0]
    #X = wt.get_input_space().make_theano_batch()
    #Y = wt.fprop(X)
    #f = theano.function([X[0], X[1]], Y)
    #y = f(d['train'].X1[0], d['train'].X2[0])
    #print y, y.shape
    #print (d['train'].y[0] * y).sum()
    #print len(d['train'].y[0])
    cost = SeqTaggerCost()
    words = T.matrix('words', dtype='int64')
    features = T.matrix('features', dtype='int64')
    targets = T.matrix('targets', dtype='float32')
    cost_expression = cost.expr(wt, ((words, features), targets))
    fn = theano.function(inputs=[words, features, targets],
                         outputs=cost_expression)
    print fn(d['train'].X1[0], d['train'].X2[0], d['train'].y[0])


if __name__ == "__main__":
    #predict_test()
    train_brown_pos()
    #load_and_predict()
    #train_ner()
