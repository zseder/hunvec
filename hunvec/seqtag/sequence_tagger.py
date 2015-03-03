import functools
from itertools import izip

import numpy

import theano
import theano.tensor as T

from pylearn2.models.model import Model
from pylearn2.space import CompositeSpace
from pylearn2.utils import sharedX
from pylearn2.termination_criteria import MonitorBased, And, EpochCounter
from pylearn2.train import Train
from pylearn2.train_extensions.best_params import MonitorBasedSaveBest
from pylearn2.training_algorithms import learning_rule
from pylearn2.training_algorithms.sgd import SGD, LinearDecay
from pylearn2.training_algorithms.sgd import MonitorBasedLRAdjuster

from hunvec.seqtag.word_tagger import WordTaggerNetwork
from hunvec.cost.seq_tagger_cost import SeqTaggerCost
from hunvec.utils.viterbi import viterbi


class SequenceTaggerNetwork(Model):
    def __init__(self, hdims, edim, dataset, w2i, t2i, featurizer,
                 max_epochs=100, use_momentum=False, lr_decay=1.,
                 valid_stop=False, reg_factors=None, dropout=False):

        super(SequenceTaggerNetwork, self).__init__()

        self.vocab_size = dataset.vocab_size
        self.window_size = dataset.window_size
        self.total_feats = dataset.total_feats
        self.feat_num = dataset.feat_num
        self.n_classes = dataset.n_classes
        self.max_epochs = max_epochs

        self.w2i = w2i
        self.t2i = t2i
        self.featurizer = featurizer

        self.input_space = CompositeSpace([
            dataset.data_specs[0].components[0],
            dataset.data_specs[0].components[1],
        ])
        self.output_space = dataset.data_specs[0].components[2]

        self.input_source = ('words', 'features')
        self.target_source = 'targets'

        self.tagger = WordTaggerNetwork(self.vocab_size, self.window_size,
                                        self.total_feats, self.feat_num,
                                        hdims, edim, self.n_classes)

        A_value = numpy.random.uniform(low=-.1, high=.1,
                                       size=(self.n_classes + 2,
                                             self.n_classes))
        self.A = sharedX(A_value, name='A')
        self.prepare_tagging()
        self.use_momentum = use_momentum
        self.lr_decay = lr_decay
        self.valid_stop = valid_stop
        self.reg_factors = reg_factors

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
        d['w2i'] = self.w2i
        d['t2i'] = self.t2i
        d['featurizer'] = self.featurizer
        return d

    def fprop(self, data):
        tagger_out = self.tagger.fprop(data)
        probs = T.concatenate([self.A, tagger_out])
        return probs

    def dropout_fprop(self, data, default_input_include_prob=0.5,
                      input_include_probs=None, default_input_scale=2.0,
                      input_scales=None, per_example=True):
        tagger_out = self.tagger.dropout_fprop(
            data, default_input_include_prob, input_include_probs,
            default_input_scale, input_scales, per_example)
        probs = T.concatenate([self.A, tagger_out])
        return probs

    @functools.wraps(Model.get_lr_scalers)
    def get_lr_scalers(self):
        d = self.tagger.get_lr_scalers()
        return d

    @functools.wraps(Model.get_params)
    def get_params(self):
        return self.tagger.get_params() + [self.A]

    def create_adjustors(self):
        initial_momentum = .5
        final_momentum = .99
        start = 1
        saturate = self.max_epochs / 3
        self.momentum_adjustor = learning_rule.MomentumAdjustor(
            final_momentum, start, saturate)
        self.momentum_rule = learning_rule.Momentum(initial_momentum,
                                                    nesterov_momentum=True)

        self.learning_rate_adjustor = LinearDecay(
            start, saturate * 10000, self.lr_decay)
        self.learning_rate_adjustor = MonitorBasedLRAdjuster(
            low_trigger=1., shrink_amt=.9, channel_name='train_objective')

    def create_algorithm(self, data, save_best_path=None):
        self.dataset = data
        self.create_adjustors()
        term = EpochCounter(max_epochs=self.max_epochs)
        cost_crit = MonitorBased(channel_name='valid_objective',
                                 prop_decrease=0., N=10)
        if self.valid_stop:
            term = And(criteria=[cost_crit, term])

        #(layers, A_weight_decay)
        coeffs = None
        if self.reg_factors:
            rf = self.reg_factors
            coeffs = ([[rf, rf], rf, rf], rf)
        cost = SeqTaggerCost(coeffs)

        self.mbsb = MonitorBasedSaveBest(channel_name='valid_objective',
                                         save_path=save_best_path)

        learning_rule = (self.momentum_rule if self.use_momentum else None)
        self.algorithm = SGD(batch_size=1, learning_rate=0.01,
                             termination_criterion=term,
                             monitoring_dataset=data,
                             cost=cost,
                             learning_rule=learning_rule,
                             #update_callbacks=[self.learning_rate_adjustor],
                             )
        self.trainer = Train(dataset=data['train'], model=self,
                             algorithm=self.algorithm,
                             extensions=[self.learning_rate_adjustor])
        self.algorithm.setup(self, self.dataset['train'])

    def train(self):
        while True:
            self.algorithm.train(dataset=self.dataset['train'])
            self.monitor.report_epoch()
            self.monitor()
            self.mbsb.on_monitor(self, self.dataset['valid'], self.algorithm)
            if not self.algorithm.continue_learning(self):
                break
            if self.use_momentum:
                self.momentum_adjustor.on_monitor(self, self.dataset['valid'],
                                                  self.algorithm)
            self.learning_rate_adjustor.on_monitor(self, self.dataset['valid'],
                                                   self.algorithm)

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
            yield numpy.array([[e] for e in best_path])

    def get_score(self, dataset, mode='pwp'):
        tagged = self.tag_seq(dataset.X1, dataset.X2)
        gold = dataset.y
        good, bad = 0., 0.
        if mode == 'pwp':
            for t, g in izip(tagged, gold):
                good += sum(t == g)
                bad += sum(t != g)
            return good / (good + bad)
        elif mode == 'f1':
            return self.f1c.count_score(gold, tagged)
