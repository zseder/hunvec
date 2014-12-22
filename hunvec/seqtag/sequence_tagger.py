import numpy
import functools

import theano

from pylearn2.models.model import Model
from pylearn2.space import CompositeSpace, IndexSequenceSpace
from pylearn2.space import VectorSequenceSpace
from pylearn2.utils import sharedX
from pylearn2.costs.cost import Cost, DefaultDataSpecsMixin
from pylearn2.termination_criteria import EpochCounter
from pylearn2.training_algorithms.sgd import SGD
from pylearn2.train import Train

from word_tagger import WordTaggerNetwork
from word_tagger_dataset import WordTaggerDataset


class SeqTaggerCost(DefaultDataSpecsMixin, Cost):
    # Here it is assumed that we are doing supervised learning
    supervised = True

    def expr(self, model, data, **kwargs):
        space, source = self.get_data_specs(model)
        space.validate(data)

        inputs, targets = data
        outputs = model.fprop(inputs)
        # TODO word-wise softmax cost should come here
        return theano.tensor.mean(theano.tensor.sqr(targets - outputs))


class SequenceTaggerNetwork(Model):
    def __init__(self, vocab_size, window_size, feat_num, hdim, edim,
                 n_classes, max_epochs=100):

        super(SequenceTaggerNetwork, self).__init__()

        self.vocab_size = vocab_size
        self.window_size = window_size
        self.feat_num = feat_num
        self.n_classes = n_classes
        self.max_epochs = max_epochs

        self.input_space = CompositeSpace([
            IndexSequenceSpace(max_labels=vocab_size, dim=window_size),
            IndexSequenceSpace(max_labels=feat_num, dim=window_size)
        ])
        self.output_space = VectorSequenceSpace(dim=n_classes)

        self.input_source = ('words', 'features')
        self.target_source = 'targets'

        self.tagger = WordTaggerNetwork(vocab_size, window_size, feat_num,
                                        hdim, edim, n_classes)

        # start is included as 0th row to make T.scan easy
        A_value = numpy.random.uniform(low=-.1, high=.1,
                                       size=(self.n_classes + 1,
                                             self.n_classes))
        self.A = sharedX(A_value, name='A')
        end = numpy.random.uniform(low=-.1, high=.1, size=self.n_classes)
        self.end = sharedX(end, name='start')

    def fprop(self, data):
        tagger_out = self.tagger.fprop(data)

        # since we put starting probabilities into A[0,:], we only need
        # a mask, so iteration is easier
        start_mask = numpy.zeros(self.n_classes + 1, dtype=numpy.float32)
        start_mask[0] = 1.0

        # tout: tagger output
        fn = lambda res_probs, tout: theano.dot(res_probs * self.A, tout)
        (probs_after_A, updates) = theano.scan(
            fn=fn,
            sequences=[tagger_out],
            outputs_info=start_mask
        )

        # use $ transitions
        probs_after_A[-1].dot(self.end)

        # run a theano.nnet.softmax on every row of probs_after_A
        # TODO maybe a cost function would need this?
        res = theano.tensor.nnet.softmax(probs_after_A)

        return res

    @functools.wraps(Model.get_params)
    def get_params(self):
        return self.tagger.get_params() + [self.A]

    @functools.wraps(Model.get_monitoring_channels)
    def get_monitoring_channels(self, data):
        rval = Model.get_monitoring_channels(self, data)
        # TODO add own channels
        return rval

    def create_algorithm(self, data, save_best_path=None):
        self.dataset = data
        epoch_cnt_crit = EpochCounter(max_epochs=self.max_epochs)
        algorithm = SGD(batch_size=1, learning_rate=.1,
                        termination_criterion=epoch_cnt_crit,
                        monitoring_dataset=data,
                        cost=SeqTaggerCost(),
                        )
        self.trainer = Train(dataset=data, model=self,
                             algorithm=algorithm)


def test_data():
    params = {
        "vocab_size": 10,
        "window_size": 3,
        "feat_num": 2,
        "hdim": 10,
        "edim": 10,
        "n_classes": 2
    }
    X = [numpy.array([[0, 1, 2], [3, 5, 4]], dtype=numpy.int32),
         numpy.array([[0, 1, 0], [1, 1, 1]], dtype=numpy.int32)]
    y = numpy.array([[0], [0]], dtype=numpy.int32)
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
    st.trainer.main_loop()


if __name__ == "__main__":
    test()
