import functools

import theano
import theano.tensor as T
from pylearn2.costs.cost import Cost, DefaultDataSpecsMixin


class SeqTaggerCost(DefaultDataSpecsMixin, Cost):
    supervised = True

    def expr(self, model, data, **kwargs):
        ## compute score as Collobert did
        space, source = self.get_data_specs(model)
        space.validate(data)
        seq_score, NF = self.compute_costs(model, data, **kwargs)
        return -(seq_score - NF) + T.sum(T.sqr(abs(model.A)))

    def compute_costs(self, model, data, **kwargs):
        inputs, targets = data
        tagger_out = model.fprop(inputs)

        start = model.A[0]
        end = model.A[1]
        A = model.A[2:]

        seq_score = self.cost_seq(start, end, A, tagger_out, targets)
        return seq_score, T.max(tagger_out[-1])

    def cost_seq(self, start, end, A, tagger_out, targets):
        # compute gold seq's score with using A and tagger_out
        gold_seq = targets.argmax(axis=1)
        seq_score = start[gold_seq[0]] + end[gold_seq[-1]]

        # tagger_out_scores
        tout_chooser = lambda gold_index, i: tagger_out[i][gold_index]
        tout_seq_scores, updates = theano.scan(
            fn=tout_chooser,
            sequences=[gold_seq, T.arange(gold_seq.shape[0])],
            outputs_info=None
        )
        seq_score += tout_seq_scores.sum()

        # A matrix scores
        A_chooser = lambda i, next_i: A[i][next_i]
        A_seq_scores, updates = theano.scan(
            fn=A_chooser,
            sequences=[gold_seq[:-1], gold_seq[1:]],
            outputs_info=None
        )
        seq_score += A_seq_scores.sum()

        return seq_score

    @functools.wraps(Cost.get_monitoring_channels)
    def get_monitoring_channels(self, model, data, **kwargs):
        d = Cost.get_monitoring_channels(self, model, data, **kwargs)
        costs = self.compute_costs(model, data, **kwargs)
        d['cost_seq_score'] = -costs[0]
        d['NF'] = -costs[1]

        inputs, targets = data
        out = model.fprop(inputs)

        same = lambda c, t: T.sum(T.eq(T.argmax(c), T.argmax(t)))
        notsame = lambda c, t: T.sum(T.neq(T.argmax(c), T.argmax(t)))
        o, u = theano.scan(fn=same, sequences=[out, targets],
                           outputs_info=None)
        good = T.sum(o)
        o, u = theano.scan(fn=notsame, sequences=[out, targets],
                           outputs_info=None)
        bad = T.sum(o)

        d['Prec'] = T.cast(good / (good + bad), dtype='floatX')

        return d
