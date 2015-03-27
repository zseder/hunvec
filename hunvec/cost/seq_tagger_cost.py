import functools

import theano
import theano.tensor as T
from pylearn2.costs.cost import Cost, DefaultDataSpecsMixin


class SeqTaggerCost(DefaultDataSpecsMixin, Cost):
    supervised = True

    def __init__(self, regularization_costs=None, dropout=False):
        super(SeqTaggerCost, self).__init__()
        self.reg = regularization_costs
        self.dropout = dropout

    def expr(self, model, data, **kwargs):
        ## compute score as Collobert did
        space, source = self.get_data_specs(model)
        space.validate(data)
        seq_score, _, _, end = self.compute_costs(model, data, **kwargs)
        NF = T.max(end)
        sc = -(seq_score - NF)
        if self.reg is not None:
            sc += model.tagger.get_weight_decay(self.reg[0])
            sc += self.reg[1] * T.sum(T.sqr(abs(model.A)))
        return sc

    def compute_costs(self, model, data, **kwargs):
        inputs, targets = data
        if not self.dropout:
            outputs = model.fprop(inputs)
        else:
            outputs = model.dropout_fprop(inputs)

        # unpack A and tagger_out from outputs
        start = outputs[0]
        end = outputs[1]
        A = outputs[2:2 + model.n_classes]
        tagger_out = outputs[2 + model.n_classes:]

        seq_score = self.cost_seq(start, end, A, tagger_out, targets)
        start_M, combined, end_M = self.combined_scores(
            start, end, A, tagger_out)
        return seq_score, start_M, combined, end_M

    def cost_seq(self, start, end, A, tagger_out, targets):
        # compute gold seq's score with using A and tagger_out
        gold_seq = targets.flatten()
        seq_score = start[gold_seq[0]] + end[gold_seq[-1]]

        # tagger_out_scores
        tout_chooser = lambda gold_i, i, tagger_out: tagger_out[i][gold_i]
        tout_seq_scores, updates = theano.scan(
            fn=tout_chooser,
            sequences=[gold_seq, T.arange(gold_seq.shape[0])],
            non_sequences=[tagger_out],
            outputs_info=None
        )
        seq_score += tout_seq_scores.sum()

        # A matrix scores
        A_chooser = lambda i, next_i, A: A[i][next_i]
        A_seq_scores, updates = theano.scan(
            fn=A_chooser,
            sequences=[gold_seq[:-1], gold_seq[1:]],
            non_sequences=[A],
            outputs_info=None
        )
        seq_score += A_seq_scores.sum()

        return seq_score

    def combine_A_tout_scanner(self, tagger_out, prev_res, A):
        # create a new matrix from A, add prev_res to every column
        A_t_ = A.dimshuffle((1, 0))
        A_t = A_t_ + prev_res

        # original logadd from Collobert
        #log_added = T.log(T.exp(A_t).sum(axis=1))

        # dummy approximation
        log_added = A_t.max(axis=1)

        # more sophisticated approximation
        # TODO where does it come from?
        #log_added = A_t.max(axis=1) + T.log((A_t + 1e-4 - A_t.max(axis=1)).sum(axis=1))

        new_res = log_added + tagger_out
        return new_res

    def combined_scores(self, start, end, A, tagger_out):
        # compute normalizer factor NF for this given training data
        start_M = tagger_out[0] + start
        combined_probs, updates = theano.scan(
            fn=self.combine_A_tout_scanner,
            sequences=[tagger_out[1:]],
            non_sequences=[A],
            outputs_info=[start_M]
        )

        end_M = combined_probs[-1] + end
        return start_M, combined_probs[:-1], end_M

    @functools.wraps(Cost.get_monitoring_channels)
    def get_monitoring_channels(self, model, data, **kwargs):
        d = Cost.get_monitoring_channels(self, model, data)
        if self.dropout:
            self.dropout = False
            d['nodrop_obj'] = self.expr(model, data, **kwargs)
            self.dropout = True
        return d
