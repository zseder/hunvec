import theano
from theano import tensor

from pylearn2.utils import wraps
from pylearn2.models.mlp import Layer, Sigmoid


class HierarchicalSoftmax(Sigmoid):
    def __init__(self, vocab_size, **kwargs):
        dim = vocab_size
        super(HierarchicalSoftmax, self).__init__(dim=dim,
                                                  **kwargs)

    @wraps(Layer.cost)
    def cost(self, Y, Y_hat):
        zeros = tensor.eq(Y, 0)
        ones = tensor.eq(Y, 1)
        probs = zeros * Y_hat + ones * (1 - Y_hat)
        result, _ = theano.scan(fn=lambda vec: -tensor.sum(
            tensor.log2(vec.nonzero_values())),
            outputs_info=None,
            sequences=probs)
        return result.mean()

    @wraps(Layer.get_layer_monitoring_channels)
    def get_layer_monitoring_channels(self, state_below=None, state=None,
                                      target=None):

        rval = Sigmoid.get_layer_monitoring_channels(self, state_below, state,
                                                     target)

        if target is not None:
            rval['nll'] = self.cost(Y_hat=state, Y=target)
            rval['ppl'] = 2 ** (rval['nll'])

        return rval
