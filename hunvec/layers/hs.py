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
        other = tensor.lt(Y, 0)
        probs = zeros * Y_hat + ones * (1 - Y_hat) + other
        row_probs = tensor.prod(probs, axis=1)
        return tensor.sum(row_probs)

    @wraps(Layer.get_layer_monitoring_channels)
    def get_layer_monitoring_channels(self, state_below=None, state=None,
                                      target=None):

        rval = Sigmoid.get_layer_monitoring_channels(self, state_below, state,
                                                     target)

        if target is not None:
            rval['nll'] = self.cost(Y_hat=state, Y=target)
            rval['ppl'] = 2 ** (rval['nll'] / tensor.log(2))

        return rval
