from theano import tensor
from pylearn2.utils import wraps
from pylearn2.models.mlp import Layer, Sigmoid


class HierarchicalSoftmax(Sigmoid):
    def __init__(self, vocab_size, **kwargs):
        dim = vocab_size
        super(HierarchicalSoftmax, self).__init__(dim=dim, **kwargs)

    @wraps(Layer.cost)
    def cost(self, Y, Y_hat):
        # TODO iterate through Y, where Y>=0, compute kl and multiply them
        # at the end
        zeros = tensor.eq(Y, 0)
        ones = tensor.eq(Y, 1)
        other = tensor.lt(Y, 0)
        probs = zeros * Y_hat + ones * (1 - Y_hat) + other
        row_probs = tensor.prod(probs, axis=1)
        return tensor.sum(row_probs)

        return super(HierarchicalSoftmax, self).cost(Y, Y_hat)
