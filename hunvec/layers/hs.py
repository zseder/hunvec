from pylearn2.models.mlp import Sigmoid


class HierarchicalSoftmax(Sigmoid):
    def __init__(self, vocab_size, **kwargs):
        # TODO set dim based on vocab size
        super(HierarchicalSoftmax, self).__init__(**kwargs)
        pass

    def cost_XXX(self, Y, Y_hat):
        # TODO iterate through Y, where Y==1, compute kl and multiply them
        # at the end
        pass
