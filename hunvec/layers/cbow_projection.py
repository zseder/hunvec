from pylearn2.models.mlp import Layer
from pylearn2.sandbox.nlp.models.mlp import ProjectionLayer
from pylearn2.utils import wraps


class CBowProjectionLayer(ProjectionLayer):

    @wraps(Layer.fprop)
    def fprop(self, state_below):
        z = ProjectionLayer.fprop(self, state_below)
        bs = z.shape[0]
        d = self.dim
        words = z.shape[1] / d
        return z.reshape((bs, words, d)).mean(axis=0)
