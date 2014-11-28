from pylearn2.models.mlp import Layer
from pylearn2.sandbox.nlp.models.mlp import ProjectionLayer
from pylearn2.utils import wraps
from pylearn2.space import VectorSpace


class CBowProjectionLayer(ProjectionLayer):
    @wraps(Layer.set_input_space)
    def set_input_space(self, space):
        ProjectionLayer.set_input_space(self, space)
        self.output_space = VectorSpace(self.input_dim)

    @wraps(Layer.fprop)
    def fprop(self, state_below):
        z = ProjectionLayer.fprop(self, state_below)
        bs = z.shape[0]
        d = self.dim
        words = z.shape[1] / d
        return z.reshape((bs, words, d)).mean(axis=0)
