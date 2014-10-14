import sys

import theano
import numpy

from pylearn2.utils import serial

inputs = numpy.array([[0, 1, 2], [1, 2, 3], [2, 3, 4], [111, 26, 41]])

model_path = sys.argv[1]
model = serial.load(model_path)
results = model.fprop(theano.shared(inputs, name='inputs')).eval()
for i in xrange(len(results)):
    input_ = inputs[i, :]
    res = results[i]
    print input_
    sorted_indices = res.argsort()[-5:]
    print sorted_indices
