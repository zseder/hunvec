import sys

import theano
import numpy

from pylearn2.utils import serial

inputs = numpy.array([[0, 1, 2]])

model_path = sys.argv[1]
model = serial.load(model_path)
print model.fprop(theano.shared(inputs, name='inputs')).eval()
