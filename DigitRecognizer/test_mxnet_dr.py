"""
Train mnist, see more explanation at http://mxnet.io/tutorials/python/mnist.html
"""
import os
import argparse
import logging
logging.basicConfig(level=logging.DEBUG)
import mxnet as mx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def read_data():
    """
    read data into numpy
    """
    valid_df = pd.read_csv("test.csv")
    x_valid = valid_df.values.astype('float32')

    return x_valid


def to4d(img):
    """
    reshape to 4D arrays
    """
    return img.reshape(img.shape[0], 1, 28, 28).astype(np.float32)/255


if __name__ == '__main__':
    x_valid = read_data()
    x_valid = to4d(x_valid)
    valid_iter = mx.io.NDArrayIter(x_valid)

    pre_fix = 'models/lenet/'
    iteration = 20
    sym, arg_params, aux_params = mx.model.load_checkpoint(pre_fix, iteration)
    model_load = mx.mod.Module(symbol = sym)
    model_load.bind(data_shapes = [('data', (1, 1, 28, 28))])
    model_load.set_params(arg_params, aux_params)

    num_batch = 12
    prob = model_load.predict(valid_iter).asnumpy()
    print len(prob)
    length = len(prob)
    ImageId = np.arange(length) + 1
    Label = [prob[i].argmax() for i in range(length)]

    for i in range(num_batch):
        # assert max(prob) > 0.99, "Low prediction accuracy."
        print 'Classified as %d with probability %f' % (prob[i].argmax(), max(prob[i]))

    save = pd.DataFrame({'ImageId': ImageId, 'Label': Label})
    save.to_csv('submission.csv', columns=['ImageId', 'Label'], index=False, header=False)
