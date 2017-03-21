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

def get_symbol_mlp(num_classes=10, **kwargs):
    data = mx.symbol.Variable('data')
    data = mx.sym.Flatten(data=data)
    fc1  = mx.symbol.FullyConnected(data = data, name='fc1', num_hidden=128)
    act1 = mx.symbol.Activation(data = fc1, name='relu1', act_type="relu")
    fc2  = mx.symbol.FullyConnected(data = act1, name = 'fc2', num_hidden = 64)
    act2 = mx.symbol.Activation(data = fc2, name='relu2', act_type="relu")
    fc3  = mx.symbol.FullyConnected(data = act2, name='fc3', num_hidden=num_classes)
    mlp  = mx.symbol.SoftmaxOutput(data = fc3, name = 'softmax')    
    return mlp

def get_symbol_lenet(num_classes=10, add_stn=False, **kwargs):
    data = mx.symbol.Variable('data')
    if(add_stn):
        data = mx.sym.SpatialTransformer(data=data, loc=get_loc(data), target_shape = (28,28), 
                                         transform_type="affine", sampler_type="bilinear")
    # first conv
    conv1 = mx.symbol.Convolution(data=data, kernel=(5,5), num_filter=20)
    tanh1 = mx.symbol.Activation(data=conv1, act_type="tanh")
    pool1 = mx.symbol.Pooling(data=tanh1, pool_type="max", kernel=(2,2), stride=(2,2))

    # second conv
    conv2 = mx.symbol.Convolution(data=pool1, kernel=(5,5), num_filter=50)
    tanh2 = mx.symbol.Activation(data=conv2, act_type="tanh")
    pool2 = mx.symbol.Pooling(data=tanh2, pool_type="max", kernel=(2,2), stride=(2,2))
    
    # first fullc
    flatten = mx.symbol.Flatten(data=pool2)
    fc1 = mx.symbol.FullyConnected(data=flatten, num_hidden=500)
    tanh3 = mx.symbol.Activation(data=fc1, act_type="tanh")
    
    # second fullc
    fc2 = mx.symbol.FullyConnected(data=tanh3, num_hidden=num_classes)

    # loss
    lenet = mx.symbol.SoftmaxOutput(data=fc2, name='softmax')

    return lenet

def save_model(model_prefix, rank=0):
    if model_prefix is None:
        return None
    dst_dir = os.path.dirname(model_prefix)
    if not os.path.isdir(dst_dir):
        os.mkdir(dst_dir)

    return mx.callback.do_checkpoint(model_prefix if rank == 0 else "%s-%d" % (model_prefix, rank))

def read_data():
    """
    read data into numpy
    """
    train_df = pd.read_csv("train.csv")
    valid_df = pd.read_csv("test.csv")
    x_train = train_df.drop(['label'], axis=1).values.astype('float32')
    Y_train = train_df['label'].values
    x_valid = valid_df.values.astype('float32')

    return (x_train, Y_train, x_valid)


def to4d(img):
    """
    reshape to 4D arrays
    """
    return img.reshape(img.shape[0], 1, 28, 28).astype(np.float32)/255


if __name__ == '__main__':
    network = get_symbol_lenet()
    x_train, Y_train, x_valid = read_data()
    x_train = to4d(x_train)
    x_valid = to4d(x_valid)
    batch_size = 64
    train_iter = mx.io.NDArrayIter(x_train, Y_train, batch_size, shuffle=True)
    valid_iter = mx.io.NDArrayIter(x_valid, batch_size = batch_size)
    # create model
    model = mx.mod.Module(context = mx.gpu(0), symbol = network)
    initializer = mx.init.Xavier(rnd_type='gaussian', factor_type="in", magnitude=2)
    # callbacks that run after each batch
    batch_end_callbacks = [mx.callback.Speedometer(64, 100)]

    # save model
    checkpoint = save_model('models/lenet/')
    model.fit(train_iter,
              num_epoch          = 20,
              initializer        = initializer,
              batch_end_callback = batch_end_callbacks,
              epoch_end_callback = checkpoint,
              allow_missing      = True)
    print '******* train done ***************'
    # prob = model.predict(x_train[4, 0, :, :])[0]
    num_batch = 12
    prob = model.predict(valid_iter, num_batch = num_batch).asnumpy()
    for i in range(num_batch):
        # assert max(prob) > 0.99, "Low prediction accuracy."
        print 'Classified as %d with probability %f' % (prob[i].argmax(), max(prob[i]))
