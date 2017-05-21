
import pdb
import theano
import theano.tensor as T
import numpy as np
import os
import gzip
import six.moves.cPickle as Pickle
import timeit

class LogisticRegression(object):
    def __init__(self, input, n_in, n_out):
        self.input = input
        self.W = theano.shared(
            value=np.zeros(
                (n_in, n_out),
                dtype=theano.config.floatX
            ),
            name='W',
            borrow=True
        )
        self.b = theano.shared(
            value=np.zeros(
                (n_out,),
                dtype=theano.config.floatX
            ),
            name='W',
            borrow=True
        )
        self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W)+self.b)
        self.y_pred = T.argmax(self.p_y_given_x, axis = 1)

    def negative_log_likelihood(self, y):
        res = -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])
        return res

    def errors(self, y):
        if y.ndim != self.y_pred or not y.dtype.startswith('int'):
            raise ValueError('y is wrong. please use the correct y.')
        else:
            return T.mean(T.neq(y, self.y_pred))

def load_data(): # loading the MNIST dataset.
    url = 'http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz'
    from six.moves import urllib
    datadir = os.path.join('.', 'data')
    urllib.request.urlretrieve(url, datadir)
    with gzip.open(datadir, 'rb') as f:
        train_set, valid_set, test_set = Pickle.load(f)

    def shared_dataset(data_xy, borrow=True):
        x,y=data_xy
        features = theano.shared(
            np.asarray(x, dtype=theano.config.floatX),
            borrow=borrow
        )
        labels = theano.shared(np.asarray(y, dtype=theano.config.floatX), borrow=borrow)
        return features, T.cast(labels, 'int32')

    train_set_x, train_set_y = shared_dataset(train_set)
    valid_set_x, valid_set_y = shared_dataset(valid_set)
    test_set_x, test_set_y = shared_dataset(test_set)

    return [(train_set_x, train_set_y), (valid_set_x, valid_set_y), (test_set_x, test_set_y)]


