
import pdb
import theano
import theano.tensor as T
import gzip
import numpy as np
import os

class LogisticRegression(object):
    def __init__(self, input, n_in, n_out):
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
            name='b',
            borrow=True
        )
        self.input = input
        self.params = [self.W, self.b]
        self.p_y_given_x = T.nnet.softmax(T.dot(self.input, self.W) + self.b)
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)
    def negative_log_likelihood(self, y):
        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])

    def cost(self, y):
        if y.ndim != self.y_pred or not y.dtype.startswith('int'):
            raise ValueError('Wrong input y.')
        else:
            return T.mean(T.neq(y, self.y_pred))

def load_data():
    url1 = 'http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz'
    from six.moves import urllib
    import six.moves.cPickle as Pickle

    dataset = os.getcwd() + '/data/'
    urllib.request.urlretrieve(url1, dataset)
    with gzip.open(dataset, 'rb') as f:
        train_set, valid_set, test_set = Pickle.load(f)

    def shared_dataset(data_xy, borrow=True):
        x, y = data_xy
        xx = theano.shared(
            value=np.asarray(x, dtype=theano.config.floatX),
            name='xx',
            borrow=borrow
        )
        yy = theano.shared(
            value=np.asarray(y, dtype=theano.config.floatX),
            name='yy',
            borrow=borrow
        )
        return xx, yy
    train_set_x, train_set_y = shared_dataset(train_set)
    valid_set_x, valid_set_y = shared_dataset(valid_set)
    test_set_x, test_set_y = shared_dataset(test_set)

    return [(train_set_x, train_set_y), (valid_set_x, valid_set_y), (test_set_x, test_set_y)]
