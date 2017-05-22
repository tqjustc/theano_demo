
import pdb
import theano
import theano.tensor as T
import gzip
import numpy as np

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
