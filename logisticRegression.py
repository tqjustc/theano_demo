
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

def sgd_optimization_mnist(learning_rate=0.13, n_epochs = 1000,
                           batch_size=600):
    dataset = load_data()
    train_set_x, train_set_y = dataset[0]
    valid_set_x, valid_set_y = dataset[1]
    test_set_x, test_set_y = dataset[2]

    n_train_batches = train_set_x.shape[0] / batch_size
    n_test_batches = test_set_x.shape[0] / batch_size
    n_valid_batches = valid_set_x.shape[0] / batch_size

    print('... build models')

    index = T.lscalar()

    x = T.matrix('x')
    y = T.ivector('y')

    classifier = LogisticRegression(x, 28*28, 10)

    test_model = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens={
            x: test_set_x[index*batch_size:(index+1)*batch_size],
            y: test_set_y[index*batch_size:(index+1)*batch_size]
        }
    )

    validate_model = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens={
            x: valid_set_x[index * batch_size:(index + 1) * batch_size],
            y: valid_set_y[index * batch_size:(index + 1) * batch_size]
        }
    )

    cost = classifier.negative_log_likelihood(y)

    gW = T.grad(cost, wrt=classifier.W)
    gb = T.grad(cost, wrt=classifier.b)

    updates = [
        (classifier.W, classifier.W - learning_rate*gW),
        (classifier.b, classifier.b - learning_rate*gb)
    ]

    train_model = theano.function(
        inputs=[index],
        outputs=cost,
        updates=updates,
        givens={
            x: train_set_x[index*batch_size:(index+1)*batch_size],
            y: train_set_y[index*batch_size:(index+1)*batch_size]
        }
    )

    print('... model training.')

    patience = 5000
    patience_increase = 2
    improvement_threshold = 0.995
    validation_frequency = min(n_train_batches, patience // 2)
    best_validation_loss = np.inf
    test_score = 0.
    start_time = timeit.default_timer()

    done_looping = False
    epoch = 0



