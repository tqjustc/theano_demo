
import pdb
import theano
import theano.tensor as T
import gzip
import numpy as np
import os
import sys
import timeit

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

    def errors(self, y):
        if y.ndim != self.y_pred or not y.dtype.startswith('int'):
            raise ValueError('Wrong input y.')
        else:
            return T.mean(T.neq(y, self.y_pred))

def load_data():
    url1 = 'http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz'
    from six.moves import urllib
    import six.moves.cPickle as Pickle

    dataset = os.getcwd() + '/data/mnist.pkl.gz'
    if not os.path.exists(os.getcwd()+'/data/'):
        os.mkdir(os.getcwd()+'/data/')

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

class HiddenLayer(object):
    def __init__(self, rng, input, n_in, n_out, W = None, b = None, activation=T.tanh):
        self.input = input
        if W is None:
            W_values = np.asarray(
                rng.uniform(
                    low=-np.sqrt(6./(n_in+n_out)),
                    high = np.sqrt(6./(n_in+n_out)),
                    size=(n_in, n_out)
                ),
                dtype=theano.config.floatX
            )

            if activation == T.nnet.sigmoid:
                W_values *= 4
            W = theano.shared(
                value=W_values,
                name='W',
                borrow=True
            )
        if b is None:
            b = theano.shared(
                value=np.zeros((n_out,), dtype=theano.config.floatX),
                name='b',
                borrow=True
            )
        self.W = W
        self.b = b
        self.params = [self.W, self.b]
        lin_output = T.dot(self.input, self.W) + self.b
        self.output = (lin_output if activation is None else activation(lin_output))

class MLP(object):
    def __init__(self, rng, input, n_in, n_hidden, n_out):
        self.hiddenLayer = HiddenLayer(rng, input=input, n_in=n_in, n_out=n_hidden, activation=T.tanh)
        self.logRegressionLayer = LogisticRegression(input=self.hiddenLayer.output, n_in=n_hidden, n_out=n_out)
        self.L1 = (
            abs(self.hiddenLayer.W).sum() + abs(self.logRegressionLayer.W).sum()
        )
        self.L2_sqr = (
            (self.hiddenLayer.W ** 2).sum() + (self.logRegressionLayer.W**2).sum()
        )
        self.negative_log_likelihood = (
            self.logRegressionLayer.negative_log_likelihood
        )
        self.errors = self.logRegressionLayer.errors
        self.params = self.hiddenLayer.params + self.logRegressionLayer.params

        self.input = input

def test_mlp(learning_rate = 0.01, L1_reg = 0.00, L2_reg=0.0001, n_epochs=1000, batch_size=20, n_hidden=500):
    datasets = load_data()
    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

    n_train_batches = train_set_x.get_value(borrow=True).shape[0] // batch_size
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] // batch_size
    n_test_batches = test_set_x.get_value(borrow=True).shape[0] // batch_size

    print('...building the model')

    index = T.lscalar()
    x = T.matrix('x')
    y = T.matrix('y')

    rng = np.random.RandomState(1234)

    classifier = MLP(rng=rng, input=x, n_in = 28*28, n_hidden=n_hidden, n_out=10)

    cost=(
        classifier.negative_log_likelihood(y) + L1_reg * classifier.L1 + L2_reg * classifier.L2_sqr
    )

    test_model = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens={
            x: test_set_x[index*batch_size:(index+1)*batch_size],
            y: test_set_y[index * batch_size:(index + 1) * batch_size]
        }
    )

    validate_model = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens={
            x: valid_set_x[index * batch_size:(index + 1) * batch_size],
            y: valid_set_x[index * batch_size:(index + 1) * batch_size]
        }
    )

    gparams = [T.grad(cost, param) for param in classifier.params]
    updates = [(param, param - learning_rate*gparam) for param, gparam in zip(classifier.params, gparams)]

    train_model = theano.function(
        inputs=[index],
        outputs=cost,
        updates=updates,
        givens={
            x: train_set_x[index*batch_size:(index+1)*batch_size],
            y: train_set_y[index*batch_size:(index+1)*batch_size]
        }
    )

    print('...training.')

    patience = 10000
    patience_increase = 2
    improvement_threshold = 0.995
    validation_frequency = min(n_train_batches, patience // 2)

    best_validation_loss = np.inf
    best_iter = 0
    test_score = 0
    start_time = timeit.default_timer()

    epoch = 0
    done_looping = False

    while(epoch < n_epochs) and (not done_looping):
        epoch += 1
        for minibatch_index in range(n_train_batches):
            minibatch_avg_cost = train_model(minibatch_index)
            iter = (epoch-1) * n_train_batches + minibatch_index

            if (iter+1) % validation_frequency == 0:
                validation_losses = [validate_model[i] for i in range(n_valid_batches)]
                this_validation_loss = np.mean(validation_losses)
                print(
                    'epoch %i, minibatch %i/%i, validation error %f %%' % (
                        epoch,
                        minibatch_index+1,
                        n_train_batches,
                        this_validation_loss*100.
                    )
                )
                if this_validation_loss < best_validation_loss:
                    if(this_validation_loss < best_validation_loss * improvement_threshold):
                        patience = max(patience, iter*patience_increase)
                    best_validation_loss = this_validation_loss
                    best_iter = iter

                    test_losses = [test_model[i] for i in range(n_test_batches)]
                    test_score = np.mean(test_losses)
                    print((
                        '    epoch %i, minibatch %i/%i, test error of '
                        'best model %f %%') %
                          (epoch, minibatch_index+1, n_train_batches, test_score*100.)
                          )
            if patience <= iter:
                done_looping = True
                break
    end_time = timeit.default_timer()
    print(('Optimization complete. Best validation score of %f %% '
           'obtained at iteration %i, with test performance %f %%') %
          (best_validation_loss * 100., best_iter + 1, test_score * 100.))
    print(('The code for file ' +
           os.path.split(__file__)[1] +
           ' ran for %.2fm' % ((end_time - start_time) / 60.)))

if __name__ == '__main__':
    test_mlp()