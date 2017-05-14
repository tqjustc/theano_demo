import pdb
import theano
import theano.tensor as T

state = theano.shared(0)
inc = T.iscalar('inc')
accumulator = theano.function([inc], state, updates = [(state, state+inc)])
