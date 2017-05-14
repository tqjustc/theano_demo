import pdb
import theano
import theano.tensor as T

state = theano.shared(0)
inc = T.iscalar('inc')
accumulator = theano.function([inc], state, updates = [(state, state+inc)])

print(state.get_value())
accumulator(2)
print (state.get_value())
accumulator(3)
print (state.get_value())