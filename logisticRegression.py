import pdb
import theano
import theano.tensor as T

x_sym = T.dvector('x_sym')
s_sym = 1/(1+T.exp(-1*x_sym))
fn = theano.function([x_sym], s_sym)

print(fn([0,1]))
