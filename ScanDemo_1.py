import pdb
import theano
import theano.tensor as T
import numpy as np

vec1 = T.dvector('vec1')
vec2 = T.dvector('vec2')

def oneStep(i, vec1, vec2):
    return vec1[i] * vec2[i]

result, updates = theano.scan(fn=oneStep, sequences=T.arange(vec1.shape[0]), non_sequences=[vec1, vec2])

fn1 = theano.function(
    inputs=[vec1, vec2],
    outputs=result,
    updates=updates
)


v1 = np.random.random((1,3)).flatten()
v2 = np.random.random((1,3)).flatten()
print(fn1(v1,v2))
print(v1*v2)