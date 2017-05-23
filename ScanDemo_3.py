
'''
using scan to implement mat+scalar
'''

import pdb
import theano
import theano.tensor as T
import numpy as np


x = T.dmatrix('x')
v = T.dscalar('v')

def rowAction(rowVal, v):
    return v + rowVal

def matAction(rowVec, v):
    result, updates=  theano.scan(
        fn=rowAction,
        sequences=rowVec,
        non_sequences=v
    )
    return result

result, updates = theano.scan(
    fn = matAction,
    sequences=x,
    non_sequences=v
)

fn1 = theano.function(
    inputs=[x, v],
    outputs=result,
    updates=updates
)

xx = np.random.random((3,4))
vv = np.random.rand()

print(fn1(xx,vv))
print(xx+vv)

