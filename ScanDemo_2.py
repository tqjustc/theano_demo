

'''
using scan to implement Mat.sum()
'''

import pdb
import theano
import theano.tensor as T
import numpy as np

mat = T.dmatrix('mat')

def rowSum(val, pre):
    return pre + val

def matSum(row, pre):
    result, updates = theano.scan(
        fn=rowSum,
        sequences=row,
        outputs_info=T.constant(0, dtype=theano.config.floatX)
    )
    return result[-1] + pre

result, updates = theano.scan(
    fn = matSum,
    sequences=mat,
    outputs_info=T.constant(0, dtype=theano.config.floatX)
)

fn1 = theano.function(
    inputs=[mat],
    outputs=result[-1],
    updates=updates
)

mat1 = np.random.random((2,3))
print(mat1)
print(fn1(mat1))
print(mat1.sum())

