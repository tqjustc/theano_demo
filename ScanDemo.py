
import pdb
import theano
import theano.tensor as T
import numpy as np
from theano.ifelse import ifelse
from theano import scan_module

'''
1. Loop for every row
2. For a given row, loop through the elements.
   2.1 Count the number of elements > 0.5
   2.2 If we come across any element < 0.5, STOP

Output:
For row 1 the count is 0, as no element(starting from left) is greater than 0.
For row 2 the count is 1, (0.6 > 0.5 but we stop at 0.3)
For row 3 the count is 3, as all elements starting from left are greater than 0.5
'''

x = T.matrix('x')

def myFunc(val, preval):
    val1 = preval
    preval = ifelse(T.gt(val, 0.5), preval+1, preval)
    return preval, scan_module.until(T.eq(val1, preval))

def oneRow(row):
    result, updates = theano.scan(fn = myFunc,
                                  sequences=[row],
                                  outputs_info=0
                                  )
    return result[-1]

result, updates = theano.scan(fn=oneRow, sequences=[x])

fn1 = theano.function(inputs=[x], outputs=result)

xx = np.random.random((3,4))
print(xx)

print(fn1(xx))