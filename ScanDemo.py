
import pdb
import theano
import theano.tensor as T
import numpy
from theano.ifelse import ifelse
from theano import scan_module

x = T.matrix("x")


# Function to loop through every element in a given row
def scanElemInRow(value, preVal):
    tmpVal = preVal
    preVal = ifelse(T.gt(value, 0.5), preVal + 1, preVal)
    return preVal, scan_module.until(T.eq(preVal, tmpVal))


# Function to loop through every row of a matrix
def scanRow(row):
    count, updates = theano.scan(scanElemInRow,
                                 sequences=[row],
                                 outputs_info=T.constant(0, dtype=theano.config.floatX),

                                 non_sequences=None)
    count = count[-1]
    return count


y, updates = theano.scan(scanRow,
                         sequences=[x],

                         outputs_info=None,
                         non_sequences=None)

fn1 = theano.function([x], y)

arr = numpy.asarray([[0.1, 0.2, 0.3],
                     [0.6, 0.3, 0.9],
                     [0.6, 0.7, 0.8]])
print(fn1(arr))




print(arr.sum())


print('end.')