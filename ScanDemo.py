
import pdb
import theano
import theano.tensor as T
import numpy as np


H = T.dtensor3('H')
Z = T.dtensor3('Z')
ZT = T.transpose(Z, (0, 2, 1))
HZ = T.batched_dot(H, ZT)
result = T.reshape(HZ, (HZ.shape[0], HZ.shape[2]))
func1 = theano.function(inputs=[H, Z], outputs=result)


# input:
hMat = np.random.random((2, 1, 10))
zMat = np.random.random((2, 3, 10))

res = func1(hMat, zMat)
print(res)

print('Numpy Testing :')
# verify it using Numpy only
resNp = np.zeros((hMat.shape[0], zMat.shape[1]))
for i in range(hMat.shape[0]):
    zi = zMat[i,:,:]
    resNp[i, :] = np.dot(hMat[i, :, :], np.transpose(zi, (1, 0)))
print(resNp)

print('end.')