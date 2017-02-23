__author__ = 'Haohan Wang'

import numpy as np

np.random.seed(1)

C = np.zeros([4, 4])

minL = 0.21
maxL = 0.30

c = np.random.random()*(maxL-minL) + minL
C[:,:] = c

for i in range(4):
    C[i,i] = 1

C[0, 3] = 2*(c*c)
C[3, 0] = 2*(c*c)

C[1, 2] = 0
C[2, 1] = 0

print C

print np.linalg.inv(C)

X = np.random.multivariate_normal(np.zeros([4]), C, size=1000000)

print X.shape

np.savetxt('../data/graph.csv', X, delimiter=',')