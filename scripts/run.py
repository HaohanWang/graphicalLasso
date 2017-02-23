
import sys
sys.path.append('../')

import numpy as np

import matplotlib.pylab as plt
from models.graphicalLasso import GraphicalLasso as GL

np.random.seed(1)

X = np.loadtxt('../data/graph.csv', delimiter=',')

G = np.ones([4, 4])
G[0, 3] = 0
G[3, 0] = 0

for lam in [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]:
    gl=GL(rho=lam)
    gl.fit(X)

    g = gl.A
    # print len(np.where(g!=0)[0])
    #
    # print g
    g[g!=0] = 1

    print g

    print np.mean(np.square(G-g))
