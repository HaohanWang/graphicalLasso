__author__ = 'Haohan Wang'

import numpy as np

scI = np.array([2, 8])
sI = np.array([0,1,3,4,5,6,7,9,10,11,12,13,14,15])

def graphicalLasso(c):
    C = np.zeros([4, 4])
    C[:,:] = c
    for i in range(4):
        C[i,i] = 1

    C[0, 3] = 2*c**2
    C[3, 0] = 2*c**2

    C[1, 2] = 0
    C[2, 1] = 0

    G = np.kron(C, C)

    G2 = G[sI,:]
    G2 = G2[:,sI]
    m = -np.inf
    for ind in scI:
        G1 = G[ind,sI]
        v = np.dot(G1, np.linalg.inv(G2))
        l = np.linalg.norm(v, ord=1)
        m = max(m, l)
    return m < 1


def DtraceLasso(c):
    C = np.zeros([4, 4])
    C[:,:] = c
    for i in range(4):
        C[i,i] = 1

    C[0, 3] = 2*c**2
    C[3, 0] = 2*c**2

    C[1, 2] = 0
    C[2, 1] = 0

    G = 0.5*(np.kron(C, np.identity(C.shape[0])) + (np.kron(np.identity(C.shape[0]), C)))

    G2 = G[sI,:]
    G2 = G2[:,sI]
    m = -np.inf
    for ind in scI:
        G1 = G[ind,sI]
        v = np.dot(G1, np.linalg.inv(G2))
        l = np.linalg.norm(v, ord=1)
        m = max(m, l)
    return m < 1


if __name__ == '__main__':
    m = np.inf
    n = -np.inf
    for i in range(1000):
        c = i/1000.0
        if not DtraceLasso(c):
            m = min(m, c)
            n = max(n, c)
    print m, n