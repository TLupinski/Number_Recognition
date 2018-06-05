import numpy as np
import pylab
from pylab import *
import cv2


def decode_n_best(res, n, a):
    score = np.zeros((n))
    best = np.flip(np.argsort(res[0]),0)
    label = np.zeros((n,len(res)),dtype=np.int32)
    labels = np.zeros((n,len(res)),dtype=np.int32)
    for i in range(n):
        label[i][0] = int(best[i])
        score[i] = res[0][label[i][0]]
    for i in range(1,len(res)):
        scores = np.zeros((n*a))
        for j in range(n):
            for k in range(a):
                scores[j*a + k] = score[j] * res[i][k]
        best = np.flip(np.argsort(scores),0)
        for j in range(n*a-1):
            assert scores[best[j]]>=scores[best[j+1]]
        for j in range(n):
            m = best[j]
            af = m%a
            bf = (m-af)/a
            labels[j] = label[bf] 
            labels[j][i] = af
        for j in range(n):
            label[j] = labels[j]
            score[j] = scores[best[j]]
    return label, score

bs = 25
features_dec_inp = 128
go_token = -1
end_token = 2
minibatch_size = 1
res = np.random.rand(5,10)
alphabet = '0123456789'
print(res)
decode_n_best(res,5,len(alphabet))
