import sys, os
sys.path.append(os.pardir)
import numpy as np
from common.gradient import *

'''
other way of doing softmax func for 2d array

def softmax(x):
    if x.ndim == 2:
        x = x.T
        x = x-np.max(x, axis = 0)
        y = np.exp(x) / np.sum(np.exp(x), axis=0)
        return y.T
    x = x - np.max(x)
    return np.exp(x) / np.sum(np.exp(x))

x = np.array([[1, 2, 3], [4, 5, 6]])
print(softmax(x))
print("============")
x = x - np.max(x,axis=1).reshape(2,1)
print(x)
print("============")
print(np.exp(x) / np.sum(np.exp(x),axis=1).reshape(2,1)
'''


def f(x):
    return x

x = np.array([[1,2], [3,4]])
print(x.shape)
print(numerical_gradient(f,x))