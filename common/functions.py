import numpy as np

def sigmoid(x):
    return 1/(1+np.exp(-x))

def softmax(x):
    if x.ndim == 2:
        x = x.T
        x = x - np.max(x, axis = 0)
        y = np.exp(x) / np.sum(np.exp(x), axis=0)
        return y.T
    # why 2d is diff, ex) x is (2,3) 2x3 -> np.max(x) becomes (2,) which is 1x2 arr, so it cant be subtracted
    # other way
    # x = x - np.max(x,axis=1).reshape(2,1)
    # np.exp(x) / np.sum(np.exp(x),axis=1).reshape(2,1)
    x = x - np.max(x)
    return np.exp(x) / np.sum(np.exp(x))

def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1,t.size)
        y = y.resahpe(1,y.size)
    

    batch_size = y.shape[0]
    return -np.sum(t*np.log(y + 1e-7)) / batch_size

def sigmoid_grad(x):
    return (1 - sigmoid(x)) * sigmoid(x)