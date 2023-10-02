import numpy as np

def _numerical_gradient_1d(f, x):
    h=1e-4
    grad = np.zeros_like(x)

    for idx in range(x.size):
        tmp_val = x[idx]
        x[idx] = tmp_val + h
        fxh1 = f(x) # f(x+h)

        x[idx] = tmp_val - h
        fxh2 = f(x) # f(x-h)
        grad[idx] = (fxh1-fxh2)/(2*h)
        x[idx] = tmp_val

    return grad

def numerical_gradient_2d(f,X):
    if X.ndim == 1:
        return _numerical_gradient_1d(f,X)
    else:
        grad = np.zeros_like(X)
        # enumerate() returns index and element as a tuple
        '''
        example
        (0, array([1, 2, 3]))
        (1, array([4, 5, 6]))
        '''
        for idx, x in enumerate(X):
            grad[idx] = _numerical_gradient_1d(f,x)

        return grad

def numerical_gradient(f,x):
    h = 1e-4
    grad = np.zeros_like(x)
    '''
    https://numpy.org/doc/stable/reference/generated/numpy.nditer.html
    nditer() makes its np.array parameter as iterator, efficient for more than 2d array
    flags - "c_index" for 1d array, access index like c a[0], a[1], ... / "multi_index" for more than 2d array, access index as tuple
    op_flags - readwrite indicates the operand will be read from and written to.
    '''
    it = np.nditer(x, flags=["multi_index"], op_flags=["readwrite"])
    while not it.finished:
        idx = it.multi_index
        tmp_val = x[idx]
        x[idx] = float(tmp_val) + h
        fxh1 = f(x) # f(x+h)

        x[idx] = tmp_val - h
        fxh2 = f(x) # f(x-h)
        grad[idx] = (fxh1-fxh2) / (2*h)
        
        x[idx] = tmp_val
        it.iternext()
    
    return grad
