import numpy as np
x = np.array([[[[1,2]]], [[[3,4]]]])
print(x.shape)
y = x.reshape(x.shape[0], -1)
print(y.shape)
print(y)