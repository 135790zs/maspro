import numpy as np

N = 2

W = np.random.randint(low=0, high=5, size=(N, N))
x = np.random.random(size=(N,))


print(W)
print(x)
# Find out which dim is receiving
# Obtain activations for receiving per weight

a = np.repeat(np.expand_dims(x, axis=0), N, axis=0).T
print(a)
