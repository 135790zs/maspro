import numpy as np

# N = 2

# W = np.random.randint(low=0, high=5, size=(N, N))
# x = np.random.random(size=(N,))


# print(W)
# print(x)
# # Find out which dim is receiving
# # Obtain activations for receiving per weight

# a = np.repeat(np.expand_dims(x, axis=0), N, axis=0).T
# print(a)
# w = 6
# N = 4

# F = np.zeros(shape=(w, N))
# F += np.random.randint(low=0, high=2, size=F.shape)
# print(F)
# print("newest: ", F[-1, :])

# W = np.random.randint(low=0, high=5, size=(4, 4))


# R = np.asarray([.9, 1, .1])
# P = np.repeat(np.expand_dims(R, axis=0), R.size, axis=0)
# Z = (P.T - P)
# Z[Z == 0] = np.inf
# Z = 1/Z
# print(Z * 0.01)

def get_2d_from_3d(M, a):
    N = M.shape[-1]
    R = np.repeat(np.expand_dims(np.arange(N), axis=0), N, axis=0)
    return M[a, R.T, R]


# def insert_2d_in_3d(M, a, val=1):
#     N = M.shape[-1]
#     R = np.repeat(np.expand_dims(np.arange(N), axis=0), N, axis=0)
#     M[a, R.T, R] = val
#     return M

# A = np.random.randint(low=0, high=9, size=(5, 4, 4))
# d = np.random.randint(low=0, high=5, size=(4, 4))

# print(A)
# print(d)
# print(insert_2d_in_3d(A, d, val=100))

# Repeat inner array

N = 3
MAL = 4
MAX = 5

MALS = np.random.randint(low=0, high=MAL, size=(N, N))

x = np.random.randint(low=0, high=9, size=(MAX, N))
xs = np.repeat(np.expand_dims(x, axis=1), x.shape[-1], axis=1)  # May need transpose?

xceiled = get_2d_from_3d(M=xs, a=MALS)

print(xs)
print("##########")
print(MALS)
print("##########")
print(xceiled)




exit()
x = np.random.randint(low=0, high=9, size=(MAX, N))

# xs = np.repeat(np.expand_dims(x, axis=0), , axis=0)

M = np.random.randint(low=0, high=3, size=(4, 2))
print(M)
N = M.shape[-1]
R = np.repeat(np.expand_dims(np.arange(N), axis=0), N, axis=0)
a = np.random.randint(low=0, high=4, size=(2, 2))
print(a)
print(M[a, R.T, R])
# print(a)

# print(M[a])
