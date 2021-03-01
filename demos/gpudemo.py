# from __future__ import division
# from numba import cuda, float32
import numpy
import torch
# import math
print(torch.cuda.is_available())



# size = 4
# # Host code
# data1 = numpy.random.random(size=(size, size))
# data2 = numpy.random.random(size=(size, size))
# data3 = numpy.zeros(shape=(size, size))
# # data1 = numpy.ones(shape=(size,))
# # data2 = numpy.ones(shape=(size, size*2))

# @cuda.jit
# def fast_matmul(A, B, C):
#     # Define an array in the shared memory
#     # The size and type of the arrays must be known at compile time
#     sA = cuda.shared.array(shape=(size, size), dtype=float32)
#     sB = cuda.shared.array(shape=(size, size), dtype=float32)

#     x, y = cuda.grid(2)

#     tx = cuda.threadIdx.x
#     ty = cuda.threadIdx.y
#     bpg = cuda.gridDim.x    # blocks per grid

#     if x >= C.shape[0] and y >= C.shape[1]:
#         # Quit if (x, y) is outside of valid C boundary
#         return

#     # Each thread computes one element in the result matrix.
#     # The dot product is chunked into dot products of TPB-long vectors.
#     tmp = 0.
#     for i in range(bpg):
#         # Preload data into shared memory
#         sA[tx, ty] = A[x, ty + i * size]
#         sB[tx, ty] = B[tx + i * size, y]

#         # Wait until all threads finish preloading
#         cuda.syncthreads()

#         # Computes partial product on the shared memory
#         for j in range(size):
#             tmp += sA[tx, j] * sB[j, ty]

#         # Wait until all threads finish computing
#         cuda.syncthreads()

#     C[x, y] = tmp

# threadsperblock = 256
# blockspergrid = math.ceil(data1.shape[0] / threadsperblock)

# fast_matmul[blockspergrid, threadsperblock](data1, data2, data3)

# print(data1)
