import torch
import opt_einsum
import time

def sbrj_sbrji_sbrji(a, b):
    start = time.time()
    res = opt_einsum.contract("sbrj, sbrji -> sbrji",
                              a,
                              b,
                              backend='torch')
    # b = b.reshape(-1,)
    # res2 = torch.reshape(a * torch.reshape(b, (-1, b.shape[-1], b.shape[-2])), b.shape)
    # res2 = (a * b.T).T
    b_shape = b.shape
    # b_ = torch.reshape(b, (-1, b.shape[-1] * b.shape[-2]))
    # ab_ = a * b_
    # ab = torch.reshape(ab, b_shape)
    res2 = a[...,  None] * b
    print(res == res2)
    print(1000*(time.time()-start))
    return res2
