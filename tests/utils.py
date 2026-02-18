from typing import Sequence
import numpy as np
import array_api_compat as api

backends = [api.array_namespace(np.zeros(1))]

#import torch as tr
#tr.set_default_dtype(tr.float64)
#backends.append(api.array_namespace(tr.zeros(1)))

#import cupy as cp
#backends.append(api.array_namespace(cp.zeros(1)))

def prime_factorization(num: int) -> Sequence[int]:
    facs = []
    i = 2
    while i * i <= num:
        while num % i == 0:
            facs.append(i)
            num //= i
        i += 1
    if num > 1:
        facs.append(num)
    return facs

def rand_data(xp, *shape: int):
    data = np.random.rand(*shape)
    if api.is_cupy_namespace(xp):
        return xp.asarray(data)
    elif api.is_torch_namespace(xp):
        return xp.asarray(data)
    else:
        return data

def plot(*x):
    import matplotlib.pyplot as plt
    rows = len(x) // 3 + (1 if len(x) % 3 > 0 else 0)
    fig, axes = plt.subplots(rows, 3, figsize=(15, 3*rows))
    for data, ax in zip(x, axes.flatten()):
        if len(data.shape) == 1:
            ax.plot(data)
        elif len(data.shape) == 2:
            ax.imshow(data)
        else:
            raise ValueError("Unsupported data shape for plotting.")
    plt.show()
