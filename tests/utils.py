from typing import Sequence, Any
import numpy as np
import array_api_compat as api

backends = []
backends = [api.array_namespace(np.zeros(1))]

# import torch as tr
# tr.set_default_dtype(tr.float64)
# backends.append(api.array_namespace(tr.zeros(1)))
# backends.append(api.array_namespace(tr.zeros(1, device="cuda")))

# import cupy as cp
# backends.append(api.array_namespace(cp.zeros(1)))


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


def get_grid(ts, sizes: Sequence[int], lower: float, upper: float):
    dims = [ts.dimension(size) for size in sizes]
    domains = [ts.domain(lower, upper) for _ in sizes]
    return ts.uniform_grid(dims, domains)


def get_idxs(ts, grid):
    xp = ts.namespace
    idxs = xp.zeros(
        [len(grid.dims), *[dim.size() for dim in grid.dims]], dtype=ts.index_type
    )
    for i, dim in enumerate(grid.dims):
        cut = (
            *(xp.newaxis,) * i,
            slice(None),
            *(xp.newaxis,) * (len(grid.dims) - i - 1),
        )
        idxs[i] += xp.arange(dim.size(), dtype=ts.index_type)[cut]
    return idxs


def rand_cores(ts, shape, rank: int = 10):
    xp = ts.namespace
    cores = []
    for i in range(len(shape)):
        left = 1 if i == 0 else rank
        right = 1 if i == len(shape) - 1 else rank
        cores.append(xp.asarray(rand_data(xp, left, *shape.middle(i), right)))
    return cores


def assert_relative_error_less(
    testcase: Any,
    ts,
    exact: Any,
    approx: Any,
    tol: float,
    *,
    use_abs: bool = False,
) -> None:
    xp = ts.namespace
    if use_abs:
        diff = xp.sum(xp.abs(approx - exact) ** 2)
        norm = xp.sum(xp.abs(exact) ** 2)
    else:
        diff = xp.sum((approx - exact) ** 2)
        norm = xp.sum(exact**2)
    if float(norm) == 0.0:
        testcase.assertLess(float(diff), tol)
    else:
        testcase.assertLess(float(diff / norm), tol)


def assert_exact(testcase: Any, ts, exact: Any, approx: Any, tol: float = 1e-7) -> None:
    xp = ts.namespace
    diff = abs(xp.sum((exact - approx) ** 2))
    testcase.assertLess(diff, tol)


def shift_matrix(xp, rows: int, cols: int, shift: int):
    mat = xp.zeros((rows, cols))
    for row in range(rows):
        col = row + shift
        if 0 <= col < cols:
            mat[row, col] = 1.0
    return mat


def exact_toeplitz(xp, size: int, mode: str):
    if mode == "full":
        exact = xp.zeros((2 * size, size, size))
        for i in range(size):
            for j in range(size):
                exact[size + i - j, i, j] = 1.0
        return exact

    exact = xp.zeros((size, size, size))
    for i in range(size):
        for j in range(size):
            if mode == "lower" and i >= j:
                exact[i - j, i, j] = 1.0
            elif mode == "upper" and i < j:
                exact[size + i - j, i, j] = 1.0
            elif mode == "circular":
                exact[(i - j) % size, i, j] = 1.0
    return exact


def exact_slice_operator(xp, size: int, slc: slice):
    start = 0 if slc.start is None else int(slc.start)
    stop = size if slc.stop is None else int(slc.stop)
    step = 1 if slc.step is None else int(slc.step)

    cols = list(range(start, stop, step))
    exact = xp.zeros((len(cols), size))
    for row, col in enumerate(cols):
        exact[row, col] = 1.0
    return exact


def plot(*x):
    import matplotlib.pyplot as plt

    rows = len(x) // 3 + (1 if len(x) % 3 > 0 else 0)
    fig, axes = plt.subplots(rows, 3, figsize=(15, 3 * rows))
    for data, ax in zip(x, axes.flatten()):
        if len(data.shape) == 1:
            ax.plot(data)
        elif len(data.shape) == 2:
            ax.imshow(data)
        else:
            raise ValueError("Unsupported data shape for plotting.")
    plt.show()

