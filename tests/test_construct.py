from typing import Any
import unittest
from itertools import product

from trainsum import TrainSum
from trainsum.typing import TrainShape
from utils import (
    backends,
    rand_data,
    get_grid,
    get_idxs,
    assert_exact,
)


class TestConstruct(unittest.TestCase):
    def setUp(self):
        self.trainsum = [TrainSum(backend) for backend in backends]

    def check_data_construct(self, ts: TrainSum, shape: TrainShape, data: Any) -> None:
        train = ts.tensortrain(shape, data)
        approx = train.to_tensor()
        assert_exact(self, ts, data, approx, 1e-5)

    def check_cross_construct(
        self,
        ts: TrainSum,
        shape: TrainShape,
        grid: Any,
        func: Any,
        start_idxs: Any = None,
    ) -> None:
        train = ts.tensortrain(shape, func, start_idxs)

        idxs = get_idxs(ts, grid)
        exact = func(idxs)

        approx = train.to_tensor()
        assert_exact(self, ts, exact, approx, 1e-5)

    def test_data(self):
        sizes = [(1024,), (120, 1024), (324, 120)]
        for ts, sizes in product(self.trainsum, sizes):
            xp = ts.namespace
            grid = get_grid(ts, sizes, -10.0, 10.0)
            idxs = get_idxs(ts, grid)
            coords = grid.to_coords(idxs)

            exact = xp.exp(-0.5 * xp.sum(coords**2, axis=0))
            shape = ts.trainshape(*grid.dims, mode="block")
            with ts.exact():
                self.check_data_construct(ts, shape, exact)
            with ts.decomposition(max_rank=15, cutoff=1e-10, ncores=2):
                self.check_data_construct(ts, shape, exact)
            with ts.variational(max_rank=15, cutoff=1e-10, nsweeps=1, ncores=2):
                self.check_data_construct(ts, shape, exact)

    def test_func(self):
        sizes = [(1024,), (120, 1024), (324, 120)]
        for ts, sizes in product(self.trainsum, sizes):
            xp = ts.namespace
            grid = get_grid(ts, sizes, -10.0, 10.0)

            func = lambda idxs: xp.exp(
                -0.5 * xp.sqrt(xp.sum(grid.to_coords(idxs) ** 2, axis=0))
            )
            shape = ts.trainshape(*grid.dims, mode="block")
            with ts.cross(max_rank=32, eps=1e-10):
                self.check_cross_construct(ts, shape, grid, func)

    def test_func_with_start_idxs(self):
        sizes = [(1024,), (120, 1024)]
        for ts, sizes in product(self.trainsum, sizes):
            xp = ts.namespace
            grid = get_grid(ts, sizes, -10.0, 10.0)

            func = lambda idxs: xp.exp(
                -0.25 * xp.sum(grid.to_coords(idxs) ** 2, axis=0)
            )
            start_idxs = xp.asarray(
                [
                    [0, size // 2]
                    for size in sizes
                ],
                dtype=ts.index_type,
            )
            shape = ts.trainshape(*grid.dims, mode="block")
            with ts.cross(max_rank=32, eps=1e-10):
                self.check_cross_construct(ts, shape, grid, func, start_idxs)

    def test_explicit(self):
        sizes = [(1024,), (120, 1024), (324, 120)]
        for ts, sizes in product(self.trainsum, sizes):
            xp = ts.namespace
            grid = get_grid(ts, sizes, -10.0, 10.0)

            shape = ts.trainshape(*grid.dims, mode="interleaved")
            cores = []
            for i in range(len(shape)):
                left = 1 if i == 0 else 16
                right = 1 if i == len(shape) - 1 else 16
                cores.append(rand_data(xp, left, *shape.middle(i), right))
            train = ts.tensortrain(shape, cores)
            for ref_core, core in zip(cores, train.cores):
                self.assertTrue(xp.all(xp.equal(ref_core, core)))


if __name__ == "__main__":
    unittest.main()
