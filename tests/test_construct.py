from typing import Sequence, Any
import unittest
from itertools import product

from trainsum import TrainSum
from trainsum.typing import UniformGrid, TrainShape
from utils import backends, rand_data

class TestConstruct(unittest.TestCase):

    def setUp(self):
        self.trainsum = [TrainSum(backend) for backend in backends]

    def get_grid(self, ts, sizes: Sequence[int], lower: float, upper: float):
        dims = [ts.dimension(size) for size in sizes]
        domains = [ts.domain(lower, upper) for _ in sizes]
        return ts.uniform_grid(dims, domains)

    def get_idxs(self, ts, grid: UniformGrid):
        xp = ts.namespace
        idxs = xp.zeros([len(grid.dims), *[dim.size() for dim in grid.dims]],
                        dtype=ts.index_type)
        for i, dim in enumerate(grid.dims):
            cut = (*(xp.newaxis,) * i, slice(None), *(xp.newaxis,) * (len(grid.dims) - i - 1))
            idxs[i] += xp.arange(dim.size(), dtype=ts.index_type)[cut]
        return idxs

    def check_data_construct(self, ts: TrainSum, shape: TrainShape, data: Any) -> None:
        xp = ts.namespace
        train = ts.tensortrain(shape, data)
        approx = train.to_tensor()
        diff = abs(xp.sum((data - approx)**2))
        self.assertLess(diff, 1e-5)

    def check_cross_construct(self, ts: TrainSum, shape: TrainShape, grid: UniformGrid, func: Any) -> None:
        xp = ts.namespace
        train = ts.tensortrain(shape, func)

        idxs = self.get_idxs(ts, grid)
        exact = func(idxs)

        approx = train.to_tensor()
        diff = abs(xp.sum((exact - approx)**2))
        self.assertLess(diff, 1e-5)

    def test_data(self):
        sizes = [(1024,), (120, 1024), (324, 120)]
        for ts, sizes in product(self.trainsum, sizes):
            xp = ts.namespace
            grid = self.get_grid(ts, sizes, -10.0, 10.0)
            idxs = self.get_idxs(ts, grid)
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
            grid = self.get_grid(ts, sizes, -10.0, 10.0)

            func = lambda idxs: xp.exp(-0.5 * xp.sqrt(xp.sum(grid.to_coords(idxs)**2, axis=0)))
            shape = ts.trainshape(*grid.dims, mode="block")
            with ts.cross(max_rank=32, eps=1e-10):
                self.check_cross_construct(ts, shape, grid, func)

    def test_explicit(self):
        sizes = [(1024,), (120, 1024), (324, 120)]
        for ts, sizes in product(self.trainsum, sizes):
            xp = ts.namespace
            grid = self.get_grid(ts, sizes, -10.0, 10.0)

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
