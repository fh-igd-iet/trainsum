from typing import Sequence
import unittest
from itertools import product

from trainsum import TrainSum
from trainsum.typing import UniformGrid
from utils import backends

class TestMinMax(unittest.TestCase):

    def setUp(self):
        self.trainsum = [TrainSum(backend) for backend in backends]
        self.sizes = [(120,), (280,), (1024,), (120, 1024), (324, 120)]

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

    def test_gauss(self) -> None:
        for sizes, ts in product(self.sizes, self.trainsum):
            xp = ts.namespace
            grid = self.get_grid(ts, sizes, -10, 10)
            idxs = self.get_idxs(ts, grid)
            coords = grid.to_coords(idxs)
            
            shape = ts.trainshape(*grid.dims)
            data = xp.exp(-xp.sum(coords**2, axis=0))
            train = ts.tensortrain(shape, data)

            res = ts.min_max(train, 16)
            min_val = xp.min(data)
            max_val = xp.max(data)

            self.assertLess(abs(min_val-res.min_val), 1e-6)
            self.assertLess(abs(max_val-res.max_val), 1e-6)

if __name__ == "__main__":
    unittest.main()
