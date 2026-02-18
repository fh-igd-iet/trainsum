import unittest
from typing import Any
from itertools import product

from trainsum import TrainSum
from trainsum.backend import get_index_dtype
from utils import backends

class TestUniformGrid(unittest.TestCase):

    def setUp(self) -> None:
        self.lower = -1.0
        self.upper = 1.0
        self.trainsum = [TrainSum(backend) for backend in backends]
        self.sizes = [(120,), (256,), (20, 120), (128, 128)]

    def index_tensor(self, xp: Any, dims: tuple[int, ...]) -> Any:
        int_type = get_index_dtype(xp)
        idxs = xp.zeros((len(dims), *[size for size in dims]), dtype=int_type)
        for i, size in enumerate(dims):
            cut = [xp.newaxis] * i + [slice(None)] + [xp.newaxis] * (len(dims)-i-1)
            idxs[i,...] += xp.arange(size, dtype=int_type)[*cut]
        return idxs

    def coord_tensor(self, xp: Any, dims: tuple[int, ...]) -> Any:
        coords = xp.zeros((len(dims), *[size for size in dims]))
        for i, size in enumerate(dims):
            cut = [xp.newaxis] * i + [slice(None)] + [xp.newaxis] * (len(dims)-i-1)
            coords[i,...] += xp.linspace(self.lower, self.upper, size)[*cut]
        return coords

    def test_construction(self):
        for ts, sizes in product(self.trainsum, self.sizes):
            dims = [ts.dimension(size) for size in sizes]
            domains = [ts.domain(self.lower, self.upper) for _ in sizes]
            spacings = [(domain.diff / (dim.size()-1)) for dim, domain in zip(dims, domains)]
            grid = ts.uniform_grid(dims, domains)

            self.assertTrue(all(d1==d2 for d1, d2 in zip(grid.dims, dims)))
            self.assertTrue(all(d1==d2 for d1, d2 in zip(grid.domains, domains)))
            self.assertTrue(all(d1==d2 for d1, d2 in zip(grid.spacings, spacings)))

        for ts in self.trainsum:
            with self.assertRaises(ValueError):
                dims = (ts.dimension(10),)
                domains = (ts.domain(1.0, 2.0), ts.domain(1.0, 2.0))
                grid = ts.uniform_grid(dims, domains)

    def test_to_idxs(self):
        for (xp, ts), sizes in product(zip(backends, self.trainsum), self.sizes):

            ref_idxs = self.index_tensor(xp, sizes)
            ref_coords = self.coord_tensor(xp, sizes)

            dims = [ts.dimension(size) for size in sizes]
            domains = [ts.domain(self.lower, self.upper) for _ in sizes]
            grid = ts.uniform_grid(dims, domains)
            idxs = grid.to_idxs(ref_coords)

            self.assertTrue(xp.all(idxs == ref_idxs))
            self.assertRaises(ValueError, grid.to_idxs, xp.zeros((10, 0)))

    def test_to_coords(self):
        for (xp, ts), sizes in product(zip(backends, self.trainsum), self.sizes):

            ref_idxs = self.index_tensor(xp, sizes)
            ref_coords = self.coord_tensor(xp, sizes)

            dims = [ts.dimension(size) for size in sizes]
            domains = [ts.domain(self.lower, self.upper) for _ in sizes]
            grid = ts.uniform_grid(dims, domains)
            coords = grid.to_coords(ref_idxs)

            self.assertTrue(xp.max(xp.abs(coords-ref_coords)) < 1e-5)
            self.assertRaises(ValueError, grid.to_coords, xp.zeros((10, 0), dtype=xp.int64))
    
if __name__ == '__main__':
    unittest.main()
