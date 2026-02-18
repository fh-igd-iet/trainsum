from typing import Sequence
import unittest
from itertools import product

from trainsum import TrainSum
from trainsum.typing import UniformGrid
from utils import backends

class TestEigsolver(unittest.TestCase):

    def setUp(self):
        self.trainsum = [TrainSum(backend) for backend in backends]
        self.sizes = [(1024,)]

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

    def test_solve(self) -> None:
        for ts, sizes in product(self.trainsum, self.sizes):
            xp = ts.namespace
            grid = self.get_grid(ts, sizes, -20.0, 20.0)
            idxs = self.get_idxs(ts, grid)
            coords = grid.to_coords(idxs)

            shape = ts.trainshape(*sizes)
            data = xp.exp(-0.001*xp.sum(coords**2, axis=0))
            data /= xp.sum(data**2)
            guess = ts.tensortrain(shape, data)
            pot = ts.polyval(grid, [1.0, 0.0, 0.0], 0.0)
            with ts.exact():
                lap_op = -2*ts.shift(grid.dims[0], 0)
                lap_op += ts.shift(grid.dims[0], 1)
                lap_op += ts.shift(grid.dims[0], -1)
                lap_op *= -0.5/grid.spacings[0]

            decomp = ts.svdecomposition(max_rank=15, cutoff=1e-15)
            strat = ts.sweeping_strategy(ncores=2, nsweeps=20)
            solver = ts.lanczos(subspace=3, nsteps=1, eps=1e-8)

            lap_map = ts.linear_map("ij,j->i", lap_op, guess.shape)
            pot_map = ts.linear_map("i,i->i", pot, guess.shape)

            eigsolver = ts.eigsolver(lap_map, pot_map,
                                     solver=solver,
                                     decomposition=decomp,
                                     strategy=strat)
            eigvals = []
            def call(lrange, res):
                eigvals.append(res.value)
                return False
            guess = eigsolver(guess, callback=call)

            mat = lap_op.to_tensor()
            mat += xp.eye(mat.shape[0]) * pot.to_tensor()
            exact = xp.linalg.eigh(mat)[0][0]
            self.assertTrue(abs(eigvals[-1] - exact) < 1e-6)

if __name__ == "__main__":
    unittest.main()
