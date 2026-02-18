
from typing import Sequence
import unittest
from itertools import product

from trainsum import TrainSum
from trainsum.typing import UniformGrid
from utils import backends

class TestLinsolver(unittest.TestCase):

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
            data = 2*xp.exp(-xp.sum(coords**2, axis=0)) + 1
            train = ts.tensortrain(shape, data)
            lmap = ts.linear_map("i,i->i", train, train.shape)
            rhs = ts.full(shape, 1.0)
            guess = ts.full(shape, 1.0)

            solver = ts.gmres(subspace=3, nsteps=1, eps=1e-8)

            decomp = ts.svdecomposition(max_rank=15, cutoff=1e-15)
            strat = ts.sweeping_strategy(ncores=2, nsweeps=20)
            linsolver = ts.linsolver(rhs,
                                     lmap,
                                     solver=solver,
                                     decomposition=decomp,
                                     strategy=strat,
                                     method="dmrg")
            guess = linsolver(guess)
            diff = xp.sum((train.to_tensor() * guess.to_tensor() - rhs.to_tensor())**2)
            self.assertLess(diff, 1e-7)

            decomp = ts.svdecomposition(max_rank=2, cutoff=1e-15)
            strat = ts.sweeping_strategy(ncores=1, nsweeps=10)

            linsolver = ts.linsolver(rhs,
                                     lmap,
                                     solver=solver,
                                     decomposition=decomp,
                                     strategy=strat,
                                     method="amen")
            guess = linsolver(guess)
            diff = xp.sum((train.to_tensor() * guess.to_tensor() - rhs.to_tensor())**2)
            self.assertLess(diff, 1e-7)

if __name__ == "__main__":
    unittest.main()
