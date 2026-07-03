import unittest
from itertools import product

from trainsum import TrainSum
from utils import backends, get_grid, get_idxs, assert_relative_error_less


class TestLinsolver(unittest.TestCase):
    def setUp(self):
        self.trainsum = [TrainSum(backend) for backend in backends]
        self.sizes = [(1024,)]

    def test_gmres(self) -> None:
        for ts in self.trainsum:
            xp = ts.namespace
            ctype = xp.__array_namespace_info__().dtypes()["complex128"]

            mat = xp.asarray(
                [[2.0 + 0.0j, 1.0j],
                 [-1.0j, 3.0 + 0.0j]], dtype=ctype
            )
            rhs = xp.asarray([1.0 + 2.0j, -0.5 + 0.25j], dtype=ctype)

            solver = ts.gmres(subspace=2, nsteps=2, eps=1e-12)
            res = solver(lambda vec: mat @ vec, rhs)
            exact = xp.linalg.solve(mat, rhs)

            assert_relative_error_less(self, ts, exact, res.array, 1e-12, use_abs=True)
            self.assertLessEqual(len(res.residuals), 2)
            self.assertLess(res.residuals[-1], 1e-12)

    def test_solve(self) -> None:
        for ts, sizes in product(self.trainsum, self.sizes):
            xp = ts.namespace
            grid = get_grid(ts, sizes, -20.0, 20.0)
            idxs = get_idxs(ts, grid)
            coords = grid.to_coords(idxs)

            shape = ts.trainshape(*sizes)
            data = 2 * xp.exp(-xp.sum(coords**2, axis=0)) + 1
            train = ts.tensortrain(shape, data)
            lmap = ts.linear_map("i,i->i", train, train.shape)
            rhs = ts.full(shape, 1.0)

            solver = ts.gmres(subspace=3, nsteps=1, eps=1e-8)

            decomp = ts.svdecomposition(max_rank=15, cutoff=1e-15)
            strat = ts.sweeping_strategy(ncores=2, nsweeps=20)
            linsolver = ts.linsolver(
                rhs,
                lmap,
                solver=solver,
                decomposition=decomp,
                strategy=strat,
                method="dmrg",
            )
            guess = ts.full(shape, 1.0)
            guess = linsolver(guess)
            diff = xp.sum(
                (train.to_tensor() * guess.to_tensor() - rhs.to_tensor()) ** 2
            )
            self.assertLess(diff, 1e-7)

            decomp = ts.svdecomposition(max_rank=2, cutoff=1e-15)
            strat = ts.sweeping_strategy(ncores=1, nsweeps=2)

            linsolver = ts.linsolver(
                rhs,
                lmap,
                solver=solver,
                decomposition=decomp,
                strategy=strat,
                method="amen",
            )
            guess = ts.full(shape, 1.0)
            guess = linsolver(guess)
            diff = xp.sum(
                (train.to_tensor() * guess.to_tensor() - rhs.to_tensor()) ** 2
            )
            self.assertLess(diff, 1e-7)


if __name__ == "__main__":
    unittest.main()
