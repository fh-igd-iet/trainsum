import unittest
from itertools import product

from trainsum import TrainSum
from utils import backends, get_grid, get_idxs, assert_relative_error_less


class TestEigsolver(unittest.TestCase):
    def setUp(self):
        self.trainsum = [TrainSum(backend) for backend in backends]
        self.sizes = [(1024,)]

    def test_lanczos(self) -> None:
        for ts in self.trainsum:
            xp = ts.namespace
            ctype = xp.__array_namespace_info__().dtypes()["complex128"]

            mat = xp.asarray(
                [[2.0 + 0.0j, 1.0 - 1.0j],
                 [1.0 + 1.0j, 4.0 + 0.0j]], dtype=ctype
            )
            guess = xp.asarray([1.0 + 0.0j, 0.25 + 0.5j], dtype=ctype)

            solver = ts.lanczos(subspace=2, nsteps=2, eps=1e-12)
            res = solver(lambda vec: mat @ vec, guess)
            vals, vecs = xp.linalg.eigh(mat)
            exact_val = vals[0]
            exact_vec = vecs[:, 0]

            vec = res.array / xp.sqrt(xp.sum(xp.conj(res.array) * res.array))
            phase = exact_vec[0] / vec[0] if vec[0] != 0 else 1.0

            self.assertLess(abs(res.value - exact_val), 1e-12)
            assert_relative_error_less(
                self, ts, exact_vec, vec * phase, 1e-12, use_abs=True
            )

    def test_solve(self) -> None:
        for ts, sizes in product(self.trainsum, self.sizes):
            xp = ts.namespace
            grid = get_grid(ts, sizes, -20.0, 20.0)
            idxs = get_idxs(ts, grid)
            coords = grid.to_coords(idxs)

            shape = ts.trainshape(*sizes)
            data = xp.exp(-0.001 * xp.sum(coords**2, axis=0))
            data /= xp.sum(data**2)
            guess = ts.tensortrain(shape, data)
            pot = ts.polyval(grid, [1.0, 0.0, 0.0], 0.0)
            with ts.exact():
                lap_op = -2*ts.shift(grid.dims[0], 0)
                lap_op += ts.shift(grid.dims[0], 1)
                lap_op += ts.shift(grid.dims[0], -1)
                lap_op *= -0.5 / grid.spacings[0]

            decomp = ts.svdecomposition(max_rank=15, cutoff=1e-15)
            strat = ts.sweeping_strategy(ncores=2, nsweeps=20)
            solver = ts.lanczos(subspace=3, nsteps=1, eps=1e-8)

            lap_map = ts.linear_map("ij,j->i", lap_op, guess.shape)
            pot_map = ts.linear_map("i,i->i", pot, guess.shape)

            eigsolver = ts.eigsolver(
                lap_map, pot_map, solver=solver, decomposition=decomp, strategy=strat
            )
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
