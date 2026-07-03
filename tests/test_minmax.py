import unittest
from itertools import product

from trainsum import TrainSum
from utils import backends, get_grid, get_idxs


class TestMinMax(unittest.TestCase):
    def setUp(self):
        self.trainsum = [TrainSum(backend) for backend in backends]
        self.sizes = [(120,), (280,), (1024,), (120, 1024), (324, 120)]

    def test_gauss(self) -> None:
        for sizes, ts in product(self.sizes, self.trainsum):
            xp = ts.namespace
            grid = get_grid(ts, sizes, -10, 10)
            idxs = get_idxs(ts, grid)
            coords = grid.to_coords(idxs)

            shape = ts.trainshape(*grid.dims)
            data = xp.exp(-xp.sum(coords**2, axis=0))
            train = ts.tensortrain(shape, data)

            res = ts.min_max(train, 16)
            min_val = xp.min(data)
            max_val = xp.max(data)

            self.assertLess(abs(min_val - res.min_val), 1e-6)
            self.assertLess(abs(max_val - res.max_val), 1e-6)

    def test_constant_train(self) -> None:
        for ts in self.trainsum:
            shape = ts.trainshape(64)
            train = ts.full(shape, 3.5)

            res = ts.min_max(train, 1)

            self.assertLess(abs(res.min_val - 3.5), 1e-12)
            self.assertLess(abs(res.max_val - 3.5), 1e-12)


if __name__ == "__main__":
    unittest.main()
