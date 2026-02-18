import unittest
from itertools import product

from trainsum import TrainSum
from utils import backends

class TestExp(unittest.TestCase):

    def setUp(self):
        self.trainsum = [TrainSum(backend) for backend in backends]
        self.sizes = [120, 256, 280, 1024]

    def test_exp(self):
        vals = [-1.0, 0.0, 1.0]
        offsets = [0.0, 0.5]
        for ts, size, val, offset in product(self.trainsum, self.sizes, vals, offsets):
            xp = ts.namespace
            dim = ts.dimension(size)
            domain = ts.domain(-1.0, 1.0)
            grid = ts.uniform_grid(dim, domain)

            exact = xp.linspace(domain.lower, domain.upper, dim.size())
            exact = xp.exp(val * (exact - offset))
            train = ts.exp(grid, val, offset)

            approx = train.to_tensor()
            diff = abs(xp.sum((exact - approx)**2))
            self.assertLess(diff, 1e-7)

if __name__ == "__main__":
    unittest.main()
