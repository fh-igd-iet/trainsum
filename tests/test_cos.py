import unittest
from itertools import product

from trainsum import TrainSum
from utils import backends

class TestCos(unittest.TestCase):

    def setUp(self):
        self.trainsum = [TrainSum(backend) for backend in backends]
        self.sizes = [120, 256, 280, 1024]

    def test_cos(self):
        vals = [100, 10, 1]
        offsets = [0.0, 0.5]
        for ts, size, val, offset in product(self.trainsum, self.sizes, vals, offsets):
            xp = ts.namespace
            dim = ts.dimension(size)
            domain = ts.domain(-1.0, 1.0)
            grid = ts.uniform_grid(dim, domain)

            x = xp.linspace(domain.lower, domain.upper, dim.size())
            exact = xp.cos(val*(x-offset))
            train = ts.cos(grid, val, offset)

            approx = train.to_tensor()
            diff = abs(xp.sum((exact - approx)**2))
            self.assertLess(diff, 1e-7)

if __name__ == "__main__":
    unittest.main()
