import unittest
from itertools import product

from trainsum import TrainSum
from utils import backends

class TestPolyval(unittest.TestCase):

    def setUp(self):
        self.trainsum = [TrainSum(backend) for backend in backends]
        self.sizes = [120, 256, 280, 1024]

    def test_polyval(self):
        coeffs = [[1.0], [0.8, 1.0], [1.0, 0.2, 0.1]]
        offsets = [0.0, 0.5]
        for ts, size, coeffs, offset in product(self.trainsum, self.sizes, coeffs, offsets):
            xp = ts.namespace
            dim = ts.dimension(size)
            domain = ts.domain(-1.0, 1.0)
            grid = ts.uniform_grid(dim, domain)

            x = xp.linspace(domain.lower, domain.upper, dim.size())
            exact = xp.zeros_like(x)
            for i, coeff in enumerate(reversed(coeffs)):
                exact += coeff * (x-offset)**i
            train = ts.polyval(grid, coeffs, offset)

            approx = train.to_tensor()
            diff = abs(xp.sum((exact - approx)**2))
            self.assertLess(diff, 1e-7)

if __name__ == "__main__":
    unittest.main()
