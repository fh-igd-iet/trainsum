import unittest
from itertools import product

from trainsum import TrainSum
from utils import backends

class TestQftShift(unittest.TestCase):

    def setUp(self):
        self.trainsum = [TrainSum(backend) for backend in backends]
        self.sizes = [120, 256, 280, 1024]

    def test_qftshift(self):
        for ts, size in product(self.trainsum, self.sizes):
            xp = ts.namespace
            dim = ts.dimension(size)
            domain = ts.domain(-1.0, 2.0)
            grid = ts.uniform_grid(dim, domain)

            coeffs = [1.0, 0.2, 0.1]
            offset = 0.2
            x = xp.linspace(domain.lower, domain.upper, dim.size())
            exact = xp.zeros_like(x)
            for i, coeff in enumerate(reversed(coeffs)):
                exact += coeff * (x-offset)**i
            train = ts.polyval(grid, coeffs, offset)

            exact = xp.fft.fftshift(exact) # type: ignore
            train = ts.qftshift(train)
            approx = train.to_tensor()
            diff = abs(xp.sum((exact - approx)**2) / xp.sum(exact**2))
            self.assertLess(diff, 1e-7)

            exact = xp.fft.ifftshift(exact) # type: ignore
            train = ts.iqftshift(train)
            approx = train.to_tensor()
            diff = abs(xp.sum((exact - approx)**2) / xp.sum(exact**2))
            self.assertLess(diff, 1e-7)

if __name__ == "__main__":
    unittest.main()
