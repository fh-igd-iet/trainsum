import unittest
from itertools import product

from trainsum import TrainSum
from utils import backends

class TestQftShift(unittest.TestCase):

    def setUp(self):
        self.trainsum = [TrainSum(backend) for backend in backends]
        self.sizes = [120, 256, 280, 1024]

    def test_qftshift(self):
        vals = [1.0, 2.0, 11.0]
        for ts, size, val in product(self.trainsum, self.sizes, vals):
            xp = ts.namespace
            dim = ts.dimension(size)

            exact = xp.fft.fftfreq(size, d=val) # type: ignore
            train = ts.qftfreq(dim, val)

            approx = train.to_tensor()
            diff = abs(xp.sum((exact - approx)**2) / xp.sum(exact**2))
            self.assertLess(diff, 1e-7)

if __name__ == "__main__":
    unittest.main()
