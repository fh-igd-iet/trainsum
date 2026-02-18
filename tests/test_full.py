import unittest
from itertools import product

from trainsum import TrainSum
from utils import backends

class TestFull(unittest.TestCase):

    def setUp(self):
        self.trainsum = [TrainSum(backend) for backend in backends]
        self.sizes = [120, 256, 280, 1024]

    def test_full(self):
        vals = [-1.0, 0.0, 1.0]
        for ts, size, val in product(self.trainsum, self.sizes, vals):
            xp = ts.namespace

            shape = ts.trainshape(size)
            exact = xp.full([size], val)
            train = ts.full(shape, val)
            approx = train.to_tensor()
            diff = abs(xp.sum((exact - approx)**2))
            self.assertLess(diff, 1e-7)

if __name__ == "__main__":
    unittest.main()
