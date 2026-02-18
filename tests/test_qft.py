import unittest
from itertools import product
from scipy.linalg import dft

from trainsum import TrainSum
from utils import backends

class TestQft(unittest.TestCase):

    def setUp(self):
        self.trainsum = [TrainSum(backend) for backend in backends]
        self.sizes = [120, 256, 280, 1024]

    def test_qft(self):
        for ts, size in product(self.trainsum, self.sizes):
            xp = ts.namespace
            dim = ts.dimension(size)
            decomp = ts.svdecomposition(max_rank=16, cutoff=0.0)

            exact = xp.asarray(dft(dim.size(), "sqrtn"))
            train = ts.qft(dim, decomp)

            approx = train.to_tensor()
            diff = abs(xp.sum((exact - approx)**2) / xp.sum(exact**2))
            self.assertLess(diff, 1e-4)

    def test_iqft(self):
        for ts, size in product(self.trainsum, self.sizes):
            xp = ts.namespace
            dim = ts.dimension(size)
            decomp = ts.svdecomposition(max_rank=16, cutoff=0.0)

            exact = xp.conj(xp.asarray(dft(dim.size(), "sqrtn"))).T
            train = ts.iqft(dim, decomp)

            approx = train.to_tensor()
            diff = abs(xp.sum((exact - approx)**2) / xp.sum(exact**2))
            self.assertLess(diff, 1e-4)

if __name__ == "__main__":
    unittest.main()
