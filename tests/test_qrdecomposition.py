import unittest
from itertools import product

from trainsum import TrainSum
from utils import backends, rand_data

class TestSVDecomposition(unittest.TestCase):

    def setUp(self):
        self.trainsum = [TrainSum(backend) for backend in backends]
        self.sizes = [(120, 120), (256, 250), (120, 7), (256, 800)]

    def test_construction(self):
        for ts in self.trainsum:
            decomp = ts.qrdecomposition()
    
    def test_decompose(self):
        for ts, sizes in product(self.trainsum, self.sizes):
            xp = ts.namespace
            mat = rand_data(xp, *sizes)
            decomp = ts.qrdecomposition()

            res = decomp.left(mat)
            q, r = xp.linalg.qr(mat, mode='reduced')
            self.assertTrue(xp.max(res.left - q) < 1e-10)
            self.assertTrue(xp.max(res.right - r) < 1e-10)

            res = decomp.right(mat)
            r, q = xp.linalg.qr(mat.T, mode='reduced')
            q, r = q.T, r.T
            self.assertTrue(xp.max(res.left - q) < 1e-10)
            self.assertTrue(xp.max(res.right - r) < 1e-10)

if __name__ == '__main__':
    unittest.main()
