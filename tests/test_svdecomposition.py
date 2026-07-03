import unittest
from itertools import product
import numpy as np

from trainsum import TrainSum
from utils import backends, rand_data


class TestSVDecomposition(unittest.TestCase):
    def setUp(self):
        self.trainsum = [TrainSum(backend) for backend in backends]
        self.sizes = [(120, 120), (256, 250), (120, 7), (256, 800)]

    def test_construction(self):
        for ts in self.trainsum:
            decomp = ts.svdecomposition(max_rank=2, cutoff=1e-10)
            self.assertRaises(ValueError, ts.svdecomposition, max_rank=-1, cutoff=1e-10)
            self.assertRaises(ValueError, ts.svdecomposition, max_rank=2, cutoff=-1e-10)

    def test_decompose(self):
        mrank = 2
        cutoff = 1e-10
        for ts, sizes in product(self.trainsum, self.sizes):
            xp = ts.namespace
            decomp = ts.svdecomposition(max_rank=mrank, cutoff=cutoff)

            mat = xp.asarray(rand_data(xp, *sizes))
            res = decomp.left(mat)
            u, s, vh = xp.linalg.svd(mat, full_matrices=False)
            numel = min(mrank, xp.sum(s > cutoff))

            ref_left = u[:, :numel]
            ref_right = s[:numel, xp.newaxis] * vh[:numel, :]
            self.assertTrue(xp.max(res.left - ref_left) < 1e-10)
            self.assertTrue(xp.max(res.right - ref_right) < 1e-10)

            res = decomp.right(mat)
            ref_left = u[:, :numel] * s[xp.newaxis, :numel]
            ref_right = vh[:numel, :]
            self.assertTrue(xp.max(res.left - ref_left) < 1e-10)
            self.assertTrue(xp.max(res.right - ref_right) < 1e-10)

    def test_randomized_decompose(self):
        rank = 2
        for ts, sizes in product(self.trainsum, self.sizes):
            xp = ts.namespace
            self.assertRaises(ValueError, ts.rand_svdecomposition, max_rank=0)

            left = xp.asarray(rand_data(xp, sizes[0], rank))
            right = xp.asarray(rand_data(xp, rank, sizes[1]))
            mat = left @ right

            np.random.seed(0)
            decomp = ts.rand_svdecomposition(max_rank=rank + 1)
            res = decomp.left(mat)
            approx = res.left @ res.right
            diff = xp.sum((mat - approx) ** 2)
            norm = xp.sum(mat**2)
            self.assertLess(float(diff / norm), 1e-10)

            np.random.seed(0)
            res = decomp.right(mat)
            approx = res.left @ res.right
            diff = xp.sum((mat - approx) ** 2)
            norm = xp.sum(mat**2)
            self.assertLess(float(diff / norm), 1e-10)


if __name__ == "__main__":
    unittest.main()
