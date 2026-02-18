from typing import Any
import unittest
from math import prod
from itertools import product

from trainsum import TrainSum
from trainsum.digit import Digit
from trainsum.backend import get_index_dtype
from utils import prime_factorization, backends

class TestDimension(unittest.TestCase):

    def setUp(self):
        self.trainsum = [TrainSum(backend) for backend in backends]
        self.sizes = [120, 256]

    def digits(self, idf: int, size: int) -> list[Digit]:
        bases = prime_factorization(size)
        factors = [prod(bases[i+1:]) for i in range(len(bases))]
        return [Digit(idf, i, base, factor) for i, (base, factor) in enumerate(zip(bases, factors))]

    def digit_tensor(self, xp: Any, size: int) -> Any:
        int_type = get_index_dtype(xp)
        bases = prime_factorization(size)
        idxs = xp.arange(size, dtype=int_type)
        digits = xp.zeros((len(bases), *idxs.shape), dtype=int_type)
        for i, base in enumerate(reversed(bases)):
            digits[len(bases)-i-1,...] = idxs % base
            idxs //= base
        return digits

    def index_tensor(self, xp: Any, size: int) -> Any:
        int_type = get_index_dtype(xp)
        return xp.arange(0, size, dtype=int_type)

    def test_construction(self):
        for ts, size in product(self.trainsum, self.sizes):
            dim = ts.dimension(size)
            ref_digits = self.digits(dim.idf, size)
            self.assertEqual(len(dim), len(ref_digits))
            self.assertTrue(all(d1==d2 for d1, d2 in zip(dim, ref_digits)))

        for ts in self.trainsum:
            self.assertRaises(ValueError, ts.dimension, 0)
            self.assertRaises(ValueError, ts.dimension, -1)
            self.assertRaises(ValueError, ts.dimension, [])
            self.assertRaises(ValueError, ts.dimension, [-1, 2])
            self.assertRaises(ValueError, ts.dimension, [0, 2])

    def test_unique_idf(self):
        for ts in self.trainsum:
            dim1 = ts.dimension(10)
            dim2 = ts.dimension(10)
            self.assertNotEqual(dim1.idf, dim2.idf)

    def test_eq(self):
        for ts in self.trainsum:
            dims = [ts.dimension(size) for size in self.sizes]
            for i in range(len(dims)):
                self.assertEqual(dims[i], dims[i])
                for j in range(i+1, len(dims)):
                    self.assertNotEqual(dims[i], dims[j])

    def test_to_idxs(self):
        for (xp, ts), size in product(zip(backends, self.trainsum), self.sizes):
            ref_idxs = self.index_tensor(xp, size)

            dim = ts.dimension(size)
            digits = self.digit_tensor(xp, size)
            idxs = dim.to_idxs(digits)
            self.assertTrue(xp.all(idxs == ref_idxs))

    def test_to_digits(self):
        for (xp, ts), size in product(zip(backends, self.trainsum), self.sizes):
            ref_digits = self.digit_tensor(xp, size)

            dim = ts.dimension(size)
            idxs = self.index_tensor(xp, size)
            digits = dim.to_digits(idxs)
            self.assertTrue(xp.all(digits == ref_digits))
    
if __name__ == '__main__':
    unittest.main()
