from typing import Sequence, Any
import unittest
from itertools import product
from math import prod

from trainsum import TrainSum
from trainsum.trainshape import TrainShape
from trainsum.typing import Dimension, Digit
from utils import backends

class TestTrainShape(unittest.TestCase):

    def setUp(self):
        self.trainsum = [TrainSum(backend) for backend in backends]
        self.sizes = [(120,), (256,), (120, 7, 10), (17, 256, 800)]
        self.modes = ("block", "interleaved")

    def block_digits(self, dims: Sequence[Dimension]) -> Sequence[Sequence[Digit]]:
        return [(d,) for dim in dims for d in dim]

    def interleaved_digits(self, dims: Sequence[Dimension]) -> Sequence[Sequence[Digit]]:
        digits = []
        for i in range(max(len(dim) for dim in dims)):
            digits.append([])
            for dim in dims:
                if i < len(dim):
                    digits[-1].append(dim[i])
        return digits

    def exact_ranks(self, shape: Sequence[Sequence[Digit]]) -> Sequence[int]:
        left = [prod(d.base for d in shape[0])]
        for i in range(1, len(shape)-1):
            left.append(left[-1] * prod(d.base for d in shape[i]))

        right = [prod(d.base for d in shape[-1])]
        for i in range(1, len(shape)-1)[::-1]:
            right.append(right[-1] * prod(d.base for d in shape[i]))
        right.reverse()

        return [min(l, r) for l, r in zip(left, right)]

    def get_shapes(self, ts: Any, dims: Sequence[Dimension], mode: str) -> tuple[TrainShape, Sequence[Sequence[Digit]]]:
        if mode == "block":
            return ts.trainshape(*dims, mode=mode), self.block_digits(dims)
        elif mode == "interleaved":
            return ts.trainshape(*dims, mode=mode), self.interleaved_digits(dims)
        raise ValueError(f"Invalid mode: {mode}")

    # -------------------------------------
    # tests

    def test_construction(self):
        for ts, sizes, mode in product(self.trainsum, self.sizes, self.modes):
            dims = [ts.dimension(size) for size in sizes]
            shape, dgts = self.get_shapes(ts, dims, mode)
            self.assertRaises(ValueError, ts.trainshape, *dims, mode="invalid_mode")

            tmp = [ts.dimension(size) for size in sizes]
            self.assertRaises(ValueError, ts.trainshape, *dims, digits=dgts[:-1]) # not all digits
            self.assertRaises(ValueError, ts.trainshape, *tmp, digits=dgts) # digits don't match dims

            self.assertEqual(len(shape), len(dgts))
            self.assertEqual(shape.dims, dims)
            self.assertTrue(all(d1==d2 for d1, d2 in zip(dgts, shape.digits)))
            self.assertTrue(all(d1==d2 for d1, d2 in zip(dgts, ts.trainshape(*dims, digits=dgts).digits)))
    
    def test_middle(self):
        for ts, sizes, mode in product(self.trainsum, self.sizes, self.modes):
            dims = [ts.dimension(size) for size in sizes]
            shape, dgts = self.get_shapes(ts, dims, mode)
            for i, dgts in enumerate(shape.digits):
                self.assertEqual(shape.middle(i), [d.base for d in dgts])

    def test_exact_ranks(self):
        for ts, sizes, mode in product(self.trainsum, self.sizes, self.modes):
            dims = [ts.dimension(size) for size in sizes]
            shape, _ = self.get_shapes(ts, dims, mode)
            branks = self.exact_ranks(shape.digits)
            shape.ranks = None
            self.assertEqual(shape.ranks, branks)
            self.assertEqual([shape.left_rank(i) for i in range(len(shape))], [1, *branks])
            self.assertEqual([shape.right_rank(i) for i in range(len(shape))], [*branks, 1])

    def test_max_ranks(self):
        mrank = 2
        for ts, sizes, mode in product(self.trainsum, self.sizes, self.modes):
            dims = [ts.dimension(size) for size in sizes]
            shape, _ = self.get_shapes(ts, dims, mode)
            branks = [mrank] * (len(shape)-1)
            shape.ranks = mrank
            self.assertEqual(max(shape.ranks), mrank)
            self.assertEqual([shape.left_rank(i) for i in range(len(shape))], [1, *branks])
            self.assertEqual([shape.right_rank(i) for i in range(len(shape))], [*branks, 1])

    def test_reverse(self):
        for ts, sizes, mode in product(self.trainsum, self.sizes, self.modes):
            dims = [ts.dimension(size) for size in sizes]
            shape, _ = self.get_shapes(ts, dims, mode)
            shape_rev = shape.reverse()
            rev_dgts = [list(reversed(digits)) for digits in reversed(shape.digits)]
            self.assertEqual(shape_rev.digits, rev_dgts)
            self.assertEqual(shape_rev.ranks, shape.ranks[::-1])

if __name__ == '__main__':
    unittest.main()
