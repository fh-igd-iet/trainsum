import unittest
from itertools import product
from trainsum import TrainSum
from utils import backends

class TestLinearIntegerEquation(unittest.TestCase):
    def setUp(self):
        self.trainsum = [TrainSum(backend) for backend in backends]
        self.ndims = [1, 2, 3]
        self.sizes = [120, 256, 280, 1024]

    def test_init(self):
        for ts, ndim, size in product(self.trainsum, self.ndims, self.sizes):
            dim = ts.dimension(size)
            eq = ts.linear_integer_equation((dim,), (1,), rhs=0)

            # One entry in evaluated per dimension
            self.assertEqual(len(eq.evaluated), 1)
            # One entry per digit in the dimension
            self.assertEqual(len(eq.evaluated[0]), len(dim))
            # All digits are initially None (unevaluated)
            self.assertTrue(all(v is None for v in eq.evaluated[0]))

if __name__ == "__main__":
    unittest.main()
