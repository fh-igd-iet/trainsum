import unittest
from itertools import product

from trainsum import TrainSum
from utils import backends

class TestShift(unittest.TestCase):

    def setUp(self):
        self.trainsum = [TrainSum(backend) for backend in backends]
        self.sizes = [20, 64, 18]

    def shift_matrix(self, xp, size: int, shift: int):
        if shift == 0:
            return xp.eye(size)
        shift_matrix = xp.zeros((size, size))
        sign = 1 if shift > 0 else -1
        shift = abs(shift)
        if sign < 0:
            idx = lambda i: (i+shift, i)
        else:
            idx = lambda i: (i, i+shift)
        for i in range(size):
            if i + shift < size:
                shift_matrix[idx(i)] = 1.0
        return shift_matrix

    def test_shift(self):
        for ts, size in product(self.trainsum, self.sizes):
            xp = ts.namespace

            shifts = list(range(-size+1, size-1))
            circular = [False, True]
            dim = ts.dimension(size)

            for shift, circ in product(shifts, circular):
                exact = self.shift_matrix(xp, dim.size(), shift)
                if circ:
                    tmp = -size+shift if shift >= 0 else size+shift
                    exact += self.shift_matrix(xp, dim.size(), tmp)

                train = ts.shift(dim, shift, circular=circ)
                approx = train.to_tensor()
                diff = abs(xp.sum((exact - approx)**2))
                self.assertLess(diff, 1e-7)

if __name__ == "__main__":
    unittest.main()
