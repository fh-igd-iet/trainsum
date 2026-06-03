import unittest
from itertools import product

from trainsum import TrainSum
from utils import backends

from typing import Sequence

class TestWavelet(unittest.TestCase):
    def setUp(self):
        self.coeffs = [[float(i) for i in range(1, 53)],
                       [float(i) for i in range(1, 20)],
                       [float(i) for i in range(1, 32)],
                       [float(i) for i in range(1, 40)]]
        self.trainsum = [TrainSum(backend) for backend in backends]
        self.sizes = [64, 100, 128, 2**10]
        #self.sizes = [8]

    def ground_truth(self, xp, coefficients: Sequence[float], n: int):
        n_coeffs = len(coefficients)
        mat = xp.zeros((n, n))
        for i, j in enumerate(range(0, n, 2)):
            js = [x % n for x in range(j, j + n_coeffs)]
            mat[i, js] = coefficients
        odds = xp.ones(n_coeffs)
        odds[1::2] = odds[1::2] * -1
        rcoeffs = coefficients[::-1] * odds
        for i, j in enumerate(range(0, n, 2)):
            i += n // 2
            js = [x % n for x in range(j, j + n_coeffs)]
            mat[i, js] = rcoeffs
        return mat

    def test_wavelet(self):
        for coeffs, ts, size in product(self.coeffs, self.trainsum, self.sizes):
            xp = ts.namespace
            dim = ts.dimension(size)

            train = ts.dwt(dim, coeffs)
            approx = train.to_tensor()

            exact = self.ground_truth(xp, coeffs, size)
            diff = abs(xp.sum((exact - approx) ** 2))

            self.assertLess(diff, 1e-10)


if __name__ == "__main__":
    unittest.main()
