import unittest

from trainsum import TrainSum
from utils import backends


class TestRangeIntegerEquation(unittest.TestCase):
    def test_binary_train_covers_full_range(self):
        for backend in backends:
            ts = TrainSum(backend)
            xp = ts.namespace

            dim = ts.dimension(16)
            shape = ts.trainshape(dim)
            eq = ts.range_integer_equation((dim,), lower=(0,), upper=(16,))

            train = ts.binary_train(shape, [eq])
            exact = xp.ones((16,))
            approx = train.to_tensor()

            diff = abs(xp.sum((exact - approx) ** 2))
            self.assertLess(diff, 1e-7)


if __name__ == "__main__":
    unittest.main()
