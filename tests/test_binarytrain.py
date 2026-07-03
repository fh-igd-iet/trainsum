import unittest

from trainsum import TrainSum
from utils import backends, assert_exact


class TestBinaryTrain(unittest.TestCase):
    def setUp(self):
        self.trainsum = [TrainSum(backend) for backend in backends]

    def test_linear_integer_equation_constraint(self):
        for ts in self.trainsum:
            xp = ts.namespace
            dim1 = ts.dimension(8)
            dim2 = ts.dimension(8)
            shape = ts.trainshape(dim1, dim2)
            eq = ts.linear_integer_equation((dim1, dim2), (1, -1), rhs=0)

            train = ts.binary_train(shape, [eq])
            exact = xp.eye(dim1.size())
            assert_exact(self, ts, exact, train.to_tensor())

    def test_modulo_integer_equation(self):
        for ts in self.trainsum:
            xp = ts.namespace
            dim = ts.dimension(16)
            shape = ts.trainshape(dim)
            eq = ts.modulo_integer_equation((dim,), (4,), rhs=2)

            train = ts.binary_train(shape, [eq])
            exact = xp.zeros((dim.size(),))
            exact[2::4] = 1.0
            assert_exact(self, ts, exact, train.to_tensor())

    def test_binary_train_multiple_equations_are_summed(self):
        for ts in self.trainsum:
            xp = ts.namespace
            dim1 = ts.dimension(16)
            dim2 = ts.dimension(16)
            shape = ts.trainshape(dim1, dim2)

            eq1 = ts.linear_integer_equation((dim1, dim2), (1, 1), rhs=3)
            eq2 = ts.range_integer_equation((dim1, dim2), lower=(1, 0), upper=(4, 3))

            train = ts.binary_train(shape, [eq1, eq2])
            exact = xp.zeros((dim1.size(), dim2.size()))
            for i in range(dim1.size()):
                for j in range(dim2.size()):
                    if i + j == 3:
                        exact[i, j] += 1.0
                    if 1 <= i < 4 and 0 <= j < 3:
                        exact[i, j] += 1.0
            assert_exact(self, ts, exact, train.to_tensor())

    def test_binary_train_no_solution(self):
        for ts in self.trainsum:
            xp = ts.namespace
            dim = ts.dimension(8)
            shape = ts.trainshape(dim)
            eq = ts.linear_integer_equation((dim,), (1,), rhs=dim.size())

            train = ts.binary_train(shape, [eq])
            exact = xp.zeros((dim.size(),))
            assert_exact(self, ts, exact, train.to_tensor())


if __name__ == "__main__":
    unittest.main()
