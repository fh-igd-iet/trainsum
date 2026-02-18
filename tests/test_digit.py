import unittest
from math import prod

from trainsum.digit import Digit
from utils import prime_factorization

class TestDigit(unittest.TestCase):

    def setUp(self) -> None:
        idfs = [0, 1, 2]
        sizes = [6, 10, 15]
        self.digits = []
        for idf, size in zip(idfs, sizes):
            bases = prime_factorization(size)
            factors = [prod(bases[i+1:]) for i in range(len(bases))]
            for i in range(len(bases)):
                self.digits.append(Digit(idf, i, bases[i], factors[i]))

    def test_construction(self):
        self.assertRaises(ValueError, Digit, 0, -1, 2, 1)
        self.assertRaises(ValueError, Digit, 0, 0, -1, 1)
        self.assertRaises(ValueError, Digit, 0, 0, 0, 1)
        self.assertRaises(ValueError, Digit, 0, 0, 1, 1)
        self.assertRaises(ValueError, Digit, 0, 0, 1, 0)
        self.assertRaises(ValueError, Digit, 0, 0, 1, -1)

    def test_eq(self):
        for i in range(len(self.digits)):
            self.assertEqual(self.digits[i], self.digits[i])
            for j in range(i + 1, len(self.digits)):
                self.assertNotEqual(self.digits[i], self.digits[j])

    def test_hash(self):
        self.assertEqual(len(self.digits), len(set(self.digits)))
        self.assertEqual(len(self.digits), len(set(self.digits+self.digits)))

if __name__ == '__main__':
    unittest.main()
