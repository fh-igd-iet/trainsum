import unittest

from trainsum import TrainSum
from utils import backends

class TestDomain(unittest.TestCase):

    def setUp(self):
        self.trainsum = [TrainSum(backend) for backend in backends]

    def test_construction(self):
        for ts in self.trainsum:
            self.assertRaises(ValueError, ts.domain, 1.0, 1.0)
            self.assertRaises(ValueError, ts.domain, 2.0, 1.0)
            lower = 1.0
            upper = 2.0
            domain = ts.domain(lower, upper)
            self.assertEqual(domain.lower, lower)
            self.assertEqual(domain.upper, upper)
            self.assertEqual(domain.diff, upper - lower)

    def test_eq(self):
        for ts in self.trainsum:
            domain1 = ts.domain(1.0, 2.0)
            domain2 = ts.domain(1.0, 2.0)
            domain3 = ts.domain(0.0, 2.0)
            self.assertEqual(domain1, domain2)
            self.assertNotEqual(domain1, domain3)

if __name__ == '__main__':
    unittest.main()
