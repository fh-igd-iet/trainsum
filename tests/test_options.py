import unittest

from trainsum import TrainSum
from trainsum.typing import OptionType
from utils import backends

class TestTensorTrain(unittest.TestCase):

    def setUp(self):
        self.trainsum = [TrainSum(backend) for backend in backends]

    def test_options(self) -> None:

        opts = [(lambda ts: ts.exact(), OptionType.EINSUM),
                (lambda ts: ts.decomposition(max_rank=10), OptionType.EINSUM),
                (lambda ts: ts.variational(max_rank=10), OptionType.EINSUM),
                (lambda ts: ts.evaluation(), OptionType.EVALUATE),
                (lambda ts: ts.cross(max_rank=10, eps=1e-10), OptionType.CROSS)]

        for ts in self.trainsum:

            for (ofunc, otype) in opts:
                with ofunc(ts) as opts1:
                    with ofunc(ts) as opts2:
                        self.assertEqual(opts2, ts.get_options(otype))
                    self.assertEqual(opts1, ts.get_options(otype))

                opt = ofunc(ts)
                ts.set_options(opt)
                self.assertEqual(opt, ts.get_options(otype))

if __name__ == "__main__":
    unittest.main()
