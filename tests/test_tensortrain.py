from typing import Sequence, Any
import unittest
from itertools import product
from copy import deepcopy

from trainsum import TrainSum
from trainsum.typing import UniformGrid, TrainShape
from utils import backends, rand_data

class TestTensorTrain(unittest.TestCase):

    def setUp(self):
        self.trainsum = [TrainSum(backend) for backend in backends]
        self.sizes = [(120,), (280,), (1024,), (120, 1024), (324, 120)]

    def get_grid(self, ts, sizes: Sequence[int], lower: float, upper: float):
        dims = [ts.dimension(size) for size in sizes]
        domains = [ts.domain(lower, upper) for _ in sizes]
        return ts.uniform_grid(dims, domains)

    def get_idxs(self, ts, grid: UniformGrid):
        xp = ts.namespace
        idxs = xp.zeros([len(grid.dims), *[dim.size() for dim in grid.dims]],
                        dtype=ts.index_type)
        for i, dim in enumerate(grid.dims):
            cut = (*(xp.newaxis,) * i, slice(None), *(xp.newaxis,) * (len(grid.dims) - i - 1))
            idxs[i] += xp.arange(dim.size(), dtype=ts.index_type)[cut]
        return idxs
    
    def rand_cores(self, ts: TrainSum, shape: TrainShape):
        xp = ts.namespace
        cores = []
        for i in range(len(shape)):
            left = 1 if i == 0 else 10
            right = 1 if i == len(shape) - 1 else 10
            cores.append(xp.asarray(rand_data(xp, left, *shape.middle(i), right)))
        return cores

    def left_contract(self, ts: TrainSum, train: Any, idx: int):
        xp = ts.namespace
        tmp = xp.ones((1, 1))
        for i in range(idx):
            idxs = [i+1 for i in range(len(train.shape.middle(i)))]
            mid = xp.tensordot(train.cores[i], train.cores[i], axes=(idxs, idxs))
            tmp = xp.tensordot(tmp, mid, axes=([0, 1], [0, 2]))
        return tmp

    def right_contract(self, ts: TrainSum, train: Any, idx: int):
        xp = ts.namespace
        tmp = xp.ones((1, 1))
        for i in range(len(train.shape)-1, idx+1, -1):
            idxs = [i+1 for i in range(len(train.shape.middle(i)))]
            mid = xp.tensordot(train.cores[i], train.cores[i], axes=(idxs, idxs))
            tmp = xp.tensordot(mid, tmp, axes=([1, 3], [0, 1]))
        return tmp


    def test_extend(self) -> None:
        for ts, sizes1, sizes2 in product(self.trainsum, self.sizes, self.sizes):
            xp = ts.namespace

            shape1 = ts.trainshape(*sizes1, mode="block")
            cores1 = self.rand_cores(ts, shape1)
            train1 = ts.tensortrain(shape1, cores1)

            shape2 = ts.trainshape(*sizes2, mode="interleaved")
            cores2 = self.rand_cores(ts, shape2)
            train2 = ts.tensortrain(shape2, cores2)

            res = deepcopy(train1)
            res.extend(train2)
            core_iter = iter(res.cores)

            self.assertEqual(list(train1.shape.dims) + list(train2.shape.dims), res.shape.dims)
            for core in train1.cores:
                self.assertTrue(xp.all(xp.equal(core, next(core_iter))))
            for core in train2.cores:
                self.assertTrue(xp.all(xp.equal(core, next(core_iter))))

    def test_conj(self) -> None:
        for ts, sizes in product(self.trainsum, self.sizes):
            xp = ts.namespace
            ctype = xp.__array_namespace_info__().dtypes()["complex128"]

            shape = ts.trainshape(*sizes, mode="block")
            cores = self.rand_cores(ts, shape)
            train = ts.tensortrain(shape, cores)
            train.dtype = ctype

            res = train.conj()
            core_iter = iter(res.cores)

            self.assertEqual(train.shape.dims, res.shape.dims)
            for core in train.cores:
                self.assertTrue(xp.all(xp.equal(xp.conj(core), next(core_iter))))

    def test_normalize(self) -> None:
        for ts, sizes in product(self.trainsum, self.sizes):
            xp = ts.namespace

            shape = ts.trainshape(*sizes, mode="block")
            cores = self.rand_cores(ts, shape)
            train = ts.tensortrain(shape, cores)

            for i in range(len(shape)):
                train.normalize(i)

                left = self.left_contract(ts, train, i)
                eye_left = xp.eye(left.shape[0])
                diff = xp.sum((left - eye_left)**2)
                self.assertLess(diff, 1e-7)

                right = self.right_contract(ts, train, i)
                eye_right = xp.eye(right.shape[0])
                diff = xp.sum((right - eye_right)**2)
                self.assertLess(diff, 1e-7)

    def test_truncate(self) -> None:
        for ts, sizes in product(self.trainsum, self.sizes):
            shape = ts.trainshape(*sizes, mode="block")
            cores = self.rand_cores(ts, shape)
            train = ts.tensortrain(shape, cores)

            with ts.decomposition(max_rank=5):
                train.truncate()
            self.assertLessEqual(max(train.shape.ranks), 5)

            with ts.variational(max_rank=5):
                train.truncate()
            self.assertLessEqual(max(train.shape.ranks), 5)

    def test_transform(self) -> None:
        for ts, sizes in product(self.trainsum, self.sizes):
            xp = ts.namespace
            grid = self.get_grid(ts, sizes, -10.0, 10.0)
            idxs = self.get_idxs(ts, grid)
            coords = grid.to_coords(idxs)

            data = xp.sum(coords**2, axis=0)
            shape = ts.trainshape(*sizes, mode="block")
            with ts.variational(max_rank=15, cutoff=1e-10, nsweeps=1, ncores=2):
                train = ts.tensortrain(shape, data)

            func = lambda x: x**2
            with ts.cross(max_rank=32, eps=1e-10):
                res = train.transform(func)

            exact = func(train.to_tensor())
            approx = res.to_tensor()
            diff = xp.sum((exact - approx)**2) / xp.sum(exact**2)
            self.assertLess(diff, 1e-5)

if __name__ == "__main__":
    unittest.main()
