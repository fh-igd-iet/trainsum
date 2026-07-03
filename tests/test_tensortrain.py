from typing import Any
import unittest
from itertools import product
from copy import deepcopy

from trainsum import TrainSum
from utils import (
    backends,
    get_grid,
    get_idxs,
    rand_cores,
    assert_relative_error_less,
)


class TestTensorTrain(unittest.TestCase):
    def setUp(self):
        self.trainsum = [TrainSum(backend) for backend in backends]
        self.sizes = [(120,), (280,), (1024,), (120, 1024), (324, 120)]

    def left_contract(self, ts: TrainSum, train: Any, idx: int):
        xp = ts.namespace
        tmp = xp.ones((1, 1))
        for i in range(idx):
            idxs = [i + 1 for i in range(len(train.shape.middle(i)))]
            mid = xp.tensordot(train.cores[i], train.cores[i], axes=(idxs, idxs))
            tmp = xp.tensordot(tmp, mid, axes=([0, 1], [0, 2]))
        return tmp

    def right_contract(self, ts: TrainSum, train: Any, idx: int):
        xp = ts.namespace
        tmp = xp.ones((1, 1))
        for i in range(len(train.shape) - 1, idx + 1, -1):
            idxs = [i + 1 for i in range(len(train.shape.middle(i)))]
            mid = xp.tensordot(train.cores[i], train.cores[i], axes=(idxs, idxs))
            tmp = xp.tensordot(mid, tmp, axes=([1, 3], [0, 1]))
        return tmp

    def test_extend(self) -> None:
        for ts, sizes1, sizes2 in product(self.trainsum, self.sizes, self.sizes):
            xp = ts.namespace

            shape1 = ts.trainshape(*sizes1, mode="block")
            cores1 = rand_cores(ts, shape1)
            train1 = ts.tensortrain(shape1, cores1)

            shape2 = ts.trainshape(*sizes2, mode="interleaved")
            cores2 = rand_cores(ts, shape2)
            train2 = ts.tensortrain(shape2, cores2)

            res = deepcopy(train1)
            res.extend(train2)
            core_iter = iter(res.cores)

            self.assertEqual(
                list(train1.shape.dims) + list(train2.shape.dims), res.shape.dims
            )
            for core in train1.cores:
                self.assertTrue(xp.all(xp.equal(core, next(core_iter))))
            for core in train2.cores:
                self.assertTrue(xp.all(xp.equal(core, next(core_iter))))

    def test_conj(self) -> None:
        for ts, sizes in product(self.trainsum, self.sizes):
            xp = ts.namespace
            ctype = xp.__array_namespace_info__().dtypes()["complex128"]

            shape = ts.trainshape(*sizes, mode="block")
            cores = rand_cores(ts, shape)
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
            cores = rand_cores(ts, shape)
            train = ts.tensortrain(shape, cores)

            for i in range(len(shape)):
                train.normalize(i)

                left = self.left_contract(ts, train, i)
                eye_left = xp.eye(left.shape[0])
                diff = xp.sum((left - eye_left) ** 2)
                self.assertLess(diff, 1e-7)

                right = self.right_contract(ts, train, i)
                eye_right = xp.eye(right.shape[0])
                diff = xp.sum((right - eye_right) ** 2)
                self.assertLess(diff, 1e-7)

    def test_truncate(self) -> None:
        for ts, sizes in product(self.trainsum, self.sizes):
            shape = ts.trainshape(*sizes, mode="block")
            cores = rand_cores(ts, shape)
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
            grid = get_grid(ts, sizes, -10.0, 10.0)
            idxs = get_idxs(ts, grid)
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
            diff = xp.sum((exact - approx) ** 2) / xp.sum(exact**2)
            self.assertLess(diff, 1e-5)

    def test_getitem(self) -> None:
        for ts, sizes in product(self.trainsum, self.sizes):
            xp = ts.namespace
            shape = ts.trainshape(*sizes, mode="block")
            cores = rand_cores(ts, shape)
            train = ts.tensortrain(shape, cores)
            exact = train.to_tensor()

            point = tuple(size // 2 for size in sizes)
            point_val = train[point]
            assert_relative_error_less(self, ts, exact[point], point_val[0], 1e-12)

            slice_cut = tuple(slice(size // 4, None, 2) for size in sizes)
            slice_train = train[slice_cut]
            self.assertEqual(slice_train.to_tensor().shape, exact[slice_cut].shape)
            assert_relative_error_less(
                self, ts, exact[slice_cut], slice_train.to_tensor(), 5e-5
            )

            ellipsis_train = train[..., slice(sizes[-1] // 4, None, 3)]
            assert_relative_error_less(
                self,
                ts,
                exact[..., slice(sizes[-1] // 4, None, 3)],
                ellipsis_train.to_tensor(),
                5e-5,
            )

            singletons = tuple(slice(idx, idx + 1) for idx in point)
            single_val = train[singletons]
            assert_relative_error_less(self, ts, exact[singletons], single_val, 1e-12)

            vec_cut = tuple(
                xp.asarray([0, size // 2, size - 1], dtype=ts.index_type)
                for size in sizes
            )
            vec_val = train[vec_cut]
            assert_relative_error_less(self, ts, exact[vec_cut], vec_val, 1e-12)

            idxs = xp.stack(vec_cut, axis=0)
            idx_val = train[idxs]
            assert_relative_error_less(self, ts, exact[vec_cut], idx_val, 1e-12)

    def test_assign(self) -> None:
        for ts, sizes in product(self.trainsum, self.sizes):
            shape = ts.trainshape(*sizes, mode="block")
            cores = rand_cores(ts, shape)
            base_train = ts.tensortrain(shape, cores)

            cuts = [tuple(slice(16, None, step) for _ in sizes) for step in [2, 4]]
            cuts.append((Ellipsis, slice(16, None, 3)))
            if len(sizes) > 1:
                cuts.append((slice(16, None, 3), slice(None)))

            for cut in cuts:
                train = deepcopy(base_train)
                exact = train.to_tensor()
                exact[cut] = exact[cut] ** 2

                decomp = ts.qrdecomposition()
                with ts.decomposition(decomposition=decomp):
                    sl_train = train[cut] ** 2
                    train[cut] = sl_train
                approx = train.to_tensor()

                assert_relative_error_less(self, ts, exact, approx, 5e-5)


if __name__ == "__main__":
    unittest.main()
