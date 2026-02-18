import unittest
from copy import deepcopy
import opt_einsum as oe

from trainsum import TrainSum
from trainsum.typing import Dimension, TensorTrain, TrainShape
from utils import backends

class TestEinsum(unittest.TestCase):

    def setUp(self):
        self.trainsum = [TrainSum(backend) for backend in backends]

    def poly_val(self, ts: TrainSum, dim: Dimension, coeffs: list[float]) -> TensorTrain:
        domain = ts.domain(-1, 1)
        grid = ts.uniform_grid(dim, domain)
        vec = ts.polyval(grid, coeffs, 0.0)
        return vec

    def compare_vals(self, ts:TrainSum, msg: str, eq_str: str, guess: float | TensorTrain, *ops: TensorTrain) -> None:
        xp = ts.namespace
        if not isinstance(guess, (int, float)):
            res = guess.to_tensor()
        else:
            res = guess
        exact = oe.contract(eq_str, *[op.to_tensor() for op in ops])
        diff = xp.sum((res - exact)**2)
        self.assertLess(diff, 1e-6, f"{msg} relative error {diff:.2E} is too large")

    def compare(self, ts: TrainSum, eq_str: str, *ops: TensorTrain, result: None | TrainShape = None) -> None:
        shapes = [op.shape for op in ops]

        if eq_str.split("->")[1]  == "":
            expr = ts.einsum_expression(eq_str, *shapes)
            res = expr(*ops)
            self.compare_vals(ts, "full", eq_str, res, *ops)
            return

        max_rank, cutoff, ncores, nsweeps = 5, 1e-10, 2, 1

        with ts.exact():
            exact_expr = ts.einsum_expression(eq_str, *shapes)

        with ts.decomposition(max_rank=max_rank, cutoff=cutoff, ncores=ncores):
            decomp_expr = ts.einsum_expression(eq_str, *shapes, result_shape=result)

        with ts.variational(max_rank=max_rank, cutoff=cutoff, ncores=ncores, nsweeps=nsweeps):
            var_expr = ts.einsum_expression(eq_str, *shapes, result_shape=result)

        with ts.evaluation():
            ev_expr = ts.evaluate_expression(eq_str, *shapes)

        exact_train = exact_expr(*ops)
        self.compare_vals(ts, "exact", eq_str, exact_train, *ops)
        
        decomp_train = decomp_expr(*ops)
        self.compare_vals(ts, "decomp", eq_str, decomp_train, *ops)

        var_train = var_expr(*ops)
        self.compare_vals(ts, "variational", eq_str, var_train, *ops)
        if not isinstance(var_train, TensorTrain):
            raise ValueError("variational expression should return a TensorTrain")

        xp = ts.namespace
        dims = var_train.shape.dims
        idxs = xp.zeros((len(dims), *[dim.size() for dim in dims]), dtype=ts.index_type)
        for i, dim in enumerate(dims):
            cut = [xp.newaxis] * i + [slice(None)] + [xp.newaxis] * (len(dims)-i-1)
            idxs[i,...] += xp.arange(dim.size(), dtype=ts.index_type)[*cut]

        approx = ev_expr(idxs, *ops)
        exact = oe.contract(eq_str, *[op.to_tensor() for op in ops])
        diff = xp.sum((approx - exact)**2) / xp.sum(exact**2)
        self.assertLess(diff, 1e-13, f"eval relative error {diff:.2E} is too large")

    def test_case1(self) -> None:
        for ts in self.trainsum:
            dim1, dim2, dim3 = [ts.dimension(2**6) for _ in range(3)]
            poly1 = self.poly_val(ts, dim1, [1.0, 0.0, 0.0])
            poly2 = self.poly_val(ts, dim2, [1.0, 1.0, 0.0])
            poly3 = self.poly_val(ts, dim3, [0.1, 1.0, 0.0])

            tmp = ts.full(poly2.shape, 1.0)
            op1 = deepcopy(poly1)
            op1.extend(tmp)

            tmp = ts.full(poly1.shape, 1.0)
            op2 = deepcopy(tmp)
            op2.extend(poly2)

            op3 = deepcopy(poly1)
            op3.extend(poly2)
            op3.extend(poly3)

            self.compare(ts, "ab->", op1)
            self.compare(ts, "ab->a", op1)
            self.compare(ts, "ab->b", op1)

            self.compare(ts, "ab,a->", op1, poly3)
            self.compare(ts, "ab,b->", op1, poly3)
            self.compare(ts, "ab,a->a", op1, poly3)
            self.compare(ts, "ab,b->a", op1, poly3)
            self.compare(ts, "ab,a->b", op1, poly3)
            self.compare(ts, "ab,b->b", op1, poly3)
            self.compare(ts, "ab,a->ab", op1, poly3)
            self.compare(ts, "ab,b->ab", op1, poly3)

            self.compare(ts, "ab,bc->ac", op1, op2)
            self.compare(ts, "ab,ab->a", op1, op2)
            self.compare(ts, "ab,ab->ab", op1, op2)
            self.compare(ts, "ab,ab->b", op1, op2)
            self.compare(ts, "ab,ba->a", op1, op2)

            self.compare(ts, "ab,ac->bc", op1, op2)
            self.compare(ts, "ab,cb->ac", op1, op2)

            self.compare(ts, "abc->", op3)
            self.compare(ts, "abc->ab", op3)
            self.compare(ts, "abc,ab->abc", op3, op1)
            self.compare(ts, "abc,ab->ab", op3, op1)
            self.compare(ts, "abc,ab->ac", op3, op1) 
            self.compare(ts, "abc,ab->bc", op3, op1)
            self.compare(ts, "abc,ab->a", op3, op1)
            self.compare(ts, "abc,ab->b", op3, op1)
            self.compare(ts, "abc,ab->c", op3, op1)

            self.compare(ts, "abc,ac->ac", op3, op1)
            self.compare(ts, "abc,ca->ca", op3, op1)
            self.compare(ts, "abc,ad->bcd", op3, op1)
            self.compare(ts, "abc,bd->acd", op3, op1)
            self.compare(ts, "abc,ba->abc", op3, op1)

    def test_case2(self) -> None:
        for ts in self.trainsum:
            xp = ts.namespace
            dim = ts.dimension(120)
            mat = ts.shift(dim, 2)
            shape = TrainShape(dim, [[*dim]])
            data = xp.exp(-xp.linspace(-10, 10, dim.size())**2)
            data = xp.reshape(data, (1, *[d.base for d in dim], 1))
            vec = ts.tensortrain(shape, [data])

            res_shape = ts.trainshape(dim)
            self.compare(ts, "ab,b->a", mat, vec, result=res_shape)
            self.compare(ts, "a,ab->b", vec, mat, result=res_shape)

    def test_case3(self) -> None:
        for ts in self.trainsum:
            dim = ts.dimension(2**8)

            mat = ts.shift(dim, 2)
            vec = self.poly_val(ts, dim, [1.0, 0.0, 0.0])

            self.compare(ts, "ab,b->a", mat, vec)
            self.compare(ts, "a,ab->b", vec, mat)

    def test_case4(self) -> None:
        for ts in self.trainsum:
            dim1 = ts.dimension(2**5)
            dim2 = ts.dimension(2**5)

            op1 = self.poly_val(ts, dim1, [1.0, 0.0, 0.0])
            op2 = self.poly_val(ts, dim2, [1.0, 1.0, 0.0, 0.0])

            cshapes  = [(d,) for d in dim1[:-1]]
            cshapes += [(dim1[-1], dim2[0])]
            cshapes += [(d,) for d in dim2[1:]]
            shape = TrainShape([dim1, dim2], cshapes)
            data  = [dat for dat in op1.cores[:-1]]
            data += [oe.contract("iaj,mbn->iabn", op1.cores[-1], op2.cores[0])]
            data += [dat for dat in op2.cores[1:]]
            op3 = ts.tensortrain(shape, data)

            self.compare(ts, "a,ab,b->ab", op1, op3, op2)

if __name__ == "__main__":
    unittest.main()
