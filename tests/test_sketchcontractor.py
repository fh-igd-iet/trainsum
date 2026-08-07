import unittest
from copy import deepcopy
from math import prod
from statistics import median
from string import ascii_lowercase
from timeit import repeat as timeit_repeat

import numpy as np
from utils import backends

from trainsum import TrainSum
from trainsum.trainsum import Direction


class TestSketchContractor(unittest.TestCase):
    def setUp(self) -> None:
        self.trainsum = [TrainSum(backend) for backend in backends]
        self.modes = ["interleaved", "block"]

    def polyval(
        self,
        ts: TrainSum,
        ndims: int,
        dim: int = 2**8,
        dims=None,
        mode: str = "interleaved",
        shape=None,
    ):
        if dims is None and shape is None:
            shape = ts.trainshape(*[ts.dimension(dim) for _ in range(ndims)], mode=mode)
        elif shape is None:
            shape = ts.trainshape(*dims, mode=mode)

        full = ts.full(shape, 1.0)
        trains = []
        coeff_sets = (
            [1.0, 0.5, 0.25],
            [0.5, -0.25, 0.1],
            [1.0, 0.2, -0.05],
        )
        for k in range(ndims):
            grid = ts.uniform_grid(shape.dims[k], ts.domain(-1, 1))
            trains.append(ts.polyval(grid, coeff_sets[k % len(coeff_sets)], 0.0))

        chars = ascii_lowercase[:ndims]
        eq = ",".join(chars) + f",{chars}->{chars}"
        with ts.exact():
            res = ts.einsum(eq, *trains, full)
        return res

    def case1_operands(self, ts: TrainSum, mode: str):
        poly1 = self.polyval(ts, 1, dim=2**10, mode=mode)
        poly2 = self.polyval(ts, 1, dim=2**10, mode=mode)
        poly3 = self.polyval(ts, 1, dim=2**10, mode=mode)

        tmp = ts.full(poly2.shape, 1.0)
        op1 = deepcopy(poly1)
        op1.extend(tmp)

        tmp = ts.full(poly1.shape, 1.0)
        op2 = deepcopy(tmp)
        op2.extend(poly2)

        op3 = deepcopy(poly1)
        op3.extend(poly2)
        op3.extend(poly3)
        return op1, op2, op3, poly3

    def irregular_matvec_operands(self, ts: TrainSum, mode: str):
        dim_out = ts.dimension(2**8 - 1)
        dim_in = ts.dimension(2**10)
        mat = ts.shift((dim_out, dim_in), 2)
        vec = self.polyval(ts, 1, dims=[dim_in], mode=mode)
        return mat, vec

    def triple_matvec_operands(self, ts: TrainSum, mode: str):
        d = 10
        dim_out = ts.dimension(2 ** (d - 2))
        dim_mid = ts.dimension(2 ** (d - 1))
        dim_in = ts.dimension(2**d)
        shape_a = ts.trainshape(dim_out, dim_mid, mode=mode)
        shape_b = ts.trainshape(dim_mid, dim_in, mode=mode)
        A = self.polyval(ts, 2, dims=[dim_out, dim_mid], mode=mode, shape=shape_a)
        B = self.polyval(ts, 2, dims=[dim_mid, dim_in], mode=mode, shape=shape_b)
        x = self.polyval(ts, 1, dims=[dim_in], mode=mode)
        return A, B, x

    def higher_order_operands(self, ts: TrainSum, mode: str):
        dim_a = ts.dimension(2**3)
        dim_b = ts.dimension(2**4)
        dim_c = ts.dimension(2**5)
        dim_d = ts.dimension(2**3)
        dim_e = ts.dimension(2**4)

        lhs = self.polyval(
            ts,
            3,
            dims=[dim_a, dim_b, dim_c],
            mode=mode,
            shape=ts.trainshape(dim_a, dim_b, dim_c, mode=mode),
        )
        rhs = self.polyval(
            ts,
            3,
            dims=[dim_c, dim_d, dim_e],
            mode=mode,
            shape=ts.trainshape(dim_c, dim_d, dim_e, mode=mode),
        )
        return lhs, rhs

    def to_tensor(self, ts: TrainSum, train) -> np.ndarray:
        if isinstance(train, list):
            dense = [ts.tensortrain(t.shape, t.data).to_tensor() for t in train]
            return sum(dense) / len(dense)
        return ts.tensortrain(train.shape, train.data).to_tensor()

    def timed(self, fn, repeat: int = 5) -> float:
        return median(timeit_repeat(fn, number=1, repeat=repeat))

    def assert_orthonormal(
        self, ts: TrainSum, train, direction: Direction, tol: float = 1e-7
    ) -> None:
        xp = ts.namespace
        cores = train.cores
        if direction == Direction.TO_RIGHT:
            for core in cores[:-1]:
                mat = xp.reshape(core, (prod(core.shape[:-1]), core.shape[-1]))
                gram = xp.matmul(xp.permute_dims(xp.conj(mat), (1, 0)), mat)
                eye = xp.eye(gram.shape[0], dtype=gram.dtype, device=train.device)
                self.assertLess(np.linalg.norm(np.asarray(gram - eye)), tol)
        else:
            for core in cores[1:]:
                mat = xp.reshape(core, (core.shape[0], prod(core.shape[1:])))
                gram = xp.matmul(mat, xp.permute_dims(xp.conj(mat), (1, 0)))
                eye = xp.eye(gram.shape[0], dtype=gram.dtype, device=train.device)
                self.assertLess(np.linalg.norm(np.asarray(gram - eye)), tol)

    def exact_einsum_expression(self, ts: TrainSum, equation: str, *ops):
        with ts.exact():
            return ts.einsum_expression(equation, *(op.shape for op in ops))

    def exact_einsum(self, ts: TrainSum, equation: str, *ops):
        with ts.exact():
            return ts.einsum(equation, *ops)

    def approx_einsum_expression(
        self,
        ts: TrainSum,
        equation: str,
        *ops,
        result_shape=None,
        direction=Direction.TO_RIGHT,
    ):
        with ts.sketching(
            sketch_stack_size=4,
            sketch_rank=4,
            sketch_seed=0,
            sketch_random_distribution="uniform",
            sketch_mode="flattened",
            direction=direction,
        ):
            return ts.einsum_expression(
                equation, *(op.shape for op in ops), result_shape=result_shape
            )

    def approx_einsum(
        self,
        ts: TrainSum,
        equation: str,
        *ops,
        result_shape=None,
        direction=Direction.TO_RIGHT,
    ):
        with ts.sketching(
            sketch_stack_size=4,
            sketch_rank=4,
            sketch_seed=0,
            sketch_random_distribution="uniform",
            sketch_mode="flattened",
            direction=direction,
        ):
            return ts.einsum(equation, *ops, result_shape=result_shape)

    def run_case(
        self,
        ts: TrainSum,
        mode: str,
        equation: str,
        *ops,
        result_shape=None,
        rel_err_bound: float = 1.0,
        direction=Direction.TO_RIGHT,
    ) -> None:
        approx_expr = self.approx_einsum_expression(
            ts, equation, *ops, result_shape=result_shape, direction=direction
        )
        exact_expr = self.exact_einsum_expression(ts, equation, *ops)
        approx = approx_expr(*ops)
        exact = exact_expr(*ops)
        approx_dense = approx.to_tensor()
        exact_dense = exact.to_tensor()
        rel_err = np.linalg.norm(approx_dense - exact_dense) / np.linalg.norm(
            exact_dense
        )

        exact_setup_time = self.timed(lambda: self.exact_einsum(ts, equation, *ops))
        sketch_setup_time = self.timed(
            lambda: self.approx_einsum(
                ts, equation, *ops, result_shape=result_shape, direction=direction
            )
        )
        exact_time = self.timed(lambda: exact_expr(*ops))
        sketch_time = self.timed(lambda: approx_expr(*ops))

        print(
            f"{mode:11s} {equation:14s} rel_err={rel_err:.3e} \t|\t"
            f"setup exact={exact_setup_time:.1e}s sketch={sketch_setup_time:.1e}s "
            f"(ratio={exact_setup_time / sketch_setup_time:.2}) \t"
            f"call exact={exact_time:.1e}s sketch={sketch_time:.1e}s "
            f"(ratio={exact_time / sketch_time:.2})"
        )
        # print("\texact :", [core.shape for core in exact.cores])
        # print("\tsketch:", [core.shape for core in approx.cores])

        self.assertTrue(np.isfinite(rel_err))
        self.assertLess(rel_err, rel_err_bound)

    def test_sketch_cases_with_timings(self) -> None:
        for ts in self.trainsum:
            for mode in self.modes:
                op1, op2, op3, vec = self.case1_operands(ts, mode)

                with ts.exact():
                    mat = op1 * op1 * op1 * op1 * op1 * op1

                self.run_case(ts, mode, "i->i", vec)
                self.run_case(ts, mode, "ab,a->a", mat, vec)
                self.run_case(
                    ts,
                    mode,
                    "ab,b->a",
                    *self.irregular_matvec_operands(ts, mode),
                    result_shape=ts.trainshape(ts.dimension(2**8 - 1), mode=mode),
                )

                self.run_case(ts, mode, "ab,bc->ac", op1, op2)
                self.run_case(ts, mode, "ab,a->ab", op1, vec)
                self.run_case(ts, mode, "ab,ab->a", op1, op2)
                self.run_case(ts, mode, "abc,ab->ac", op3, op1)
                self.run_case(ts, mode, "abc,ab->bc", op3, op1)
                self.run_case(
                    ts, mode, "ab,bc,c->a", *self.triple_matvec_operands(ts, mode)
                )
                self.run_case(
                    ts, mode, "abc,cde->abde", *self.higher_order_operands(ts, mode)
                )

    def test_sketch_directions(self) -> None:
        print("\nTesting sketching in both directions...")
        for ts in self.trainsum:
            for mode in self.modes:
                ops = self.irregular_matvec_operands(ts, mode)
                result_shape = ts.trainshape(ts.dimension(2**8 - 1), mode=mode)
                for direction in (Direction.TO_RIGHT, Direction.TO_LEFT):
                    self.run_case(
                        ts,
                        mode,
                        "ab,b->a",
                        *ops,
                        result_shape=result_shape,
                        direction=direction,
                    )
                    approx = self.approx_einsum(
                        ts,
                        "ab,b->a",
                        *ops,
                        result_shape=result_shape,
                        direction=direction,
                    )
                    self.assert_orthonormal(ts, approx, direction)


if __name__ == "__main__":
    unittest.main()
