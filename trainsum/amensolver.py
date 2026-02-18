# Copyright© 2025-2026 Gesellschaft zur Förderung der angewandten Forschung e.V.
# acting on behalf of its Fraunhofer Institut für Graphische Datenverarbeitung.
# Licensed under the EUPL. See LICENSE.txt.

from typing import Generator, Sequence, Callable, Optional
from copy import deepcopy
from string import ascii_lowercase
from math import prod

from .backend import ArrayLike, shape
from .localrange import LocalRange
from .trainbase import TrainBase
from .linearmap import LinearMap
from .locallinsolver import LocalLinSolver, LocalLinSolverResult
from .utils import get_device_dtype, namespace_of_trains

from .trainshape import TrainShape, change_dims
from .innergenerator import InnerGenerator
from .sweepingstrategy import SweepingStrategy
from .einsumequation import EinsumEquation
from .einsumcontraction import EinsumContraction
from .contractor import OptimizeKind, DEFAULT_OPTIMIZER
from .generatorcallabletype import GeneratorCallableType
from .direction import Direction
from .tensordecomposition import TensorDecomposition
from .operationspace import einsum_operation_shape
from .matrixdecomposition import MatrixDecomposition
from .normalization import Normalization
from .svdecomposition import SVDecomposition

class AMEnSolver[T: ArrayLike, S: LocalLinSolverResult]:
    
    @property
    def strategy(self) -> SweepingStrategy:
        return self._strategy
    @strategy.setter
    def strategy(self, value: SweepingStrategy) -> None:
        if value.ncores != 1:
            raise ValueError("AMEnSolver only supports single-core sweeping strategies.")
        self._strategy = deepcopy(value)

    solver: LocalLinSolver[S]
    optimizer: OptimizeKind
    decomposition: MatrixDecomposition
    _strategy: SweepingStrategy
    _maps: Sequence[LinearMap]
    _rhs_gen: InnerGenerator
    _rhs: TrainBase[T]

    def __init__(
            self,
            solver: LocalLinSolver[S],
            rhs: TrainBase[T],
            *maps: LinearMap[T],
            strategy: SweepingStrategy = SweepingStrategy(),
            decomposition: MatrixDecomposition = SVDecomposition(max_rank=2),
            optimizer: OptimizeKind = DEFAULT_OPTIMIZER) -> None:

        self.solver = deepcopy(solver)
        self.strategy = strategy
        self.optimizer = deepcopy(optimizer)
        self.decomposition = deepcopy(decomposition)
        self._maps = maps
        self._rhs = deepcopy(rhs)
        self._rhs_gen = self._get_gen(rhs.shape)

    def __call__(
            self,
            guess: TrainBase[T],
            callback: Optional[Callable[[LocalRange, S], bool]] = None,
            ) -> TrainBase[T]:
        guess = deepcopy(guess)
        gen = self._gen(guess)
        next(gen)  # warm up
        for _, lrange in self.strategy(guess.shape):
            loc_res = gen.send(lrange)
            if callback is not None:
                if callback(lrange, loc_res):
                    break
        return guess

    def _gen(
            self,
            guess: TrainBase[T],
            ) -> Generator[S, LocalRange]:
        xp = namespace_of_trains(guess)
        device, dtype = get_device_dtype([guess])
        decomp = TensorDecomposition(self.decomposition)

        map_gens = [lin_map(guess) for lin_map in self._maps]
        rhs_gen = self._rhs_gen(guess, self._rhs)
        gen_type = GeneratorCallableType.FULL

        loc_res = None
        direc = Direction.TO_RIGHT
        idx = -1
        try:
            while True:
                lrange = yield loc_res # type: ignore
                if lrange.end - lrange.begin != 1:
                    raise ValueError("AMEn only supports single-core sweeps.")
                if lrange.begin == 0:
                    direc = Direction.TO_RIGHT
                elif lrange.end == len(guess.shape):
                    direc = Direction.TO_LEFT
                else:
                    direc = Direction.TO_LEFT if lrange.begin < idx else Direction.TO_RIGHT
                idx = lrange.begin

                guess.normalize(idx)
                funcs = [gen.send((lrange, gen_type)) for gen in map_gens]
                func = lambda x: sum((f(x) for f in funcs), start=xp.zeros_like(x))
                lrhs = rhs_gen.send(((lrange, gen_type)))
                loc_res = self.solver(func, lrhs, guess.data[idx])
                tmp = loc_res.array

                if direc == Direction.TO_RIGHT:

                    num_left = len(tmp.shape)-1
                    lops = []
                    for gen in map_gens:
                        op_data = gen.send((lrange, GeneratorCallableType.LEFT))(tmp)
                        op_data = xp.reshape(op_data, (*tmp.shape[:-1], prod(op_data.shape[num_left:])))
                        lops.append(op_data)
                    lrhs = rhs_gen.send((lrange, GeneratorCallableType.LEFT))

                    right = sum(shape(op)[-1] for op in lops) + shape(lrhs)[-1]
                    exact = xp.zeros((*tmp.shape[:-1], right), device=device, dtype=dtype)

                    start = 0
                    for op in lops:
                        end = start + prod(op.shape[num_left:])
                        exact[...,start:end] = op
                        start = end
                    exact[...,start:] = lrhs
                    u = decomp.left(exact, -1).left
                    numel = u.shape[-1]

                    lshape = guess.data[idx].shape
                    left = xp.zeros((*lshape[:-1], lshape[-1]+numel), device=device, dtype=dtype)
                    left[...,:-numel] = tmp
                    left[...,-numel:] = u

                    rshape = guess.data[idx+1].shape
                    right = xp.zeros((rshape[0]+numel, *rshape[1:]), device=device, dtype=dtype)
                    right[:-numel,:,:] = guess.data[idx+1]

                    guess.set_data(slice(idx, idx+2), [left, right], [Normalization.LEFT, Normalization.NONE])

                else:  # Direction.TO_LEFT

                    num_right = len(tmp.shape)-1
                    lops = []
                    for gen in map_gens:
                        op_data = gen.send((lrange, GeneratorCallableType.RIGHT))(tmp)
                        op_data = xp.reshape(op_data, (prod(op_data.shape[:-num_right]), *tmp.shape[1:]))
                        lops.append(op_data)
                    lrhs = rhs_gen.send((lrange, GeneratorCallableType.RIGHT))

                    left = sum(shape(op)[0] for op in lops) + shape(lrhs)[0]
                    exact = xp.zeros((left, *tmp.shape[1:]), device=device, dtype=dtype)

                    start = 0
                    for op in lops:
                        end = start + op.shape[0]
                        exact[start:end,...] = op
                        start = end
                    exact[start:,...] = lrhs
                    vh = decomp.right(exact, 1).right
                    numel = vh.shape[0]

                    rshape = guess.data[idx].shape
                    right = xp.zeros((rshape[0]+numel, *rshape[1:]), device=device, dtype=dtype)
                    right[:-numel,...] = tmp
                    right[-numel:,...] = vh

                    lshape = guess.data[idx-1].shape
                    left = xp.zeros((*lshape[:-1], lshape[-1]+numel), device=device, dtype=dtype)
                    left[...,:-numel] = guess.data[idx-1]

                    guess.set_data(slice(idx-1, idx+1), [left, right], [Normalization.NONE, Normalization.RIGHT])


        finally:
            for gen in map_gens:
                gen.close()
            rhs_gen.close()

    def _get_gen(self, state: TrainShape) -> InnerGenerator:
        dims = ascii_lowercase[:len(state.dims)]
        eq = EinsumEquation(f"{dims},{dims}->{dims}", state, state)
        op_shape = einsum_operation_shape(eq)
        eq_ = EinsumEquation(f"{dims},{dims}->", state, state)
        dim_map = {}
        for in_shape, out_shape in zip(eq.shapes, eq_.shapes):
            for in_dim, out_dim in zip(in_shape.dims, out_shape.dims):
                dim_map[in_dim] = out_dim
        new_op_shape_dims = [dim_map[dim] for dim in op_shape.dims]
        contr = EinsumContraction(eq_, op_shape=change_dims(op_shape, new_op_shape_dims))
        return InnerGenerator(contr, 0, self.optimizer)
