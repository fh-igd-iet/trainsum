# Copyright© 2025-2026 Gesellschaft zur Förderung der angewandten Forschung e.V.
# acting on behalf of its Fraunhofer Institut für Graphische Datenverarbeitung.
# Licensed under the EUPL. See LICENSE.txt.

from typing import Generator, Sequence, Optional, Callable
from copy import deepcopy
from string import ascii_lowercase

from .backend import ArrayLike
from .localrange import LocalRange
from .trainbase import TrainBase
from .linearmap import LinearMap
from .localvector import local_vector
from .locallinsolver import LocalLinSolver, LocalLinSolverResult
from .matrixdecomposition import MatrixDecomposition
from .utils import namespace_of_trains

from .trainshape import TrainShape, change_dims
from .innergenerator import InnerGenerator
from .sweepingstrategy import SweepingStrategy
from .svdecomposition import SVDecomposition
from .einsumequation import EinsumEquation
from .einsumcontraction import EinsumContraction
from .contractor import OptimizeKind, DEFAULT_OPTIMIZER
from .generatorcallabletype import GeneratorCallableType
from .operationspace import einsum_operation_shape

class LinSolver[T: ArrayLike, S: LocalLinSolverResult]:

    solver: LocalLinSolver[S]
    decomposition: MatrixDecomposition
    strategy: SweepingStrategy
    optimizer: OptimizeKind
    _maps: Sequence[LinearMap]
    _rhs_gen: InnerGenerator
    _rhs: TrainBase[T]

    def __init__(
            self,
            solver: LocalLinSolver[S],
            rhs: TrainBase[T],
            *maps: LinearMap[T],
            decomposition: MatrixDecomposition = SVDecomposition(),
            strategy: SweepingStrategy = SweepingStrategy(),
            optimizer: OptimizeKind = DEFAULT_OPTIMIZER) -> None:
        self.solver = deepcopy(solver)
        self.decomposition = deepcopy(decomposition)
        self.strategy = deepcopy(strategy)
        self.optimizer = deepcopy(optimizer)
        self._maps = maps
        self._rhs = deepcopy(rhs)
        self._rhs_gen = self._get_gen(rhs.shape)

    def __call__(
            self,
            guess: TrainBase[T],
            expr: bool = False,
            callback: Optional[Callable[[LocalRange, S], bool]] = None,
            ) -> TrainBase[T]:
        guess = deepcopy(guess)
        gen = self._gen(guess, expr)
        next(gen)  # warm up
        for i, lrange in self.strategy(guess.shape):
            loc_res = gen.send(lrange)
            if callback is not None:
                if callback(lrange, loc_res):
                    break
        return guess

    def _gen(
            self,
            guess: TrainBase[T],
            expr: bool,
            ) -> Generator[S, LocalRange]:
        xp = namespace_of_trains(guess)

        lvec = local_vector(guess, self.decomposition)
        rhs_gen = self._rhs_gen(guess, self._rhs, expr=expr)
        map_gens = [lin_map(guess, expr=expr) for lin_map in self._maps]
        gen_type = GeneratorCallableType.FULL

        loc_res = None
        data = xp.empty(0)
        try:
            while True:
                lrange = yield loc_res # type: ignore
                vec = lvec.send((lrange, data))
                funcs = [gen.send((lrange, gen_type)) for gen in map_gens]
                func = lambda x: sum((f(x) for f in funcs[1:]), start=funcs[0](x))
                lrhs = rhs_gen.send((lrange, gen_type))
                loc_res = self.solver(func, lrhs, vec) # type: ignore
                data = loc_res.array
        finally:
            for gen in map_gens:
                gen.close()
            lvec.close()
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
