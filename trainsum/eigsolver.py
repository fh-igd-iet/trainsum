# Copyright© 2025-2026 Gesellschaft zur Förderung der angewandten Forschung e.V.
# acting on behalf of its Fraunhofer Institut für Graphische Datenverarbeitung.
# Licensed under the EUPL. See LICENSE.txt.

from typing import Generator, Optional, Sequence, Callable
from copy import deepcopy
from string import ascii_lowercase

from .backend import ArrayLike
from .localrange import LocalRange
from .trainbase import TrainBase
from .linearmap import LinearMap
from .localvector import local_vector
from .localeigsolver import LocalEigSolver, LocalEigSolverResult
from .matrixdecomposition import MatrixDecomposition
from .utils import namespace_of_trains

from .trainshape import TrainShape, change_dims
from .innergenerator import InnerGenerator
from .sweepingstrategy import SweepingStrategy
from .svdecomposition import SVDecomposition
from .einsumequation import EinsumEquation
from .contractor import OptimizeKind, DEFAULT_OPTIMIZER
from .generatorcallabletype import GeneratorCallableType
from .operationspace import einsum_operation_shape
from .einsumcontraction import EinsumContraction

class EigSolver[T: ArrayLike, S: LocalEigSolverResult]:

    solver: LocalEigSolver[S]
    decomposition: MatrixDecomposition
    strategy: SweepingStrategy
    optimizer: OptimizeKind
    eps: float
    _maps: Sequence[LinearMap]
    _inner_gen: InnerGenerator

    def __init__(
            self,
            solver: LocalEigSolver[S],
            *maps: LinearMap[T],
            decomposition: MatrixDecomposition = SVDecomposition(),
            strategy: SweepingStrategy = SweepingStrategy(),
            optimizer: OptimizeKind = DEFAULT_OPTIMIZER,
            eps: float = 1e-10) -> None:

        self.solver = deepcopy(solver)
        self.decomposition = deepcopy(decomposition)
        self.strategy = deepcopy(strategy)
        self.optimizer = deepcopy(optimizer)
        self.eps = eps
        self._maps = maps
        self._inner_gen = self._orth_gen(maps[0].result)

    def __call__(
            self,
            guess: TrainBase[T],
            states: Sequence[TrainBase[T]] = [], 
            expr: bool = False,
            callback: Optional[Callable[[LocalRange, S], bool]] = None,
            ) -> TrainBase[T]:
        guess = deepcopy(guess)
        gen = self._gen(guess, states, expr, callback)
        next(gen)  # warm up

        val = 0.0
        eps = []
        for i, lrange in self.strategy(guess.shape):
            loc_res = gen.send(lrange)
            if len(eps) == i:
                continue
            if i < 2:
                val = loc_res.value
                continue
            eps.append(abs(loc_res.value - val))
            val = loc_res.value
            if eps[-1] < self.eps:
                break
        return guess

    def _gen(
            self,
            guess:
            TrainBase[T],
            states: Sequence[TrainBase[T]],
            expr: bool,
            callback: Optional[Callable[[LocalRange, S], bool]] = None,
            ) -> Generator[S, LocalRange]:
        xp = namespace_of_trains(guess)

        inner_generators = [deepcopy(self._inner_gen) for _ in range(len(states))]
        #if not expr:
        #    [lmap.calc_expressions(self.strategy, guess.shape) for lmap in self._maps]
        #    [igen.calc_expressions(self.strategy, guess.shape, state.shape) for igen, state in zip(inner_generators, states)]

        lvec = local_vector(guess, self.decomposition)
        map_gens = [lin_map(guess, expr=expr) for lin_map in self._maps]
        inner_gens = [igen(guess, state, expr=expr) for igen, state in zip(inner_generators, states)]
        gen_type = GeneratorCallableType.FULL

        loc_res = None
        data = xp.empty(0)
        try:
            while True:
                lrange = yield loc_res # type: ignore
                vec = lvec.send((lrange, data))
                funcs = [gen.send((lrange, gen_type)) for gen in map_gens]
                func = lambda x: sum((f(x) for f in funcs[1:]), start=funcs[0](x))
                lstates = [gen.send((lrange, gen_type)) for gen in inner_gens]
                loc_res = self.solver(func, vec, lstates) # type: ignore
                if callback is not None:
                    if callback(lrange, loc_res):
                        break
                data = loc_res.array
        finally:
            for gen in map_gens:
                gen.close()
            lvec.close()

    def _orth_gen(self, state: TrainShape) -> InnerGenerator:
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
