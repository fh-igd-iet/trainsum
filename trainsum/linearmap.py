# Copyright© 2025-2026 Gesellschaft zur Förderung der angewandten Forschung e.V.
# acting on behalf of its Fraunhofer Institut für Graphische Datenverarbeitung.
# Licensed under the EUPL. See LICENSE.txt.

from typing import Callable, Generator, Sequence
from copy import deepcopy

from .backend import ArrayLike
from .localrange import LocalRange
from .trainshape import TrainShape
from .trainbase import TrainBase
from .einsumcontraction import EinsumContraction
from .sweepingstrategy import SweepingStrategy
from .contractor import OptimizeKind, DEFAULT_OPTIMIZER
from .einsumequation import EinsumEquation
from .linearmapgenerator import LinearMapGenerator
from .generatorcallabletype import GeneratorCallableType

class LinearMap[T: ArrayLike]:

    optimizer: OptimizeKind
    _ops: Sequence[TrainBase]

    @property
    def result(self) -> TrainShape:
        return self._map_gen.result_shape

    def __init__(
            self,
            eq: str,
            *ops: TrainShape | TrainBase[T],
            optimizer: OptimizeKind = DEFAULT_OPTIMIZER) -> None:
        self.optimizer = deepcopy(optimizer)
        self._ops, self._map_gen = self._get_map_gen(eq, *ops)

    def __call__(
            self,
            guess: TrainBase[T],
            expr: bool = False
            ) -> Generator[Callable[[T], T], tuple[LocalRange, GeneratorCallableType]]:
        return self._map_gen(*self._ops, guess, expr=expr)

    def calc_expressions(
            self,
            strat: SweepingStrategy,
            guess: TrainShape | TrainBase) -> None:
        guess_shape = guess if isinstance(guess, TrainShape) else guess.shape
        self._map_gen.calc_expressions(strat, *[op.shape for op in self._ops], guess_shape)

    def _get_map_gen(self, eq: str, *ops: TrainShape | TrainBase[T]) -> tuple[Sequence[TrainBase], LinearMapGenerator]:
        idx = self._ref_idx(*ops)
        ops_, guess = self._split_ops(idx, *ops)

        tmp = eq.split("->")
        op_strs = tmp[0].split(",")
        guess_str = op_strs.pop(idx)
        eq = ",".join(op_strs) +f",{guess_str}->{tmp[1]}"

        einsum_eq = EinsumEquation(eq, *[op.shape for op in ops_], guess)
        contr = EinsumContraction(einsum_eq)
        return ops_, LinearMapGenerator(contr, idx, optimizer=self.optimizer)

    def _split_ops(self, idx: int, *ops: TrainShape | TrainBase) -> tuple[list[TrainBase], TrainShape]:
        ops_ = list(deepcopy(op) for op in ops)
        ref = ops_.pop(idx)
        return ops_, ref # type: ignore

    def _ref_idx(self, *ops: TrainShape | TrainBase[T]) -> int:
        idx = -1
        for i, op in enumerate(ops):
            if isinstance(op, TrainShape):
                idx = i
                break
        if idx == -1 or sum(isinstance(op, TrainShape) for op in ops) != 1:
            raise ValueError("One of the operands must be a TrainShape.")
        return idx
