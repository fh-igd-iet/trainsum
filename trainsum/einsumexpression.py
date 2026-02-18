# Copyright© 2025-2026 Gesellschaft zur Förderung der angewandten Forschung e.V.
# acting on behalf of its Fraunhofer Institut für Graphische Datenverarbeitung.
# Licensed under the EUPL. See LICENSE.txt.

from typing import Literal, Optional
from dataclasses import dataclass

from .backend import ArrayLike
from .direction import Direction
from .trainshape import TrainShape
from .matrixdecomposition import MatrixDecomposition
from .svdecomposition import SVDecomposition
from .sweepingstrategy import SweepingStrategy
from .trainbase import TrainBase
from .einsumcontraction import EinsumContraction
from .einsumequation import EinsumEquation
from .contractor import OptimizeKind, DEFAULT_OPTIMIZER
from .utils import get_shapes

from .fullcontractor import FullContractor
from .exactcontractor import ExactContractor
from .decompositioncontractor import DecompositionContractor
from .variationalcontractor import VariationalContractor

@dataclass(frozen=True, init=False)
class EinsumExpression[T: ArrayLike]:

    equation: str
    optimizer: OptimizeKind
    method: Literal["exact", "decomposition", "variational"]
    decomposition: Optional[MatrixDecomposition]
    strategy: Optional[SweepingStrategy]
    result_shape: Optional[TrainShape]
    _expr: FullContractor | ExactContractor | DecompositionContractor | tuple[DecompositionContractor, VariationalContractor]

    def __init__(
                self,
                equation: str,
                *operands: TrainShape | TrainBase[T],
                method: Literal["exact", "decomposition", "variational"] = "decomposition",
                decomposition: Optional[MatrixDecomposition] = SVDecomposition(max_rank=25, cutoff=1e-12),
                strategy: Optional[SweepingStrategy] = SweepingStrategy(ncores=2, mode="connected", nsweeps=1),
                optimizer: OptimizeKind = DEFAULT_OPTIMIZER,
                result_shape: Optional[TrainShape] = None) -> None:

        object.__setattr__(self, "equation", equation)
        object.__setattr__(self, "optimizer", optimizer)
        object.__setattr__(self, "method", method)
        object.__setattr__(self, "decomposition", decomposition)
        object.__setattr__(self, "strategy", strategy)
        object.__setattr__(self, "result_shape", result_shape)

        eq = EinsumEquation(equation, *get_shapes(*operands))
        contr = EinsumContraction(eq, result=result_shape)

        if contr.result_shape is None:
            full_expr = FullContractor(contr, optimizer=optimizer)
            full_expr.calc_expressions(*operands)
            object.__setattr__(self, "_expr", full_expr)
            return
        elif method == "exact":
            exact_expr = ExactContractor(contr, optimizer=optimizer)
            exact_expr.calc_expressions(*operands)
            object.__setattr__(self, "_expr", exact_expr)
            return

        if decomposition is None or strategy is None:
            raise ValueError(f"Method '{method}' requires a MatrixDecomposition "\
                             f"instance and a SweepingStrategy instance.")

        if method == "decomposition" or method == "variational":
            decomp_expr = DecompositionContractor(contr,
                                                  optimizer=optimizer,
                                                  decomposition=decomposition,
                                                  strategy=strategy,
                                                  direction=Direction.TO_RIGHT)
            _, guess_shape = decomp_expr.calc_expressions(*operands)

        if method == "decomposition":
            object.__setattr__(self, "_expr", decomp_expr)
            return
        if method == "variational":
            expr = VariationalContractor(contr,
                                         optimizer=optimizer,
                                         decomposition=decomposition,
                                         strategy=strategy)
            expr.calc_expressions(guess_shape, *operands)
            object.__setattr__(self, "_expr", (decomp_expr, expr))

    def __call__(self, *operands: TrainBase[T]) -> float | TrainBase[T]:
        if isinstance(self._expr, tuple):
            return self._expr[1](self._expr[0](*operands), *operands)
        return self._expr(*operands)
