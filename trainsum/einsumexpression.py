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
from .normationcontractor import NormationContractor
from .sketchcontractor import SketchContractor


@dataclass(frozen=True, init=False)
class EinsumExpression[T: ArrayLike]:
    equation: str
    optimizer: OptimizeKind
    method: Literal["exact", "decomposition", "variational", "normation", "sketching"]
    decomposition: Optional[MatrixDecomposition]
    strategy: Optional[SweepingStrategy]
    result_shape: Optional[TrainShape]
    normation_max_rank: Optional[int]
    normation_cutoff: Optional[float]
    direction: Optional[Direction]
    sketch_stack_size: int
    sketch_rank: int
    sketch_seed: int | None
    sketch_random_distribution: Literal["gaussian", "uniform", "stiefel"]
    sketch_mode: Literal["flattened", "stacked"]
    _expr: (
        FullContractor
        | ExactContractor
        | DecompositionContractor
        | tuple[DecompositionContractor, VariationalContractor]
        | NormationContractor
        | SketchContractor
    )

    def __init__(
        self,
        equation: str,
        *operands: TrainShape | TrainBase[T],
        method: Literal[
            "exact", "decomposition", "variational", "normation", "sketching"
        ] = "decomposition",
        decomposition: Optional[MatrixDecomposition] = SVDecomposition(
            max_rank=25, cutoff=1e-12
        ),
        strategy: Optional[SweepingStrategy] = SweepingStrategy(
            ncores=2, mode="connected", nsweeps=1
        ),
        optimizer: OptimizeKind = DEFAULT_OPTIMIZER,
        result_shape: Optional[TrainShape] = None,
        normation_max_rank: Optional[int] = 50,
        normation_cutoff: Optional[float] = 1e-15,
        direction: Direction = Direction.TO_RIGHT,
        sketch_stack_size: int = 4,
        sketch_rank: int = 6,
        sketch_seed: int | None = None,
        sketch_random_distribution: Literal[
            "gaussian", "uniform", "stiefel"
        ] = "gaussian",
        sketch_mode: Literal["flattened", "stacked"] = "flattened",
    ) -> None:

        object.__setattr__(self, "equation", equation)
        object.__setattr__(self, "optimizer", optimizer)
        object.__setattr__(self, "method", method)
        object.__setattr__(self, "decomposition", decomposition)
        object.__setattr__(self, "strategy", strategy)
        object.__setattr__(self, "result_shape", result_shape)
        object.__setattr__(self, "direction", direction)
        object.__setattr__(self, "normation_max_rank", normation_max_rank)
        object.__setattr__(self, "normation_cutoff", normation_cutoff)
        object.__setattr__(self, "sketch_stack_size", sketch_stack_size)
        object.__setattr__(self, "sketch_rank", sketch_rank)
        object.__setattr__(self, "sketch_seed", sketch_seed)
        object.__setattr__(
            self, "sketch_random_distribution", sketch_random_distribution
        )
        object.__setattr__(self, "sketch_mode", sketch_mode)

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

        if method == "sketching":
            sketch_expr = SketchContractor(
                contr,
                optimizer=optimizer,
                P=sketch_stack_size,
                sketch_rank=sketch_rank,
                seed=sketch_seed,
                random_distribution=sketch_random_distribution,
                sketch_mode=sketch_mode,
                direction=direction,
            )
            sketch_expr.calc_expressions(*operands)
            object.__setattr__(self, "_expr", sketch_expr)
            return

        if method == "normation":
            if (
                normation_max_rank is None
                or normation_cutoff is None
                or decomposition is None
            ):
                raise ValueError(
                    f"Method '{method}' requires a MatrixDecomposition "
                    f"instance, a max_rank and a cutoff."
                )

            norm_expr = NormationContractor(
                contr,
                optimizer=optimizer,
                decomposition=decomposition,
                max_rank=normation_max_rank,
                relative_cutoff=normation_cutoff,
                direction=direction,
            )
            norm_expr.calc_expressions(*operands)
            object.__setattr__(self, "_expr", norm_expr)
            return

        if decomposition is None or strategy is None:
            raise ValueError(
                f"Method '{method}' requires a MatrixDecomposition "
                f"instance and a SweepingStrategy instance."
            )

        if method == "decomposition" or method == "variational":
            decomp_expr = DecompositionContractor(
                contr,
                optimizer=optimizer,
                decomposition=decomposition,
                strategy=strategy,
                direction=direction,
            )
            _, guess_shape = decomp_expr.calc_expressions(*operands)

        if method == "decomposition":
            object.__setattr__(self, "_expr", decomp_expr)
            return
        if method == "variational":
            expr = VariationalContractor(
                contr,
                optimizer=optimizer,
                decomposition=decomposition,
                strategy=strategy,
            )
            expr.calc_expressions(guess_shape, *operands)
            object.__setattr__(self, "_expr", (decomp_expr, expr))

    def __call__(self, *operands: TrainBase[T]) -> float | complex | TrainBase[T]:
        if isinstance(self._expr, tuple):
            return self._expr[1](self._expr[0](*operands), *operands)
        return self._expr(*operands)
