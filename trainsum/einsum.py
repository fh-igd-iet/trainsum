# Copyright© 2025-2026 Gesellschaft zur Förderung der angewandten Forschung e.V.
# acting on behalf of its Fraunhofer Institut für Graphische Datenverarbeitung.
# Licensed under the EUPL. See LICENSE.txt.

from types import NoneType
from typing import Callable
from .backend import ArrayLike, ArrayNamespace
from .options import (
    get_options,
    OptionType,
    ExactOptions,
    DecompositionOptions,
    VariationalOptions,
    NormationOptions,
)
from .trainshape import TrainShape
from .trainbase import TrainBase
from .utils import namespace_of_trains
from .einsumexpression import EinsumExpression as _EinsumExpression


class EinsumExpression[T: ArrayLike]:
    _expr: Callable

    def __init__(
        self,
        xp: ArrayNamespace[T],
        eq: str,
        *ops: TrainShape,
        result_shape: NoneType | TrainShape = None,
    ) -> None:
        opts = get_options(xp, OptionType.EINSUM)
        if isinstance(opts, ExactOptions):
            self._expr = _EinsumExpression(
                eq, *ops, method="exact", optimizer=opts.optimizer
            )
        elif isinstance(opts, DecompositionOptions):
            self._expr = _EinsumExpression(
                eq,
                *ops,
                method="decomposition",
                decomposition=opts.decomposition,
                strategy=opts.strategy,
                optimizer=opts.optimizer,
                result_shape=result_shape,
                direction=opts.direction,
            )
        elif isinstance(opts, VariationalOptions):
            self._expr = _EinsumExpression(
                eq,
                *ops,
                method="variational",
                decomposition=opts.decomposition,
                strategy=opts.strategy,
                optimizer=opts.optimizer,
                result_shape=result_shape,
            )
        elif isinstance(opts, NormationOptions):
            self._expr = _EinsumExpression(
                eq,
                *ops,
                method="normation",
                decomposition=opts.decomposition,
                optimizer=opts.optimizer,
                normation_max_rank=opts.max_rank,
                normation_cutoff=opts.cutoff,
                direction=opts.direction,
            )
        else:
            raise RuntimeError("Invalid options type.")

    def __call__(
        self,
        *ops: TrainBase[T],
    ) -> float | complex | TrainBase[T]:
        return self._expr(*ops)


def einsum[T: ArrayLike](
    eq: str,
    *ops: TrainBase[T],
    result_shape: NoneType | TrainShape = None,
) -> float | complex | TrainBase[T]:
    xp = namespace_of_trains(*ops)
    expr = EinsumExpression(
        xp, eq, *[op.shape for op in ops], result_shape=result_shape
    )
    return expr(*ops)
