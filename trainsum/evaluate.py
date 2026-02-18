# Copyright© 2025-2026 Gesellschaft zur Förderung der angewandten Forschung e.V.
# acting on behalf of its Fraunhofer Institut für Graphische Datenverarbeitung.
# Licensed under the EUPL. See LICENSE.txt.

from typing import Callable

from .backend import ArrayLike, namespace_of_arrays, ArrayNamespace
from .options import get_options, OptionType
from .trainshape import TrainShape
from .trainbase import TrainBase
from .evaluationexpression import EvaluationExpression as _EvaluationExpression

class EvaluateExpression[T: ArrayLike]:

    _expr: Callable

    def __init__(
        self,
        xp: ArrayNamespace[T],
        eq: str,
        *ops: TrainShape,
        ) -> None:
        opts = get_options(xp, OptionType.EVALUATE)
        self._expr = _EvaluationExpression(
                eq, *ops,
                chunk_size=opts.chunk_size,
                optimizer=opts.optimizer)

    def __call__(
            self,
            idxs: T,
            *ops: TrainBase[T],
            ) -> T:
        return self._expr(idxs, *ops)


def evaluate[T: ArrayLike](eq: str, idxs: T, *trains: TrainBase[T]) -> T:
    xp = namespace_of_arrays(idxs)
    shapes = [train.shape for train in trains]
    expr = EvaluateExpression(xp, eq, *shapes)
    return expr(idxs, *trains)
