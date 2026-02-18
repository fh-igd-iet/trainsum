# Copyright© 2025-2026 Gesellschaft zur Förderung der angewandten Forschung e.V.
# acting on behalf of its Fraunhofer Institut für Graphische Datenverarbeitung.
# Licensed under the EUPL. See LICENSE.txt.

from typing import Callable


from .backend import ArrayLike
from .options import get_options, OptionType
from .trainbase import TrainBase
from .utils import symbol_generator, namespace_of_trains
from .crossinterpolation import CrossInterpolation
from .evaluation import Evaluation
from .min_max import min_max
from .einsumcontraction import EinsumContraction
from .einsumequation import EinsumEquation

def transform[T: ArrayLike](train: TrainBase[T], func: Callable[[T], T]) -> TrainBase[T]:
    xp = namespace_of_trains(train)
    opts = get_options(xp, OptionType.CROSS)

    mx_res = min_max(train, 8)
    if mx_res.max_val > mx_res.min_val:
        idxs = mx_res.max_idxs
    else:
        idxs = mx_res.min_idxs

    sgen = symbol_generator()
    chars = "".join(next(sgen) for _ in range(len(train.shape.dims)))
    eq = EinsumEquation(f"{chars}->{chars}", train.shape)
    contr = EinsumContraction(eq)
    expr = Evaluation(contr, optimizer="greedy")

    func_ = lambda idxs: func(expr(idxs, train))
    cross = CrossInterpolation(solver=opts.solver, strategy=opts.strategy, eps=opts.eps)
    res = cross(xp, func_, train.shape, idxs)
    return res
