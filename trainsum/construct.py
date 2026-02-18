# Copyright© 2025-2026 Gesellschaft zur Förderung der angewandten Forschung e.V.
# acting on behalf of its Fraunhofer Institut für Graphische Datenverarbeitung.
# Licensed under the EUPL. See LICENSE.txt.

from typing import Callable, overload, Optional

from .backend import ArrayLike, ArrayNamespace
from .trainshape import TrainShape
from .trainbase import TrainBase
from .options import get_options, OptionType
from .crossinterpolation import CrossInterpolation
from .utils import symbol_generator
from .to_train import to_train
from .einsum import einsum

@overload
def construct[T: ArrayLike](shape: TrainShape, func: Callable[[T], T], xp: ArrayNamespace[T], start_idxs: Optional[T] = None, /) -> TrainBase[T]: ...
@overload
def construct[T: ArrayLike](shape: TrainShape, data: T, /) -> TrainBase[T]: ...
# implementation
def construct[T: ArrayLike](
        shape: TrainShape,
        data: T | Callable[[T], T],
        xp: Optional[ArrayNamespace[T]] = None,
        start_idxs: Optional[T] = None, /
        ) -> TrainBase[T]:
    if isinstance(data, Callable):
        if xp is None:
            raise ValueError("Array namespace must be provided when data is a function.")
        opts = get_options(xp, OptionType.CROSS)
        cross = CrossInterpolation(solver=opts.solver, strategy=opts.strategy, eps=opts.eps)
        return cross(xp, data, shape, start_idxs)
    else:
        sgen = symbol_generator()
        chars = "".join(next(sgen) for _ in range(len(shape.dims)))
        eq = f"{chars}->{chars}"
        ref = to_train(shape, data)
        res = einsum(eq, ref, result_shape=shape)
        if not isinstance(res, TrainBase):
            raise RuntimeError("Unexpected result type from EinsumExpression.")
        return res
