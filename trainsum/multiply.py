# Copyright© 2025-2026 Gesellschaft zur Förderung der angewandten Forschung e.V.
# acting on behalf of its Fraunhofer Institut für Graphische Datenverarbeitung.
# Licensed under the EUPL. See LICENSE.txt.

from .backend import ArrayLike
from .trainbase import TrainBase
from .utils import symbol_generator
from .einsum import einsum

def multiply[T: ArrayLike](train1: TrainBase[T], train2: TrainBase[T]) -> TrainBase[T]:
    sgen = symbol_generator()
    chars = "".join(next(sgen) for _ in range(len(train1.shape.dims)))
    eq = f"{chars},{chars}->{chars}"
    res = einsum(eq, train1, train2)
    if not isinstance(res, TrainBase):
        raise RuntimeError("Unexpected result type from EinsumExpression.")
    return res
