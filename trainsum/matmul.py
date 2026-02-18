# Copyright© 2025-2026 Gesellschaft zur Förderung der angewandten Forschung e.V.
# acting on behalf of its Fraunhofer Institut für Graphische Datenverarbeitung.
# Licensed under the EUPL. See LICENSE.txt.

from .backend import ArrayLike
from .trainbase import TrainBase
from .utils import symbol_generator
from .einsum import einsum

def matmul[T: ArrayLike](train1: TrainBase[T], train2: TrainBase[T]) -> TrainBase[T]:
    if len(train1.shape.dims) == 1 and len(train2.shape.dims) == 1:
        raise ValueError("Operands of matmul cannot have both only one dimension (would be a inner product).")
    sgen = symbol_generator()
    chars1 = "".join(next(sgen) for _ in range(len(train1.shape.dims)))
    chars2 = "".join(next(sgen) for _ in range(len(train2.shape.dims[1:])))
    eq = f"{chars1},{chars1[-1]}{chars2}->{chars1[:-1]}{chars2}"
    res = einsum(eq, train1, train2)
    if not isinstance(res, TrainBase):
        raise RuntimeError("Unexpected result type from EinsumExpression.")
    return res
