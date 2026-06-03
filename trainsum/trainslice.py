# Copyright© 2025-2026 Gesellschaft zur Förderung der angewandten Forschung e.V.
# acting on behalf of its Fraunhofer Institut für Graphische Datenverarbeitung.
# Licensed under the EUPL. See LICENSE.txt.

from .backend import ArrayLike
from .trainbase import TrainBase
from .slice import slice_operator

from .utils import namespace_of_trains, symbol_generator
from .einsum import einsum


def trainslice[T: ArrayLike](
    train: TrainBase[T],
    cut: tuple[slice, ...],
) -> TrainBase[T]:
    if len(cut) != len(train.shape.dims):
        raise ValueError("The number of slices must match the number of dimensions.")

    xp = namespace_of_trains(train)
    ops = [slice_operator(xp, dim, slc) for dim, slc in zip(train.shape.dims, cut)]

    sgen = symbol_generator()
    in_chars = "".join(next(sgen) for _ in range(len(train.shape.dims)))
    out_chars = "".join(next(sgen) for _ in range(len(train.shape.dims)))
    eq = (
        ",".join(f"{out}{inp}" for inp, out in zip(in_chars, out_chars))
        + f",{in_chars}->{out_chars}"
    )
    res = einsum(eq, *ops, train)

    if not isinstance(res, TrainBase):
        raise ValueError("The result of slicing is not a tensor train.")
    return res
