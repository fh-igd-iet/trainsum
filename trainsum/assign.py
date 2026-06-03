# Copyright© 2025-2026 Gesellschaft zur Förderung der angewandten Forschung e.V.
# acting on behalf of its Fraunhofer Institut für Graphische Datenverarbeitung.
# Licensed under the EUPL. See LICENSE.txt.

from trainsum.exactaddition import ExactAddition
from .backend import ArrayLike
from .trainbase import TrainBase
from .slice import slice_operator, slice_vector

from .utils import namespace_of_trains, symbol_generator
from .einsum import einsum
from .add import add
from .full import full


def assign[T: ArrayLike](
    train: TrainBase[T], cut: tuple[slice, ...], value: TrainBase[T]
) -> TrainBase[T]:
    if len(cut) != len(train.shape.dims):
        raise ValueError("The number of slices must match the number of dimensions.")
    exact_add = ExactAddition()

    xp = namespace_of_trains(train)
    ops = [slice_operator(xp, dim, slc) for dim, slc in zip(train.shape.dims, cut)]
    slc_ops = [slice_vector(xp, dim, slc) for dim, slc in zip(train.shape.dims, cut)]
    # slc_ops = [exact_add(full(xp, op.shape, 1.0), op) for op in slc_ops]

    sgen = symbol_generator()
    in_chars = "".join(next(sgen) for _ in range(len(train.shape.dims)))
    out_chars = "".join(next(sgen) for _ in range(len(train.shape.dims)))
    tmp = full(xp, train.shape, 1.0)

    eq = ",".join(char for char in in_chars) + f",{in_chars}->{in_chars}"
    slc = einsum(eq, *slc_ops, tmp)
    if not isinstance(slc, TrainBase):
        raise ValueError("The result of slicing is not a tensor train.")
    slc.data[0][...] *= -1.0
    slc = exact_add(tmp, slc)

    eq = f"{in_chars},{in_chars}->{in_chars}"
    org = einsum(eq, slc, train)

    eq = (
        ",".join(f"{inp}{out}" for inp, out in zip(in_chars, out_chars))
        + f",{out_chars},{in_chars}->{out_chars}"
    )
    new = einsum(eq, *ops, tmp, value, result_shape=train.shape)
    if not isinstance(org, TrainBase) or not isinstance(new, TrainBase):
        raise ValueError("The result of slicing is not a tensor train.")
    res = add(org, new)
    return res
