# Copyright© 2025-2026 Gesellschaft zur Förderung der angewandten Forschung e.V.
# acting on behalf of its Fraunhofer Institut für Graphische Datenverarbeitung.
# Licensed under the EUPL. See LICENSE.txt.

from typing import Literal

from .backend import ArrayLike, ArrayNamespace, get_index_dtype
from .dimension import Dimension
from .trainshape import TrainShape
from .trainbase import TrainBase

def toeplitz[T: ArrayLike](
        xp: ArrayNamespace[T],
        dim: Dimension,
        mode: Literal["full", "lower", "upper", "circular"]
        ) -> TrainBase:
    if mode == "full":
        dim = Dimension([2, *[d.base for d in dim]])

    out = []
    for digit in dim:
        data = xp.zeros((2, digit.base, digit.base, digit.base, 2))
        for j in range(digit.base):
            data[:,j,...] = shift_core(xp, j, digit.base)
        out.append(data)
    out[-1] = xp.reshape(out[-1][...,0], [*out[-1].shape[:-1], 1])
    if mode == "full":
        out[0] = xp.zeros([1, 2, 2])
        out[0][0,:,0] = xp.asarray([0.0, 1.0])
        out[0][0,:,1] = xp.asarray([1.0, 0.0])
        dims = [Dimension([d.base for d in dim]),
                Dimension([d.base for d in dim[1:]]),
                Dimension([d.base for d in dim[1:]])]
        shape = TrainShape(dims, [(dims[0][0],)] + [d for d in zip(dims[0][1:], dims[1], dims[2])])
    else:
        w = {"lower": xp.asarray([1.0, 0.0]),
             "upper": xp.asarray([0.0, 1.0]),
             "circular": xp.asarray([1.0, 1.0])
             }[mode]
        tmp = xp.tensordot(w, out[0], axes=([0], [0]))
        out[0] = xp.reshape(tmp, [1, *tmp.shape])
        dims = [Dimension([d.base for d in dim]),
                Dimension([d.base for d in dim]),
                Dimension([d.base for d in dim])]
        shape = TrainShape(dims, [d for d in zip(dims[0], dims[1], dims[2])])
    return TrainBase(shape, out)

def shift[T: ArrayLike](
        xp: ArrayNamespace[T],
        dim: Dimension,
        shift: int,
        circular: bool = False
        ) -> TrainBase:
    dims = Dimension([d.base for d in dim]), Dimension([d.base for d in dim])
    shape = TrainShape(dims, [d for d in zip(*dims)])

    if shift == 0:
        data = [xp.eye(digit.base).reshape(1, digit.base, digit.base, 1) for digit in dim]
        return TrainBase(shape, data)
    elif abs(shift) >= dim.size():
        raise ValueError("Shift value must be less than dimension size")

    if shift < 0:
        shift = abs(shift)
        w = xp.asarray([1.0, 0.0]) if not circular else xp.asarray([1.0, 1.0])
    else:
        shift = dim.size() - shift
        w = xp.asarray([0.0, 1.0]) if not circular else xp.asarray([1.0, 1.0])

    idxs = xp.asarray([[shift]], dtype=get_index_dtype(xp))
    digits = dim.to_digits(idxs)[:,0]

    data = []
    for num, digit in zip(digits, dim):
        data.append(shift_core(xp, int(num[0]), digit.base))
    tmp = xp.tensordot(w, data[0], axes=([0], [0]))
    data[0] = xp.reshape(tmp, (1, *tmp.shape))
    data[-1] = data[-1][...,0:1]
    return TrainBase(shape, data)

def shift_core[T: ArrayLike](
        xp: ArrayNamespace[T],
        digit: int,
        base: int
        ) -> T:
    core = xp.zeros((2, base, base, 2))
    for c_in in range(2):
        for j in range(base):
            s = j + digit + c_in
            i = s % base
            c_out = s // base  # 0 or 1
            #core = set_data(core, (c_out, i, j, c_in), 1.0)
            core[c_out,i,j,c_in] = 1.0
    return core
