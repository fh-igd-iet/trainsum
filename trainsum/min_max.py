# Copyright© 2025-2026 Gesellschaft zur Förderung der angewandten Forschung e.V.
# acting on behalf of its Fraunhofer Institut für Graphische Datenverarbeitung.
# Licensed under the EUPL. See LICENSE.txt.

from dataclasses import dataclass
from math import prod

from .backend import ArrayLike, get_index_dtype, size, shape
from .trainbase import TrainBase
from .utils import namespace_of_trains, symbol_generator
from .exactaddition import ExactAddition
from .evaluationexpression import EvaluationExpression
from .exactaddition import ExactAddition
from .full import full

@dataclass
class MinMaxResult[T: ArrayLike]:
    """
    Result of the min_max algorithm. Contains the values and indices of the minimum and maximum
    values found in a tensor train.
    """

    #: Indices of the minimum value
    min_idxs: T
    #: Minimum value
    min_val: float
    #: Indices of the maximum value
    max_idxs: T
    #: Maximum value
    max_val: float

def min_max[T: ArrayLike](train: TrainBase[T], num: int) -> MinMaxResult[T]:
    xp = namespace_of_trains(train)

    sgen = symbol_generator()
    dim_chars = "".join(next(sgen) for _ in range(len(train.shape.dims)))
    ev_expr = EvaluationExpression(f"{dim_chars}->{dim_chars}", train.shape)

    right_idxs = right_sweep(train, num)
    right_idxs = xp.reshape(right_idxs, (len(train.shape.dims), 1))
    right_val = float(ev_expr(right_idxs, train)[0])

    left_idxs = left_sweep(train, num)
    left_idxs = xp.reshape(left_idxs, (len(train.shape.dims), 1))
    left_val = float(ev_expr(left_idxs, train)[0])

    if abs(left_val) < abs(right_val):
        idxs, val = right_idxs, right_val
    else:
        idxs, val = left_idxs, left_val

    tmp = full(xp, train.shape, -val)
    mod_train = ExactAddition()(train, tmp)

    mod_right_idxs = right_sweep(mod_train, num)
    mod_right_idxs = xp.reshape(mod_right_idxs, (len(train.shape.dims), 1))
    mod_right_val = float(ev_expr(mod_right_idxs, train)[0])

    mod_left_idxs = left_sweep(mod_train, num)
    mod_left_idxs = xp.reshape(mod_left_idxs, (len(train.shape.dims), 1))
    mod_left_val = float(ev_expr(mod_left_idxs, train)[0])

    if abs(mod_left_val) < abs(mod_right_val):
        mod_idxs, mod_val = mod_right_idxs, mod_right_val
    else:
        mod_idxs, mod_val = mod_left_idxs, mod_left_val

    if val < mod_val:
        return MinMaxResult(idxs, val, mod_idxs, mod_val)
    return MinMaxResult(mod_idxs, mod_val, idxs, val)

def right_sweep[T: ArrayLike](train: TrainBase[T], num: int) -> T:
    xp = namespace_of_trains(train)
    tshape = train.shape
    ndims = len(tshape.dims)
    max_middle = max(prod(tshape.middle(i)) for i in range(len(tshape)))

    cur_idxs = xp.zeros((ndims, max_middle*num),
                        device=train.device, dtype=get_index_dtype(xp))
    tmp_idxs = xp.zeros((ndims, max_middle*num),
                        device=train.device, dtype=get_index_dtype(xp))
    dim_map = {dim.idf: i for i, dim in enumerate(train.shape.dims)}

    cur_num = 1
    mat = xp.ones((1, 1), device=train.device, dtype=train.dtype)
    train.normalize(0)
    for i, tn in enumerate(train.data):
        middle = prod(tshape.middle(i))
        mat = xp.tensordot(mat, tn, axes=([1],[0]))
        mat = xp.reshape(mat, (prod(shape(mat)[:-1]), mat.shape[-1]))

        for j in range(middle):
            cut = (slice(None), slice(j, cur_num*middle, middle))
            tmp_idxs[cut] = cur_idxs[:,:cur_num]
        for digit in tshape.digits[i]:
            idx = dim_map[digit.idf]
            for j in range(digit.base):
                cut = (idx, slice(j, cur_num*middle, digit.base))
                tmp_idxs[cut] += digit.factor * j

        norms = xp.sqrt(xp.sum(xp.pow(mat, 2), axis=1))
        cur_num = min(num, size(norms))
        norm_idxs = xp.argsort(norms, descending=True)[:cur_num]# topk behaviour
        cur_idxs[:,:size(norm_idxs)] = xp.take(tmp_idxs, norm_idxs, axis=1)
        mat = xp.take(mat, norm_idxs, axis=0)
    return cur_idxs[:,0]

def left_sweep[T: ArrayLike](train: TrainBase[T], num: int) -> T:
    xp = namespace_of_trains(train)
    tshape = train.shape
    ndims = len(tshape.dims)
    max_middle = max(prod(tshape.middle(i)) for i in range(len(tshape)))

    cur_idxs = xp.zeros((ndims, max_middle*num),
                        device=train.device, dtype=get_index_dtype(xp))
    tmp_idxs = xp.zeros((ndims, max_middle*num),
                        device=train.device, dtype=get_index_dtype(xp))
    dim_map = {dim.idf: i for i, dim in enumerate(train.shape.dims)}

    cur_num = 1
    mat = xp.ones((1, 1), device=train.device, dtype=train.dtype)
    train.normalize(-1)
    for i, tn in zip(reversed(range(len(tshape))), reversed(train.data)):
        middle = prod(tshape.middle(i))
        idx = len(tn.shape) - 1
        mat = xp.tensordot(tn, mat, axes=([idx],[0]))
        mat = xp.reshape(mat, (mat.shape[0], prod(shape(mat)[1:])))

        for j in range(middle):
            cut = (slice(None), slice(j*cur_num, (j+1)*cur_num))
            tmp_idxs[cut] = cur_idxs[:,:cur_num]
        for digit in tshape.digits[i]:
            idx = dim_map[digit.idf]
            off = (cur_num*middle)//digit.base
            for j in range(digit.base):
                cut = (idx, slice(j*off, (j+1)*off))
                tmp_idxs[cut] += digit.factor * j
        norms = xp.sqrt(xp.sum(xp.pow(mat, 2), axis=0))
        cur_num = min(num, size(norms))
        norm_idxs = xp.argsort(norms, descending=True)[:cur_num]# topk behaviour
        cur_idxs[:,:size(norm_idxs)] = xp.take(tmp_idxs, norm_idxs, axis=1)
        mat = xp.take(mat, norm_idxs, axis=1)
    return cur_idxs[:,0]
