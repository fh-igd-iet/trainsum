# Copyright© 2025-2026 Gesellschaft zur Förderung der angewandten Forschung e.V.
# acting on behalf of its Fraunhofer Institut für Graphische Datenverarbeitung.
# Licensed under the EUPL. See LICENSE.txt.

from math import prod
from typing import Optional, Sequence, Callable, Generator
from collections import deque
from copy import deepcopy
from random import randint

from .backend import ArrayLike, namespace_of_arrays, get_index_dtype, shape, ArrayNamespace
from .digit import Digit
from .dimension import Dimension
from .trainshape import TrainShape
from .trainbase import TrainBase
from .localrange import LocalRange
from .sweepingstrategy import SweepingStrategy
from .utils import namespace_of_trains
from .matrixleastsquares import MatrixLeastSquares
from .full import full

from .functionalcore import FunctionalCore
from .picore import PiCore

class CrossInterpolation[T: ArrayLike]:

    solver: MatrixLeastSquares
    eps: float
    _strat: SweepingStrategy

    @property
    def strategy(self) -> SweepingStrategy:
        return self._strat
    @strategy.setter
    def strategy(self, strat: SweepingStrategy) -> None:
        if strat.ncores != 2 or not strat.min_size is None:
            raise NotImplementedError("Only SweepingStrategy with ncores=2 and min_size=0 is supported")
        self._strat = strat

    def __init__(
            self,
            solver: MatrixLeastSquares,
            strategy: SweepingStrategy = SweepingStrategy(ncores=2, nsweeps=10),
            eps: float = 1e-6) -> None:
        self.solver = solver
        self.strategy = strategy
        self.eps = eps

    def __call__(
            self,
            xp: ArrayNamespace[T],
            func: Callable[[T], T],
            tshape: TrainShape,
            start_idxs: Optional[T] = None,
            ) -> TrainBase[T]:
        train = full(xp, tshape, 0.0)
        for lrange, eps in self._gen(func, train, start_idxs):
            #print(f"eps={eps:.2e} at range {lrange.begin}:{lrange.end}", end='\n', flush=True)
            pass
        return train

    def _gen(
            self,
            func: Callable[[T], T],
            train: TrainBase[T],
            start_idxs: Optional[T] = None,
            ) -> Generator[tuple[LocalRange, float]]:

        max_rank = 2*self._strat.nsweeps+1
        cross_gen = cross_interpolation(func, train, self.solver, max_rank, start_idxs)
        eps = deepcopy(self.eps)
        try:
            errs = deque[float]()
            for _, lrange in self._strat(train.shape):
                err_val = cross_gen.send(lrange)
                errs.append(err_val)
                if len(errs) == len(train.shape):
                    errs.popleft()

                mean = sum(errs)/len(errs)
                yield lrange, mean
                if mean < eps:
                    break
            raise GeneratorExit

        except GeneratorExit:
            cross_gen.close()

    def _get_strat(self, shape: TrainShape) -> Generator[tuple[int, LocalRange]]:
        right1 = [LocalRange(begin=i, end=i+2) for i in range(0, len(shape), 2) if i+2 <= len(shape)]
        right2 = [LocalRange(begin=i, end=i+2) for i in range(1, len(shape), 2) if i+2 <= len(shape)]
        for i in range(4*self._strat.nsweeps):
            if i%2 == 0:
                for j in range(len(right1)):
                    yield i//2, right1[j]
            else:
                for j in range(len(right2)):
                    yield i//2, right2[j]

def cross_interpolation[T: ArrayLike](
        func: Callable[[T], T],
        train: TrainBase[T],
        solver: MatrixLeastSquares,
        max_rank: int,
        start_idxs: Optional[T] = None,
        ) -> Generator[float, LocalRange]:
    gen = cross_interpolation_gen(func, train, solver, max_rank, start_idxs)
    next(gen)
    return gen

def cross_interpolation_gen[T: ArrayLike](
        func: Callable[[T], T],
        train: TrainBase[T],
        solver: MatrixLeastSquares,
        max_rank: int,
        start_idxs: Optional[T] = None,
        ) -> Generator[float, LocalRange]:

    def func_(x: T) -> T:
        xp = namespace_of_arrays(x)
        x_ = xp.reshape(x, (x.shape[0], prod(shape(x)[1:])))
        res = func(x_)
        return xp.reshape(res, x.shape[1:])

    xp = namespace_of_trains(train)
    int_type = get_index_dtype(xp)
    dims, digits = train.shape.dims, train.shape.digits
    idx_array = lambda sh: xp.empty((len(dims), sh),
                                     device=train.device,
                                     dtype=int_type)

    if start_idxs is None:
        sidxs = xp.asarray(random_idxs(dims), dtype=int_type)[:,xp.newaxis]
    else:
        sidxs = xp.astype(start_idxs, int_type)

    fcores = [FunctionalCore(dims, digits[0], idx_array(1), idx_array(max_rank))]
    for dgts in digits[1:-1]:
        fcores.append(FunctionalCore(dims, dgts, idx_array(max_rank), idx_array(max_rank)))
    fcores.append(FunctionalCore(dims, digits[-1], idx_array(max_rank), idx_array(1)))
    for i in range(shape(sidxs)[1]):
        idxs = xp.reshape(sidxs[:,i], (sidxs.shape[0], 1))
        add_idxs(dims, fcores, idxs)

    picores = [PiCore(fcores[i], fcores[i+1], func_, train.dtype) for i in range(len(fcores)-1)]

    err_val = xp.inf
    try:
        while True:
            data = yield err_val
            if data is None:
                raise ValueError("Expected SweepingData, got None")
            elif data.end-data.begin != 2:
                raise ValueError(f"Expected SweepingData with ncores=2, got {data}")

            lcore, rcore, picore = fcores[data.begin], fcores[data.end-1], picores[data.begin]
            picore.update_rows(lcore, rcore, func_)
            picore.update_cols(lcore, rcore, func_)
            err = picore.get_error(solver)

            idxs = get_index(xp.argmax(err), err.shape)
            left = (lcore.left.data[:,idxs[0]] + lcore.middle.data[:,idxs[1]])
            right = (rcore.middle.data[:,idxs[2]] + rcore.right.data[:,idxs[3]])

            lcore.right.add(right)
            picore.set_col(right)
            rcore.left.add(left)
            picore.set_row(left)

            err_val = float(xp.linalg.vector_norm(err) / xp.linalg.vector_norm(picore._data[...]))

    except GeneratorExit:
        for i in range(len(picores)-1):
            lcore, rcore, picore = fcores[i], fcores[i+1], picores[i]
            picore.update_rows(lcore, rcore, func_)
            picore.update_cols(lcore, rcore, func_)

        data = [picores[0].left()]
        for i in range(1, len(fcores)):
            data.append(picores[i-1].result_right(solver))
        train.set_data(slice(0, len(data)), data)

def random_idxs(dims: Sequence[Dimension]) -> list[int]:
    return [randint(0, dim.size()-1) for dim in dims]

def add_idxs[T: ArrayLike](dims: Sequence[Dimension], fcores: Sequence[FunctionalCore[T]], idxs: T) -> None:
    xp = namespace_of_arrays(idxs)
    int_type = get_index_dtype(xp)
    digits = [dim.to_digits(idxs[i]) for i, dim in enumerate(dims)]
    didxs = [dim_idxs(dims, fcore) for fcore in fcores]

    for i, fcore in enumerate(fcores):

        tmp = xp.zeros((len(dims)), device=fcore.device, dtype=int_type)
        for j in range(i):
            tmp += eval_middle(fcores[j], didxs[j], digits)
        tmp = xp.reshape(tmp, (tmp.shape[0], 1))
        fcore.left.add(tmp)

        tmp = xp.zeros((len(dims)), device=fcore.device, dtype=int_type)
        for j in range(len(fcores)-1, i, -1):
            tmp += eval_middle(fcores[j], didxs[j], digits)
        tmp = xp.reshape(tmp, (tmp.shape[0], 1))
        fcore.right.add(tmp)

def eval_middle[T: ArrayLike](fcore: FunctionalCore[T], dim_idxs: Sequence[int], digits: Sequence[T]) -> T:
    middle_idxs = [int(digits[dim_idx][digit.idx][0]) for dim_idx, digit in zip(dim_idxs, fcore)]
    view = [fcore.middle.data.shape[0]] + [digit.base for digit in fcore]
    xp = namespace_of_arrays(fcore.middle.data)
    return xp.reshape(fcore.middle.data, view)[:,*middle_idxs]

def dim_idxs(dims: Sequence[Dimension], digits: Sequence[Digit]) -> Sequence[int]:
    idfs = [dim.idf for dim in dims]
    return [idfs.index(digit.idf) for digit in digits]

def get_index(idx: int, shape: Sequence[int]) -> Sequence[int]:
    res = []
    for dim in reversed(shape):
        res.append(int(idx % dim))
        idx //= dim
    return list(reversed(res))
