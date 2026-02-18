# Copyright© 2025-2026 Gesellschaft zur Förderung der angewandten Forschung e.V.
# acting on behalf of its Fraunhofer Institut für Graphische Datenverarbeitung.
# Licensed under the EUPL. See LICENSE.txt.

from typing import Generator

from .backend import ArrayLike
from .matrixdecomposition import MatrixDecomposition, MatrixDecompositionResult
from .tensordecomposition import TensorDecomposition
from .supercore import SuperCore
from .sweepingstrategy import LocalRange
from .trainbase import TrainBase
from .utils import namespace_of_trains

def local_vector[T: ArrayLike, S: MatrixDecompositionResult](
        train: TrainBase[T],
        decomposition: MatrixDecomposition[T, S],
        ) -> Generator[T, tuple[LocalRange, T]]:
    gen = local_vector_gen(train, decomposition)
    next(gen)
    return gen

def local_vector_gen[T: ArrayLike, S: MatrixDecompositionResult](
        train: TrainBase[T],
        decomposition: MatrixDecomposition[T, S],
        ) -> Generator[T, tuple[LocalRange, T]]:
    xp = namespace_of_trains(train)
    supercore = SuperCore()
    decomp = TensorDecomposition(decomposition)
    try:
        while True:
            if len(supercore.shapes) == 0:
                sweep_data, data = yield xp.zeros(0)
            else:
                sweep_data, data = yield supercore.data
                view = [data.shape[0], *supercore.data.shape[1:-1], data.shape[-1]]
                supercore.data = xp.reshape(data, view)

            begin, end = sweep_data.begin, sweep_data.end
            if len(supercore.shapes) == 0:
                train.normalize(begin, end)
                for i in range(begin, end):
                    supercore.add_right(train.shape.digits[i], train.data[i])
                continue


            idx0 = train.shape.digits.index(supercore.shapes[0])
            idx1 = train.shape.digits.index(supercore.shapes[-1])

            slc = slice(idx0, min(begin, idx1+1))
            data = []
            norm = []
            for i in range(idx0, min(begin, idx1+1)):
                cres = supercore.cut_left(decomp)
                data.append(cres.data)
                norm.append(cres.norm)
                idx0 += 1
            train.set_data(slc, data, norm)

            slc = slice(max(end-1, idx0-1)+1, idx1+1)
            data = []
            norm = []
            for i in range(idx1, max(end-1, idx0-1), -1):
                cres = supercore.cut_right(decomp)
                data.append(cres.data)
                norm.append(cres.norm)
                idx1 -= 1
            data.reverse()
            norm.reverse()
            train.set_data(slc, data, norm)

            if len(supercore.shapes) == 0:
                train.normalize(begin, end)
                for i in range(begin, end):
                    supercore.add_right(train.shape.digits[i], train.data[i])
                continue

            for i in range(idx0-1, begin-1, -1):
                supercore.add_left(train.shape.digits[i], train.data[i])
            for i in range(idx1+1, end):
                supercore.add_right(train.shape.digits[i], train.data[i])

    except GeneratorExit:
        if len(supercore.shapes) == 0:
            return
        idxs, data, norm = [], [], []
        while len(supercore.shapes) > 0:
            cres = supercore.cut_left(decomp)
            idxs.append(train.shape.digits.index(cres.shape))
            data.append(cres.data)
            norm.append(cres.norm)
        train.set_data(slice(idxs[0], idxs[-1]+1), data, norm)
