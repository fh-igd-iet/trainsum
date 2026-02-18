# Copyright© 2025-2026 Gesellschaft zur Förderung der angewandten Forschung e.V.
# acting on behalf of its Fraunhofer Institut für Graphische Datenverarbeitung.
# Licensed under the EUPL. See LICENSE.txt.

from copy import deepcopy

from .backend import ArrayLike
from .direction import Direction
from .sweepingstrategy import SweepingStrategy
from .trainbase import TrainBase
from .matrixdecomposition import MatrixDecomposition, MatrixDecompositionResult
from .svdecomposition import SVDecomposition
from .utils import trains_match
from .tensordecomposition import TensorDecomposition
from .additioncore import AdditionCore

class DecompositionAddition[T: ArrayLike, S: MatrixDecompositionResult]:

    @property
    def decomposition(self) -> MatrixDecomposition[T, S]:
        return self._mat_decomp
    @decomposition.setter
    def decomposition(self, value: MatrixDecomposition[T, S]) -> None:
        self._mat_decomp = deepcopy(value)
        self._tn_decomp = TensorDecomposition(value)
    
    strategy: SweepingStrategy
    _mat_decomp: MatrixDecomposition[T, S]
    _tn_decomp: TensorDecomposition[T, S]

    def __init__(
            self,
            strategy: SweepingStrategy,
            decomposition: MatrixDecomposition[T, S] = SVDecomposition[T]()) -> None:
        self.strategy = deepcopy(strategy)
        self.decomposition = deepcopy(decomposition)

    def __call__(
            self,
            train1: TrainBase[T],
            train2: TrainBase[T],
            *trains: TrainBase[T],
            direction: Direction = Direction.TO_RIGHT,
            ) -> TrainBase[T]:
        trains = (train1, train2, *trains)
        trains_match(*trains)
        if direction == Direction.TO_RIGHT:
            [train.normalize(0) for train in trains]
            return self._to_right(*trains)
        else:
            [train.normalize(-1) for train in trains]
            return self._to_left(*trains)

    def _to_right(self, *trains: TrainBase[T]) -> TrainBase[T]:
        shape = trains[0].shape
        acore = AdditionCore[T](shape, Direction.TO_RIGHT)
        lranges = self.strategy.right_sweep(shape)

        data = []
        begin, end = lranges[0].begin, 0
        for lrange in lranges:
            for i in range(begin, lrange.begin):
                tmp = acore.cut(self._tn_decomp)
                data.append(tmp.data)
            for i in range(max(end, lrange.begin), lrange.end):
                acore.add(i, *trains)
            begin, end = lrange.begin, lrange.end

        for i in range(begin, len(shape)):
            tmp = acore.cut(self._tn_decomp)
            data.append(tmp.data)

        return TrainBase(shape, data, copy_data=False)

    def _to_left(self, *trains: TrainBase[T]) -> TrainBase[T]:
        shape = trains[0].shape
        acore = AdditionCore[T](shape, Direction.TO_LEFT)
        lranges = self.strategy.left_sweep(shape)

        data = []
        begin, end = lranges[0].end, lranges[0].end
        for lrange in lranges:
            for i in range(end-1, lrange.end-1, -1):
                tmp = acore.cut(self._tn_decomp)
                data.append(tmp.data)
            for i in range(min(begin, lrange.end)-1, lrange.begin-1, -1):
                acore.add(i, *trains)
            begin, end = lrange.begin, lrange.end

        for i in range(end-1, -1, -1):
            tmp = acore.cut(self._tn_decomp)
            data.append(tmp.data)
        data.reverse()

        return TrainBase(shape, data, copy_data=False)
