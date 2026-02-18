# Copyright© 2025-2026 Gesellschaft zur Förderung der angewandten Forschung e.V.
# acting on behalf of its Fraunhofer Institut für Graphische Datenverarbeitung.
# Licensed under the EUPL. See LICENSE.txt.

from typing import Generator, Literal, Sequence
from types import NoneType
from dataclasses import dataclass
from math import prod

from .localrange import LocalRange
from .trainshape import TrainShape

@dataclass
class SweepingStrategy:
    """
    Strategy for sweeping through a tensor train. Used heavily in the arithmetic algorithms and solvers.
    """

    min_size: NoneType | int = None
    #: Number of cores that are contracted together to create a super-core.
    ncores: int = 2
    #: Mode defines the overlap of the consecutive super-cores. In "connected" mode, the super-cores overlap by one core, while in "interleaved" mode, they overlap by ncores-1.
    mode: Literal["connected", "interleaved"] = "connected"
    #: Number of sweeps to perform. If None, the strategy will sweep indefinitely.
    nsweeps: int = 1

    def __call__(self, shape: TrainShape) -> Generator[tuple[int, LocalRange]]:
        """Generator that yields the local ranges for each sweep."""
        nsweeps = 0
        begin, end = 1, len(shape)
        while True:
            if self.ncores >= len(shape):
                yield nsweeps, LocalRange(begin=0, end=len(shape))
            else:

                if begin != 0:
                    begin = 0
                elif self.mode == "connected":
                    begin = end - 1
                else:
                    begin = 1
                for lrange in self._sweep_to_right(shape, begin):
                    begin, end = lrange.begin, lrange.end
                    yield nsweeps, lrange

                if end != len(shape):
                    end = len(shape)
                elif self.mode == "connected":
                    end = min(begin + 1, len(shape) - 1)
                else:
                    end = len(shape) - 1

                for lrange in self._sweep_to_left(shape, end):
                    begin, end = lrange.begin, lrange.end
                    yield nsweeps, lrange

            nsweeps += 1
            if self.nsweeps == nsweeps:
                break

    def right_sweep(self, shape: TrainShape) -> Sequence[LocalRange]:
        """Returns the local ranges for a single right sweep."""
        if self.ncores >= len(shape):
            return [LocalRange(begin=0, end=len(shape))]

        sweep = list(self._sweep_to_right(shape, 0))
        if sweep[-1].end != len(shape):
            sweep.append(LocalRange(begin=len(shape)-self.ncores, end=len(shape)))
        return sweep

    def left_sweep(self, shape: TrainShape) -> Sequence[LocalRange]:
        """Returns the local ranges for a single left sweep."""
        if self.ncores >= len(shape):
            return [LocalRange(begin=0, end=len(shape))]

        sweep = list(self._sweep_to_left(shape, len(shape)))
        if sweep[-1].begin != 0:
            sweep.append(LocalRange(begin=0, end=self.ncores))
        return sweep

    def _sweep_to_right(self, shape: TrainShape, idx: int) -> Generator[LocalRange]:
        if self.min_size is not None and prod(dim.size() for dim in shape.dims) < 0:
            raise ValueError("min_size is too large for the given shape.")

        end = idx
        while end < len(shape):
            if self.min_size is not None:
                while end < len(shape) and self._get_size(shape, idx, end) < self.min_size:
                    end += 1
            if end - idx < self.ncores:
                end = idx + self.ncores

            if end > len(shape):
                break

            yield LocalRange(begin=idx, end=end)
            idx = max(idx+1, end-1) if self.mode == "connected" else idx + 1

    def _sweep_to_left(self, shape: TrainShape, end: int) -> Generator[LocalRange]:
        if self.min_size is not None and prod(dim.size() for dim in shape.dims) < 0:
            raise ValueError("min_size is too large for the given shape.")

        idx = end-1
        while idx > -1:
            if self.min_size is not None:
                while idx > 0 and self._get_size(shape, idx, end) < self.min_size:
                    idx -= 1
            if end - idx < self.ncores:
                idx = end - self.ncores

            if idx < 1:
                break

            yield LocalRange(begin=idx, end=end)
            end = min(idx+1, end-1) if self.mode == "connected" else end - 1

    def _get_size(self, shape: TrainShape, begin: int, end: int) -> int:
        size = 1
        for i in range(begin, end):
            size *= prod(shape.middle(i))
        size *= shape.left_rank(begin)
        size *= shape.right_rank(end-1)
        return size

    def __repr__(self) -> str:
        return (f"SweepingStrategy(ncores={self.ncores}, mode='{self.mode}', "
                f"nsweeps={self.nsweeps}, min_size={self.min_size})")
