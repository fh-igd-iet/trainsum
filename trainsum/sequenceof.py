# Copyright© 2025-2026 Gesellschaft zur Förderung der angewandten Forschung e.V.
# acting on behalf of its Fraunhofer Institut für Graphische Datenverarbeitung.
# Licensed under the EUPL. See LICENSE.txt.

import sys
from typing import Protocol, overload, SupportsIndex, Iterator, Sequence, Any
from dataclasses import dataclass

class SequenceOfLike[T](Protocol):

    def __len__(self) -> int: ...
    def __iter__(self) -> Iterator[T]: ...
    def __reversed__(self) -> Iterator[T]: ...
    @overload
    def __getitem__(self, idx: SupportsIndex) -> T: ...
    @overload
    def __getitem__(self, idx: slice) -> Sequence[T]: ...
    def __contains__(self, item: Any) -> bool: ...
    def __eq__(self, other: Any) -> bool: ...
    def index(self, value: T, start: SupportsIndex = 0, stop: SupportsIndex = sys.maxsize) -> int: ...
    def count(self, value: T) -> int: ...

@dataclass(frozen=True, init=False)
class SequenceOf[T](Sequence):

    _seq_data: list[T]

    def __init__(self, data: Sequence[T]) -> None:
        object.__setattr__(self, "_seq_data", data)

    #-------------------------------------------------------------------------
    #container behaviour

    def __len__(self) -> int:
        return len(self._seq_data)

    def __iter__(self) -> Iterator[T]:
        return self._seq_data.__iter__()

    def __reversed__(self) -> Iterator[T]:
        return self._seq_data.__reversed__()

    @overload
    def __getitem__(self, idx: SupportsIndex) -> T: ...
    @overload
    def __getitem__(self, idx: slice) -> Sequence[T]: ...
    #implementation
    def __getitem__(self, idx: SupportsIndex | slice) -> T | Sequence[T]:
        return self._seq_data[idx]

    def __contains__(self, item: Any) -> bool:
        return len(self) > 0\
               and isinstance(item, type(self._seq_data[0]))\
               and item in self._seq_data

    def __eq__(self, other: Any) -> bool:
        return isinstance(other, type(self))\
               and self._seq_data == other._seq_data

    def index(self, value: T, start: SupportsIndex = 0, stop: SupportsIndex = sys.maxsize) -> int:
        return self._seq_data.index(value, start, stop)

    def count(self, value: T) -> int:
        return self._seq_data.count(value)
