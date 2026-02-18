# Copyright© 2025-2026 Gesellschaft zur Förderung der angewandten Forschung e.V.
# acting on behalf of its Fraunhofer Institut für Graphische Datenverarbeitung.
# Licensed under the EUPL. See LICENSE.txt.

from typing import Self
from dataclasses import dataclass

@dataclass(kw_only=True, frozen=True)
class LocalRange:
    begin: int
    end: int

    def __init__(self, *, begin: int, end: int) -> None:
        if begin < 0:
            raise ValueError("LocalRange begin must be non-negative.")
        elif end <= begin:
            raise ValueError("LocalRange end must be greater than begin.")
        object.__setattr__(self, "begin", begin)
        object.__setattr__(self, "end", end)

    def __hash__(self) -> int:
        return hash((self.begin, self.end))

    def __lt__(self, other: Self) -> bool:
        if self.begin != other.begin:
            return self.begin < other.begin
        return self.end < other.end
