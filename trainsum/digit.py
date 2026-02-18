# Copyright© 2025-2026 Gesellschaft zur Förderung der angewandten Forschung e.V.
# acting on behalf of its Fraunhofer Institut für Graphische Datenverarbeitung.
# Licensed under the EUPL. See LICENSE.txt.

from __future__ import annotations
from typing import Sequence
from dataclasses import dataclass

@dataclass(frozen=True, init=False)
class Digit:
    """
    A digit represents a single component of a quantized dimension.
    """

    #-------------------------------------------------------------------------
    #members & properties

    #: The identifier of the dimension this digit belongs to.
    idf: int

    #: The index of the digit within its dimension.
    idx: int

    #: The base of the digit, i.e., the number of possible values it can take.
    base: int

    #: The factor of the digit, i.e., the product of the bases of all subsequent digits.
    factor: int

    #----------------------------------------------------------------------
    #constructor

    def __init__(self, idf: int, idx: int, base: int, factor: int) -> None:
        self._check_input(idx, base, factor)
        object.__setattr__(self, "idf", idf)
        object.__setattr__(self, "base", base)
        object.__setattr__(self, "idx", idx)
        object.__setattr__(self, "factor", factor)

    def _check_input(self, idx: int, base: int, factor: int) -> None:
        if factor < 1:
            raise ValueError(f"Factor must be positive, but got {factor}")
        if base < 2:
            raise ValueError(f"Base must be greater one, but got {base}")
        if idx < 0:
            raise ValueError(f"Index must be positive, but got {idx}")

    #----------------------------------------------------------------------
    #methods

    def __eq__(self, other) -> bool:
        return isinstance(other, Digit)\
               and self.base == other.base\
               and self.idx == other.idx\
               and self.factor == other.factor\
               and self.idf == other.idf

    def __hash__(self) -> int:
        return hash((self.idf, self.base, self.idx, self.factor))

    def __str__(self) -> str:
        return f"Digit(base={self.base},factor={self.factor},idx={self.idx},idf={self.idf})"

Digits = Sequence[Digit]

def digits_similar(d1: Digit, d2: Digit) -> bool:
    return d1.idx == d2.idx and d1.base == d2.base and d1.factor == d2.factor
