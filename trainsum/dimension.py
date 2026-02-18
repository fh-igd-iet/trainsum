# Copyright© 2025-2026 Gesellschaft zur Förderung der angewandten Forschung e.V.
# acting on behalf of its Fraunhofer Institut für Graphische Datenverarbeitung.
# Licensed under the EUPL. See LICENSE.txt.

from typing import Sequence, overload
from dataclasses import dataclass
from copy import deepcopy
from math import prod
from types import NoneType

from .backend import ArrayLike, namespace_of_arrays, device, DType
from .digit import Digit, digits_similar
from .sequenceof import SequenceOf
from .backend import get_index_dtype

@dataclass(frozen=True, init=False)
class Dimension[T: ArrayLike](SequenceOf[Digit]):
    """
    A quantized dimension, represented as a sequence of digits. If a integer is provided during initialization,
    it will be factorized into its prime factors to create the digits. Otherwise, the provided sequence of integers
    will be used as the bases of the digits.
    """

    #-------------------------------------------------------------------------
    #private members

    #: The identifier of the dimension, shared by all digits in the dimension.
    idf: int

    #-------------------------------------------------------------------------
    #constructor

    @overload
    def __init__(self,
                 size: int, /,
                 idf: int | NoneType = None) -> None: ...
    @overload
    def __init__(self,
                 bases: Sequence[int], /,
                 idf: int | NoneType = None) -> None: ...
    def __init__(self,
                 size: int | Sequence[int], /,
                 idf: int | NoneType = None) -> None:
        if isinstance(size, int):
            bases = prime_factorization(size)
        else:
            bases = size

        if isinstance(idf, NoneType):
            idf = unique_identifier()
        object.__setattr__(self, "idf", idf)

        self._check_bases(bases)
        factors = [prod(bases[i+1:]) for i in range(len(bases))]
        digits = []
        for i, base, factor in zip(range(len(bases)), bases, factors):
            digits.append(Digit(idf, i, base, factor))
        super().__init__(digits)

    def _check_bases(self, bases: Sequence[int]) -> None:
        if len(bases) == 0:
            raise ValueError("Dimension must have at least one digit")
        if any(b < 2 for b in bases):
            raise ValueError("All bases must be at least 2")

    #-------------------------------------------------------------------------
    #methods

    def size(self) -> int:
        """Calculate the size of the dimension, i.e., the product of the bases of all digits."""
        return prod(d.base for d in self)

    def to_idxs(self, digits: T) -> T:
        """
        Convert digit indices with shape (len(Dimension), ...) to dimension indices with shape (...).
        """
        xp = namespace_of_arrays(digits)
        int_type = get_index_dtype(xp)
        self._check_dtype(int_type, digits)
        if digits.shape[0] != len(self):
            raise ValueError(f"Expect a tensor of shape ({len(self)}, ...)")
        trans = xp.asarray([d.factor for d in self],
                            dtype=int_type,
                            device=device(digits))
        idxs = digits * xp.reshape(trans, (len(self), *[1]*(len(digits.shape)-1)))
        return xp.sum(idxs, axis=0)

    def to_digits(self, idxs: T) -> T:
        """
        Convert dimension indices with shape (...) to digit indices with shape (len(Dimension), ...).
        """
        xp = namespace_of_arrays(idxs)
        int_type = get_index_dtype(xp)
        self._check_dtype(int_type, idxs)
        digits = xp.zeros((len(self), *idxs.shape),
                          dtype=int_type,
                          device=device(idxs))
        idxs = deepcopy(idxs)
        for i, digit in enumerate(reversed(self)):
            digits[(len(self)-i-1, ...)] = idxs % digit.base
            idxs = idxs // digit.base
        return digits

    def _check_dtype(self, index_dtype: DType, inp: ArrayLike) -> None:
        if inp.dtype != index_dtype:
            raise ValueError(f"Input should have dtype={index_dtype}")

    #-------------------------------------------------------------------------
    #some magic

    def __str__(self) -> str:
        return f"Dimension({str([d.base for d in self])})"

    def __hash__(self) -> int:
        return hash((*self[:],))

    def __eq__(self, other) -> bool:
        return isinstance(other, Dimension)\
           and len(self) == len(other)\
           and all(digits_similar(d1, d2) for d1, d2 in zip(self, other))
_idf = 0
def unique_identifier() -> int:
    global _idf
    _idf += 1
    return _idf

def prime_factorization(num: int) -> Sequence[int]:
    facs = []
    i = 2
    while i * i <= num:
        while num % i == 0:
            facs.append(i)
            num //= i
        i += 1
    if num > 1:
        facs.append(num)
    return facs

Dimensions = Sequence[Dimension]

def dimensions_similar(dim1: Dimension, dim2: Dimension) -> bool:
    return all(digits_similar(d1, d2) for d1, d2 in zip(dim1, dim2))
