# Copyright© 2025-2026 Gesellschaft zur Förderung der angewandten Forschung e.V.
# acting on behalf of its Fraunhofer Institut für Graphische Datenverarbeitung.
# Licensed under the EUPL. See LICENSE.txt.

from typing import Literal, Optional, Sequence, overload
from types import NoneType
from copy import deepcopy
from math import prod

from .digit import Digit, Digits, digits_similar
from .dimension import Dimension, dimensions_similar

class TrainShape:
    """
    Class for representing the structure of a quantics tensor train.
    The shape consists of one or multiple dimensions, which are each
    represented as a sequence of digits. The digits are grouped into
    cores that are connected linearly. 
    """

    _dims: Sequence[Dimension]
    _ranks: list[int]
    _digits: list[Digits]

    @property
    def dims(self) -> Sequence[Dimension]:
        """Dimensions of the shape. Cannot be set."""
        return self._dims

    @property
    def digits(self) -> Sequence[Digits]:
        """Digits that are grouped to cores. Cannot be set."""
        return self._digits

    @property
    def ranks(self) -> Sequence[int]:
        """
        Ranks between the cores. The length of ranks is one less than the number of cores.
        Setting the ranks to None will set the ranks to the maximum possible values, for
        representing a tensor exactly. The ranks can also be set to an integer, which will
        set all ranks to the minimum of the given integer and the maximum possible value.
        Otherwise, the ranks can be set individually by providing a sequence of integers.
        """
        return self._ranks
    @ranks.setter
    def ranks(self, ranks: NoneType | int | Sequence[int] = None, /) -> None:
        if ranks is None:
            self._ranks = self._exact_ranks()
        elif isinstance(ranks, int):
            exact_ranks = self._exact_ranks()
            self._ranks = [min(rank, ranks) for rank in exact_ranks]
        else:
            if len(ranks) != len(self._ranks):
                raise ValueError(f"Length of ranks {len(ranks)} does not match length of shape {len(self._ranks)}.")
            self._ranks = list(ranks)

    def __init__(
            self,
            dims: Dimension | Sequence[Dimension],
            cores: Sequence[Digits],
            ranks: NoneType | int | Sequence[int] = 1) -> None:
        if isinstance(dims, Dimension):
            dims = [dims]
        self._cores_match_dims(dims, cores)
        self._distinct_cores(cores)

        self._dims = list(deepcopy(dim) for dim in dims)
        self._digits = list(deepcopy(core) for core in cores)
        self._ranks = [0] * (len(cores)-1)
        self.ranks = ranks

    # ------------------------------------------------------------------------
    # shape getter

    def left_rank(self, idx: int) -> int:
        """Get the left rank of the core at index idx."""
        idx = transform_index(idx, len(self))
        if idx == 0:
            return 1
        return self._ranks[idx-1]

    def right_rank(self, idx: int) -> int:
        """Get the right rank of the core at index idx."""
        idx = transform_index(idx, len(self))
        if idx == len(self)-1:
            return 1
        return self._ranks[idx]

    def middle(self, idx: int) -> Sequence[int]:
        """Get the bases of the digits in the core at index idx."""
        idx = transform_index(idx, len(self))
        return [d.base for d in self._digits[idx]]

    def reverse(self) -> TrainShape:
        """return a new TrainShape with reversed order of cores and digits."""
        cores = [list(reversed(digits)) for digits in reversed(self._digits)]
        ranks = list(reversed(self._ranks))
        return TrainShape(self._dims, cores, ranks)

    # ------------------------------------------------------------------------
    # magic stuff

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, TrainShape):
            return False
        flag = len(self.dims) == len(other.dims)\
           and len(self) == len(other)\
           and all(dim1==dim2 for dim1, dim2 in zip(self.dims, other.dims))
        if not flag:
            return False
        idf_map = {dim1.idf: dim2.idf for dim1, dim2 in zip(self.dims, other.dims)}
        return all(_similar(dgts1, dgts2, idf_map) for dgts1, dgts2 in zip(self.digits, other.digits))

    def __len__(self) -> int:
        return len(self._digits)

    def __repr__(self) -> str:
        res = ""
        for i, digits in enumerate(self._digits[:-1]):
            res += f"[{','.join(str(d.base) for d in digits)}]"
            res += f"--{self._ranks[i]}--"
        res += f"[{','.join(str(d.base) for d in self._digits[-1])}]"
        return res

    # ------------------------------------------------------------------------
    # helper

    def _exact_ranks(self) -> list[int]:
        if len(self._digits) == 1:
            return []
        tmp1 = [prod(self.middle(0))]
        for i in range(1, len(self)-1):
            tmp1.append(tmp1[-1]*prod(self.middle(i)))
        tmp2 = [prod(self.middle(-1))]
        for i in reversed(range(1, len(self)-1)):
            tmp2.append(tmp2[-1]*prod(self.middle(i)))
        return [min(t1, t2) for t1, t2 in zip(tmp1, reversed(tmp2))]

    def _cores_match_dims(self, dims: Sequence[Dimension], cores: Sequence[Digits]):
        core_digits = set(digit for core in cores for digit in core)
        dim_digits = set(digit for dim in dims for digit in dim)
        if core_digits != dim_digits:
            raise ValueError("Digits in cores and dimensions do not match")

    def _distinct_cores(self, cores: Sequence[Digits]) -> None:
        digits = [digit for core in cores for digit in core]
        ref = set(digits)
        if len(digits) != len(ref):
            raise ValueError("Cores must have distinct digits")

def transform_index(idx: int, length: int) -> int:
    if idx < 0:
        idx += length
    if idx < 0 or idx >= length:
        raise IndexError(f"Index {idx} out of range for sequence of length {length}.")
    return idx

def change_dims(shape: TrainShape, dims: Sequence[Dimension]) -> TrainShape:
    """Change the dimensions of a TrainShape."""
    if len(dims) != len(shape.dims):
        raise ValueError("New dimensions must have the same length as the old ones.")
    if not all(dimensions_similar(old_dim, new_dim) for old_dim, new_dim in zip(shape.dims, dims)):
        raise ValueError("New dimensions must be similar to the old ones.")
    idf_map = {old_dim.idf: new_dim.idf for old_dim, new_dim in zip(shape.dims, dims)}
    cores = []
    for i in range(len(shape)):
        digits = [Digit(idf_map[digit.idf], digit.idx, digit.base, digit.factor) for digit in shape.digits[i]]
        cores.append(digits)
    return TrainShape(dims, cores)

@overload
def trainshape(*dims: Dimension, digits: Sequence[Digits]) -> TrainShape: ...
@overload
def trainshape(*dims: int | Dimension, mode: Literal["full", "block", "interleaved"] = "block") -> TrainShape: ...
# implementation
def trainshape(
        *dims: int | Dimension,
        mode: Literal["full", "block", "interleaved"] = "block",
        digits: Optional[Sequence[Digits]] = None
        ) -> TrainShape:
    dims = tuple([Dimension(dim) if isinstance(dim, int) else dim for dim in dims])
    if digits is not None:
        return TrainShape(dims, digits)
    if mode == "block":
        return TrainShape(dims, [(d,) for dim in dims for d in dim])
    elif mode == "interleaved":
        length = max(len(dim) for dim in dims)
        digits = []
        for i in range(length):
            tmp = []
            for dim in dims:
                if i < len(dim):
                    tmp.append(dim[i])
            digits.append(tmp)
        return TrainShape(dims, digits)
    elif mode == "full":
        digits = []
        for dim in dims:
            digits.extend(dim)
        return TrainShape(dims, [digits])
    raise ValueError("Invalid mode. Choose 'block' or 'interleaved'.")

def _similar(digit1: Digits, digit2: Digits, idf_map: dict[int, int]) -> bool:
    if len(digit1) != len(digit2):
        return False
    return all(digits_similar(d1, d2) and d2.idf == idf_map[d1.idf] for d1, d2 in zip(digit1, digit2))
