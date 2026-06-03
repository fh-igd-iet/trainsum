# Copyright© 2025-2026 Gesellschaft zur Förderung der angewandten Forschung e.V.
# acting on behalf of its Fraunhofer Institut für Graphische Datenverarbeitung.
# Licensed under the EUPL. See LICENSE.txt.

from typing import Self, Callable, overload, Sequence, Optional, Any
from types import EllipsisType
from copy import deepcopy

from .backend import ArrayLike, Device, DType, ArrayNamespace, get_index_dtype
from .trainshape import TrainShape
from .trainbase import TrainBase
from .utils import symbol_generator, namespace_of_trains

from .full import full
from .add import add
from .multiply import multiply
from .matmul import matmul
from .evaluate import evaluate
from .transform import transform
from .construct import construct
from .to_tensor import to_tensor
from .conj import conj
from .truncate import truncate
from .trainslice import trainslice
from trainsum.assign import assign

IndexType = slice | EllipsisType


class TensorTrain[S: ArrayLike]:
    """
    N-dimensional tensor train. Main class for representing and manipulating tensor trains.\
Should not be instantiated directly, but rather through the `tensortrain` function.
    """

    _base: TrainBase[S]

    @property
    def device(self) -> Device:
        """Get and set the device."""
        return self._base.device

    @device.setter
    def device(self, device: Device) -> None:
        self._base.device = device

    @property
    def dtype(self) -> DType:
        """Get and set the data type."""
        return self._base.dtype

    @dtype.setter
    def dtype(self, dtype: DType) -> None:
        self._base.dtype = dtype

    @property
    def shape(self) -> TrainShape:
        """Return the shape of the tensor train with the current ranks."""
        return self._base.shape

    @property
    def cores(self) -> Sequence[S]:
        """Tensor cores."""
        return self._base.data

    def __init__(self, base: TrainBase[S], copy_data: bool = True) -> None:
        self._base = deepcopy(base) if copy_data else base

    # ------------------------------------------------------------------------
    # multiplication

    def __imul__(self, other: int | float | Self, /) -> Self:
        if isinstance(other, TensorTrain):
            base = multiply(self._base, other._base)
            self._base = base
        else:
            self._base.data[0][...] *= other
        return self

    def __mul__(self, other: int | float | Self, /) -> Self:
        return deepcopy(self).__imul__(other)

    def __rmul__(self, other: int | float | Self, /) -> Self:
        return deepcopy(self).__imul__(other)

    # ------------------------------------------------------------------------
    # multiplication

    def __imatmul__(self, other: Self, /) -> Self:
        base = matmul(self._base, other._base)
        self._base = base
        return self

    def __matmul__(self, other: Self, /) -> Self:
        return deepcopy(self).__imatmul__(other)

    # ------------------------------------------------------------------------
    # addition

    def __iadd__(self, other: int | float | Self, /) -> Self:
        if isinstance(other, TensorTrain):
            self._base = add(self._base, other._base)
        else:
            xp = namespace_of_trains(self._base)
            base = full(xp, self._base.shape, other)
            self._base = add(self._base, base)
        return self

    def __add__(self, other: int | float | Self, /) -> Self:
        return deepcopy(self).__iadd__(other)

    def __radd__(self, other: int | float | Self, /) -> Self:
        return deepcopy(self).__iadd__(other)

    # ------------------------------------------------------------------------
    # subtraction

    def __isub__(self, other: int | float | Self, /) -> Self:
        return self.__iadd__(-1 * other)

    def __sub__(self, other: int | float | Self, /) -> Self:
        return deepcopy(self).__iadd__(-1 * other)

    def __rsub__(self, other: int | float | Self, /) -> Self:
        return (-self).__iadd__(other)

    # ------------------------------------------------------------------------
    # divide

    def __itruediv__(self, other: int | float | Self, /) -> Self:
        other = 1.0 / other
        return self.__imul__(other)

    def __truediv__(self, other: int | float | Self, /) -> Self:
        return deepcopy(self).__itruediv__(other)

    def __rtruediv__(self, other: int | float | Self, /) -> Self:
        base = transform(self._base, lambda x: 1.0 / x)  # type: ignore
        return type(self)(base, copy_data=False) * other

    # ------------------------------------------------------------------------
    # unary operators

    def __pos__(self) -> Self:
        return self.__mul__(1)

    def __neg__(self) -> Self:
        return self.__mul__(-1)

    # ------------------------------------------------------------------------
    # cross based

    def __ipow__(self, power: int, /) -> Self:
        if power == 2:
            return self.__imul__(self)

        self._base = transform(self._base, lambda x: x**power)
        return self

    def __pow__(self, power: int, /) -> Self:
        return deepcopy(self).__ipow__(power)

    def __abs__(self) -> Self:
        base = transform(self._base, lambda x: abs(x))
        return type(self)(base, copy_data=False)

    # ------------------------------------------------------------------------
    # getter & setter

    @overload
    def __getitem__(self, cut: int | S | tuple[int, ...] | tuple[S, ...], /) -> S: ...
    @overload
    def __getitem__(
        self, cut: IndexType | tuple[IndexType, ...], /
    ) -> TensorTrain[S]: ...
    # implementation
    def __getitem__(self, cut: Any, /) -> S | TensorTrain[S]:
        xp = namespace_of_trains(self._base)
        if not isinstance(cut, tuple):
            cut = (cut,)

        if any(isinstance(c, (slice, EllipsisType)) for c in cut):
            cut = self._get_cut(cut)
            base = trainslice(self._base, cut)
            return type(self)(base, copy_data=False)

        if all(isinstance(c, int) for c in cut):
            if len(cut) != len(self._base.shape.dims):
                raise IndexError(
                    "Number of indices must match the number of dimensions."
                )
            idxs = xp.asarray(cut, dtype=get_index_dtype(xp))[:, xp.newaxis]
        else:
            idxs = xp.stack(list(cut), axis=0)

        sgen = symbol_generator()
        chars = "".join(next(sgen) for _ in range(len(self._base.shape.dims)))
        eq = f"{chars}->{chars}"
        return evaluate(eq, idxs, self._base)

    def _get_cut(self, cut: tuple[IndexType, ...]) -> tuple[slice, ...]:
        nellps = sum(isinstance(val, EllipsisType) for val in cut)
        if nellps > 1:
            raise IndexError("Only one ellipsis is allowed.")
        if nellps == 1:
            ellps_idx = cut.index(Ellipsis)
            num_missing = len(self._base.shape.dims) - (len(cut) - 1)
            cut = cut[:ellps_idx] + (slice(None),) * num_missing + cut[ellps_idx + 1 :]
        ncut = []
        for val in cut:
            if isinstance(val, int):
                val = slice(val, val + 1)
            ncut.append(val)
        return tuple(ncut)

    def __setitem__(
        self, cut: IndexType | tuple[IndexType, ...], value: S | TensorTrain[S], /
    ) -> None:

        if not isinstance(cut, tuple):
            cut = (cut,)
        cut = self._get_cut(cut)
        self._base = assign(self._base, cut, value._base)

    # ------------------------------------------------------------------------
    # other

    def to_tensor(self) -> S:
        """Construct the full tensor from the tensor train."""
        return to_tensor(self._base)

    def extend(self, *trains: Self, copy_data: bool = True) -> None:
        """Extend the tensor train by fusing it with another tensor train."""
        self._base.extend(*(train._base for train in trains), copy_data=copy_data)

    def conj(self) -> Self:
        """Return the complex conjugate of the tensor train."""
        base = conj(self._base)
        return type(self)(base, copy_data=False)

    def normalize(self, idx: int, /) -> None:
        """Create the canonical form of the tensor train with respect to the core at index idx."""
        self._base.normalize(idx)

    def truncate(self) -> None:
        """Reduce the ranks of the tensor train according to the current einsum options."""
        self._base = truncate(self._base)

    def transform(self, func: Callable[[S], S]) -> Self:
        """Perform an element-wise transformation of the tensor train defined by some function."""
        base = transform(self._base, func)
        return type(self)(base, copy_data=False)

    def reverse(self) -> Self:
        """Return a new tensor train with reversed order of cores and digits."""
        self._base.reverse()
        return self

    def permute_dims(self, order: Sequence[int]) -> None:
        """Permute the dimensions of the tensor train according to the given order."""
        self._base.permute_dims(order)

    def __repr__(self) -> str:
        return f"TensorTrain: {self._base.shape}"


@overload
def tensortrain[T: ArrayLike](
    shape: TrainShape, func: Callable[[T], T], start_idxs: T, xp: ArrayNamespace[T], /
) -> TensorTrain[T]: ...
@overload
def tensortrain[T: ArrayLike](
    shape: TrainShape, data: Sequence[T], /
) -> TensorTrain[T]: ...
@overload
def tensortrain[T: ArrayLike](shape: TrainShape, data: T, /) -> TensorTrain[T]: ...
# implementation
def tensortrain[T: ArrayLike](
    shape: TrainShape,
    data: T | Sequence[T] | Callable[[T], T],
    start_idxs: Optional[T] = None,
    xp: Optional[ArrayNamespace[T]] = None,
    /,
) -> TensorTrain[T]:
    if isinstance(data, Sequence):
        base = TrainBase(shape, data, copy_data=False)
    elif isinstance(data, Callable):
        if xp is None:
            raise ValueError(
                "Array namespace must be provided when data is a function."
            )
        base = construct(shape, data, xp, start_idxs)
    else:
        base = construct(shape, data)
    return TensorTrain(base, copy_data=False)
