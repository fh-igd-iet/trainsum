# Copyright© 2025-2026 Gesellschaft zur Förderung der angewandten Forschung e.V.
# acting on behalf of its Fraunhofer Institut für Graphische Datenverarbeitung.
# Licensed under the EUPL. See LICENSE.txt.

from typing import Self, Callable, overload, Sequence, Optional
from copy import deepcopy

from trainsum.utils import namespace_of_trains

from .backend import ArrayLike, Device, DType, ArrayNamespace, namespace_of_arrays
from .trainshape import TrainShape
from .trainbase import TrainBase
from .utils import symbol_generator

from .full import full
from .add import add
from .multiply import multiply
from .matmul import matmul
from .evaluate import evaluate
from .transform import transform
from .construct import construct
from .to_tensor import to_tensor
from .fuse import fuse
from .conj import conj
from .truncate import truncate

class TensorTrain[S: ArrayLike]:
    """
    N-dimensional tensor train. Main class for representing and manipulating tensor trains.
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
        """3-dimensional tensor cores."""
        return self._base.data

    def __init__(self,
                 base: TrainBase[S],
                 copy_data: bool = True) -> None:
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
    # divide

    def __itruediv__(self, other: int | float | Self, /) -> Self:
        other = 1.0/other
        return self.__imul__(other)

    def __truediv__(self, other: int | float | Self, /) -> Self:
        return deepcopy(self).__itruediv__(other)

    def __rtruediv__(self, other: int | float | Self, /) -> Self:
        base = transform(self._base, lambda x: 1.0/x) # type: ignore
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

        self._base = transform(self._base, lambda x: x ** power)
        return self

    def __pow__(self, power: int, /) -> Self:
        return deepcopy(self).__ipow__(power)

    def __abs__(self) -> Self:
        base = transform(self._base, lambda x: abs(x))
        return type(self)(base, copy_data=False)

    # ------------------------------------------------------------------------
    # getter & setter

    def __getitem__(self, cut: tuple[S], /) -> S:
        sgen = symbol_generator()
        chars = "".join(next(sgen) for _ in range(len(self._base.shape.dims)))
        eq = f"{chars}->{chars}"
        xp = namespace_of_arrays(*cut)
        idxs = xp.stack(list(cut), axis=0)
        return evaluate(eq, idxs, self._base)

    # ------------------------------------------------------------------------
    # other

    def to_tensor(self) -> S:
        """Construct the full tensor from the tensor train."""
        return to_tensor(self._base)

    def extend(self, train: Self, /) -> None:
        """Extend the tensor train by fusing it with another tensor train."""
        self._base = fuse(self._base, train._base)

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

    def __repr__(self) -> str:
        return f"TensorTrain: {self._base.shape}"


@overload
def tensortrain[T: ArrayLike](shape: TrainShape, func: Callable[[T], T], start_idxs: T, xp: ArrayNamespace[T], /) -> TensorTrain[T]: ...
@overload
def tensortrain[T: ArrayLike](shape: TrainShape, data: Sequence[T], /) -> TensorTrain[T]: ...
@overload
def tensortrain[T: ArrayLike](shape: TrainShape, data: T, /) -> TensorTrain[T]: ...
# implementation
def tensortrain[T: ArrayLike](
        shape: TrainShape,
        data: T | Sequence[T] | Callable[[T], T],
        start_idxs: Optional[T] = None,
        xp: Optional[ArrayNamespace[T]] = None,
        /) -> TensorTrain[T]:
    if isinstance(data, Sequence):
        base = TrainBase(shape, data, copy_data=False)
    elif isinstance(data, Callable):
        if xp is None:
            raise ValueError("Array namespace must be provided when data is a function.")
        base = construct(shape, data, xp, start_idxs)
    else:
        base = construct(shape, data)
    return TensorTrain(base, copy_data=False)
