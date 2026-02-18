# Copyright© 2025-2026 Gesellschaft zur Förderung der angewandten Forschung e.V.
# acting on behalf of its Fraunhofer Institut für Graphische Datenverarbeitung.
# Licensed under the EUPL. See LICENSE.txt.

from typing import Any, Sequence, overload, Optional, Self
from copy import deepcopy

from .backend import ArrayLike, namespace_of_arrays, to_device, device, shape
from .qrdecomposition import QRDecomposition
from .tensordecomposition import TensorDecomposition

from .normalization import Normalization
from .trainshape import TrainShape, transform_index

class TrainBase[T: ArrayLike]:

    _shape: TrainShape
    _data: list[T]
    _device: Any
    _dtype: Any
    _norm: list[Normalization]

    @property
    def device(self) -> Any:
        return self._device
    @device.setter
    def device(self, device: Any) -> None:
        self._device = device
        for i in range(len(self._data)):
            self._data[i] = to_device(self._data[i], device)

    @property
    def dtype(self) -> Any:
        return self._dtype
    @dtype.setter
    def dtype(self, dtype: Any) -> None:
        self._dtype = dtype
        xp = namespace_of_arrays(self._data[0])
        for i in range(len(self._data)):
            self._data[i] = xp.asarray(self._data[i], dtype=dtype, device=self._device)

    @property
    def shape(self) -> TrainShape:
        self._shape.ranks = [shape(tn)[-1] for tn in self._data[:-1]]
        return self._shape

    @property
    def data(self) -> Sequence[T]:
        return self._data

    def __init__(self,
                 shape: TrainShape,
                 data: Sequence[T],
                 norm: Optional[Sequence[Normalization]] = None,
                 copy_data: bool = True) -> None:
        self._check_shape_vs_data(shape, data)
        self._check_dtype_and_device(data)
        self._check_ranks(data)
        if copy_data:
            data = list(deepcopy(core_data) for core_data in data)

        self._shape = deepcopy(shape)
        self._data = list(data)
        self._device = device(data[0])
        self._dtype = data[0].dtype
        if norm is None:
            norm = [Normalization.NONE for _ in data]
        self._norm = list(norm)

    @overload
    def set_data(self,
                 idx: int,
                 data: T, /,
                 norm: Optional[Normalization] = None) -> None: ...
    @overload
    def set_data(self,
                 cut: slice,
                 data: Sequence[T], /,
                 norm: Optional[Sequence[Normalization]] = None) -> None: ...
    # implementation
    def set_data(self,
                 idx: int | slice,
                 data: T | Sequence[T], /,
                 norm: Optional[Normalization | Sequence[Normalization]] = None
                 ) -> None:
        if isinstance(idx, int)\
        and isinstance(data, type(self._data[0]))\
        and isinstance(norm, (Normalization, type(None))):
            if idx < 0 or idx >= len(self._data):
                raise IndexError("Core index out of range.")
            if data.shape != self._data[idx].shape:
                raise ValueError("New data shape does not match existing core shape.")
            self._data[idx] = data
            if norm is None:
                norm = Normalization.NONE
            self._norm[idx] = norm
        elif not isinstance(idx, int)\
        and not isinstance(data, type(self._data[0]))\
        and not isinstance(norm, Normalization):
            self._data[idx] = list(data) # type: ignore
            for i in range(idx.start, idx.stop-1):
                if self._data[i].shape[-1] != self._data[i+1].shape[0]:
                    raise ValueError(f"Core ranks do not match between adjacent cores at position {i}.")
            if norm is None:
                norm = [Normalization.NONE for _ in range(idx.stop - idx.start)]
            self._norm[idx] = norm
        else:
            raise TypeError("Invalid arguments for set_data.")

    def normalize(self, begin: int, end: Optional[int] = None) -> None:
        begin = transform_index(begin, len(self._data))
        if end is None:
            end = begin+1
        else:
            end = transform_index(end-1, len(self._data))+1
        if end < begin or end > len(self._data):
            raise IndexError("End index out of range.")

        xp = namespace_of_arrays(self._data[0])
        qr = TensorDecomposition(QRDecomposition[T]())

        for i in range(begin):
            res = qr.left(self._data[i], -1)
            self._data[i], r = res.left, res.right
            idx = len(r.shape)-1
            self._data[i+1] = xp.tensordot(r, self._data[i+1],
                                           axes=([idx], [0]))
        for i in range(len(self._data)-1, end-1, -1):
            res = qr.right(self._data[i], 1)
            l, self._data[i] =  res.left, res.right
            idx = len(self._data[i-1].shape)-1
            self._data[i-1] = xp.tensordot(self._data[i-1], l,
                                           axes=([idx], [0]))
    
    def reverse(self) -> None:
        xp = namespace_of_arrays(self._data[0])
        self._data.reverse()
        for i in range(len(self._data)):
            num = len(self._data[i].shape)
            self._data[i] = xp.permute_dims(self._data[i], range(num)[::-1])
        self._shape = self._shape.reverse()
        self._norm.reverse()

    def extend(self, train: Self, /) -> Self:
        dims = [*self.shape.dims, *train.shape.dims]
        digits = [*self.shape.digits, *train.shape.digits]
        shape = TrainShape(dims, digits)
        data = [*self.data, *train.data]
        self._base = TrainBase(shape, data)
        return self

    def _check_shape_vs_data(self, shape: TrainShape, data: Sequence[ArrayLike]) -> None:
        if len(shape) != len(data):
            raise ValueError("Number of cores in shape and data must match.")
        for i, dat in enumerate(data):
            if tuple(dat.shape[1:-1]) != tuple(shape.middle(i)):
                raise ValueError("Core data shape does not match core shape.")

    def _check_dtype_and_device(self, data: Sequence[ArrayLike]) -> None:
        dev = device(data[0])
        dtype = data[0].dtype
        for core_data in data:
            if device(core_data) != dev:
                raise ValueError("All core tensors must be on the same device.")
            if core_data.dtype != dtype:
                raise ValueError("All core tensors must have the same dtype.")

    def _check_ranks(self, data: Sequence[ArrayLike]) -> None:
        for i in range(len(data)-1):
            if data[i].shape[-1] != data[i+1].shape[0]:
                raise ValueError(f"Core ranks do not match between adjacent cores at position {i}.")
        if data[0].shape[0] != 1 or data[-1].shape[-1] != 1:
            raise ValueError("First and last cores must have rank 1 on the outside")

