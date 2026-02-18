# Copyright© 2025-2026 Gesellschaft zur Förderung der angewandten Forschung e.V.
# acting on behalf of its Fraunhofer Institut für Graphische Datenverarbeitung.
# Licensed under the EUPL. See LICENSE.txt.

from .storagetensor import StorageTensor
from .backend import Device, DType, ArrayLike, get_index_dtype, namespace_of_arrays, size, shape

class ValueSet[T: ArrayLike]:
    _data: StorageTensor
    _set: set[tuple[int, ...]]

    @property
    def data(self) -> T:
        return self._data[...]

    @property
    def ndims(self) -> int:
        return self._data.max_shape[0]

    @property
    def size(self) -> int:
        return self._data.shape[1]

    @property
    def max_size(self) -> int:
        return self._data.max_shape[1]

    @property
    def dtype(self) -> DType:
        return self._data.dtype

    @property
    def device(self) -> Device:
        return self._data.device

    def __init__(self, storage: T) -> None:
        if storage.ndim != 2:
            raise ValueError("Storage tensor has incorrect shape")
        self._data = StorageTensor(storage)
        self._data.add_to_dim(0, shape(storage)[0])
        self._set = set()

    def add(self, vals: T) -> int:
        xp = namespace_of_arrays(vals)
        info = xp.__array_namespace_info__()
        btype = info.dtypes()["bool"]

        if vals.ndim == 1:
            vals = xp.reshape(vals, (size(vals), 1))
        self._check_vals(vals)

        to_add = xp.ones(vals.shape[1:], dtype=btype)
        for j in range(size(to_add)):
            key = tuple(int(val) for val in vals[:,j])
            size_ = len(self._set)
            self._set.add(key)
            to_add[j] = len(self._set) > size_
        vals = vals[:,to_add]
        added = shape(vals)[1]
        if size(vals) == 0:
            return 0

        self._data.add_to_dim(1, added)
        self._data[:,-added:] = vals
        return added

    def _check_vals(self, vals: ArrayLike) -> None:
        xp = namespace_of_arrays(vals)
        if vals.dtype != get_index_dtype(xp):
            raise ValueError("Expect a tensor of dtype torch.int64")
        if vals.ndim != 2 or vals.shape[0] != self.ndims:
            raise ValueError(f"Expect a tensor of shape ({self.ndims}, k) for some k")

    def _check_storage(self, storage: T) -> None:
        xp = namespace_of_arrays(storage[...])
        if storage.ndim != 2:
            raise ValueError("Storage tensor has incorrect shape")
        elif storage.dtype != get_index_dtype(xp):
            raise ValueError(f"Storage tensor has incorrect dtype, expected {get_index_dtype(xp)}")
