# Copyright© 2025-2026 Gesellschaft zur Förderung der angewandten Forschung e.V.
# acting on behalf of its Fraunhofer Institut für Graphische Datenverarbeitung.
# Licensed under the EUPL. See LICENSE.txt.

from math import prod
from typing import Any
import array_api_compat as api
from array_api_compat import to_device, device
from array_api_compat import size as _size

from .array_namespace import ArrayNamespace, ArrayLike, Device, DType


def get_namespace(obj: Any) -> ArrayNamespace:
    if not api.is_array_api_obj(obj):
        try:
            obj = obj.zeros(1)
        except:
            raise TypeError("Provided object is not a recognized array or namespace.")
    return api.array_namespace(obj) # type: ignore

def namespace_of_arrays[T: ArrayLike](*arrays: T) -> ArrayNamespace[T]:
    return api.array_namespace(*arrays) # type: ignore

def get_index_dtype(xp: ArrayNamespace) -> Any:
    info = xp.__array_namespace_info__()
    dtypes = info.dtypes(kind=None)
    for name in ["uint64", "int64", "uin32", "int32", "uint16", "int16"]:
        if name in dtypes:
            return dtypes[name]
    raise ValueError("No suitable index dtype found")

def shape_size(shape: tuple[int | None, ...]) -> int:
    if any(s is None for s in shape):
        raise ValueError("Shape contains None dimension(s), cannot compute size.")
    return prod(s for s in shape) # type: ignore

def size(array: ArrayLike) -> int:
    val = _size(array)
    if val is None:
        raise ValueError("Array size is unknown (None).")
    return val

def shape(array: ArrayLike) -> tuple[int, ...]:
    shp = array.shape
    if any(s is None for s in shp):
        raise ValueError("Array shape contains None dimension(s).")
    return shp  # type: ignore
