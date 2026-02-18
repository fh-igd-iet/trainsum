# Copyright© 2025-2026 Gesellschaft zur Förderung der angewandten Forschung e.V.
# acting on behalf of its Fraunhofer Institut für Graphische Datenverarbeitung.
# Licensed under the EUPL. See LICENSE.txt.

from typing import Any, Optional, Type, overload
import h5py
import numpy as np

from .backend import ArrayNamespace, ArrayLike, to_device
from .digit import Digit
from .dimension import Dimension
from .domain import Domain
from .uniformgrid import UniformGrid
from .trainshape import TrainShape
from .trainbase import TrainBase

@overload
def write(group: h5py.Group, obj: Digit) -> None: ...
@overload
def write(group: h5py.Group, obj: Dimension) -> None: ...
@overload
def write(group: h5py.Group, obj: TrainShape) -> None: ...
@overload
def write(group: h5py.Group, obj: Domain) -> None: ...
@overload
def write(group: h5py.Group, obj: UniformGrid) -> None: ...
@overload
def write(group: h5py.Group, obj: TrainBase) -> None: ...
#implementation
def write(group: h5py.Group, obj: Any) -> None:
    if isinstance(obj, Digit):
        group.attrs["idf"] = obj.idf
        group.attrs["idx"] = obj.idx
        group.attrs["base"] = obj.base
        group.attrs["factor"] = obj.factor
    elif isinstance(obj, Dimension):
        group.attrs["idf"] = obj.idf
        group.attrs["prc"] = len(obj)
        group.attrs["bases"] = [digit.base for digit in obj]
    elif isinstance(obj, TrainShape):
        for i, dim in enumerate(obj.dims):
            dgroup = group.create_group(f"dim{i}")
            write(dgroup, dim)
        for i, digits in enumerate(obj.digits):
            cgroup = group.create_group(f"core{i}")
            for j, digit in enumerate(digits):
                dgroup = cgroup.create_group(f"digit{j}")
                write(dgroup, digit)
    elif isinstance(obj, Domain):
        group.attrs["lower"] = obj.lower
        group.attrs["upper"] = obj.upper
    elif isinstance(obj, UniformGrid):
        for i, dim, domain in zip(range(obj.ndims), obj.dims, obj.domains):
            dgroup = group.create_group(f"dim{i}")
            write(dgroup, dim)
            write(dgroup, domain)
    elif isinstance(obj, (TrainBase)):
        sgroup = group.create_group("shape")
        write(sgroup, obj.shape)
        for i, data in enumerate(obj.data):
            group.create_dataset(f"data{i}", data=np.asarray(to_device(data, "cpu")))

@overload
def read(group: h5py.Group, cls: Type[Digit]) -> Digit: ...
@overload
def read(group: h5py.Group, cls: Type[Dimension]) -> Dimension: ...
@overload
def read(group: h5py.Group, cls: Type[TrainShape]) -> TrainShape: ...
@overload
def read(group: h5py.Group, cls: Type[Domain]) -> Domain: ...
@overload
def read(group: h5py.Group, cls: Type[UniformGrid]) -> UniformGrid: ...
@overload
def read[T: ArrayLike](group: h5py.Group, cls: Type[TrainBase[T]], xp: Optional[ArrayNamespace[T]]) -> TrainBase[T]: ...
#implementation
def read(group: h5py.Group, cls: Any, xp: Optional[ArrayNamespace] = None) -> Any:
    if cls == Digit:
        return Digit(int(get_attr(group, "idf")),
                     int(get_attr(group, "idx")),
                     int(get_attr(group, "base")),
                     int(get_attr(group, "factor")))
    elif cls == Dimension:
        return Dimension(get_attr(group, "bases"),
                         idf=int(get_attr(group, "idf")))
    elif cls == TrainShape:
        dims = []
        while f"dim{len(dims)}" in group.keys():
            dgroup = group[f"dim{len(dims)}"]
            assert isinstance(dgroup, h5py.Group)
            dims.append(read(dgroup, Dimension))
        cores = []
        while f"core{len(cores)}" in group.keys():
            cgroup = group[f"core{len(cores)}"]
            assert isinstance(cgroup, h5py.Group)
            digits = []
            while f"digit{len(digits)}" in cgroup.keys():
                dgroup = cgroup[f"digit{len(digits)}"]
                assert isinstance(dgroup, h5py.Group)
                digits.append(read(dgroup, Digit))
            cores.append(digits)
        return TrainShape(dims, cores)
    elif cls == Domain:
        return Domain(float(get_attr(group, "lower")),
                      float(get_attr(group, "upper")))
    elif cls == UniformGrid:
        dims = []
        domains = []
        while f"dim{len(dims)}" in group.keys():
            dgroup = group[f"dim{len(dims)}"]
            assert isinstance(dgroup, h5py.Group)
            dims.append(read(dgroup, Dimension))
            domains.append(read(dgroup, Domain))
        return UniformGrid(dims, domains)
    elif cls == TrainBase:
        if xp is None:
            raise ValueError("Array namespace must be provided to read TrainBase.")
        sgroup = group["shape"]
        assert isinstance(sgroup, h5py.Group)
        shape = read(sgroup, TrainShape)
        data = []
        while f"data{len(data)}" in group.keys():
            dataset = group[f"data{len(data)}"]
            assert isinstance(dataset, h5py.Dataset)
            data.append(xp.asarray(np.asarray(dataset)))
        return TrainBase(shape, data)

    raise ValueError("Invalid class.")

def get_attr(group: h5py.Group, name: str) -> Any:
    return group.attrs[name]
