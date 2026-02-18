# Copyright© 2025-2026 Gesellschaft zur Förderung der angewandten Forschung e.V.
# acting on behalf of its Fraunhofer Institut für Graphische Datenverarbeitung.
# Licensed under the EUPL. See LICENSE.txt.

from typing import Sequence
from dataclasses import dataclass

from .backend import ArrayNamespace, Device, DType
from .trainshape import TrainShape
from .trainbase import TrainBase
from .utils import namespace_of_trains, get_device_dtype

@dataclass(frozen=True, init=False)
class ContractorInput:

    idx_map: dict[int, int]
    shapes: Sequence[TrainShape]
    _device: None | Device
    _dtype: None | DType
    _xp: None | ArrayNamespace

    def __init__(self, *ops: TrainShape | TrainBase):
        idx = 0
        shapes = []
        idx_map = {}
        bases = []
        for i, op in enumerate(ops):
            if isinstance(op, TrainShape):
                shapes.append(op)
                idx_map[i] = idx
                idx += 1
            elif isinstance(op, TrainBase):
                bases.append(op)
        if len(bases) > 0:
            xp = namespace_of_trains(*bases)
            device, dtype = get_device_dtype(bases)
            object.__setattr__(self, "_device", device)
            object.__setattr__(self, "_dtype", dtype)
            object.__setattr__(self, "_xp", xp)
        else:
            object.__setattr__(self, "_device", None)
            object.__setattr__(self, "_dtype", None)
            object.__setattr__(self, "_xp", None)
        object.__setattr__(self, "shapes", shapes)
        object.__setattr__(self, "idx_map", idx_map)

    def check_operands(self, *ops: TrainBase) -> None:
        if len(self.shapes) != len(ops):
            raise ValueError("Number of operand shapes do not match contraction")
        for ref_op, op in zip(self.shapes, (op.shape for op in ops)):
            if not ref_op == op:
                raise ValueError("Operand shapes do not match contraction")
        if self._device is None and self._dtype is None:
            ref_device = ops[0].device
            ref_dtype = ops[0].dtype
        else:
            ref_device = self._device
            ref_dtype = self._dtype
        if not all(op.device == ref_device for op in ops):
            raise ValueError("All operand devices must match")
        if not all(op.dtype == ref_dtype for op in ops):
            raise ValueError("All operand dtypes must match")

    def infos(self, *ops: TrainBase) -> tuple[ArrayNamespace, Device, DType]:
        if self._xp is None or self._device is None or self._dtype is None:
            xp = namespace_of_trains(*ops)
            device, dtype = get_device_dtype(ops)
            return xp, device, dtype
        return self._xp, self._device, self._dtype
