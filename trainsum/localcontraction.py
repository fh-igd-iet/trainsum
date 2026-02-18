# Copyright© 2025-2026 Gesellschaft zur Förderung der angewandten Forschung e.V.
# acting on behalf of its Fraunhofer Institut für Graphische Datenverarbeitung.
# Licensed under the EUPL. See LICENSE.txt.

from copy import deepcopy
from typing import Sequence
from dataclasses import dataclass, field

from .backend import ArrayLike
from .operandstring import OperandString
from .trainshape import TrainShape
from .trainbase import TrainBase


@dataclass(frozen=True)
class LocalContraction:

    operands: Sequence[OperandString] = field(default_factory=list)
    result: OperandString = field(default_factory=OperandString)

    train_idxs: Sequence[int] = field(default_factory=list)
    core_idxs: Sequence[int] = field(default_factory=list)

    def add_operand(self, train_idx: int, core_idx: int, op_str: OperandString) -> None:
        object.__setattr__(self, "train_idxs", [*self.train_idxs, train_idx])
        object.__setattr__(self, "core_idxs", [*self.core_idxs, core_idx])
        object.__setattr__(self, "operands", [*self.operands, op_str])

    def set_result(self, res_str: OperandString) -> None:
        object.__setattr__(self, "result", res_str)

    def get_data[T: ArrayLike](self,
                 *trains: TrainBase[T],
                 idx_map: dict[int, int] = {}) -> Sequence[T]:
        """Get the data arrays for the operands involved in this contraction."""
        out = []
        for op_idx, core_idx in zip(self.train_idxs, self.core_idxs):
            if op_idx in idx_map:
                idx = idx_map[op_idx]
                out.append(trains[idx].data[core_idx])
        return out

    def get_shapes(self, *ops: TrainShape | TrainBase) -> Sequence[Sequence[int]]:
        """Get the shapes of the data tensors for the operands involved in this contraction."""
        out = []
        for op_idx, core_idx in zip(self.train_idxs, self.core_idxs):
            op = ops[op_idx]
            if not isinstance(op, TrainShape):
                out.append(op.data[core_idx].shape)
            else:
                out.append((op.left_rank(core_idx), *op.middle(core_idx), op.right_rank(core_idx)))
        return out

    def get_constants[T: ArrayLike](self,
                      *ops: TrainShape | TrainBase[T]
                      ) -> tuple[Sequence[int], Sequence[Sequence[int] | T]]:
        idxs = []
        out = []
        for i, (op_idx, core_idx) in enumerate(zip(self.train_idxs, self.core_idxs)):
            op = ops[op_idx]
            if isinstance(op, TrainShape):
                out.append([op.left_rank(core_idx),
                            *op.middle(core_idx),
                            op.right_rank(core_idx)])
            else:
                out.append(op.data[core_idx])
                idxs.append(i)
        return idxs, out

    def __str__(self) -> str:
        ops_str = ",".join(str(op) for op in self.operands)
        res_str = str(self.result)
        return f"{ops_str} -> {res_str}"

def fuse_local_contractions(ops: Sequence[LocalContraction]) -> LocalContraction:
    if len(ops) == 0:
        raise ValueError("Cannot fuse zero LocalContractions.")
    elif len(ops) == 1:
        return ops[0]
    contr = deepcopy(ops[0])
    res = ops[0].result
    for i in range(1, len(ops)):
        prev_right = ops[i-1].result.right
        cur_left = ops[i].result.left
        if set(prev_right) != set(cur_left):
            raise ValueError("LocalContractions have incompatible contraction indices.")
        res = OperandString(left=res.left,
                            middle=res.middle+ops[i].result.middle,
                            right=ops[i].result.right)
        for op_str, train_idx, core_idx in zip(ops[i].operands, ops[i].train_idxs, ops[i].core_idxs):
            contr.add_operand(train_idx, core_idx, op_str)
    contr.set_result(res)
    return contr
