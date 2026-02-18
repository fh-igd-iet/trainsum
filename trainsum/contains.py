# Copyright© 2025-2026 Gesellschaft zur Förderung der angewandten Forschung e.V.
# acting on behalf of its Fraunhofer Institut für Graphische Datenverarbeitung.
# Licensed under the EUPL. See LICENSE.txt.

from dataclasses import dataclass
from typing import Sequence
from .trainshape import TrainShape

@dataclass(kw_only=True, frozen=True)
class ContainsResult:
    found: bool
    reverse: bool
    core_idxs: Sequence[int]

def contains(
        ref: TrainShape,
        other: TrainShape
        ) -> ContainsResult:
    found, idxs = _contains(ref, other)
    rev = False
    if not found:
        found, idxs = _contains(ref, other.reverse())
        rev = True
    if not found:
        return ContainsResult(found=False, reverse=False, core_idxs=[])
    return ContainsResult(found=True, reverse=rev, core_idxs=idxs)

def _contains(
        ref: TrainShape,
        other: TrainShape
        ) -> tuple[bool, Sequence[int]]:
    idx = 0
    map_idxs = []
    for i, digits in enumerate(ref.digits):
        if idx >= len(other):
            break
        flags = [digit in digits for digit in other.digits[idx]]
        if not any(flags):
            continue
        elif not all(flags):
            raise ValueError("Train cores do not align with operation space.")
        else:
            while all(flags):
                map_idxs.append(i)
                idx += 1
                if idx >= len(other):
                    break
                flags = [digit in digits for digit in other.digits[idx]]
    if idx < len(other):
        return False, map_idxs
    return True, map_idxs
