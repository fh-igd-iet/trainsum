# Copyright© 2025-2026 Gesellschaft zur Förderung der angewandten Forschung e.V.
# acting on behalf of its Fraunhofer Institut für Graphische Datenverarbeitung.
# Licensed under the EUPL. See LICENSE.txt.

from typing import Self
from dataclasses import dataclass

@dataclass(kw_only=True)
class OperandString:
    left: str = ""
    right: str = ""
    middle: str = ""

    def reverse(self) -> Self:
        return type(self)(left="".join(reversed(self.right)),
                          middle="".join(reversed(self.middle)),
                          right="".join(reversed(self.left)))

    def __str__(self) -> str:
        return f"{self.left}{self.middle}{self.right}"

