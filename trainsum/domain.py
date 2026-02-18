# Copyright© 2025-2026 Gesellschaft zur Förderung der angewandten Forschung e.V.
# acting on behalf of its Fraunhofer Institut für Graphische Datenverarbeitung.
# Licensed under the EUPL. See LICENSE.txt.

class Domain:                                                                                                                       
    """Domnain of a variable, defined by a lower and upper bound."""

    _lower: float
    _upper: float
    _diff: float

    @property
    def lower(self) -> float:
        return self._lower

    @property
    def upper(self) -> float:
        return self._upper

    @property
    def diff(self) -> float:
        return self._diff

    def __init__(self, lower: float, upper: float):
        if lower >= upper:
            raise ValueError("Upper bound must be greater than lower bound")
        self._lower = lower
        self._upper = upper
        self._diff = upper - lower

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Domain)\
               and self.lower == other.lower\
               and self.upper == other.upper
    
    def __str__(self) -> str:
        return f"Domain({self.lower}, {self.upper})"
