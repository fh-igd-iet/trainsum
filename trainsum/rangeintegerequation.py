# Copyright© 2025-2026 Gesellschaft zur Förderung der angewandten Forschung e.V.
# acting on behalf of its Fraunhofer Institut für Graphische Datenverarbeitung.
# Licensed under the EUPL. See LICENSE.txt.

from typing import Sequence
from math import prod
import pulp as pl

from .backend import ArrayNamespace, ArrayLike, default_ftype
from .dimension import Dimension
from .integerequation import IntegerEquation, PulpSolver, PulpStandardSolver

class RangeIntegerEquation(IntegerEquation):
    """
    Range integer equation: :math:`l_i \\leq x_i \\leq u_i`. :math:`x_i` are the dimensions, :math:`l_i` and
    :math:`u_i` are the lower and upper bounds for the dimensions, respectively.
    """

    _solver: PulpSolver
    _dims: tuple[Dimension, ...]
    _evaluated: tuple[tuple[None | int, ...], ...]
    _lower : tuple[int, ...]
    _upper : tuple[int, ...]

    _solvable: bool
    _upper_solutions: list[int]
    _lower_solutions: list[int]
    _hash_val: int
    _eq_val: tuple[tuple[Dimension, ...], tuple[tuple[None | int, ...], ...], tuple[int, ...], tuple[int, ...]]


    #: Solver for the linear programming problems.
    @property
    def solver(self) -> PulpSolver:
        return self.solver
    @solver.setter
    def solver(self, value: PulpSolver) -> None:
        self.solver = value

    #: Tuple of dimensions involved in the equation.
    @property
    def dims(self) -> tuple[Dimension, ...]:
        return self._dims
    
    #: Tuple of coefficients corresponding to each dimension.
    @property
    def evaluated(self) -> tuple[tuple[None | int, ...], ...]:
        return self._evaluated
    @evaluated.setter
    def evaluated(self, value: tuple[tuple[None | int, ...], ...]):
        self._evaluated = value
        self._solve(True)

    #: Tuple of lower bounds for each dimension.
    @property
    def lower(self) -> tuple[int, ...]:
        return self._lower
    
    #: Tuple of upper bounds for each dimension.
    @property
    def upper(self) -> tuple[int, ...]:
        return self._upper

    def __init__(
            self,
            dims: tuple[Dimension, ...],
            lower: tuple[int, ...],
            upper: tuple[int, ...],
            solver: PulpSolver = PulpStandardSolver(msg=False)):
        if len(lower) != len(dims) or len(upper) != len(dims):
            raise ValueError("Number of bounds must match the number of dimensions.")
        self._solver = solver
        self._dims = dims
        self._lower = lower
        self._upper = upper
        tmp = []
        for dim in dims:
            tmp.append(tuple(None for _ in range(len(dim))))
        self._evaluated = tuple(tuple(ev) for ev in tmp)

    def __hash__(self) -> int:
        self._solve(False)
        return self._hash_val

    def __eq__(self, other: object) -> bool:
        self._solve(False)
        if not isinstance(other, RangeIntegerEquation):
            return False
        if self._eq_val != other._eq_val:
            return False
        for i in range(self.ndim):
            if self._lower_solutions[i] != other._lower_solutions[i]\
            or self._upper_solutions[i] != other._upper_solutions[i]:
                return False
        return True

    def inner(self, other: IntegerEquation) -> float:
        """
        Calculates the inner product between this equation and another range integer equation.
        """
        if not isinstance(other, RangeIntegerEquation):
            return 0.0
        self._solve(False)
        other._solve(False)

        numel = []
        for i in range(self.ndim):
            min_idx = max(self._lower_solutions[i], other._lower_solutions[i])
            max_idx = min(self._upper_solutions[i], other._upper_solutions[i])
            if max_idx - min_idx + 1 <= 0:
                return 0.0
            numel.append(max_idx - min_idx + 1)
        return float(prod(numel))

    def has_solution(self) -> bool:
        """
        Determines whether the integer equation has a solution given the current evaluated values.
        """
        self._solve(False)
        return self._solvable

    def to_tensor[T: ArrayLike](self, xp: ArrayNamespace[T]) -> T:
        """
        Returns a tensor representation of the integer equation, where the shape of the tensor corresponds to the
        dimensions of the equation, and the values are 1.0 for indices that satisfy the equation and 0.0 for indices that do not.
        """
        shape = []
        for i in range(self.ndim):
            for j, val in enumerate(self.evaluated[i]):
                if val is None:
                    shape.append(self.dims[i][j].base)
        tn = xp.ones(shape, dtype=int)
        idx = 0
        for i in range(self.ndim):
            tmp = xp.zeros(shape, dtype=int)
            for j, val in enumerate(self.evaluated[i]):
                if val is not None:
                    tmp += self.dims[i][j].factor * val
                    continue
                idxs = xp.arange(self.dims[i][j].base)
                cut = [None] * idx + [slice(None)] + [None] * (len(shape) - idx - 1)
                tmp += self.dims[i][j].factor * idxs[*cut,...]
                idx += 1
            tn *= (tmp >= self.lower[i]) & (tmp < self.upper[i])
        tn = xp.astype(tn, default_ftype(xp))
        return tn

    def tensor_shape(self) -> tuple[int, ...]:
        """
        Returns the shape that to_tensor will product.
        """
        return super().tensor_shape()

    def factorize(
            self, *idxs: Sequence[int],
            ) -> tuple[Sequence[Sequence[int]], Sequence[IntegerEquation]]:
        """
        Factorizes the integer equation into sub-equations based on the provided indices. Each index tuple corresponds
        to a dimension and specifies which digits of that dimension to factorize over. The method returns a tuple containing two
        lists: the first list contains the indices for each dimension that were used for factorization, and the second list contains
        the resulting sub-equations after factorization.
        """
        return super().factorize(*idxs)


    def _solve(self, force: bool) -> None:
        if not force and hasattr(self, "_solvable"):
            return
        self._solvable = True
        self._upper_solutions = []
        self._lower_solutions = []
        for i in range(self.ndim):
            solvable, lower, upper = self._upper_lower(i)
            self._solvable = self._solvable and solvable
            self._lower_solutions.append(self._to_idx(i, lower))
            self._upper_solutions.append(self._to_idx(i, upper))
        self._eq_val = (self.dims, self._get_evaled(), self.lower, self.upper)
        self._hash_val = hash(self._eq_val)

    def _to_idx(self, dim_idx: int, vals: Sequence[int]) -> int:
        bases = [digit.base for digit, ev in zip(self.dims[dim_idx], self.evaluated[dim_idx]) if ev is None]
        tmp = 1
        num = 0
        for base, val in zip(bases[::-1], vals[::-1]):
            num += val*tmp
            tmp *= base
        return num

    def _upper_lower(self, idx: int) -> tuple[bool, list[int], list[int]]:
        dim, evaled = self.dims[idx], self.evaluated[idx]

        def solve_problem(sense: int) -> tuple[bool, list[int]]:
            variables = []
            factors = []
            offset = 0
            problem = pl.LpProblem("RangeEquation", sense)
            for digit, ev in zip(dim, evaled):
                if ev is None:
                    factors.append(digit.factor)
                    variables.append(
                        int(digit.factor)
                        * problem.add_variable(
                            f"x_{len(variables)}",
                            lowBound=0,
                            upBound=digit.base - 1,
                            cat="Integer",
                        )
                    )
                else:
                    offset += digit.factor * ev
            problem += pl.lpSum(variables), "Objective"
            problem += pl.lpSum(variables) >= self.lower[idx] - offset, "LowerBound"
            problem += pl.lpSum(variables) <= self.upper[idx] - offset - 1, "UpperBound"
            problem.solve(self._solver)
            solvable = pl.LpStatus[problem.status] == "Optimal"
            solution = [pl.value(var) // val for var, val in zip(variables, factors)]
            return solvable, solution

        lower_solvable, lower = solve_problem(pl.LpMinimize)
        upper_solvable, upper = solve_problem(pl.LpMaximize)
        return lower_solvable and upper_solvable, lower, upper

    def _get_evaled(self) -> tuple[tuple[bool, ...], ...]:
        evaled = []
        for ev in self.evaluated:
            evaled.append(tuple(val is None for val in ev))
        return tuple(evaled)
