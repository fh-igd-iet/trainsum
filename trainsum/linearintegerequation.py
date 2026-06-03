# Copyright© 2025-2026 Gesellschaft zur Förderung der angewandten Forschung e.V.
# acting on behalf of its Fraunhofer Institut für Graphische Datenverarbeitung.
# Licensed under the EUPL. See LICENSE.txt.

from typing import Sequence
import pulp as pl
from math import gcd

from .backend import ArrayNamespace, ArrayLike, default_ftype
from .dimension import Dimension
from .integerequation import IntegerEquation, PulpSolver, PulpStandardSolver

class LinearIntegerEquation(IntegerEquation):
    """
    Linear integer equation: :math:`\\sum^N_i c_i x_i = \\Delta`. :math:`c_i` are integer coefficients,
    :math:`x_i` are the dimensions and :math:`\\Delta` is the right hand side.
    """

    _solver: PulpSolver
    _dims: tuple[Dimension, ...]
    _evaluated: tuple[tuple[None | int, ...], ...]
    _coeffs : tuple[int, ...]
    _rhs: int

    _solvable: bool
    _effective_rhs: int
    _hash_val: int
    _eq_val: tuple[tuple[Dimension, ...], tuple[tuple[None | int, ...], ...], tuple[int, ...], int]


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

    #: Per dimension tuples, where each tuple contains the evaluated digit values (or None if not evaluated) for that dimension.
    @property
    def evaluated(self) -> tuple[tuple[None | int, ...], ...]:
        return self._evaluated
    @evaluated.setter
    def evaluated(self, value: tuple[tuple[None | int, ...], ...]):
        self._evaluated = value
        self._solve(True)

    #: Tuple of coefficients corresponding to each dimension.
    @property
    def coeffs(self) -> tuple[int, ...]:
        return self._coeffs

    #: Right-hand side constant of the equation.
    @property
    def rhs(self) -> int:
        return self._rhs

    def __init__(
            self,
            dims: tuple[Dimension, ...],
            coeffs: tuple[int, ...],
            rhs: int,
            solver: PulpSolver = PulpStandardSolver(msg=False)):
        self._solver = solver
        self._dims = tuple(dims)
        self._coeffs = tuple(coeffs)
        self._rhs = rhs
        tmp = []
        for dim in dims:
            tmp.append(tuple(None for _ in range(len(dim))))
        self._evaluated = tuple(tuple(ev) for ev in tmp)

    def __hash__(self) -> int:
        self._solve(False)
        return self._hash_val

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, LinearIntegerEquation):
            return False
        self._solve(False)
        other._solve(False)
        return self._eq_val == other._eq_val
    
    def inner(self, other: IntegerEquation) -> float:
        """
        Calculates the inner product between this equation and another equation. The inner product is defined as
        1.0 if the two equations are identical, and 0.0 otherwise.
        """
        if self == other:
            return 1.0
        return 0.0

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
        tn = xp.zeros(shape, dtype=int)
        idx = 0
        for i in range(self.ndim):
            for j, val in enumerate(self.evaluated[i]):
                if val is not None:
                    continue
                idxs = xp.arange(self.dims[i][j].base)
                cut = [None] * idx + [slice(None)] + [None] * (len(shape) - idx - 1)
                tn += self.coeffs[i] * self.dims[i][j].factor * idxs[*cut,...]
                idx += 1
        tn = xp.astype(tn == self._get_rhs(), default_ftype(xp))
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
        evaled = self._get_evaled()
        self._effective_rhs = self._get_rhs()
        self._eq_val = (self.dims, evaled, self.coeffs, self._effective_rhs)
        self._hash_val = hash(self._eq_val)

        if self._is_block(evaled):
            self._solvable = self._block_solution()
        else:
            self._solvable = self._ilp_solution()

    def _ilp_solution(self) -> bool:
        x = []
        prob = pl.LpProblem("LinearEquation", pl.LpMinimize)
        for dim, coeff, evaled in zip(self.dims, self.coeffs, self.evaluated):
            for i, val in enumerate(evaled):
                if val is not None:
                    continue
                #var = pl.LpVariable(f"x_{len(x)}",
                #                    lowBound=0,
                #                    upBound=dim[i].base-1,
                #                    cat="Integer")
                var = prob.add_variable(f"x_{len(x)}",
                                  lowBound=0,
                                  upBound=dim[i].base-1,
                                  cat="Integer")
                expr = coeff * dim[i].factor * var
                x.append(expr)

        prob += pl.lpSum(x) == self._get_rhs(), "EquationConstraint"
        prob.solve(self._solver)
        return pl.LpStatus[prob.status] == "Optimal"

    def _block_solution(self) -> bool:
        min_idx, max_idx = 0, 0
        val = gcd(*self.coeffs)
        shape = []
        for dim, ev in zip(self.dims, self._get_evaled()):
            if True not in ev:
                shape.append(1)
                continue
            idx = ev.index(True)
            shape.append(sum(digit.factor*(digit.base-1) for digit in dim[idx:])+1)
        for coeff, size in zip(self.coeffs, shape):
            if coeff > 0:
                max_idx += coeff * (size - 1)
            else:
                min_idx += coeff * (size - 1)
        return min_idx <= self._effective_rhs <= max_idx and self._effective_rhs % val == 0 
 
    def _get_rhs(self) -> int:
        rhs = self.rhs
        for dim, coeff, evaled in zip(self.dims, self.coeffs, self.evaluated):
            for i, val in enumerate(evaled):
                if val is not None:
                    rhs -= coeff * dim[i].factor * val
        return rhs

    def _get_evaled(self) -> tuple[tuple[bool, ...], ...]:
        evaled = []
        for ev in self.evaluated:
            evaled.append(tuple(val is None for val in ev))
        return tuple(evaled)

    def _is_block(self, evaled: tuple[tuple[bool, ...], ...]) -> bool:
        flags = []
        for ev in evaled:
            if all(ev):
                flags.append(True)
                continue
            elif not any(ev):
                flags.append(True)
                continue
            if False not in ev:
                flags.append(True)

            idx = ev.index(True)
            if False in ev[idx+1:]:
                flags.append(False)

        #print(flags, evaled)
        return all(flags)
