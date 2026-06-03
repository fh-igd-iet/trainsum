# Copyright© 2025-2026 Gesellschaft zur Förderung der angewandten Forschung e.V.
# acting on behalf of its Fraunhofer Institut für Graphische Datenverarbeitung.
# Licensed under the EUPL. See LICENSE.txt.

from typing import Sequence
import pulp as pl

from .backend import ArrayNamespace, ArrayLike, default_ftype
from .dimension import Dimension
from .integerequation import IntegerEquation, PulpSolver, PulpStandardSolver

class ModuloIntegerEquation(IntegerEquation):
    """
    Modulo integer equation: :math:`\\sum^N_i c_i x_i \\mod m = \\Delta`. :math:`c_i` are integer coefficients,
    :math:`x_i` are the dimensions, :math:`m` is the product of the sizes of the dimensions and :math:`\\Delta` is the right hand side.
    """

    _solver: PulpSolver
    _dims: tuple[Dimension, ...]
    _evaluated: tuple[tuple[None | int, ...], ...]
    _mods : tuple[int, ...]
    _rhs: int

    _solvable: bool
    _offsets: tuple[int, ...]
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

    #: Tuple of coefficients corresponding to each dimension.
    @property
    def evaluated(self) -> tuple[tuple[None | int, ...], ...]:
        return self._evaluated
    @evaluated.setter
    def evaluated(self, value: tuple[tuple[None | int, ...], ...]):
        self._evaluated = value
        self._solve(True)

    #: Tuple of moduli corresponding to each dimension.
    @property
    def mods(self) -> tuple[int, ...]:
        return self._mods

    #: Right-hand side constant of the equation.
    @property
    def rhs(self) -> int:
        return self._rhs

    def __init__(
            self,
            dims: tuple[Dimension, ...],
            mods: tuple[int, ...],
            rhs: int,
            solver: PulpSolver = PulpStandardSolver(msg=False)):
        if len(mods) != len(dims):
            raise ValueError("Number of coefficients must match the number of dimensions.")
        self._solver = solver
        self._dims = dims
        self._mods = mods
        self._rhs = rhs
        tmp = []
        for dim in dims:
            tmp.append(tuple(None for _ in range(len(dim))))
        self._evaluated = tuple(tuple(ev) for ev in tmp)

    def __hash__(self) -> int:
        self._solve(False)
        return self._hash_val

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ModuloIntegerEquation):
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
            tmp = xp.zeros(shape, dtype=int)
            for j, val in enumerate(self.evaluated[i]):
                if val is not None:
                    tmp += self.dims[i][j].factor * val
                    continue
                idxs = xp.arange(self.dims[i][j].base)
                cut = [None] * idx + [slice(None)] + [None] * (len(shape) - idx - 1)
                tmp += (self.dims[i][j].factor * idxs[*cut,...])
                idx += 1
            tn += tmp % self._mods[i]
        tn = xp.astype(tn == self._rhs, default_ftype(xp))
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
        self._offsets = self._get_offsets()
        self._eq_val = (self.dims, evaled, self._offsets, self._rhs)
        self._hash_val = hash(self._eq_val)
        self._solvable = self._has_solution()

    def _get_offsets(self) -> tuple[int,...]:
        offs = [0] * self.ndim
        for i, evaled in enumerate(self.evaluated):
            for j, val in enumerate(evaled):
                if val is not None:
                    offs[i] += self.dims[i][j].factor * val
            offs[i] = offs[i] % self._mods[i]
        return tuple(offs)

    def _get_evaled(self) -> tuple[tuple[bool, ...], ...]:
        evaled = []
        for ev in self.evaluated:
            evaled.append(tuple(val is None for val in ev))
        return tuple(evaled)

    def _has_solution(self) -> bool:
        prob = pl.LpProblem("ModularEquation", pl.LpMinimize)
        offs = self._get_offsets()
        rs = []
        const_sum = 0

        for dim_idx, (dim, mod, evaled, off) in enumerate(
                zip(self.dims, self._mods, self.evaluated, offs)):
            free_indices = [j for j, val in enumerate(evaled) if val is None]
            if not free_indices:
                const_sum += off
                continue

            expr = off
            max_val = off

            for j in free_indices:
                x = prob.add_variable(
                    f"x_{dim_idx}_{j}",
                    lowBound=0,
                    upBound=dim[j].base - 1,
                    cat="Integer",
                )
                expr += dim[j].factor * x
                max_val += dim[j].factor * (dim[j].base - 1)

            # One k_i and one r_i per dimension
            k = prob.add_variable(
                f"k_{dim_idx}",
                lowBound=0,
                upBound=max_val // mod,
                cat="Integer",
            )
            r = prob.add_variable(
                f"r_{dim_idx}",
                lowBound=0,
                upBound=mod - 1,
                cat="Integer",
            )
            rs.append(r)
            prob += expr == mod * k + r, f"ModConstraint_dim{dim_idx}"

        # If everything is fixed already: just check the constant sum
        if not rs:
            return const_sum == self._rhs

        effective_rhs = self._rhs - const_sum
        prob += pl.lpSum(rs) == effective_rhs, "DeltaConstraint"

        prob.solve(self._solver)
        return pl.LpStatus[prob.status] == "Optimal"
