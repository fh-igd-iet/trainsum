# Copyright© 2025-2026 Gesellschaft zur Förderung der angewandten Forschung e.V.
# acting on behalf of its Fraunhofer Institut für Graphische Datenverarbeitung.
# Licensed under the EUPL. See LICENSE.txt.

from typing import Sequence
from copy import deepcopy
from math import prod

from .backend import ArrayLike, get_index_dtype
from .direction import Direction
from .trainshape import TrainShape
from .trainbase import TrainBase
from .einsumcontraction import EinsumContraction, get_symbol_generator
from .matrixdecomposition import MatrixDecomposition
from .qrdecomposition import QRDecomposition
from .utils import check_operand_shapes, get_shapes, shape_map
from .contractor import ArrayContractor, OptimizeKind
from .contractorinput import ContractorInput
from .localcontraction import LocalContraction
from .contains import contains


class NormationContractor:
    optimizer: OptimizeKind
    decomposition: MatrixDecomposition
    direction: Direction
    _contr: EinsumContraction
    _inp: None | ContractorInput = None
    _tmp_str: str
    _cmap: Sequence[int]
    _to_right_exprs: dict[int, tuple[LocalContraction, ArrayContractor]]
    _to_left_exprs: dict[int, tuple[LocalContraction, ArrayContractor]]

    def __init__(
        self,
        contr: EinsumContraction,
        optimizer: OptimizeKind = "greedy",
        decomposition: MatrixDecomposition = QRDecomposition(),
        direction: Direction = Direction.TO_RIGHT,
        max_rank: int = 50,
        relative_cutoff: float = 1e-15,
    ) -> None:
        if contr.result_shape is None or contr.full_result_shape is None:
            raise ValueError(
                "NormationContractor cannot be used for full contractions. Use FullContractor instead."
            )

        self.optimizer = deepcopy(optimizer)
        self.decomposition = deepcopy(decomposition)
        self.direction = direction

        self.max_rank = max_rank
        self.relative_cutoff = relative_cutoff

        self._contr = deepcopy(contr)
        res = contains(contr.full_result_shape, contr.result_shape)
        self._cmap = res.core_idxs
        _sgen = get_symbol_generator(contr)
        self._tmp_str1 = next(_sgen)
        self._tmp_str2 = next(_sgen)

        self._inp = None
        self._to_right_exprs = {}
        self._to_left_exprs = {}

    def __call__[T: ArrayLike](
        self, *ops: TrainBase[T], expr: bool = False
    ) -> TrainBase[T]:

        shapes = get_shapes(*ops)
        if expr or self._inp is None:
            self.calc_expressions(*shapes)
        else:
            self._inp.check_operands(*ops)

        if self.direction == Direction.TO_RIGHT:
            [op.normalize(0) for op in ops]
            return self._contract_to_right(*ops)
        elif self.direction == Direction.TO_LEFT:
            [op.normalize(-1) for op in ops]
            return self._contract_to_left(*ops)
        raise ValueError("Direction must be either 'to_left' or 'to_right'.")

    def calc_expressions(self, *ops: TrainShape | TrainBase) -> None:
        check_operand_shapes(self._contr.operand_shapes, get_shapes(*ops))
        self._inp = ContractorInput(*ops)

        shape = self._contr.result_shape
        shape.ranks = None  # type: ignore

        self._to_right_exprs.clear()
        self._to_left_exprs.clear()

        n_contr = len(self._contr)
        for i in range(n_contr):
            max_rank = shape.ranks[i - 1] if i != 0 else 1  # type: ignore
            self._to_right_expressions(max_rank, i, *ops)

        for i in range(n_contr - 1, -1, -1):
            max_rank = shape.ranks[i + 1] if i < n_contr - 2 else 1  # type: ignore
            self._to_left_expressions(max_rank, i, *ops)

    # ------------------------------------------------------------------------
    # Contraction generators

    def _contract_to_right(self, *ops: TrainBase):
        if self._inp is None:
            raise RuntimeError("Input cannot be None here.")
        if self._contr.result_shape is None:
            raise RuntimeError("ResultShape cannot be None here.")

        xp, device, dtype = self._inp.infos(*ops)
        idx_dtype = get_index_dtype(xp)

        cores = []
        tmp = xp.ones(
            [1, *[1] * len(self._contr[0].result.left)], device=device, dtype=dtype
        )
        norm_idxs: dict[str, Sequence[idx_dtype]] = {}

        for i in range(len(self._contr)):
            lcontr, expr = self._to_right_exprs[i]

            tns = list(lcontr.get_data(*ops, idx_map=self._inp.idx_map))

            # truncate tns
            for j, op in enumerate(lcontr.operands):
                bond_dim = op.left
                if bond_dim in norm_idxs.keys():
                    tns[j] = tns[j][norm_idxs[bond_dim], ...]

            ncore = expr(tmp, *tns)

            if i < len(self._contr) - 1:
                bond_dims = lcontr.result.right
                axis_idxs = tuple(range(len(ncore.shape) - len(bond_dims)))

                norms = xp.sqrt(xp.sum(ncore**2, axis=axis_idxs))
                norms_flat = norms.flatten()

                # only keep at most max_rank many
                mrank = min(self.max_rank, prod(ncore.shape[-len(tns) :]))

                # remove if less than cutoff
                numel = xp.sum(norms / norms.max() > self.relative_cutoff)
                numel = min(numel, mrank)

                # get biggest idxs
                flat_idxs = xp.argsort(norms_flat)[-numel:]
                idxs = unravel_indices(flat_idxs, norms.shape)

                # save idxs per bond
                norm_idxs.clear()
                for j, bond_dim in enumerate(bond_dims):
                    norm_idxs[bond_dim] = idxs[j]

                ncore = ncore[..., *idxs]
                mat = xp.reshape(ncore, (prod(ncore.shape[:-1]), ncore.shape[-1]))
                res = self.decomposition.left(mat)
                left, tmp = res.left, res.right
                ncore = xp.reshape(left, (*ncore.shape[:-1], left.shape[1]))
            cores.append(ncore)

        return TrainBase(self._contr.result_shape, cores)

    def _contract_to_left(self, *ops: TrainBase):
        if self._inp is None:
            raise RuntimeError("Input cannot be None here.")
        if self._contr.result_shape is None:
            raise RuntimeError("ResultShape cannot be None here.")

        xp, device, dtype = self._inp.infos(*ops)
        idx_dtype = get_index_dtype(xp)

        cores = []
        tmp = xp.ones(
            [1, *[1] * len(self._contr[-1].result.right)], device=device, dtype=dtype
        )

        norm_idxs: dict[str, Sequence[idx_dtype]] = {}

        for i in range(len(self._contr) - 1, -1, -1):
            lcontr, expr = self._to_left_exprs[i]

            tns = list(lcontr.get_data(*ops, idx_map=self._inp.idx_map))

            # truncate tns
            for j, op in enumerate(lcontr.operands):
                bond_dim = op.right
                if bond_dim in norm_idxs.keys():
                    tns[j] = tns[j][..., norm_idxs[bond_dim]]

            ncore = expr(*tns, tmp)

            if i > 0:
                bond_dims = lcontr.result.left
                axis_idxs = tuple(range(len(bond_dims), len(ncore.shape)))

                norms = xp.sqrt(xp.sum(ncore**2, axis=axis_idxs))
                norms_flat = norms.flatten()

                # only keep at most max_rank many
                mrank = min(self.max_rank, prod(ncore.shape[-len(tns) :]))
                mrank = min(self.max_rank, prod(ncore.shape[: len(bond_dims)]))

                # remove if less than cutoff
                numel = xp.sum(norms / norms.max() > self.relative_cutoff)
                numel = min(numel, mrank)

                # get biggest idxs
                flat_idxs = xp.argsort(norms_flat)[-numel:]
                idxs = unravel_indices(flat_idxs, norms.shape)

                # save idxs per bond
                norm_idxs.clear()
                for j, bond_dim in enumerate(bond_dims):
                    norm_idxs[bond_dim] = idxs[j]

                ncore = ncore[*idxs, ...]
                mat = xp.reshape(ncore, (ncore.shape[0], prod(ncore.shape[1:])))
                res = self.decomposition.right(mat)
                tmp, right = res.left, res.right
                ncore = xp.reshape(right, (right.shape[0], *ncore.shape[1:]))
            cores.append(ncore)
        cores.reverse()

        return TrainBase(self._contr.result_shape, cores)

    # ------------------------------------------------------------------------
    # Expression builders

    def _to_right_expressions(
        self, left_rank: int, idx: int, *ops: TrainBase | TrainShape
    ) -> None:
        lcontr = self._contr[idx]

        ops_str = [f"{self._tmp_str1}{self._tmp_str2}"]
        ops_str.extend(f"{op.left}{op.middle}{op.right}" for op in lcontr.operands)

        # replace bond dims with _tmp_str2
        if self._to_right_exprs:
            last_bond_dims = self._to_right_exprs[idx - 1][0].result.right
            for bond_char in last_bond_dims:
                ops_str = [
                    op_str.replace(bond_char, self._tmp_str2) for op_str in ops_str
                ]

        res = f"{self._tmp_str1}{lcontr.result.middle}{lcontr.result.right}"
        eq = f"{','.join(ops_str)}->{res}"

        smap = shape_map(ops, lcontr)
        smap[self._tmp_str1] = left_rank
        smap[self._tmp_str2] = min(left_rank, self.max_rank)

        tns = list()
        for op in ops_str:
            tns.append([smap[char] for char in op])

        expr = ArrayContractor(eq, *tns, optimizer=self.optimizer)
        self._to_right_exprs[idx] = lcontr, expr

    def _to_left_expressions(
        self, right_rank: int, idx: int, *ops: TrainBase | TrainShape
    ) -> None:
        lcontr = self._contr[idx]

        ops_str = [f"{op.left}{op.middle}{op.right}" for op in lcontr.operands]
        ops_str.append(f"{self._tmp_str2}{self._tmp_str1}")

        # replace bond dims with _tmp_str2
        if self._to_left_exprs:
            last_bond_dims = self._to_left_exprs[idx + 1][0].result.left
            for bond_char in last_bond_dims:
                ops_str = [
                    op_str.replace(bond_char, self._tmp_str2) for op_str in ops_str
                ]

        res = f"{lcontr.result.left}{lcontr.result.middle}{self._tmp_str1}"
        eq = f"{','.join(ops_str)}->{res}"

        smap = shape_map(ops, lcontr)
        smap[self._tmp_str1] = right_rank
        smap[self._tmp_str2] = min(right_rank, self.max_rank)

        tns = list()
        for op in ops_str:
            tns.append([smap[char] for char in op])

        expr = ArrayContractor(eq, *tns, optimizer=self.optimizer)
        self._to_left_exprs[idx] = lcontr, expr


# ------------------------------------------------------------------------
# Utils
def unravel_indices[T: ArrayLike](index: T, shape: tuple[int, ...]) -> tuple[T, ...]:
    out = []
    for dim in reversed(shape):
        out.append(index % dim)
        index = index // dim
    return tuple(reversed(out))
