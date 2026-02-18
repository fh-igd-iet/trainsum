# Copyright© 2025-2026 Gesellschaft zur Förderung der angewandten Forschung e.V.
# acting on behalf of its Fraunhofer Institut für Graphische Datenverarbeitung.
# Licensed under the EUPL. See LICENSE.txt.

from typing import Sequence
from copy import deepcopy

from .backend import ArrayLike
from .direction import Direction
from .localrange import LocalRange
from .normalization import Normalization
from .trainshape import TrainShape
from .trainbase import TrainBase
from .einsumcontraction import EinsumContraction, get_symbol_generator
from .matrixdecomposition import MatrixDecomposition
from .tensordecomposition import TensorDecomposition
from .svdecomposition import SVDecomposition
from .sweepingstrategy import SweepingStrategy
from .utils import check_operand_shapes, get_shapes, shape_map
from .contractor import ArrayContractor, OptimizeKind
from .contractorinput import ContractorInput
from .localcontraction import fuse_local_contractions, LocalContraction
from .contains import contains

class DecompositionContractor:

    optimizer: OptimizeKind
    decomposition: MatrixDecomposition
    strategy: SweepingStrategy
    direction: Direction
    _contr: EinsumContraction
    _inp: None | ContractorInput = None
    _tmp_str: str
    _cmap: Sequence[int]
    _to_right_exprs: dict[tuple[int, LocalRange],
                          tuple[LocalContraction, ArrayContractor]]
    _to_right_tranges: list[LocalRange]
    _to_left_exprs: dict[tuple[LocalRange, int],
                         tuple[LocalContraction, ArrayContractor]]
    _to_left_tranges: list[LocalRange]

    def __init__(self,
                 contr: EinsumContraction,
                 optimizer: OptimizeKind = "greedy",
                 decomposition: MatrixDecomposition = SVDecomposition(),
                 strategy: SweepingStrategy = SweepingStrategy(),
                 direction: Direction = Direction.TO_RIGHT) -> None:

        if contr.result_shape is None or contr.full_result_shape is None:
            raise ValueError("DecompositionContractor cannot be used for full ccontractions. Use FullContractor instead.")

        self.optimizer = deepcopy(optimizer)
        self.decomposition = deepcopy(decomposition)
        self.strategy = deepcopy(strategy)
        self.direction = direction

        self._contr = deepcopy(contr)
        res = contains(contr.full_result_shape, contr.result_shape)
        self._cmap = res.core_idxs
        self._tmp_str = next(get_symbol_generator(contr))

        self._inp = None
        self._to_right_exprs = {}
        self._to_right_tranges = []
        self._to_left_exprs = {}
        self._to_left_tranges = []

    def __call__[T: ArrayLike](self,
            *ops: TrainBase[T],
            expr: bool = False) -> TrainBase[T]:
        shapes = get_shapes(*ops)
        if expr or self._inp is None:
            self.calc_expressions(*shapes)
        else:
            self._inp.check_operands(*ops)

        if self.direction == Direction.TO_RIGHT:
            [op.normalize(0) for op in ops]
            return self._contract_to_right(self.decomposition, *ops)
        elif self.direction == Direction.TO_LEFT:
            [op.normalize(-1) for op in ops]
            return self._contract_to_left(self.decomposition, *ops)
        raise ValueError("Direction must be either 'to_left' or 'to_right'.")

    def calc_expressions(self, *ops: TrainShape | TrainBase) -> tuple[TrainShape, TrainShape]:
        check_operand_shapes(self._contr.operand_shapes, get_shapes(*ops))
        self._inp = ContractorInput(*ops)

        right_ranges = self._get_ranges(Direction.TO_RIGHT)
        right_shape = self._to_right_guess(self.decomposition, right_ranges, *ops)
        self._to_right_expressions(right_ranges, right_shape, *ops)

        left_ranges = self._get_ranges(Direction.TO_LEFT)
        left_shape = self._to_left_guess(self.decomposition, left_ranges, *ops)
        self._to_left_expressions(left_ranges, left_shape, *ops)
        return left_shape, right_shape

    # ------------------------------------------------------------------------
    # Contraction generators

    def _contract_to_right[T: ArrayLike](
            self,
            decomp: MatrixDecomposition,
            *ops: TrainBase[T]) -> TrainBase[T]:
        if self._inp is None:
            raise RuntimeError("Input cannot be None here.")
        xp, device, dtype = self._inp.infos(*ops)

        shape = self._contr.result_shape
        data = []
        norm = []
        if shape is None:
            raise ValueError("Result shape cannot be none here.")
        tn_decomp = TensorDecomposition(decomp)

        lidx = 0
        idx = 0
        tmp = xp.ones([1, *[1]*len(self._contr[0].result.left)],
                      device=device, dtype=dtype)
        for i, trange in enumerate(self._to_right_tranges):
            lcontr, expr = self._to_right_exprs[(idx, trange)]
            tns = lcontr.get_data(*ops, idx_map=self._inp.idx_map)
            tmp = expr(tmp, *tns)
            idx = trange.end
            
            ref_idx = self._cmap[-1]+1
            if i < len(self._to_right_tranges)-1:
                ref_idx = self._to_right_tranges[i+1].begin

            while lidx < len(shape)-1 and self._cmap[lidx] < ref_idx:
                split = 1+len(shape.middle(lidx))
                #left, tmp = tn_decomp.left(tmp, split)
                res = tn_decomp.left(tmp, split)
                left, tmp = res.left, res.right
                data.append(left)
                norm.append(Normalization.LEFT)
                lidx += 1
        data.append(tmp)
        norm.append(Normalization.NONE)

        return TrainBase(shape, data, norm, copy_data=False)

    def _contract_to_left[T: ArrayLike](
            self,
            decomp: MatrixDecomposition,
            *ops: TrainBase[T]) -> TrainBase[T]:
        if self._inp is None:
            raise RuntimeError("Input cannot be None here.")
        xp, device, dtype = self._inp.infos(*ops)

        shape = self._contr.result_shape
        data = []
        norm = []
        if shape is None:
            raise ValueError("Result shape cannot be none here.")
        tn_decomp = TensorDecomposition(decomp)

        lidx = len(shape)-1
        idx = len(self._contr)
        tmp = xp.ones([*[1]*len(self._contr[-1].result.right), 1],
                      device=device, dtype=dtype)
        for i, trange in enumerate(self._to_left_tranges):
            lcontr, expr = self._to_left_exprs[(trange, idx)]
            tns = lcontr.get_data(*ops, idx_map=self._inp.idx_map)
            tmp = expr(*tns, tmp)
            idx = trange.begin
            
            ref_idx = self._cmap[0]-1
            if i < len(self._to_left_tranges)-1:
                ref_idx = self._to_left_tranges[i+1].end-1

            while lidx > 0 and self._cmap[lidx] > ref_idx:
                split = len(tmp.shape)-len(shape.middle(lidx))-1
                #tmp, right = tn_decomp.right(tmp, split)
                res = tn_decomp.right(tmp, split)
                tmp, right = res.left, res.right
                data.append(right)
                norm.append(Normalization.RIGHT)
                lidx -= 1

        data.append(tmp)
        data.reverse()
        norm.append(Normalization.NONE)
        norm.reverse()
        return TrainBase(shape, data, norm, copy_data=False)

    # ------------------------------------------------------------------------
    # Expression builders

    def _to_right_expressions(self,
                              tranges: Sequence[LocalRange],
                              guess: TrainShape,
                              *ops: TrainShape | TrainBase) -> TrainShape:
        self._to_right_exprs.clear()
        self._to_right_tranges.clear()

        self._to_right_expression(0, tranges[0], 1, *ops)
        self._to_right_tranges.append(tranges[0])
        for i in range(1, len(tranges)):
            end_idx = tranges[i-1].end
            left_size = guess.right_rank(end_idx-1)
            if (end_idx, tranges[i]) in self._to_right_exprs:
                continue
            elif end_idx >= tranges[i].end:
                continue
            self._to_right_expression(end_idx, tranges[i], left_size, *ops)
            self._to_right_tranges.append(tranges[i])
        return guess

    def _to_right_guess(
            self,
            decomp: MatrixDecomposition,
            tranges: Sequence[LocalRange],
            *ops: TrainShape | TrainBase) -> TrainShape:
        res_shape = self._contr.result_shape
        if res_shape is None:
            raise ValueError("Result shape cannot be none here.")
        tn_decomp = TensorDecomposition(decomp)
        ranks = [1]

        idx = 0
        for i, trange in enumerate(tranges):
            lcontr = fuse_local_contractions(self._contr[trange.begin:trange.end])
            smap = shape_map(ops, lcontr)
            smap[self._tmp_str] = ranks[-1]
            result = f"{self._tmp_str}{lcontr.result.middle}{lcontr.result.right}"
            tmp = [smap[char] for char in result]
            
            ref_idx = self._cmap[-1]+1
            if i < len(tranges)-1:
                ref_idx = tranges[i+1].begin

            while idx < len(res_shape)-1 and self._cmap[idx] < ref_idx:
                split = 1+len(res_shape.middle(idx))
                left, tmp = tn_decomp.left_shape(tuple(tmp), split)
                ranks.append(left[-1])
                idx += 1
        return TrainShape(res_shape.dims, res_shape.digits, ranks[1:])

    def _to_right_expression(self,
                             prev_tend: int,
                             cur_trange: LocalRange,
                             left_size: int,
                             *ops: TrainBase | TrainShape
                             ) -> None:
        if prev_tend < cur_trange.begin:
            raise ValueError("End of previous range must be at least the begin of the current range.")
        elif prev_tend >= cur_trange.end:
            raise ValueError("End of previous range must be less than the end of the current range.")

        res_contr = fuse_local_contractions(self._contr[cur_trange.begin:cur_trange.end])
        res_str = f"{self._tmp_str}{res_contr.result.middle}{res_contr.result.right}"

        if prev_tend == cur_trange.begin:
            tmp_str = f"{self._tmp_str}{res_contr.result.left}"
            contr = res_contr
        else:
            tmp_contr = fuse_local_contractions(self._contr[cur_trange.begin:prev_tend])
            tmp_str = f"{self._tmp_str}{tmp_contr.result.middle}{tmp_contr.result.right}"
            contr = fuse_local_contractions(self._contr[prev_tend:cur_trange.end])
        eq = f"{tmp_str}," + ",".join(str(op) for op in contr.operands) + f"->{res_str}"

        smap = shape_map(ops, *self._contr[cur_trange.begin:cur_trange.end])
        smap[self._tmp_str] = left_size
        tmp_shape = [smap[char] for char in tmp_str]

        _, tns = contr.get_constants(*ops)
        expr = ArrayContractor(eq, tmp_shape, *tns, optimizer=self.optimizer)
        self._to_right_exprs[(prev_tend, cur_trange)] = contr, expr

    def _to_left_expressions(self,
                             tranges: Sequence[LocalRange],
                             guess: TrainShape,
                             *ops: TrainShape | TrainBase) -> None:
        self._to_left_exprs.clear()
        self._to_left_tranges.clear()

        self._to_left_expression(tranges[-1], tranges[-1].end, guess.left_rank(-1), *ops)
        self._to_left_tranges.append(tranges[-1])
        for i in range(len(tranges)-2, -1, -1):
            begin_idx = tranges[i+1].begin
            left_size = guess.left_rank(begin_idx)
            if (tranges[i], begin_idx) in self._to_left_exprs:
                continue
            elif begin_idx <= tranges[i].begin:
                continue
            self._to_left_expression(tranges[i], begin_idx, left_size, *ops)
            self._to_left_tranges.append(tranges[i])

    def _to_left_guess(
            self,
            decomp: MatrixDecomposition,
            tranges: Sequence[LocalRange],
            *ops: TrainShape | TrainBase) -> TrainShape:
        res_shape = self._contr.result_shape
        if res_shape is None:
            raise ValueError("Result shape cannot be none here.")
        tn_decomp = TensorDecomposition(decomp)
        ranks = [1]

        idx = len(res_shape)-1
        for i, trange in enumerate(reversed(tranges)):
            lcontr = fuse_local_contractions(self._contr[trange.begin:trange.end])
            smap = shape_map(ops, lcontr)
            smap[self._tmp_str] = ranks[-1]
            result = f"{lcontr.result.left}{lcontr.result.middle}{self._tmp_str}"
            tmp = [smap[char] for char in result]
            
            ref_idx = self._cmap[0]-1
            if i < len(tranges)-1:
                ref_idx = tranges[-i-2].end-1

            while idx > 0 and self._cmap[idx] > ref_idx:
                split = len(tmp)-len(res_shape.middle(idx))-1
                tmp, right = tn_decomp.right_shape(tuple(tmp), split)
                ranks.append(right[0])
                idx -= 1
        ranks.reverse()
        return TrainShape(res_shape.dims, res_shape.digits, ranks[1:])

    def _to_left_expression(self,
                            cur_trange: LocalRange,
                            prev_tbegin: int,
                            left_size: int,
                            *ops: TrainBase | TrainShape
                            ) -> None:
        if prev_tbegin > cur_trange.end:
            raise ValueError("Begin of the previous range must be at least the end of the current range.")
        elif prev_tbegin <= cur_trange.begin:
            raise ValueError("Begin of the previous range must be greater than the begin of the current range.")

        res_contr = fuse_local_contractions(self._contr[cur_trange.begin:cur_trange.end])
        res_str = f"{res_contr.result.left}{res_contr.result.middle}{self._tmp_str}"

        if prev_tbegin == cur_trange.end:
            tmp_str = f"{res_contr.result.right}{self._tmp_str}"
            contr = res_contr
        else:
            tmp_contr = fuse_local_contractions(self._contr[prev_tbegin:cur_trange.end])
            tmp_str = f"{tmp_contr.result.left}{tmp_contr.result.middle}{self._tmp_str}"
            contr = fuse_local_contractions(self._contr[cur_trange.begin:prev_tbegin])
        eq = ",".join(str(op) for op in contr.operands) + f",{tmp_str}" + f"->{res_str}"

        smap = shape_map(ops, *self._contr[cur_trange.begin:cur_trange.end])
        smap[self._tmp_str] = left_size
        tmp_shape = [smap[char] for char in tmp_str]

        _, tns = contr.get_constants(*ops)
        expr = ArrayContractor(eq, *tns, tmp_shape, optimizer=self.optimizer)
        self._to_left_exprs[(cur_trange, prev_tbegin)] = contr, expr


    def _transform_range(self, lrange: LocalRange) -> LocalRange:
        begin = self._cmap[lrange.begin]
        end = self._cmap[lrange.end-1]
        return LocalRange(begin=begin, end=end+1)

    def _get_ranges(self, direction: Direction) -> Sequence[LocalRange]:
        if self._contr.result_shape is None:
            raise RuntimeError("Result shape cannot be none here.")
        if direction == Direction.TO_RIGHT:
            lranges = self.strategy.right_sweep(self._contr.result_shape)
        elif direction == Direction.TO_LEFT:
            lranges = self.strategy.left_sweep(self._contr.result_shape)
        tranges = [self._transform_range(lrange) for lrange in lranges]
        tranges = sorted(tranges)
        tranges = list(dict.fromkeys(tranges))
        if tranges[0].begin != 0 or tranges[-1].end != len(self._contr):
            raise RuntimeError("Provided local ranges do not cover the full contraction.")
        return tranges
