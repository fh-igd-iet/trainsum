# Copyright© 2025-2026 Gesellschaft zur Förderung der angewandten Forschung e.V.
# acting on behalf of its Fraunhofer Institut für Graphische Datenverarbeitung.
# Licensed under the EUPL. See LICENSE.txt.

# CITE: https://arxiv.org/pdf/2110.04393
# TITLE: Randomized algorithms for rounding in the Tensor-Train format
# CODE: https://github.com/SAMSI-RandTensors/randomizedTT/blob/main/TTrandomized/TTrounding_Randomize_then_Orthogonalize.m
# > Nice diagrams
# > Randomized from right-to-left, orthogonalize left-to-right
#
# CITE: https://arxiv.org/pdf/2603.11009
# TITLE: Linear-scaling Tensor Train Sketching
# > Derives errorbounds
# > Introduces Stacking
# > Introduces Stiefel for better error bounds
#
# CITE: https://arxiv.org/pdf/2504.06475
# TITLE: Successive randomized compression: A randomized algorithm for the compressed MPO-MPS product
# CODE: https://github.com/chriscamano/RandomMPOMPS/blob/main/code/tensornetwork/contraction.py#L82
# > Generalizes sketching to MPO-MPS
# > Nice diagrams


from copy import deepcopy
from functools import cache
from math import prod
from typing import Literal, Sequence

import numpy as np

from .backend import ArrayLike
from .contractor import ArrayContractor, OptimizeKind
from .contractorinput import ContractorInput
from .direction import Direction
from .einsumcontraction import EinsumContraction, get_symbol_generator
from .exactaddition import ExactAddition
from .trainbase import TrainBase
from .trainshape import TrainShape
from .utils import check_operand_shapes, get_shapes, symbol_generator


class SketchContractor:
    """
    Randomized compression for contractions (randomize-then-orthogonalize):
    1. draw a random sketch cores Omega,
    2. sweep right-to-left to build right environments W_k (partial contractions),
    3. sweep left-to-right, sketch each local unfolding, and use QR to extract new TT cores.

    Note that, left-to-right or right-to-left sweeps can be swapped, but the random environments
    must be built in the opposite direction of the orthogonalization sweep.
    """

    optimizer: OptimizeKind
    direction: Direction
    _contr: EinsumContraction
    _inp: None | ContractorInput = None

    def __init__(
        self,
        contr: EinsumContraction,
        optimizer: OptimizeKind = "greedy",
        P: int = 4,
        sketch_rank: int = 6,
        seed: int | None = None,
        random_distribution: Literal["gaussian", "uniform", "stiefel"] = "gaussian",
        sketch_mode: Literal["flattened", "stacked"] = "flattened",
        direction: Direction = Direction.TO_RIGHT,
    ) -> None:
        if contr.result_shape is None or contr.full_result_shape is None:
            raise ValueError(
                "SketchContractor cannot be used for full contractions. Use FullContractor instead."
            )
        if P <= 0:
            raise ValueError("P must be positive.")
        if sketch_rank <= 0:
            raise ValueError("sketch_rank must be positive.")

        self.optimizer = deepcopy(optimizer)
        self._contr = deepcopy(contr)
        self.P = P
        self.sketch_rank = sketch_rank
        self.seed = seed
        self.random_distribution = random_distribution
        self.sketch_mode = sketch_mode
        self.direction = direction

    def __call__[T: ArrayLike](
        self, *ops: TrainBase[T], expr: bool = False
    ) -> TrainBase[T]:
        shapes = get_shapes(*ops)
        if expr or self._inp is None:
            self.calc_expressions(*shapes)
        else:
            self._inp.check_operands(*ops)

        res = self._sketch(*ops)
        if self.sketch_mode == "stacked":
            merged = ExactAddition()(*res)
            merged.data[0][...] /= self.P
            return merged

        if isinstance(res, list):
            raise RuntimeError("Only stacked sketching may return multiple trains.")

        return res

    def calc_expressions(self, *ops: TrainShape | TrainBase) -> None:
        check_operand_shapes(self._contr.operand_shapes, get_shapes(*ops))
        self._inp = ContractorInput(*ops)

    # ------------------------------------------------------------------------
    # Contraction

    def _sketch[T: ArrayLike](
        self, *ops: TrainBase[T]
    ) -> TrainBase[T] | list[TrainBase[T]]:
        if self._inp is None:
            raise RuntimeError("Input cannot be None here.")
        xp, device, dtype = self._inp.infos(*ops)

        random_cores = self._random_cores(xp, dtype, device)
        partials = self._partial_contractions(random_cores, *ops)

        if self.sketch_mode == "stacked":
            return [
                self._sketch_from_partials(partials, *ops, sample=i)
                for i in range(self.P)
            ]

        return self._sketch_from_partials(partials, *ops)

    # ------------------------------------------------------------------------
    # Sketch cores

    def _random_cores(self, xp, dtype, device) -> list[ArrayLike]:
        """
        Build the random TT sketch cores Omega_k.
        The leading axis has size P, so we can either:
        - flatten all P samples into one larger sketch, or
        - keep them separate and average the resulting TT tensors later.
        """
        if self._contr.result_shape is None:
            raise RuntimeError("Result shape cannot be None here.")

        # TODO: NumPy RNG backend-agnostic for now;
        # switch to a shared array-namespace RNG helper.
        rng = (
            np.random.default_rng()
            if self.seed is None
            else np.random.default_rng(self.seed)
        )
        mode_dims = [
            tuple(self._contr.result_shape.middle(k)) for k in range(len(self._contr))
        ]
        ranks = [1] + [self.sketch_rank] * (len(mode_dims) - 1) + [1]
        if self.random_distribution == "stiefel":
            ranks = self._stiefel_ranks(ranks, mode_dims)

        cores = []
        for k, dims in enumerate(mode_dims):
            r_prev, r_next = ranks[k], ranks[k + 1]
            shape = (self.P, r_prev, *dims, r_next)
            if self.random_distribution == "gaussian":
                arr = rng.normal(size=shape) / np.sqrt(max(r_prev, 1))
            elif self.random_distribution == "uniform":
                arr = rng.uniform(low=-1.0, high=1.0, size=shape) / np.sqrt(
                    max(r_prev, 1)
                )
            else:
                arr = self._random_stiefel(rng, shape) / np.sqrt(max(r_prev, 1))
            cores.append(xp.asarray(arr, dtype=dtype, device=device))
        return cores

    # ------------------------------------------------------------------------
    # Partial contractions

    def _partial_contractions[T: ArrayLike](
        self,
        random_cores: Sequence[T],
        *ops: TrainBase[T],
    ) -> list[T]:
        if self.direction == Direction.TO_RIGHT:
            return self._right_partial_contractions(random_cores, *ops)
        if self.direction == Direction.TO_LEFT:
            return self._left_partial_contractions(random_cores, *ops)

    def _right_partial_contractions[T: ArrayLike](
        self,
        random_cores: Sequence[T],
        *ops: TrainBase[T],
    ) -> list[T]:
        sgen = get_symbol_generator(self._contr)
        idx_map = (
            self._inp.idx_map
            if self._inp is not None
            else {i: i for i in range(len(ops))}
        )

        P_sym = next(sgen)  # stack size symbol
        SR_sym = [next(sgen) for _ in range(2)]  # sketch rank symbol

        xp, device, dtype = self._inp.infos(*ops)

        # Right-to-left sweep: build suffix environments for reconstruction.
        Ws = []

        # Running suffix environment with the left bond left open.
        tmp = xp.ones((self.P, 1, 1), device=device, dtype=dtype)
        for k in range(len(self._contr) - 1, 0, -1):
            ctnr = self._contr[k].result

            # Fold site k into the suffix environment.
            tmp = self._contract_local_with_extras(
                self._contr[k],
                *ops,
                extras=(
                    (f"{P_sym}{SR_sym[1]}{ctnr.right}", tmp),
                    (f"{P_sym}{SR_sym[0]}{ctnr.middle}{SR_sym[1]}", random_cores[k]),
                ),
                result=f"{P_sym}{SR_sym[0]}{ctnr.left}",
                idx_map=idx_map,
            )
            Ws.append(tmp)

        Ws.reverse()
        return Ws

    def _left_partial_contractions[T: ArrayLike](
        self,
        random_cores: Sequence[T],
        *ops: TrainBase[T],
    ) -> list[T]:
        sgen = get_symbol_generator(self._contr)
        idx_map = (
            self._inp.idx_map
            if self._inp is not None
            else {i: i for i in range(len(ops))}
        )

        P_sym = next(sgen)  # stack size symbol
        SR_sym = [next(sgen) for _ in range(2)]  # sketch rank symbols

        xp, device, dtype = self._inp.infos(*ops)

        # Left-to-right sweep: build prefix environments for reconstruction.
        Vs = []

        tmp = xp.ones((self.P, 1, 1), device=device, dtype=dtype)
        for k in range(len(self._contr) - 1):
            ctnr = self._contr[k].result
            tmp = self._contract_local_with_extras(
                self._contr[k],
                *ops,
                extras=(
                    (f"{P_sym}{ctnr.left}{SR_sym[0]}", tmp),
                    (f"{P_sym}{SR_sym[0]}{ctnr.middle}{SR_sym[1]}", random_cores[k]),
                ),
                result=f"{P_sym}{ctnr.right}{SR_sym[1]}",
                idx_map=idx_map,
            )
            Vs.append(tmp)
        return Vs

    # ------------------------------------------------------------------------
    # Reconstruction from partials

    def _sketch_from_partials[T: ArrayLike](
        self,
        partials: Sequence[T],
        *ops: TrainBase[T],
        sample: int | None = None,
    ) -> TrainBase[T]:
        if self.direction == Direction.TO_RIGHT:
            return self._sketch_from_right_partials(partials, *ops, sample=sample)
        if self.direction == Direction.TO_LEFT:
            return self._sketch_from_left_partials(partials, *ops, sample=sample)
        raise ValueError("Direction must be either 'to_left' or 'to_right'.")

    def _sketch_from_right_partials[T: ArrayLike](
        self,
        Ws: Sequence[T],
        *ops: TrainBase[T],
        sample: int | None = None,
    ) -> TrainBase[T]:
        if self._contr.result_shape is None:
            raise RuntimeError("Result shape cannot be None here.")
        if self._inp is None:
            raise RuntimeError("Input cannot be None here.")

        xp, device, dtype = self._inp.infos(*ops)
        orig_ranks = [
            tuple(self._contr.result_shape.middle(k)) for k in range(len(self._contr))
        ]

        # sample=None means flatten all P samples into one sketch.
        sgen = get_symbol_generator(self._contr)
        P_sym = next(sgen) if sample is None else ""  # stack size symbol
        SR_sym = [next(sgen) for _ in range(2)]  # sketch rank symbols

        W = Ws if sample is None else [Wi[sample] for Wi in Ws]
        P = self.P if sample is None else 1
        tail = 2 if sample is None else 1

        new_cores = []
        idx_map = self._inp.idx_map
        tmp = xp.ones((1, 1), device=device, dtype=dtype)

        # Reconstruct left-to-right: sketch, flatten, QR, project.
        for k, lcontr in enumerate(self._contr):
            cntr = lcontr.result
            tmp = self._contract_local_with_extras(
                lcontr,
                *ops,
                extras=((f"{SR_sym[0]}{cntr.left}", tmp),),
                result=f"{SR_sym[0]}{cntr.middle}{cntr.right}",
                idx_map=idx_map,
            )

            if k == len(self._contr) - 1:
                break

            # Apply the random sketch and keep the normalization consistent.
            sketch_eq = (
                f"{SR_sym[0]}{cntr.middle}{cntr.right},"
                f"{P_sym}{SR_sym[1]}{cntr.right}->"
                f"{SR_sym[0]}{cntr.middle}{P_sym}{SR_sym[1]}"
            )
            sketch = self._apply_contract(sketch_eq, tmp, W[k]) / np.sqrt(P)

            # QR on the sketched unfolding gives the next left-orthonormal core.
            sketch = xp.reshape(
                sketch, (prod(sketch.shape[:-tail]), prod(sketch.shape[-tail:]))
            )
            Q, _ = xp.linalg.qr(sketch, mode="reduced")
            ncore = xp.reshape(Q, (tmp.shape[0], *orig_ranks[k], Q.shape[1]))
            new_cores.append(ncore)

            # Project onto the new core to get the interface tensor for the next site.
            eq = (
                f"{SR_sym[0]}{cntr.middle}{SR_sym[1]},"
                f"{SR_sym[0]}{cntr.middle}{cntr.right}->"
                f"{SR_sym[1]}{cntr.right}"
            )
            tmp = self._apply_contract(eq, xp.conj(ncore), tmp)

        new_cores.append(xp.reshape(tmp, (-1, *orig_ranks[-1], 1)))
        return TrainBase(self._contr.result_shape, new_cores, copy_data=False)

    def _sketch_from_left_partials[T: ArrayLike](
        self,
        Vs: Sequence[T],
        *ops: TrainBase[T],
        sample: int | None = None,
    ) -> TrainBase[T]:
        if self._contr.result_shape is None:
            raise RuntimeError("Result shape cannot be None here.")
        if self._inp is None:
            raise RuntimeError("Input cannot be None here.")

        xp, device, dtype = self._inp.infos(*ops)
        orig_ranks = [
            tuple(self._contr.result_shape.middle(k)) for k in range(len(self._contr))
        ]

        sgen = get_symbol_generator(self._contr)
        P_sym = next(sgen) if sample is None else ""  # stack size symbol
        SR_sym = [next(sgen) for _ in range(2)]  # sketch rank symbols

        V = Vs if sample is None else [Vi[sample] for Vi in Vs]
        P = self.P if sample is None else 1
        head = 2 if sample is None else 1

        new_cores = []
        idx_map = self._inp.idx_map
        tmp = xp.ones((1, 1), device=device, dtype=dtype)

        # Reconstruct right-to-left: sketch, flatten, QR, project.
        for k in range(len(self._contr) - 1, -1, -1):
            lcontr = self._contr[k]
            cntr = lcontr.result
            tmp = self._contract_local_with_extras(
                lcontr,
                *ops,
                extras=((f"{cntr.right}{SR_sym[0]}", tmp),),
                result=f"{cntr.left}{cntr.middle}{SR_sym[0]}",
                idx_map=idx_map,
            )

            if k == 0:
                break

            sketch_eq = (
                f"{P_sym}{cntr.left}{SR_sym[1]},"
                f"{cntr.left}{cntr.middle}{SR_sym[0]}->"
                f"{P_sym}{SR_sym[1]}{cntr.middle}{SR_sym[0]}"
            )
            sketch = self._apply_contract(sketch_eq, V[k - 1], tmp) / np.sqrt(P)
            sketch = xp.reshape(
                sketch, (prod(sketch.shape[:head]), prod(sketch.shape[head:]))
            )
            Q, _ = xp.linalg.qr(xp.permute_dims(sketch, (1, 0)), mode="reduced")
            ncore = xp.reshape(
                xp.permute_dims(Q, (1, 0)),
                (Q.shape[1], *orig_ranks[k], tmp.shape[-1]),
            )
            new_cores.append(ncore)

            eq = (
                f"{cntr.left}{cntr.middle}{SR_sym[0]},"
                f"{SR_sym[1]}{cntr.middle}{SR_sym[0]}->"
                f"{cntr.left}{SR_sym[1]}"
            )
            tmp = self._apply_contract(eq, tmp, xp.conj(ncore))

        new_cores.append(xp.reshape(tmp, (1, *orig_ranks[0], -1)))
        new_cores.reverse()
        return TrainBase(self._contr.result_shape, new_cores, copy_data=False)

    # ------------------------------------------------------------------------
    # Local contraction helpers

    def _contract_local_with_extras[T: ArrayLike](
        self,
        lcontr,
        *ops: TrainBase[T],
        extras: Sequence[tuple[str, T]] = (),
        result: str | None = None,
        idx_map: dict[int, int] | None = None,
    ) -> T:
        if idx_map is None:
            idx_map = {i: i for i in range(len(ops))}

        # data = [extra | local operands]
        data = [arr for _, arr in extras]
        data.extend(lcontr.get_data(*ops, idx_map=idx_map))

        # data = [extra eq | local eq]
        inputs = [eq for eq, _ in extras]
        inputs.extend(str(op) for op in lcontr.operands)

        # output = extra result or local result
        output = str(lcontr.result) if result is None else result

        return self._apply_contract(f"{','.join(inputs)}->{output}", *data)

    def _apply_contract[T: ArrayLike](self, eq: str, *arrays: T) -> T:
        # Rename einsum labels so equivalent contractions
        # share the same cached ArrayContractor expression.
        eq = self._normalize_einsum(eq)
        shapes = tuple(tuple(arr.shape) for arr in arrays)
        return self._contract_expr(eq, shapes, self.optimizer)(*arrays)

    # ------------------------------------------------------------------------
    # Static helpers

    @staticmethod
    def _stiefel_ranks(
        ranks: Sequence[int], mode_dims: Sequence[Sequence[int]]
    ) -> list[int]:
        stiefel_ranks = list(ranks)
        stiefel_ranks[-1] = 1
        for k in range(len(mode_dims) - 1, -1, -1):
            stiefel_ranks[k] = min(
                stiefel_ranks[k], prod(mode_dims[k]) * stiefel_ranks[k + 1]
            )
        return stiefel_ranks

    @staticmethod
    def _random_stiefel(rng: np.random.Generator, size: Sequence[int]) -> np.ndarray:
        """
        Stiefel manifolds, which replace iid gaussian entries with orthonormal-row blocks via QR.
        """
        if len(size) < 3:
            raise ValueError(
                "Stiefel size must include rows and at least one column axis."
            )

        nrows = size[1]
        ncols = prod(size[2:])
        if nrows > ncols:
            raise ValueError("Stiefel rows must not exceed columns.")

        # Generate Gaussian blocks, then orthonormalize their rows via QR.
        A = rng.normal(size=(size[0], nrows, ncols))
        Q, _ = np.linalg.qr(np.swapaxes(A, -1, -2))
        return np.swapaxes(Q, -1, -2).reshape(*size)

    @staticmethod
    def _normalize_einsum(eq: str) -> str:
        keep = {",", "-", ">", " "}
        unique = dict.fromkeys(ch for ch in eq if ch not in keep)
        sgen = symbol_generator()
        mapping = {ch: next(sgen) for ch in unique}
        return "".join(mapping.get(ch, ch) for ch in eq)

    @staticmethod
    @cache
    def _contract_expr(
        eq: str, shapes: tuple[tuple[int, ...], ...], optimizer: OptimizeKind
    ) -> ArrayContractor:
        return ArrayContractor(eq, *shapes, optimizer=optimizer)
