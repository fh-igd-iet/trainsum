# Copyright© 2025-2026 Gesellschaft zur Förderung der angewandten Forschung e.V.
# acting on behalf of its Fraunhofer Institut für Graphische Datenverarbeitung.
# Licensed under the EUPL. See LICENSE.txt.

from typing import Sequence, overload, Optional, Callable
from copy import deepcopy

from .backend  import ArrayLike, namespace_of_arrays, shape_size
from .matrixdecomposition import MatrixDecompositionResult, MatrixDecomposition

class TensorDecomposition[T: ArrayLike, ResType: MatrixDecompositionResult]:

    _mat_decomp: MatrixDecomposition[T, ResType]

    def __init__(self, decomposition: MatrixDecomposition[T, ResType]) -> None:
        self._mat_decomp = deepcopy(decomposition)

    def _decompose(self,
             decomp: Callable[[T], ResType],
             tn: T,
             left_dims: int | Sequence[int],
             right_dims: Optional[Sequence[int]] = None, /
             ) -> ResType:
        xp = namespace_of_arrays(tn)
        if isinstance(left_dims, int) and right_dims is None:
            rdims = tn.shape[left_dims:]
            ldims = tn.shape[:left_dims]
        elif not isinstance(left_dims, int) and right_dims is not None:
            tn = xp.permute_dims(tn, (*left_dims, *right_dims))
            rdims = tn.shape[len(left_dims):]
            ldims = tn.shape[:len(left_dims)]
        else:
            raise ValueError("Invalid input arguments")
        mat = xp.reshape(tn, (shape_size(ldims), shape_size(rdims)))
        res = decomp(mat)
        res.left = xp.reshape(res.left, (*ldims, res.left.shape[1]))
        res.right = xp.reshape(res.right, (res.right.shape[0], *rdims))
        return res

    @overload
    def left(self, tn: T, split: int, /) -> ResType: ...
    @overload
    def left(self, tn: T, left_dims: Sequence[int], right_dims: Sequence[int], /) -> ResType: ...
    # impl
    def left(self,
             tn: T,
             left_dims: int | Sequence[int],
             right_dims: Optional[Sequence[int]] = None, /
             ) -> ResType:
        return self._decompose(self._mat_decomp.left, tn, left_dims, right_dims)

    @overload
    def right(self, tn: T, split: int, /) -> ResType: ...
    @overload
    def right(self, tn: T, left_dims: Sequence[int], right_dims: Sequence[int], /) -> ResType: ...
    # impl
    def right(self,
              tn: T,
              left_dims: int | Sequence[int],
              right_dims: Optional[Sequence[int]] = None, /
              ) -> ResType:
        return self._decompose(self._mat_decomp.right, tn, left_dims, right_dims)

    def _shape(self,
               shape_func: Callable[[tuple[int, int]], tuple[tuple[int, int], tuple[int, int]]],
               shape: tuple[int, ...],
               left_dims: int | Sequence[int],
               right_dims: Optional[Sequence[int]] = None, /
               ) -> tuple[tuple[int, ...], tuple[int, ...]]:
        if isinstance(left_dims, int):
            rdims = shape[left_dims:]
            ldims = shape[:left_dims]
        elif not isinstance(left_dims, int) and right_dims is not None:
            ldims = tuple(shape[i] for i in left_dims)
            rdims = tuple(shape[i] for i in right_dims)
        else:
            raise ValueError("Invalid input arguments")
        left_shape, right_shape = shape_func((shape_size(ldims), shape_size(rdims)))
        return (*ldims, left_shape[-1]), (right_shape[0], *rdims)

    @overload
    def left_shape(self,
                   shape: tuple[int, ...],
                   split: int, /
                   ) -> tuple[tuple[int, ...], tuple[int, ...]]: ...
    @overload
    def left_shape(self,
                   shape: tuple[int, ...],
                   left_dims: Sequence[int],
                   right_dims: Sequence[int], /
                   ) -> tuple[tuple[int, ...], tuple[int, ...]]: ...
    # impl
    def left_shape(self,
             shape: tuple[int, ...],
             left_dims: int | Sequence[int],
             right_dims: Optional[Sequence[int]] = None, /
             ) -> tuple[tuple[int, ...], tuple[int, ...]]:
        return self._shape(self._mat_decomp.left_shape, shape, left_dims, right_dims)

    @overload
    def right_shape(self,
                    shape: tuple[int, ...],
                    split: int, /
                    ) -> tuple[tuple[int, ...], tuple[int, ...]]: ...
    @overload
    def right_shape(self,
                    shape: tuple[int, ...],
                    left_dims: Sequence[int],
                    right_dims: Sequence[int], /
                    ) -> tuple[tuple[int, ...], tuple[int, ...]]: ...
    # impl
    def right_shape(self,
             shape: tuple[int, ...],
             left_dims: int | Sequence[int],
             right_dims: Optional[Sequence[int]] = None, /
             ) -> tuple[tuple[int, ...], tuple[int, ...]]:
        return self._shape(self._mat_decomp.right_shape, shape, left_dims, right_dims)
