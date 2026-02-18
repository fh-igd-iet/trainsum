# Copyright© 2025-2026 Gesellschaft zur Förderung der angewandten Forschung e.V.
# acting on behalf of its Fraunhofer Institut für Graphische Datenverarbeitung.
# Licensed under the EUPL. See LICENSE.txt.

from __future__ import annotations
from typing import Callable, Literal
from itertools import product

from .backend import DType, Device, get_index_dtype, namespace_of_arrays, size, ArrayLike, device, shape
from .matrixleastsquares import MatrixLeastSquares
from .functionalcore import FunctionalCore
from .storagetensor import StorageTensor

Driver = Literal["gels"] | Literal["gelsd"]

class PiCore[T: ArrayLike]:
    _data: StorageTensor[T]

    _rows: tuple[StorageTensor[T], StorageTensor[T]]
    _cols: tuple[StorageTensor[T], StorageTensor[T]]

    _row_mask: StorageTensor[T]
    _col_mask: StorageTensor[T]

    _row_map: dict[tuple[int, ...], tuple[int, ...]]
    _col_map: dict[tuple[int, ...], tuple[int, ...]]

    @property
    def dtype(self) -> DType:
        return self._data.dtype

    @property
    def device(self) -> Device:
        return self._data.device

    def __init__(
            self,
            lcore: FunctionalCore[T],
            rcore: FunctionalCore[T],
            func: Callable[[T], T],
            dtype: DType) -> None:
        left, lmid, rmid, right = lcore.left, lcore.middle, rcore.middle, rcore.right

        xp = namespace_of_arrays(left.data[...])
        dev = device(left.data)
        data = xp.zeros((left.max_size, lmid.size, rmid.size, right.max_size),
                        device=dev, dtype=dtype)
        self._data = StorageTensor(data)

        self._data.add_to_dim(1, lmid.size)
        self._data.add_to_dim(2, rmid.size)

        self._row_map = {}
        self._col_map = {}

        int_type = get_index_dtype(xp)
        self._rows = (StorageTensor(xp.zeros(left.max_size*lmid.size, device=dev, dtype=int_type)),
                      StorageTensor(xp.zeros(left.max_size*lmid.size, device=dev, dtype=int_type)))
        self._cols = (StorageTensor(xp.zeros(right.max_size*rmid.size, device=dev, dtype=int_type)),
                      StorageTensor(xp.zeros(right.max_size*rmid.size, device=dev, dtype=int_type)))

        info = xp.__array_namespace_info__()
        btype = info.dtypes()["bool"]

        self._row_mask = StorageTensor(xp.zeros((left.max_size, lmid.size), device=dev, dtype=btype))
        self._row_mask.add_to_dim(1, lmid.size)
        self._col_mask = StorageTensor(xp.zeros((rmid.size, right.max_size), device=dev, dtype=btype))
        self._col_mask.add_to_dim(0, rmid.size)

        self.update_rows(lcore, rcore, func)
        self.update_cols(lcore, rcore, func)

        for i in range(lcore.right.size):
            self.set_col(lcore.right.data[:,i])
        for i in range(rcore.left.size):
            self.set_row(rcore.left.data[:,i])
    
    def update_rows(
            self,
            lcore: FunctionalCore,
            rcore: FunctionalCore,
            func: Callable[[T], T]) -> None:
        left, lmid, rmid, right = lcore.left, lcore.middle, rcore.middle, rcore.right
        added = left.size - self._data.shape[0]
        ndims = lmid.ndims
        if added == 0:
            return

        xp = namespace_of_arrays(self._data[...])
        int_type = get_index_dtype(xp)
        ax = xp.newaxis
        idxs = xp.zeros((ndims, added, lmid.size),
                        device=self.device, dtype=int_type)
        idxs += left.data[:,-added:,ax]
        idxs += lmid.data[:,ax,:]
        for i, j in product(range(added), range(lmid.size)):
            key = tuple(int(val) for val in idxs[:,i,j])
            self._row_map[key] = (i+self._data.shape[0], j)
        self._data.add_to_dim(0, added)
        self._row_mask.add_to_dim(0, added)
        if size(self._data[...]) == 0:
            return

        rsize = self._data.shape[3]
        idxs = idxs[:,:,:,ax,ax] + rmid.data[:,ax,ax,:,ax] + right.data[:,ax,ax,ax,:rsize]
        self._data[-added:,:,:,:] = func(idxs)

    def update_cols(
            self,
            lcore: FunctionalCore,
            rcore: FunctionalCore,
            func: Callable[[T], T]) -> None:
        left, lmid, rmid, right = lcore.left, lcore.middle, rcore.middle, rcore.right
        added = right.size - self._data.shape[3]
        ndims = lcore.middle.ndims
        if added == 0:
            return

        xp = namespace_of_arrays(self._data[...])
        int_type = get_index_dtype(xp)
        ax = xp.newaxis
        idxs = xp.zeros((ndims, rmid.size, added),
                        device=self.device, dtype=int_type)
        idxs += rmid.data[:,:,ax]
        idxs += right.data[:,ax,-added:]
        for i, j in product(range(rmid.size), range(added)):
            key = tuple(int(val) for val in idxs[:,i,j])
            self._col_map[key] = (i, j+self._data.shape[3])
        self._data.add_to_dim(3, added)
        self._col_mask.add_to_dim(1, added)
        if size(self._data[...]) == 0:
            return

        lsize = self._data.shape[0]
        idxs = left.data[:,:lsize,ax,ax,ax] + lmid.data[:,ax,:,ax,ax] + idxs[:,ax,ax,:,:]
        self._data[:,:,:,-added:] = func(idxs)

    def get_error(self, solver: MatrixLeastSquares) -> T:
        left = self._data[:,:,self._col_mask[...]]
        right = self.inv_right(solver)
        xp = namespace_of_arrays(left, right)
        approx = xp.tensordot(left, right, axes=([2],[0]))
        return xp.abs(approx - self._data[...])

    def set_row(self, vals: T) -> None:
        key = tuple(int(val) for val in vals)
        if key not in self._row_map:
            return
        left_idx, middle_idx = self._row_map[key]
        if self._row_mask[left_idx, middle_idx]:
            return
        self._rows[0].add_to_dim(0, 1)
        self._rows[0][-1] = left_idx
        self._rows[1].add_to_dim(0, 1)
        self._rows[1][-1] = middle_idx
        self._row_mask[left_idx, middle_idx] = True

    def set_col(self, vals: T) -> None:
        key = tuple(int(val) for val in vals)
        if key not in self._col_map:
            return
        middle_idx, right_idx = self._col_map[key]
        if self._col_mask[middle_idx, right_idx]:
            return
        self._cols[0].add_to_dim(0, 1)
        self._cols[0][-1] = middle_idx
        self._cols[1].add_to_dim(0, 1)
        self._cols[1][-1] = right_idx
        self._col_mask[middle_idx, right_idx] = True

    def inv_right(self, solver: MatrixLeastSquares) -> T:
        xp = namespace_of_arrays(self._data[...])
        cmask = xp.reshape(self._col_mask[...], (size(self._col_mask[...]),))
        clen = int(xp.sum(cmask))
        if clen == size(cmask):
            view = (clen, *self._data.shape[2:])
            return xp.reshape(xp.eye(clen, device=self.device, dtype=self.dtype), view)

        right = self.right()
        right = xp.reshape(right, (shape(right)[0], shape(right)[1]*shape(right)[2]))
        q = xp.linalg.qr(right.T)[0].T
        A = q[:,cmask]
        B = q[:,~cmask]
        lres = solver(A, B)
        
        res = xp.zeros((clen, self._data.shape[2]*self._data.shape[3]),
                       device=self.device, dtype=self.dtype)
        res[:,cmask] = xp.eye(clen, device=self.device, dtype=self.dtype)
        res[:,~cmask] = lres

        view = (clen, *self._data.shape[2:])
        return xp.reshape(res, view)

    def _solve(self, A: T, B: T) -> T:
        xp = namespace_of_arrays(A, B)
        lres = xp.linalg.lstsq(A, B) #type: ignore
        return lres[0]

    def left(self) -> T:
        return self._data[:,:,self._cols[0][...],self._cols[1][...]]

    def right(self) -> T:
        return self._data[self._rows[0][...],self._rows[1][...],:,:]

    def result_right(self, solver: MatrixLeastSquares) -> T:
        xp = namespace_of_arrays(self._data[...])
        int_type = get_index_dtype(xp)
        right = self.inv_right(solver)
        idxs = xp.zeros(self._cols[0].shape, device=self.device, dtype=int_type)
        idxs += self._cols[0][...] * shape(self._data[...])[3]
        idxs += self._cols[1][...]
        sorted_idxs = xp.sort(idxs)
        res_idxs = xp.searchsorted(sorted_idxs, idxs)
        return right[res_idxs,:,:]
