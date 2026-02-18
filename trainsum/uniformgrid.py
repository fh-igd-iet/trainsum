# Copyright© 2025-2026 Gesellschaft zur Förderung der angewandten Forschung e.V.
# acting on behalf of its Fraunhofer Institut für Graphische Datenverarbeitung.
# Licensed under the EUPL. See LICENSE.txt.

from typing import Any, Sequence, overload
from dataclasses import dataclass

from .backend import ArrayLike, namespace_of_arrays, get_index_dtype
from .domain import Domain
from .dimension import Dimension

@dataclass(frozen=True, init=False)
class UniformGrid[T: ArrayLike]:

    """
    Uniform spaced N-dimensional grid. The grid associates a dimension and a domain to each axis.
    The grid can be used to convert between coordinate and index representations of points in the grid.
    """

    #-------------------------------------------------------------------------
    #members & properties

    ndims: int
    dims: Sequence[Dimension]
    domains: Sequence[Domain]
    spacings: Sequence[float]

    #-------------------------------------------------------------------------
    #constructor

    
    @overload
    def __init__(self, dim: Dimension, domain: Domain, /) -> None: ...
    @overload
    def __init__(self, dims: Sequence[Dimension], domains: Sequence[Domain], /) -> None: ...
    # implementation
    def __init__(
            self,
            dims: Dimension | Sequence[Dimension],
            domains: Domain | Sequence[Domain], /
            ) -> None:
        if isinstance(dims, Dimension) and isinstance(domains, Domain):
            dims = [dims]
            domains = [domains]
        if len(dims) != len(domains): # type: ignore
            raise ValueError("Number of dimensions must match number of domains.")
        object.__setattr__(self, "ndims", len(dims))
        object.__setattr__(self, "dims", dims)
        object.__setattr__(self, "domains", domains)
        spacings = [self.domains[idx].diff / (self.dims[idx].size()-1) for idx in range(len(dims))]
        object.__setattr__(self, "spacings", spacings)

    #-------------------------------------------------------------------------
    #methods

    def to_coords(self, idxs: T) -> T:
        """Convert from index representation to coordinate representation."""
        xp = namespace_of_arrays(idxs)
        int_type = get_index_dtype(xp)
        self._check_input_tensor(idxs, int_type)
        coords = xp.zeros(idxs.shape, device=idxs.device)
        for i, (dim, domain) in enumerate(zip(self.dims, self.domains)):
            vals = domain.diff / (dim.size()-1) * idxs[i,:] + domain.lower
            #coords = set_data(coords, (i, ...), vals)
            coords[i,...] = vals
                              
        return coords

    def to_idxs(self, coords: T) -> T:
        """Convert from coordinate representation to index representation."""
        xp = namespace_of_arrays(coords)
        int_type = get_index_dtype(xp)
        self._check_input_tensor(coords, coords.dtype)
        idxs = xp.zeros(coords.shape,
                        dtype=int_type,
                        device=coords.device)
        for i, (dim, domain) in enumerate(zip(self.dims, self.domains)):
            vals = (coords[i,...] - domain.lower) / domain.diff * (dim.size()-1)
            vals = xp.asarray(xp.round(vals),
                              dtype=int_type,
                              device=coords.device)
            #idxs = set_data(idxs, (i, ...), vals)
            idxs[i,...] = vals
        return idxs

    def _check_input_tensor(self, input_tensor: ArrayLike, num_type: Any):
        if input_tensor.dtype != num_type:
            raise ValueError(f"Expected input tensor of type {num_type}, "\
                             f"got {input_tensor.dtype}")
        if input_tensor.shape[0] != self.ndims:
            raise ValueError(f"Expected input shape to be ({self.ndims}, ...), "\
                             f"got {input_tensor.shape}")
