# Copyright© 2025-2026 Gesellschaft zur Förderung der angewandten Forschung e.V.
# acting on behalf of its Fraunhofer Institut für Graphische Datenverarbeitung.
# Licensed under the EUPL. See LICENSE.txt.

from typing import Hashable, Literal, Any, Self, overload
from enum import Enum
from copy import deepcopy
import threading

from .backend import ArrayNamespace
from .contractor import OptimizeKind, DEFAULT_OPTIMIZER
from .matrixdecomposition import MatrixDecomposition
from .sweepingstrategy import SweepingStrategy
from .matrixleastsquares import MatrixLeastSquares

class OptionType(Enum):
    EINSUM = 0
    CROSS = 1
    EVALUATE = 2

class Options:

    key: Hashable

    def __init__(self, namespace: ArrayNamespace, category: OptionType):
        self.key = (namespace, category, threading.get_ident())

    def __enter__(self) -> Self:
        global _opts
        if self.key in _opts:
            self._tmp = _opts[self.key]
        else:
            self._tmp = None
        _opts[self.key] = self
        return self

    def __exit__(self, *_) -> None:
        global _opts
        if self._tmp is not None:
            _opts[self.key] = self._tmp
        else:
            del _opts[self.key]

class ExactOptions(Options):
    """
    Context manager for exact einsum options.
    """

    #: Optimizer for the local einsum contraction paths.
    optimizer: OptimizeKind

    def __init__(
            self, *,
            namespace: ArrayNamespace,
            optimizer: OptimizeKind = DEFAULT_OPTIMIZER):
        self.optimizer = deepcopy(optimizer)
        super().__init__(namespace, OptionType.EINSUM)

class EvaluationOptions(Options):
    """
    Context manager for evalaute einsum options.
    """

    #: Optimizer for the local einsum contraction paths.
    optimizer: OptimizeKind
    #: Size of the chunks that are evaluated together.
    chunk_size: int

    def __init__(
            self, *,
            namespace: ArrayNamespace,
            chunk_size: int = 1024,
            optimizer: OptimizeKind = DEFAULT_OPTIMIZER):
        self.optimizer = deepcopy(optimizer)
        self.chunk_size = chunk_size
        super().__init__(namespace, OptionType.EVALUATE)


class DecompositionOptions[T: MatrixDecomposition](Options):
    """
    Context manager for decomposition based einsum options. The decomposition and strategy are used
    to determine the ranks and the approximation of the einsum result.
    """
    #: Decomposition method for determining the approximation of the einsum result.
    decomposition: T
    #: Sweeping strategy for determining the path of the algorithm.
    strategy: SweepingStrategy
    #: Optimizer for the local einsum contraction paths.
    optimizer: OptimizeKind

    def __init__(
            self, *,
            namespace: ArrayNamespace,
            decomposition: T,
            strategy: SweepingStrategy, 
            optimizer: OptimizeKind = DEFAULT_OPTIMIZER):
        self.decomposition = deepcopy(decomposition)
        self.strategy = deepcopy(strategy)
        self.optimizer = deepcopy(optimizer)
        super().__init__(namespace, OptionType.EINSUM)

class VariationalOptions[T: MatrixDecomposition](DecompositionOptions):
    """
    Context manager for variational based einsum options.
    """
    #: Decomposition method for determining the approximation of the einsum result.
    decomposition: T
    #: Sweeping strategy for determining the path of the algorithm.
    strategy: SweepingStrategy
    #: Optimizer for the local einsum contraction paths.
    optimizer: OptimizeKind
    pass

class CrossOptions[T: MatrixLeastSquares](Options):
    """
    Context manager for cross approximation options.
    """
    #: Sweeping strategy for determining the path of the algorithm.
    strategy: SweepingStrategy
    #: Convergence criterion.
    eps: float
    #: Least squares solver for solving the local problems in the cross interpolation.
    solver: T
    def __init__(
            self, *,
            namespace: ArrayNamespace,
            strategy: SweepingStrategy,
            eps: float,
            solver: T):
        self.solver = deepcopy(solver)
        self.eps = deepcopy(eps)
        self.strategy = deepcopy(strategy)
        super().__init__(namespace, OptionType.CROSS)

_opts: dict[Any, Options] = {}

@overload
def get_options(namespace: ArrayNamespace, otype: Literal[OptionType.EINSUM]) -> ExactOptions | DecompositionOptions | VariationalOptions: ...
@overload
def get_options(namespace: ArrayNamespace, otype: Literal[OptionType.CROSS]) -> CrossOptions: ...
@overload
def get_options(namespace: ArrayNamespace, otype: Literal[OptionType.EVALUATE]) -> EvaluationOptions: ...
# implementation
def get_options(namespace: ArrayNamespace, otype: OptionType) -> Options:
    global _opts
    key = (namespace, otype, threading.get_ident())
    if key in _opts:
        return _opts[key]
    else:
        raise KeyError("No options set for the current thread.")

def set_options(opts: CrossOptions | EvaluationOptions | ExactOptions | DecompositionOptions | VariationalOptions) -> None:
    global _opts
    _opts[opts.key] = opts

