# Copyright© 2025-2026 Gesellschaft zur Förderung der angewandten Forschung e.V.
# acting on behalf of its Fraunhofer Institut für Graphische Datenverarbeitung.
# Licensed under the EUPL. See LICENSE.txt.

from typing import Literal, Sequence, overload, Any, Type, Callable, Optional
from types import NoneType
from dataclasses import dataclass
import h5py

from .backend import ArrayLike, get_index_dtype, get_namespace
from .trainshape import trainshape as _trainshape, TrainShape
from .trainbase import TrainBase as TrainBase
from .tensortrain import TensorTrain, tensortrain as _tensortrain
from .digit import Digits
from .dimension import Dimension
from .domain import Domain
from .uniformgrid import UniformGrid
from .sweepingstrategy import SweepingStrategy
from .contractor import OptimizeKind, DEFAULT_OPTIMIZER

from .matrixdecomposition import MatrixDecomposition
from .matrixleastsquares import MatrixLeastSquares
from .matrixeigenvaluedecomposition import MatrixEigenvalueDecomposition
from .svdecomposition import SVDecomposition
from .lstsqsolver import LstsqSolver
from .eighsolver import EigHSolver

from .backend import ArrayNamespace

from .full import full as _full
from .exponential import exp as _exp
from .polyval import polyval as _polyval
from .trigonometric import sin as _sin, cos as _cos
from .toeplitz import toeplitz as _toeplitz, shift as _shift
from .discrete_fourier_transform import (
    qft as _qft,
    iqft as _iqft,
    qftshift as _qftshift,
    iqftshift as _iqftshift,
    qftfreq as _qftfreq
)
from .wavelet import wavelet as _wavelet

from .io import write as _write
from .io import read as _read

from .linearmap import LinearMap as _LinearMap
from .lanczos import Lanczos
from .eigsolver import EigSolver as _EigSolver
from .localrange import LocalRange
from .localeigsolver import LocalEigSolver, LocalEigSolverResult

from .gmres import GMRES
from .linsolver import LinSolver as _LinSolver
from .amensolver import AMEnSolver as _AMEnSolver
from .locallinsolver import LocalLinSolver, LocalLinSolverResult

from .add import add
from .einsum import einsum as _einsum
from .einsum import EinsumExpression as _EinsumExpression
from .evaluate import EvaluateExpression as _EvaluationExpression, evaluate as _evaluate
from .min_max import MinMaxResult, min_max as _min_max

from .options import EvaluationOptions, ExactOptions, DecompositionOptions, OptionType, VariationalOptions, CrossOptions, set_options, get_options
from .qrdecomposition import QRDecomposition

class LinearMap[NDArray: ArrayLike]:
    """
    Linear map of qunatics tensor trains defined by an einsum expression.
    It serves as input to the EigSolver and LinSolver classes.
    """

    _map: _LinearMap

    def __init__(
            self,
            eq: str,
            *ops: TrainShape | TensorTrain[NDArray],
            optimizer: OptimizeKind = DEFAULT_OPTIMIZER) -> None:
        ops_ = [op._base if isinstance(op, TensorTrain) else op for op in ops]
        self._map = _LinearMap(eq, *ops_, optimizer=optimizer)

@dataclass(init=False)
class EigSolver[NDArray: ArrayLike, T: LocalEigSolverResult]:
    """
    Variational eigenvalue solver for quantics tensor trains. The eigenvalue equation is\
    defined by the provided linear maps. The DMRG algotihm is using the canonical format
    of tensor trains to map the global eigenvalue problem to a sequence of local problems,
    which are solved by the provided local solver. If the sweeping strategy has ncores>=2
    the decomposition is used to recover the single cores, thereby defining the rank.
    """

    @property
    def solver(self) -> LocalEigSolver[T]:
        """Iterative eigenvalue solver for the local problems."""
        return self._solver.solver
    @solver.setter
    def solver(self, value: LocalEigSolver[T]) -> None:
        self._solver.solver = value

    @property
    def decomposition(self) -> MatrixDecomposition:
        """Matrix decomposition for recovering single cores in multi-core sweeping strategies."""
        return self._solver.decomposition
    @decomposition.setter
    def decomposition(self, value: MatrixDecomposition) -> None:
        self._solver.decomposition = value

    @property
    def strategy(self) -> SweepingStrategy:
        """Sweeping strategy defining the order of the local problems and the number of cores to be optimized simultaneously."""
        return self._solver.strategy
    @strategy.setter
    def strategy(self, value: SweepingStrategy) -> None:
        self._solver.strategy = value

    @property
    def eps(self) -> float:
        """Convergence criterion."""
        return self._solver.eps
    @eps.setter
    def eps(self, value: float) -> None:
        self._solver.eps = value

    _solver: _EigSolver

    def __init__(
            self,
            *maps: LinearMap[NDArray],
            eps: float = 1e-10,
            solver: LocalEigSolver[T],
            decomposition: MatrixDecomposition,
            strategy: SweepingStrategy,
            optimizer: OptimizeKind,
            ) -> None:
        self._solver = _EigSolver(
                solver, *[m._map for m in maps],
                eps=eps,
                decomposition=decomposition,
                strategy=strategy,
                optimizer=optimizer)

    def __call__(
            self,
            guess: TensorTrain[NDArray],
            states: Sequence[TensorTrain[NDArray]] = [], 
            callback: Optional[Callable[[LocalRange, T], bool]] = None,
            ) -> TensorTrain[NDArray]:
        """
        Start with the sweeping DMRG algorithm to solve eigenvalues with an initial guess.
        The solution will be orthogonal to the provided states. The callback is called after
        each local problem is solved and can be used to monitor the convergence of the algorithm.
        Returning True from the callback will stop the algorithm.
        """
        base_states = [state._base for state in states]
        res = self._solver(guess._base, base_states, expr=True, callback=callback)
        return TensorTrain(res, copy_data=False)

@dataclass(init=False)
class LinSolver[NDArray: ArrayLike, T: LocalLinSolverResult]:
    """
    Variationally solve linear equation systems for quantics tensor trains. The equation is\
    defined by the provided linear maps and the right hand side. Using the canonical format, the
    global linear problem is mapped to a sequence of local problems, which are solved by the provided
    local solver. If the sweeping strategy has ncores>=2 the decomposition is used to recover the
    single cores, thereby defining the rank. The method can be specified to use the DMRG algorithm
    or the AMEn algorithm.
    """

    @property
    def solver(self) -> LocalLinSolver[T]:
        """Iterative linear solver for the local problems."""
        return self._solver.solver
    @solver.setter
    def solver(self, value: LocalLinSolver[T]) -> None:
        self._solver.solver = value

    @property
    def decomposition(self) -> MatrixDecomposition:
        """Matrix decomposition for recovering single cores in multi-core sweeping strategies."""
        return self._solver.decomposition
    @decomposition.setter
    def decomposition(self, value: MatrixDecomposition) -> None:
        self._solver.decomposition = value

    @property
    def strategy(self) -> SweepingStrategy:
        """Sweeping strategy defining the order of the local problems and the number of cores to be optimized simultaneously."""
        return self._solver.strategy
    @strategy.setter
    def strategy(self, value: SweepingStrategy) -> None:
        self._solver.strategy = value

    @property
    def eps(self) -> float:
        """Convergence criterion."""
        return self._solver.eps
    @eps.setter
    def eps(self, value: float) -> None:
        self._solver.eps = value

    _solver: Any

    def __init__(
            self,
            rhs: TensorTrain[NDArray],
            *maps: LinearMap[NDArray],
            method: Literal["dmrg", "amen"],
            solver: LocalLinSolver,
            decomposition,
            strategy: SweepingStrategy,
            optimizer: OptimizeKind,
            ) -> None:
        if method == "dmrg":
            self._solver = _LinSolver(
                    solver, rhs._base, *[m._map for m in maps],
                    decomposition=decomposition,
                    strategy=strategy,
                    optimizer=optimizer)
        else:
            self._solver = _AMEnSolver(
                    solver, rhs._base, *[m._map for m in maps],
                    decomposition=decomposition, 
                    strategy=strategy, optimizer=optimizer)

    def __call__(
            self,
            guess: TensorTrain[NDArray],
            callback: Optional[Callable[[LocalRange, T], bool]] = None,
            ) -> TensorTrain[NDArray]:
        """
        Start with the sweeping DMRG or AMEn algorithm to solve the linear equation system with an
        initial guess. The callback is called after each local problem is solved and can be used to
        monitor the convergence of the algorithm. Returning True from the callback will stop the algorithm.
        """
        res = self._solver(guess._base, callback=callback)
        return TensorTrain(res, copy_data=False)

class EinsumExpression:
    """Utility class for typing."""

    _expr: Callable

    def __init__(self, expr: Callable) -> None:
        self._expr = expr

    def __call__[T: TensorTrain](self, *ops: T) -> float | T:
        """Evaluate the einsum expression with the provided tensor trains as operands."""
        base = self._expr(*[op._base for op in ops])
        if isinstance(base, float):
            return base
        elif isinstance(base, TrainBase):
            return type(ops[0])(base, copy_data=False)
        raise RuntimeError("Unexpected return type from einsum expression.")

class EvaluateExpression:
    """Utility class for typing."""

    _expr: Callable

    def __init__(self, expr: Callable) -> None:
        self._expr = expr

    def __call__[T: Any](self, idxs: T, *ops: TensorTrain[T]) -> T:
        """Evaluate the einsum expression with the provided tensor trains as operands."""
        base = self._expr(idxs, *[op._base for op in ops])
        return base

#-------------------------------------------------------------------------------------------------
# Construction wrapper
@dataclass(frozen=True)
class TrainSum[NDArray: Any]:

    #: Array namespace for the underlying array library.
    namespace: ArrayNamespace[NDArray]

    #: Internally used index type.
    index_type: Any

    def __init__(self, namespace: Any) -> None:
        object.__setattr__(self, "namespace", get_namespace(namespace))
        object.__setattr__(self, "index_type", get_index_dtype(self.namespace))

        set_options(self.decomposition(max_rank=25, cutoff=1e-10, ncores=2))
        set_options(self.cross(max_rank=50, eps=1e-10, solver=LstsqSolver()))
        set_options(self.evaluation())

    #-------------------------------------------------------------------------------------------------
    # base wrapper

    def domain(self, lower: float, upper: float) -> Domain:
        """
        Domain, defining a one dimensional interval.
        """
        return Domain(lower, upper)

    @overload
    def dimension(self, size: int, /) -> Dimension[NDArray]: ...
    @overload
    def dimension(self, bases: Sequence[int], /) -> Dimension[NDArray]: ...
    # implementation
    def dimension(self, size: int | Sequence[int], /) -> Dimension[NDArray]:
        """
        Quantized dimension, defined by a sequence of digits. If an integer is provided, it will be 
        factorized into its prime factors to create the digits. Otherwise, the provided sequence of
        integers will be used as the bases of the digits.
        """
        return Dimension(size)

    @overload
    def uniform_grid(self, dim: Dimension, domain: Domain, /) -> UniformGrid[NDArray]: ...
    @overload
    def uniform_grid(self, dims: Sequence[Dimension], domains: Sequence[Domain], /) -> UniformGrid[NDArray]: ...
    # implementation
    def uniform_grid(self, dims: Any, domains: Any, /) -> UniformGrid[NDArray]:
        """
        Uniformly N-dimensional spaced grid.
        """
        return UniformGrid(dims, domains)

    @overload
    def trainshape(self, *dims: Dimension, digits: Sequence[Digits]) -> TrainShape: ...
    @overload
    def trainshape(self, *dims: int | Dimension, mode: Literal["full", "block", "interleaved"] = "block") -> TrainShape: ...
    # implementation
    def trainshape(
            self,
            *dims: int | Dimension,
            mode: Literal["full", "block", "interleaved"] = "block",
            digits: Optional[Sequence[Digits]] = None
            ) -> TrainShape:
        """
        Base class describing the shape of a quantics tensor train.
        """
        return _trainshape(*dims, mode=mode, digits=digits) # type: ignore

    def svdecomposition(self, max_rank: int, cutoff: float = 1e-10) -> SVDecomposition[NDArray]:
        """
        Singular value decomposition for matrices.
        """
        return SVDecomposition(max_rank=max_rank, cutoff=cutoff)

    def qrdecomposition(self) -> QRDecomposition[NDArray]:
        """
        QR decomposition for matrices.
        """
        return QRDecomposition()

    def sweeping_strategy(self, ncores: int, nsweeps: int) -> SweepingStrategy:
        """
        Strategy for sweeping through a tensor train.
        """
        return SweepingStrategy(ncores=ncores, nsweeps=nsweeps)

    #-------------------------------------------------------------------------------------------------
    # construction wrapper

    def full(self, shape: TrainShape, value: float) -> TensorTrain[NDArray]:
        """
        :math:`f(x)=v`.\n
        Construct a quantics tensor train where all entries are set to the given value.
        The resulting TensorTrain will have rank one.
        """
        return TensorTrain(_full(self.namespace, shape, value), copy_data=False)

    def exp(self, grid: UniformGrid, factor: float, offset: float) -> TensorTrain[NDArray]:
        """
        :math:`f(x)=e^{a(x-x_0)}`.\n
        Construct a quantics tensor train representing an exponential function.
        The provided grid has to be one-dimensional. The resulting TensorTrain will have rank one.
        """
        return TensorTrain(_exp(self.namespace, grid, factor, offset), copy_data=False)

    def polyval(self, grid: UniformGrid, coeffs: Sequence[float], offset: float) -> TensorTrain[NDArray]:
        """
        :math:`f(x)=\\sum_i{v_i(x-x_0)^i}`.\n
        Construct a quantics tensor train representing a polynomial on the given grid.
        The provided grid has to be one-dimensional.The resulting TensorTrain will have the rank len(coeffs)+1.
        """
        return TensorTrain(_polyval(self.namespace, grid, coeffs, offset), copy_data=False)

    def sin(self, grid: UniformGrid, factor: float, offset: float) -> TensorTrain[NDArray]:
        """
        :math:`f(x)=\\sin{\\left(a(x-x_0)\\right)}`.\n
        Construct a quantics tensor train representing a sine on the given grid.
        The provided grid has to be one-dimensional.The resulting TensorTrain will have the rank 2.
        """
        return TensorTrain(_sin(self.namespace, grid, factor, offset), copy_data=False)

    def cos(self, grid: UniformGrid, factor: float, offset: float) -> TensorTrain[NDArray]:
        """
        :math:`f(x)=\\cos{\\left(a(x-x_0)\\right)}`.\n
        Construct a quantics tensor train representing a cosine on the given grid.
        The provided grid has to be one-dimensional.The resulting TensorTrain will have the rank 2.
        """
        return TensorTrain(_cos(self.namespace, grid, factor, offset), copy_data=False)

    def shift(self, dim: Dimension, shift: int, circular: bool = False) -> TensorTrain[NDArray]:
        """
        Generic shift matrix. Positive shifts correspond to shifts to the right, negative
        shifts correspond to shifts to the left. If circular is set to True the shift matrix will be
        circular. The resulting tensor train is rank 2.
        """
        return TensorTrain(_shift(self.namespace, dim, shift, circular), copy_data=False)

    def toeplitz(self, dim: Dimension, mode: Literal["full", "lower", "upper", "circular"]) -> TensorTrain[NDArray]:
        """
        Three dimensional Toeplitz tensor. The tensor can be used to create toeplitz matrices
        by contracting the first dimension with a vector. With mode='full' the resulting matrix is a full
        toeplitz matrix, where every diagonal is set one value of the vector. 'lower' and 'upper' correspond
        to lower and upper triangular toeplitz matrices. mode='circular' results in the sum of the lower and
        upper triangular toeplitz matrices. The resulting tensor train is rank 2.
        """
        return TensorTrain(_toeplitz(self.namespace, dim, mode), copy_data=False)

    def wavelet(self, input_size: int, coeffs: Sequence[float]) -> TensorTrain[NDArray]:
        """
        TODO(mklein): write doc
        """
        return TensorTrain(
            _wavelet(self.namespace, input_size, coeffs), copy_data=False
        )

    @overload
    def tensortrain(self, shape: TrainShape, data: Callable[[NDArray], NDArray], start_idxs: Optional[NDArray] = None, /) -> TensorTrain[NDArray]: ...
    @overload
    def tensortrain(self, shape: TrainShape, data: Sequence[NDArray], /) -> TensorTrain[NDArray]: ...
    @overload
    def tensortrain(self, shape: TrainShape, data: NDArray, /) -> TensorTrain[NDArray]: ...
    # implementation
    def tensortrain(
            self,
            shape: TrainShape,
            data: NDArray | Sequence[NDArray] | Callable[[NDArray], NDArray],
            start_idxs: Optional[NDArray] = None,
            /) -> TensorTrain[NDArray]:
        """
        Construct generic quantics tensor trains from some provided data. If data is a function the tensor
        train will be constructed by a cross interpolation (cross context manager). If it is a tensor it will
        be decomposed (einsum context manager). A sequence of tensors are interpreted as the cores of the
        tensor train.
        """
        return _tensortrain(shape, data, start_idxs, self.namespace)# type: ignore

    #-------------------------------------------------------------------------------------------------
    # fourier transform wrapper

    def qft(self, 
            dim: Dimension,
            decomp: MatrixDecomposition = SVDecomposition(max_rank=16)
            ) -> TensorTrain[NDArray]:
        """
        Quantum fourier transformation matrix as a quantics tensor train.
        The qft is constructed approximately using a decomposition algorithm for
        a specified dimension. The decomposition can be specified to control the
        accuracy of the resulting transformation.
        """
        return TensorTrain(_qft(self.namespace, dim, decomp), copy_data=False)

    def iqft(
            self,
            dim: Dimension,
            decomp: MatrixDecomposition = SVDecomposition(max_rank=16)
            ) -> TensorTrain[NDArray]:
        """
        Inverse quantum fourier transformation matrix as a quantics tensor train.
        The construction is the same as for qft, but the result is complex conjugated.
        As for the qft, the decomposition can be specified to control the accuracy of
        the resulting transformation.
        """
        return TensorTrain(_iqft(self.namespace, dim, decomp), copy_data=False)

    def qftshift(self, train: TensorTrain[NDArray], axis: int = 0) -> TensorTrain[NDArray]:
        """
        Shift the zero frequencies to the middle of the spectrum.
        Only works for dimensions with a binary leading bit.
        """
        return TensorTrain(_qftshift(train._base, axis), copy_data=False)

    def iqftshift(self, train: TensorTrain[NDArray], axis: int = 0) -> TensorTrain[NDArray]:
        """
        Inverse of the qftshift.
        Only works for dimensions with a binary leading bit.
        """
        return TensorTrain(_iqftshift(train._base, axis), copy_data=False)

    def qftfreq(self, dim: Dimension, d: float = 1.0) -> TensorTrain[NDArray]:
        """
        Sample frequencies for the quantum fourier transform.
        """
        return TensorTrain(_qftfreq(self.namespace, dim, d), copy_data=False)

    #-------------------------------------------------------------------------------------------------
    # io wrapper

    @overload
    def write(self, group: h5py.Group, obj: UniformGrid) -> None: ...
    @overload
    def write(self, group: h5py.Group, obj: TensorTrain[NDArray]) -> None: ...
    # implementation
    def write(self, group: h5py.Group, obj: Any) -> None:
        """
        Write a quantics tensor train or a uniform grid to a hdf5 group.
        """
        if isinstance(obj, TensorTrain):
            _write(group, obj._base)
        else:
            _write(group, obj)

    @overload
    def read(self, group: h5py.Group, cls: Type[UniformGrid]) -> UniformGrid: ...
    @overload
    def read(self, group: h5py.Group, cls: Type[TensorTrain[NDArray]]) -> TensorTrain[NDArray]: ...
    # implementation
    def read(self, group: h5py.Group, cls: Any) -> Any:
        """
        Read a quantics tensor train or a uniform grid to a hdf5 group.
        """
        if cls == TensorTrain:
            base = _read(group, TrainBase, self.namespace)
            return TensorTrain(base, copy_data=False)
        return _read(group, cls)

    #-------------------------------------------------------------------------------------------------
    # tensorized solver wrapper

    def linear_map(
            self,
            eq: str,
            *ops: TrainShape | TensorTrain[NDArray],
            optimizer: OptimizeKind = DEFAULT_OPTIMIZER,
            ) -> LinearMap[NDArray]:
        """Linear map for variational solvers."""
        return LinearMap(eq, *ops, optimizer=optimizer)

    def lanczos(
            self, *,
            nsteps: int = 1,
            subspace: int = 3,
            eps: float = 1e-8,
            solver: MatrixEigenvalueDecomposition = EigHSolver()
            ) -> Lanczos:
        """
        Lanczos iterative eigenvalue solver.
        """
        return Lanczos(nsteps=nsteps, subspace=subspace, eps=eps, solver=solver)

    def gmres(
            self, *,
            nsteps: int = 10,
            subspace: int = 10,
            eps: float = 1e-8,
            solver: MatrixLeastSquares = LstsqSolver(),
            ) -> GMRES:
        """
        GMRES iterative linear solver.
        """
        return GMRES(nsteps=nsteps, subspace=subspace, eps=eps, solver=solver)

    def eigsolver[T: LocalEigSolverResult](
            self,
            *maps: LinearMap[NDArray],
            eps: float = 1e-10,
            solver: LocalEigSolver[T] = Lanczos(),
            decomposition= SVDecomposition(),
            strategy: SweepingStrategy = SweepingStrategy(ncores=2, nsweeps=10),
            ) -> EigSolver[NDArray, LocalEigSolverResult]:
        """
        Variational eigenvalue solver for quantics tensor trains.
        """
        return EigSolver(
                *maps,
                eps=eps,
                solver=solver,
                decomposition=decomposition,
                strategy=strategy,
                optimizer=DEFAULT_OPTIMIZER)

    def linsolver[T: LocalLinSolverResult](
            self,
            rhs: TensorTrain[NDArray],
            *maps: LinearMap[NDArray],
            method: Literal["dmrg", "amen"] = "dmrg",
            solver: LocalLinSolver[T] = GMRES(),
            decomposition= SVDecomposition(),
            strategy: SweepingStrategy = SweepingStrategy(ncores=2, nsweeps=10),
            optimizer: OptimizeKind = DEFAULT_OPTIMIZER,
            ) -> LinSolver[NDArray, T]:
        """
        Variational linear solver for quantics tensor trains.
        """
        return LinSolver(rhs, *maps, method=method, solver=solver,
                         decomposition=decomposition, strategy=strategy, optimizer=optimizer)


    #-------------------------------------------------------------------------------------------------
    # operation wrapper

    def einsum(
            self,
            eq: str,
            *ops: TensorTrain[NDArray],
            result_shape: NoneType | TrainShape = None
            ) -> float | TensorTrain[NDArray]:
        """
        Perform an einsum operation. The resulting tensor train will always
        be as exact as possible unless a result shape is provided. In that case, the resulting tensor
        train will be decomposed to fit the provided shape. Affected by einsum context manager.
        """
        bases = [op._base for op in ops]
        res = _einsum(eq, *bases, result_shape=result_shape)
        if not isinstance(res, TrainBase):
            return res
        return TensorTrain(res, copy_data=False)
    
    def einsum_expression(
            self,
            eq: str,
            *ops: TrainShape,
            result_shape: NoneType | TrainShape = None
            ) -> EinsumExpression:
        """
        Create an einsum expression to be used with tensor trains. The expression is optimized
        with the ranks of the provided operands. Affected by einsum context manager.
        """
        expr = _EinsumExpression(self.namespace, eq, *ops, result_shape=result_shape)
        return EinsumExpression(expr)

    def evaluate(
            self,
            eq: str,
            idxs: NDArray,
            *ops: TensorTrain[NDArray],
            ) -> NDArray:
        """
        Perform an evaluation at idxs of an einsum expression. Affected by einsum context manager.
        """
        return _evaluate(eq, idxs, *[op._base for op in ops])

    def evaluate_expression(
            self,
            eq: str,
            *ops: TrainShape,
            ) -> EvaluateExpression:
        """
        Create an evaluation expression to be used with tensor trains. The expression is optimized
        with the ranks of the provided operands. Affected by einsum context manager.
        """
        expr = _EvaluationExpression(self.namespace, eq, *ops)
        return EvaluateExpression(expr)

    def min_max(self, train: TensorTrain[NDArray], num: int) -> MinMaxResult[NDArray]:
        """
        Calculate the minimum and maximum values and indices.
        """
        return _min_max(train._base, num)

    def add(
            self,
            train1: TensorTrain[NDArray],
            train2: TensorTrain[NDArray],
            *trains: TensorTrain[NDArray]
            ) -> TensorTrain[NDArray]:
        """
        Add multiple tensor trains. Affected by einsum context manager.
        """
        base = add(train1._base, train2._base, *[train._base for train in trains])
        return TensorTrain(base, copy_data=False)

    #-------------------------------------------------------------------------------------------------
    # default options

    def exact(
            self,
            optimizer: OptimizeKind = DEFAULT_OPTIMIZER) -> ExactOptions:
        """
        Exact einsum operations.
        """
        return ExactOptions(
                namespace=self.namespace,
                optimizer=optimizer)


    @overload
    def decomposition[T: MatrixDecomposition](
            self, *,
            decomposition: T,
            ncores: int = 2,
            optimizer: OptimizeKind = DEFAULT_OPTIMIZER
            ) -> DecompositionOptions[T]: ...
    @overload
    def decomposition(
            self, *,
            max_rank: int,
            cutoff: float = 0.0,
            ncores: int = 2,
            optimizer: OptimizeKind = DEFAULT_OPTIMIZER) -> DecompositionOptions[SVDecomposition[NDArray]]: ...
    def decomposition(
            self, *,
            max_rank: int = 0,
            decomposition: Optional[Any] = None,
            cutoff: float = 0.0,
            ncores: int = 2,
            optimizer: OptimizeKind = DEFAULT_OPTIMIZER
            ) -> DecompositionOptions[Any]:
        """
        Einsum operations based on matrix decompositions. One can provide either a decomposition object
        or paramters for a singular value decomposition (max_rank, cutoff). ncores specifies the number
        of cores to be used for the sweeping strategy.
        """
        if decomposition is None:
            decomposition = SVDecomposition(max_rank=max_rank, cutoff=cutoff)
        strat = SweepingStrategy(ncores=ncores)
        return DecompositionOptions(
                namespace=self.namespace,
                decomposition=decomposition,
                strategy=strat,
                optimizer=optimizer)

    @overload
    def variational[T: MatrixDecomposition](
            self, *,
            decomposition: T,
            ncores: int = 2,
            nsweeps: int = 1,
            optimizer: OptimizeKind = DEFAULT_OPTIMIZER
            ) -> VariationalOptions[T]: ...
    @overload
    def variational(
            self, *,
            max_rank: int,
            cutoff: float = 0.0,
            ncores: int = 2,
            nsweeps: int = 1,
            optimizer: OptimizeKind = DEFAULT_OPTIMIZER) -> VariationalOptions[SVDecomposition[NDArray]]: ...
    def variational(
            self, *,
            max_rank: int = 50,
            cutoff: float = 0.0,
            decomposition: Optional[Any] = None,
            ncores: int = 2,
            nsweeps: int = 1,
            optimizer: OptimizeKind = DEFAULT_OPTIMIZER) -> VariationalOptions:
        """
        Variational einsum operations. One can provide either a decomposition object or
        paramters for a singular value decomposition (max_rank, cutoff). ncores and nsweeps
        specify the sweeping strategy.
        """
        if decomposition is None:
            decomposition = SVDecomposition(max_rank=max_rank, cutoff=cutoff)
        strat = SweepingStrategy(ncores=ncores, nsweeps=nsweeps)
        return VariationalOptions(
                namespace=self.namespace,
                decomposition=decomposition,
                strategy=strat,
                optimizer=optimizer)

    def cross(
            self, *,
            max_rank: int,
            eps: float,
            solver: Optional[MatrixLeastSquares] = None) -> CrossOptions:
        """
        Manager for cross interpolation based operations. max_rank specifies the sweeping
        strategy to be employed. eps and solver are passed to CrossOptions.
        """
        if solver is None:
            solver = LstsqSolver()
        strat = SweepingStrategy(ncores=2, nsweeps=max_rank//2-1)
        return CrossOptions(
                namespace=self.namespace,
                strategy=strat, eps=eps,
                solver=solver)

    def evaluation(
            self, *,
            chunk_size: int = 1024,
            optimizer: OptimizeKind = DEFAULT_OPTIMIZER,
            ) -> EvaluationOptions:
        """
        Manager for evaluate operations.
        """
        return EvaluationOptions(namespace=self.namespace, chunk_size=chunk_size, optimizer=optimizer)


    def set_options(self, options: ExactOptions | DecompositionOptions | VariationalOptions | CrossOptions | EvaluationOptions) -> None:
        """
        Set options globally. The options are stored in a thread local variable and are
        used by the einsum, add, cross and evaluate operations.
        """
        set_options(options)

    def get_options(self, otype: OptionType) -> ExactOptions | DecompositionOptions | VariationalOptions | CrossOptions | EvaluationOptions:
        """
        Get the current options.
        """
        return get_options(self.namespace, otype)
