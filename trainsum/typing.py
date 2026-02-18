# Copyright© 2025-2026 Gesellschaft zur Förderung der angewandten Forschung e.V.
# acting on behalf of its Fraunhofer Institut für Graphische Datenverarbeitung.
# Licensed under the EUPL. See LICENSE.txt.

"""typing has all classes used in the external API of trainsum."""

from .digit import Digit
from .dimension import Dimension
from .domain import Domain
from .uniformgrid import UniformGrid
from .trainshape import TrainShape
from .tensortrain import TensorTrain

from .sweepingstrategy import SweepingStrategy
from .svdecomposition import SVDecomposition
from .qrdecomposition import QRDecomposition

from .matrixdecomposition import MatrixDecomposition
from .matrixleastsquares import MatrixLeastSquares
from .matrixeigenvaluedecomposition import MatrixEigenvalueDecomposition
from .locallinsolver import LocalLinSolver, LocalLinSolverResult
from .localeigsolver import LocalEigSolver, LocalEigSolverResult

from .options import Options, ExactOptions, EvaluationOptions, DecompositionOptions, VariationalOptions, OptionType

from .trainsum import TrainSum, EigSolver, LinSolver, LinearMap, EinsumExpression, EvaluateExpression
