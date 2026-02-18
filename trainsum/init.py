# Copyright© 2025-2026 Gesellschaft zur Förderung der angewandten Forschung e.V.
# acting on behalf of its Fraunhofer Institut für Graphische Datenverarbeitung.
# Licensed under the EUPL. See LICENSE.txt.

from .digit import Digit
from .dimension import Dimension
from .domain import Domain
from .uniformgrid import UniformGrid
from .trainshape import TrainShape, trainshape

from .matrixdecomposition import MatrixDecomposition
from .qrdecomposition import QRDecomposition
from .svdecomposition import SVDecomposition
from .sweepingstrategy import SweepingStrategy

from .options import set_options, get_options, OptionType
