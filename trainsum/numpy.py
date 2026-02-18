# Copyright© 2025-2026 Gesellschaft zur Förderung der angewandten Forschung e.V.
# acting on behalf of its Fraunhofer Institut für Graphische Datenverarbeitung.
# Licensed under the EUPL. See LICENSE.txt.

import numpy as np
from .trainsum import TrainSum

trainsum = TrainSum[np.typing.NDArray](np)
