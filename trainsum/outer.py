from typing import Sequence
from .backend import ArrayLike
from .trainbase import TrainBase
from .trainshape import TrainShape
from .digit import Digit
from .dimension import Dimension
from .fuse import fuse

# alias for fuse
def outer[T: ArrayLike](*trains: TrainBase[T]) -> TrainBase[T]:
    return fuse(*trains)
