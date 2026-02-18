# Copyright© 2025-2026 Gesellschaft zur Förderung der angewandten Forschung e.V.
# acting on behalf of its Fraunhofer Institut für Graphische Datenverarbeitung.
# Licensed under the EUPL. See LICENSE.txt.

from .backend import ArrayLike
from .options import OptionType, get_options, ExactOptions, VariationalOptions
from .trainbase import TrainBase
from .utils import namespace_of_trains

from .exactaddition import ExactAddition
from .decompositionaddition import DecompositionAddition
from .variationaladdition import VariationalAddition

def add[T: ArrayLike](train1: TrainBase[T], train2: TrainBase[T], *trains: TrainBase[T]) -> TrainBase[T]:
    xp = namespace_of_trains(train1, train2, *trains)
    opts = get_options(xp, OptionType.EINSUM)
    if isinstance(opts, ExactOptions):
        return ExactAddition()(train1, train2, *trains)
    if opts.decomposition is None or opts.strategy is None:
        raise ValueError("Decomposition and strategy must be provided for 'decomposition' and 'variational' methods.")
    add = DecompositionAddition(opts.strategy, opts.decomposition)
    guess = add(train1, train2, *trains)
    if isinstance(opts, VariationalOptions):
        var_add = VariationalAddition(train1.shape, opts.decomposition, opts.strategy, opts.optimizer)
        guess = var_add(guess, train1, train2, *trains)
    return guess
