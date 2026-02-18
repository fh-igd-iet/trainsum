# Copyright© 2025-2026 Gesellschaft zur Förderung der angewandten Forschung e.V.
# acting on behalf of its Fraunhofer Institut für Graphische Datenverarbeitung.
# Licensed under the EUPL. See LICENSE.txt.

from copy import deepcopy

from .backend import ArrayLike
from .matrixdecomposition import MatrixDecomposition
from .svdecomposition import SVDecomposition
from .trainshape import TrainShape, change_dims
from .trainbase import TrainBase
from .einsumcontraction import EinsumContraction
from .localvector import local_vector
from .sweepingstrategy import SweepingStrategy
from .contractor import OptimizeKind
from .utils import namespace_of_trains

from .einsumequation import EinsumEquation
from .innergenerator import InnerGenerator
from .generatorcallabletype import GeneratorCallableType

class VariationalContractor:

    optimizer: OptimizeKind
    decomposition: MatrixDecomposition
    strategy: SweepingStrategy

    def __init__(self,
                 contr: EinsumContraction,
                 optimizer: OptimizeKind = "greedy",
                 decomposition: MatrixDecomposition = SVDecomposition(),
                 strategy: SweepingStrategy = SweepingStrategy()) -> None:

        if contr.result_shape is None or contr.full_result_shape is None:
            raise ValueError("Variational contractor cannot be used for full ccontractions. Use FullContractor instead.")

        self.optimizer = deepcopy(optimizer)
        self.decomposition = deepcopy(decomposition)
        self.strategy = deepcopy(strategy)
        self._inner_gen = self._get_inner_generator(contr, optimizer)

    def __call__[T: ArrayLike](
            self,
            guess: TrainBase[T],
            *operands: TrainBase[T],
            expr: bool = False) -> TrainBase[T]:
        xp = namespace_of_trains(guess, *operands)
        guess = deepcopy(guess)
        igen = self._inner_gen(guess, *operands, expr=expr)
        lvec = local_vector(guess, self.decomposition)
        data = xp.zeros(0)
        gen_type = GeneratorCallableType.FULL
        try:
            for _, lrange in self.strategy(guess.shape):
                lvec.send((lrange, data))
                data = igen.send((lrange, gen_type))
        finally:
            igen.close()
        return guess

    def calc_expressions(
            self,
            guess: TrainShape | TrainBase,
            *ops: TrainShape | TrainBase) -> None:
        guess_shape = guess.shape if isinstance(guess, TrainBase) else guess
        self._inner_gen.calc_expressions(self.strategy, guess_shape, *ops)

    def _get_inner_generator(
            self,
            contr: EinsumContraction,
            optimizer: OptimizeKind
            ) -> InnerGenerator:
        eq = f"{contr.equation.result}," + ",".join(contr.equation.operands) + "->"
        ops = [contr.result_shape, *contr.equation.shapes]
        eq = EinsumEquation(eq, *ops)
        dim_map = {}
        for in_shape, out_shape in zip(ops, eq.shapes):
            for in_dim, out_dim in zip(in_shape.dims, out_shape.dims):
                dim_map[in_dim] = out_dim
        new_op_shape_dims = [dim_map[dim] for dim in contr.operation_shape.dims]
        op_shape = change_dims(contr.operation_shape, new_op_shape_dims)
        contr = EinsumContraction(eq, op_shape)
        return InnerGenerator(contr, 0, optimizer)
