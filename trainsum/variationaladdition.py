# Copyright© 2025-2026 Gesellschaft zur Förderung der angewandten Forschung e.V.
# acting on behalf of its Fraunhofer Institut für Graphische Datenverarbeitung.
# Licensed under the EUPL. See LICENSE.txt.

from copy import deepcopy
from string import ascii_lowercase

from .backend import ArrayLike
from .matrixdecomposition import MatrixDecomposition
from .svdecomposition import SVDecomposition
from .trainshape import TrainShape
from .trainbase import TrainBase
from .einsumcontraction import EinsumContraction
from .localvector import local_vector
from .sweepingstrategy import SweepingStrategy
from .contractor import OptimizeKind
from .utils import namespace_of_trains, trains_match

from .einsumequation import EinsumEquation
from .innergenerator import InnerGenerator
from .generatorcallabletype import GeneratorCallableType

class VariationalAddition:

    optimizer: OptimizeKind
    decomposition: MatrixDecomposition
    strategy: SweepingStrategy

    def __init__(
            self,
            shape: TrainShape,
            decomposition: MatrixDecomposition = SVDecomposition(),
            strategy: SweepingStrategy = SweepingStrategy(),
            optimizer: OptimizeKind = "greedy") -> None:

        self.optimizer = deepcopy(optimizer)
        self.decomposition = deepcopy(decomposition)
        self.strategy = deepcopy(strategy)
        self._inner_gen = self._get_inner_generator(shape, optimizer)

    def _get_inner_generator(
            self,
            shape: TrainShape,
            optimizer: OptimizeKind
            ) -> InnerGenerator:
        dims = ascii_lowercase[:len(shape.dims)]
        eq = f"{dims},{dims}->"
        eq = EinsumEquation(eq, shape, shape)
        contr = EinsumContraction(eq, op_shape=eq.shapes[0])
        return InnerGenerator(contr, 0, optimizer)

    def __call__[T: ArrayLike](
            self,
            guess: TrainBase[T],
            *operands: TrainBase[T],
            expr: bool = False) -> TrainBase[T]:
        trains_match(guess, *operands)
        guess = deepcopy(guess)
        gen_type = GeneratorCallableType.FULL

        inner_gens = [deepcopy(self._inner_gen) for _ in range(len(operands))]
        igens = [igen(guess, addend, expr=expr) for igen, addend in zip(inner_gens, operands)]
        lvec = local_vector(guess, self.decomposition)

        xp = namespace_of_trains(guess, *operands)
        data = xp.zeros(0)
        try:
            for _, lrange in self.strategy(guess.shape):
                lvec.send((lrange, data))
                arg = (lrange, gen_type)
                data = sum((igen.send(arg) for igen in igens[1:]),
                           start=igens[0].send(arg))
        finally:
            [igen.close() for igen in igens]
        return guess

    def calc_expressions(
            self,
            guess: TrainShape | TrainBase,
            operand: TrainShape | TrainBase) -> None:
        guess_shape = guess.shape if isinstance(guess, TrainBase) else guess
        op_shape = operand.shape if isinstance(operand, TrainBase) else operand
        self._inner_gen.calc_expressions(self.strategy, guess_shape, op_shape)
