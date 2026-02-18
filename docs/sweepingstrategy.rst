Sweeping Strategy
=================

.. currentmodule:: trainsum.sweepingstrategy

.. autoclass:: SweepingStrategy
   :members:
   :special-members: __call__lin_map = # ... some LinearMap
guess   = # ... a guess state

# define the options
local_solver = ts.lanczos()
decomposition = ts.svdecomposition(max_rank=10)
strategy = ts.sweeping_strategy(ncores=2, nsweeps=10)

# construct the solver
solver = ts.eigsolver(
    lin_map,
    decomposition=decomposition,
    strategy=strategy,
    solver=local_solver)

# call the solver for starting the solving process
result = solver(guess)
