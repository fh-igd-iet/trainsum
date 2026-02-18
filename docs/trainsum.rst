TrainSum
========

.. currentmodule:: trainsum.trainsum

.. autoclass:: TrainSum

   .. autoattribute:: TrainSum.namespace
   .. autoattribute:: TrainSum.index_type

Basics
------

.. autosummary::
   :toctree: Trainsum/methods
   :nosignatures:

   TrainSum.dimension
   TrainSum.domain
   TrainSum.uniform_grid
   TrainSum.trainshape
   TrainSum.svdecomposition
   TrainSum.qrdecomposition
   TrainSum.sweeping_strategy

Construction
------------

.. autosummary::
   :toctree: Trainsum/methods
   :nosignatures:

   TrainSum.full
   TrainSum.exp
   TrainSum.sin
   TrainSum.cos
   TrainSum.polyval
   TrainSum.shift
   TrainSum.toeplitz
   TrainSum.tensortrain

Fourier Transform
-----------------

.. autosummary::
   :toctree: Trainsum/methods
   :nosignatures:

   TrainSum.qft
   TrainSum.iqft
   TrainSum.qftshift
   TrainSum.iqftshift
   TrainSum.qftfreq

Input/Output
------------

.. autosummary::
   :toctree: Trainsum/methods
   :nosignatures:

   TrainSum.write
   TrainSum.read

Solver
------
.. autosummary::
   :toctree: Trainsum/methods
   :nosignatures:

   TrainSum.gmres
   TrainSum.lanczos
   TrainSum.eigsolver
   TrainSum.linsolver

Operations
----------

.. autosummary::
   :toctree: Trainsum/methods
   :nosignatures:

   TrainSum.min_max
   TrainSum.add
   TrainSum.einsum
   TrainSum.einsum_expression
   TrainSum.evaluate
   TrainSum.evaluate_expression

Context Manager
---------------

.. autosummary::
   :toctree: Trainsum/methods
   :nosignatures:

   TrainSum.exact
   TrainSum.decomposition
   TrainSum.variational
   TrainSum.cross
   TrainSum.evaluation
   TrainSum.set_options
   TrainSum.get_options
