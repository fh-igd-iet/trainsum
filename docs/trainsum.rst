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
   TrainSum.rand_svdecomposition
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

Binary Tensor Trains
--------------------

.. autosummary::
   :toctree: Trainsum/methods
   :nosignatures:

   TrainSum.linear_integer_equation
   TrainSum.modulo_integer_equation
   TrainSum.range_integer_equation
   TrainSum.binary_train

Discrete Wavelet Transform
--------------------------

.. autosummary::
   :toctree: Trainsum/methods
   :nosignatures:

   TrainSum.dwt
   TrainSum.idwt


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
   TrainSum.linear_map
   TrainSum.eigsolver
   TrainSum.linsolver

Operations
----------

.. autosummary::
   :toctree: Trainsum/methods
   :nosignatures:

   TrainSum.outer
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
