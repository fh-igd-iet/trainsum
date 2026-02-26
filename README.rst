trainsum
========

Welcome to trainsum, a Python package designed for working with quantics tensor trains. The development was done by the ZAQC-team at the Fraunhofer Institute for Graphical Data Analysis (IGD). trainsum is licensed under EUPL 1.2 (similar to GPL).

The main features are:

- easy definition of N-dimensional tensor trains
- quantization of dimensions independent of their size
- einsum-operations equivalent to NumPy’s einsum function
- generic backends for NumPy, Torch and CuPy
- tensorized solver for eigenvalue equations and linear equation systems

Installation
------------
You can install trainsum using pip:

:code:`pip install trainsum`

The dependencies are:

- numpy
- array_api_compat
- opt_einsum
- hdf5

Documentation
-------------
The documentation for trainsum can be found at https://trainsum.readthedocs.io.

Citing
------
If u use trainsum in your research, please cite https://arxiv.org/abs/2602.20226.
