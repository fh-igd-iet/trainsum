import unittest
from itertools import product
import os
import h5py

from trainsum import TrainSum
from trainsum.typing import TensorTrain, UniformGrid
from utils import backends, rand_data

path = os.path.dirname(__file__)
class TestIO(unittest.TestCase):

    def setUp(self) -> None:
        self.trainsum = [TrainSum(backend) for backend in backends]
        self.sizes = [(120,), (256,18), (280,), (1024,20)]
        self.modes = ("block", "interleaved")

        os.mkdir(f"{path}/data")
        self.file = h5py.File("data/test_io.h5", "w")

    def tearDown(self) -> None:
        self.file.close()
        os.remove(f"{path}/data/test_io.h5")
        os.rmdir(f"{path}/data")

    def test_tensortrain(self) -> None:
        for ts, sizes, mode in product(self.trainsum, self.sizes, self.modes):
            xp = ts.namespace
            name = f"tensortrain_{sizes}"
            group = self.file.create_group(name)

            shape = ts.trainshape(*sizes, mode=mode)
            cores = []
            for i in range(len(shape)):
                left = 1 if i == 0 else 10
                right = 1 if i == len(shape)-1 else 10
                cores.append(rand_data(xp, left, *shape.middle(i), right))
            ref_train = ts.tensortrain(shape, cores)

            ts.write(group, ref_train)
            train = ts.read(group, TensorTrain)

            self.assertEqual(train.shape, ref_train.shape)
            for core, ref_core in zip(train.cores, ref_train.cores):
                self.assertTrue(xp.all(xp.equal(core, ref_core)))

            del self.file[name]

    def test_uniformgrid(self) -> None:
        for ts, sizes in product(self.trainsum, self.sizes):
            name = f"uniformgrid_{sizes}"
            group = self.file.create_group(name)

            dims = [ts.dimension(size) for size in sizes]
            domains = [ts.domain(0, 1) for _ in sizes]
            ref_grid = ts.uniform_grid(dims, domains)

            ts.write(group, ref_grid)
            grid = ts.read(group, UniformGrid)

            self.assertEqual(grid, ref_grid)
            for domain, ref_domain in zip(grid.domains, ref_grid.domains):
                self.assertEqual(domain, ref_domain)

            del self.file[name]

if __name__ == "__main__":
    unittest.main()
