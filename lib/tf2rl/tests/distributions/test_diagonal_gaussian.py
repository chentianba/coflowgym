import unittest
import numpy as np

from tf2rl.distributions.diagonal_gaussian import DiagonalGaussian
from tests.distributions.common import CommonDist


class TestDiagonalGaussian(CommonDist):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.dist = DiagonalGaussian(dim=cls.dim)
        cls.param = param = {
            "mean": np.zeros(shape=(1, cls.dim), dtype=np.float32),
            "log_std": np.ones(shape=(1, cls.dim), dtype=np.float32)*np.log(1.)}
        cls.params = {
            "mean": np.zeros(shape=(cls.batch_size, cls.dim), dtype=np.float32),
            "log_std": np.ones(shape=(cls.batch_size, cls.dim), dtype=np.float32)*np.log(1.)}

    def test_kl(self):
        # KL of same distribution should be zero
        np.testing.assert_array_equal(
            self.dist.kl(self.param, self.param),
            np.zeros(shape=(1,)))
        np.testing.assert_array_equal(
            self.dist.kl(self.params, self.params),
            np.zeros(shape=(self.batch_size,)))

        # Add tests with not same distribution

    def test_log_likelihood(self):
        pass

    def test_ent(self):
        pass

    def test_sample(self):
        samples = self.dist.sample(self.param)
        self.assertEqual(samples.shape, (1, self.dim))
        samples = self.dist.sample(self.params)
        self.assertEqual(samples.shape, (self.batch_size, self.dim))


if __name__ == '__main__':
    unittest.main()
