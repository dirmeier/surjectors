from jax import random, lax
from jax.scipy.special import erf
import distrax

class SolarDynamoSimulator:
    def __init__(self, **kwargs):
        self.p0_mean = kwargs.get("p0_mean", 1.0)
        self.p0_std = kwargs.get("p0_std", 1.0)
        self.alpha1_min = kwargs.get("alpha1_min", 1.3)
        self.alpha1_max = kwargs.get("alpha1_max", 1.5)
        self.alpha2_max = kwargs.get("alpha2_max", 1.65)
        self.epsilon_max = kwargs.get("epsilon_max", 0.5)
        self.alpha1 = kwargs.get("alpha1", None)
        self.alpha2 = kwargs.get("alpha2", None)

    def sample(self, key, batch_size, len_timeseries=1000):
        p_key, alpha1_key, alpha2_key, epsilon_key, key = random.split(key, 5)
        p0 = random.normal(p_key,  shape=(batch_size,)) * self.p0_std + self.p0_mean
        alpha1 = random.uniform(
            alpha1_key, shape=(batch_size,), minval=self.alpha1_min, maxval=self.alpha1_max
        )
        alpha2 = random.uniform(
            alpha2_key,  shape=(batch_size,), minval=alpha1, maxval=self.alpha2_max
        )
        epsilon_max = random.uniform(
            epsilon_key, shape=(batch_size,), minval=0, maxval=self.epsilon_max
        )
        batch = self._sample_timeseries(
            key, batch_size, p0, alpha1, alpha2, epsilon_max, len_timeseries
        )

        return p0, alpha1, alpha2, epsilon_max, batch[0].T, batch[1].T

    @staticmethod
    def babcock_leighton_fn(p, b_1=0.6, w_1=0.2, b_2=1.0, w_2=0.8):
        f = 0.5 * (1.0 + erf((p - b_1) / w_1)) * (1.0 - erf((p - b_2) / w_2))
        return f

    def babcock_leighton(self, p, alpha, epsilon):
        p = alpha * self.babcock_leighton_fn(p) * p + epsilon
        return p

    def _sample_timeseries(
        self, key, batch_size, pn, alpha_min, alpha_max, epsilon_max, len_timeseries
    ):
        a = distrax.Uniform(alpha_min, alpha_max).sample(
            seed=key, sample_shape=(len_timeseries,)
        )
        e = distrax.Uniform(0.0, epsilon_max).sample(
            seed=key, sample_shape=(len_timeseries,)
        )

        def _fn(fs, arrays):
            alpha, epsilon = arrays
            f, pn = fs
            f = self.babcock_leighton_fn(pn)
            pn = self.babcock_leighton(pn, alpha, epsilon)
            return (f, pn), (f, pn)

        _, pn = lax.scan(_fn, (pn, pn), (a, e))
        return pn
