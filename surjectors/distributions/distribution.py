class Distribution:
    def __init__(self, base, flow):
        self.flow = flow
        self.base = base

    def sample(self, rng, params, sample_shape=(1,)):
        x, log_prob = self.base.sample_and_log_prob(
            seed=rng, sample_shape=sample_shape
        )
        y, _ = self.flow.apply(params, x, "forward_and_log_det")
        return y

    def sample_and_log_prob(self, rng, params, sample_shape=(1,)):
        x, log_prob = self.base.sample_and_log_prob(
            seed=rng, sample_shape=sample_shape
        )
        y, logdet = self.flow.apply(params, x, "forward_and_log_det")
        return y, log_prob - logdet

    def log_prob(self, params, y):
        x, logdet = self.flow.apply(params, y, "inverse_and_log_det")
        logprob = self.base.log_prob(x)
        return logprob + logdet
