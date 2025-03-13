import pymc as pm

def reproduce_bug():
    print(pm.__version__)

    def _logp(value, mu):
        return -((value - mu) ** 2)

    def _random(mu, rng=None, size=None):
        return rng.normal(loc=mu, scale=1, size=size)

    with pm.Model():
        mu = pm.CustomDist("mu", 0, logp=_logp, random=_random)
        pm.CustomDist("y", mu, logp=_logp, class_name="", random=_random, observed=[1, 2])
        pm.sample_smc(draws=6, cores=2)

if __name__ == '__main__':
    reproduce_bug()