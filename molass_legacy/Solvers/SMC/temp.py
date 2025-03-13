import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pymc as pm
import pytensor.tensor as pt
from pytensor.tensor import TensorVariable


def my_model1(theta, freq):
    d_c1, a_c = theta

    d1 = 0.0075
    d2 = 0.0625

    LossE_list = []
    for k in range(0, len(freq)):
        f = freq[k]
        LossE = (f**2)*d1*d_c1 + (f)*d2*(a_c**2)
        LossE_list.append(LossE)

    stacked_LossE = np.array(LossE_list)

    return stacked_LossE

def my_loglike1(theta, freq, data, sigma):
    model = my_model1(theta, freq)
    return -(0.5 / sigma**2) * np.sum((data - model) ** 2)

# define a pytensor Op for our likelihood function
class LogLike(pt.Op):

    itypes = [pt.dvector]  # expects a vector of parameter values when called
    otypes = [pt.dscalar]  # outputs a single scalar value (the log likelihood)

    def __init__(self, loglike, data, freq, sigma):

        # add inputs as class attributes
        self.likelihood = loglike
        self.data = data
        self.freq = freq
        self.sigma = sigma

    def perform(self, node, inputs, outputs):
        # the method that is used when calling the Op
        (theta,) = inputs  # this will contain my variables

        # call the log-likelihood function
        logl = self.likelihood(theta, self.freq, self.data, self.sigma)

        outputs[0][0] = np.array(logl)  # output the log-likelihood

freq = np.arange(1,2000,5)
N = len(freq)
sigma = 0.05  # standard deviation of noise
d_c1_true = 0.003  # true gradient
a_c_true = 0.04  # true y-intercept

truemodel = my_model1([d_c1_true, a_c_true], freq)

# make data
rng = np.random.default_rng(716743)
data = sigma * rng.normal(size=N) + truemodel

# create our Op
logl1 = LogLike(my_loglike1, data, freq, sigma)

def logp(value: TensorVariable, mu: TensorVariable) -> TensorVariable:
    return -(value - mu)**2

if __name__ == '__main__':
    
    with pm.Model():
        # uniform priors 
        d_c1 = pm.Uniform('d_c1_model1',lower=0.001, upper=0.009)
        a_c = pm.Uniform("a_c_model1", lower=0.015, upper=0.09)

        mu = pm.Normal('mu',0,1)

        # convert to a tensor vector
        theta = pt.as_tensor_variable([d_c1, a_c])

        # use a Potential to "call" the Op and include it in the logp computation
        pm.Potential("likelihood", logl1(theta))

        pm.CustomDist('custom_dist',mu, logp=logp, observed=data)

        # Use custom number of draws to replace the HMC based defaults
        idata_mh = pm.smc.sample_smc(draws=100, cores=10, chains=4, return_inferencedata=True, idata_kwargs=dict(log_likelihood=True))