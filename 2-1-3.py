import pymc3 as pm
import theano.tensor as tt
import matplotlib.pyplot as plt
import numpy as np


with pm.Model() as model:
    # Calling pymc3.Deterministic is the most obvious way,
    # but not the only way, to create deterministic variables.
    # Elementary operations, like addition, exponentials etc.
    # implicitly create deterministic variables.
    # For example, the following returns a deterministic variable.
    lambda_1 = pm.Exponential("lambda_1", 1.0)
    lambda_2 = pm.Exponential("lambda_2", 1.0)
    tau = pm.DiscreteUniform("tau", lower=0, upper=10)


with pm.Model() as model:
    samples = lambda_1.random(size=20000)
    plt.hist(samples, bins=70, normed=True, histtype="stepfilled")
    plt.title("Prior distribution for $\lambda_1$")
    plt.xlim(0, 8);

# PyMC3 stochastic variables have a keyword argument observed.
# The keyword observed has a very simple role:
# fix the variable's current value to be the given data,
# typically a NumPy array or pandas DataFrame.
# For example:
data = np.array([10, 5])
with model:
    fixed_variable = pm.Poisson("fxd", 1, observed=data)
print("value: ", fixed_variable.tag.test_value)

# fix the "PyMC3 variable: observations" to the observed "dataset".
data = np.array([10, 25, 15, 20, 35])
with model:
    obs = pm.Poisson("obs", lambda_, observed=data)
print(obs.tag.test_value)