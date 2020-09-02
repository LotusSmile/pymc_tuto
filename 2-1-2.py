import pymc3 as pm
import theano.tensor as tt
import matplotlib.pyplot as plt
import numpy as np

# The test_value is used only for the model, as the starting point for sampling
# if no other start is specified.
# This initial state can be changed at variable creation
# by specifying a value for the testval parameter.

with pm.Model() as model:
    parameter = pm.Exponential("poisson_param", 1.0, testval=0.3)

print("\nparameter.tag.test_value =", parameter.tag.test_value)

# Initializing Stochastic variables
with pm.Model() as model:
    some_variable = pm.DiscreteUniform("discrete_uni_var", 0, 4)

    # beta_1 = pm.Uniform("beta_1", 0, 1)
    # beta_2 = pm.Uniform("beta_2", 0, 1)
    # ...
    N = 2
    betas = pm.Uniform("betas", 0, 1, shape=N)


# Deterministic variables
# We can create a deterministic variable similarly to how we create a stochastic variable.
# We simply call up the Deterministic class in PyMC3 and pass in the function that we desire
deterministic_variable = pm.Deterministic("deterministic variable", some_function_of_variables)

with pm.Model() as model:
    # Calling pymc3.Deterministic is the most obvious way,
    # but not the only way, to create deterministic variables.
    # Elementary operations, like addition, exponentials etc.
    # implicitly create deterministic variables.
    # For example, the following returns a deterministic variable.
    lambda_1 = pm.Exponential("lambda_1", 1.0)
    lambda_2 = pm.Exponential("lambda_2", 1.0)
    tau = pm.DiscreteUniform("tau", lower=0, upper=10)

new_deterministic_variable = lambda_1 + lambda_2


def subtract(x, y):
    return x - y


with pm.Model() as model:
    # Inside a deterministic variable,
    # the stochastic variables passed in behave
    # like scalars or NumPy arrays (if multivariable).
    stochastic_1 = pm.Uniform("U_1", 0, 1)
    stochastic_2 = pm.Uniform("U_2", 0, 1)

    det_1 = pm.Deterministic("Delta", subtract(stochastic_1, stochastic_2))


import theano.tensor as tt

with pm.Model() as theano_test:
    p1 = pm.Uniform("p", 0, 1)
    p2 = 1 - p1
    p = tt.stack([p1, p2])

    assignment = pm.Categorical("assignment", p)