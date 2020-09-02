import pymc3 as pm
import theano.tensor as tt
import matplotlib.pyplot as plt
import numpy as np


with pm.Model() as model:
    lambda_ = pm.Exponential("poisson_param", lam=1.0)
    data_generator = pm.Poisson("data", lambda_)

with model:
    data_plus_one = data_generator + 1

# We can examine the same variables outside of the model context once they have been defined,
# but to define more variables that the model will recognize they have to be within the context.
lambda_.tag.test_value

# To create a different model object with the same name as one we have used previously,
# we need only run the first block of code again.
with pm.Model() as model:
    theta = pm.Exponential("theta", 2.0)
    data_generator = pm.Poisson("data_generator", theta)

# We can also define an entirely separate model. Note that we are free to name our models whatever we like,
# so if we do not want to overwrite an old model we need only make another.

with pm.Model() as ab_testing:
    p_A = pm.Uniform("P(A)", 0, 1)
    p_B = pm.Uniform("P(B)", 0, 1)

print("lambda_.tag.test_value =", lambda_.tag.test_value)
print("data_generator.tag.test_value =", data_generator.tag.test_value)
print("data_plus_one.tag.test_value =", data_plus_one.tag.test_value)