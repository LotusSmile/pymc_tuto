import scipy.stats as stats
import numpy as np
import matplotlib.pyplot as plt
import pymc3 as pm

binomial = stats.binom

parameters = [(10, .4), (10, .9)]
colors = ["#348ABD", "#A60628"]

for i in range(2):
    N, p = parameters[i]
    _x = np.arange(N + 1)
    plt.bar(_x - 0.5, binomial.pmf(_x, N, p), color=colors[i],
            edgecolor=colors[i],
            alpha=0.6,
            label="$N$: %d, $p$: %.1f" % (N, p),
            linewidth=3)

plt.legend(loc="upper left")
plt.xlim(0, 10.5)
plt.xlabel("$k$")
plt.ylabel("$P(X = k)$")
plt.title("Probability mass distributions of binomial random variables");


# we sample p, the true proportion of cheaters, from a prior.
# Since we are quite ignorant about p, we will assign it a Uniform(0,1) prior.
N = 100
with pm.Model() as model:
    p = pm.Uniform("freq_cheating", 0, 1)

with model:
    true_answers = pm.Bernoulli("truths", p, shape=N, testval=np.random.binomial(1, 0.5, N))
    # 어떤 확률과 표본 수를 가진 분포를 모델링 하는 것으로 보임.


# If we carry out the algorithm, the next step that occurs is the first coin-flip each student makes.
# This can be modeled again by sampling 100 Bernoulli random variables with p=1/2:
# denote a 1 as a Heads and 0 a Tails.
with model:
    first_coin_flips = pm.Bernoulli("first_flips", 0.5, shape=N, testval=np.random.binomial(1, 0.5, N))
print(first_coin_flips.tag.test_value)

# Although not everyone flips a second time, we can still model the possible realization of second coin-flips:
with model:
    second_coin_flips = pm.Bernoulli("second_flips", 0.5, shape=N, testval=np.random.binomial(1, 0.5, N))
print(second_coin_flips.tag.test_value)

# Using these variables, we can return a possible realization of the observed proportion of "Yes" responses.
# We do this using a PyMC3 deterministic variable:
import theano.tensor as tt
with model:
    val = first_coin_flips*true_answers + (1 - first_coin_flips)*second_coin_flips
    observed_proportion = pm.Deterministic("observed_proportion", tt.sum(val)/float(N))
# The line fc*t_a + (1-fc)*sc contains the heart of the Privacy algorithm.
# Elements in this array are 1 if and only if i) the first toss is heads and the student cheated or ii)
# the first toss is tails, and the second is heads, and are 0 else.
# Finally, the last line sums this vector and divides by float(N), produces a proportion.
print(observed_proportion.tag.test_value)


# The researchers observe a Binomial random variable,
# with N = 100 and p = observed_proportion with value = 35

X = 35
with model:
    observations = pm.Binomial("obs", N, observed_proportion, observed=X)
# To be explained in Chapter 3!
with model:
    step = pm.Metropolis(vars=[p])
    trace = pm.sample(40000, step=step)
    burned_trace = trace[15000:]

p_trace = burned_trace["freq_cheating"][15000:]
plt.hist(p_trace, histtype="stepfilled", normed=True, alpha=0.85, bins=30,
         label="posterior distribution", color="#348ABD")
plt.vlines([.05, .35], [0, 0], [5, 5], alpha=0.3)
plt.xlim(0, 1)
plt.legend()


# Given a value for p (which from our god-like position we know),
# we can find the probability the student will answer yes:
# Thus, knowing p we know the probability a student will respond "Yes".
# In PyMC3, we can create a deterministic function to evaluate the probability of responding "Yes", given p:

with pm.Model() as model:
    p = pm.Uniform("freq_cheating", 0, 1)
    p_skewed = pm.Deterministic("p_skewed", 0.5*p + 0.25)

# This is where we include our observed 35 "Yes" responses.
# In the declaration of the pm.Binomial, we include value = 35 and observed = True.

with model:
    yes_responses = pm.Binomial("number_cheaters", 100, p_skewed, observed=35)

# Below we add all the variables of interest to a Model container and run our black-box algorithm over the model.
with model:
    # To Be Explained in Chapter 3!
    step = pm.Metropolis()
    trace = pm.sample(25000, step=step)
    burned_trace = trace[2500:]

p_trace = burned_trace["freq_cheating"]
plt.hist(p_trace, histtype="stepfilled", normed=True, alpha=0.85, bins=30,
         label="posterior distribution", color="#348ABD")
plt.vlines([.05, .35], [0, 0], [5, 5], alpha=0.2)
plt.xlim(0, 1)
plt.legend()


"""
Protip: Arrays of PyMC3 variables.
There is no reason why we cannot store multiple heterogeneous PyMC3 variables in a Numpy array. 
Just remember to set the dtype of the array to object upon initialization. For example:
"""
N = 10
x = np.ones(N, dtype=object)
with pm.Model() as model:
    for i in range(0, N):
        x[i] = pm.Exponential('x_%i' % i, (i+1.0)**2)