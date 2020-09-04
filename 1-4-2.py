import pymc3 as pm
import theano.tensor as tt
import matplotlib.pyplot as plt
import numpy as np

count_data = np.loadtxt('data/txtdata.csv')
n_count_data = len(count_data)
alpha = 1.0 / count_data.mean()

with pm.Model() as model:
    # We assign them to PyMC3's stochastic variables, so-called
    # because they are treated by the back end as random number generators.
    lambda_1 = pm.Exponential('lambda_1', lam=alpha)
    lambda_2 = pm.Exponential('lambda_2', lam=alpha)

    tau = pm.DiscreteUniform('tau', lower=0, upper=n_count_data-1)

with model:
    # switch() function assigns lambda_1 or lambda_2 as the value of lambda_,
    # depending on what side of tau we are on
    # because lambda_1, lambda_2 and tau are random, lambda_ will be random.
    # We are not fixing any variables yet.
    idx = np.arange(n_count_data)  # Index
    lambda_ = pm.math.switch(tau > idx, lambda_1, lambda_2)

with model:
    # returns thousands of random variables(stochastic) from the posterior distributions of λ1,λ2 and τ.
    observation = pm.Poisson("obs", lambda_, observed=count_data)

    # observation can be used in a way such as observation.distribution.data, but not in this tutorial.
    # 아무튼 model 컨텍스트 안에서 선언되는 pm 함수들을 통해 deterministic count_data로 시뮬레이션이 됨.

with model:
    step = pm.Metropolis()
    trace = pm.sample(10000, tune=5000, step=step)

# now all variables are determined from the stochastic generator.
lambda_1_samples = trace['lambda_1']
lambda_2_samples = trace['lambda_2']
tau_samples = trace['tau']


# histogram of the samples
ax = plt.subplot(311)
ax.set_autoscaley_on(False)

plt.hist(lambda_1_samples, histtype='stepfilled', bins=30, alpha=0.85,
         label="posterior of $\lambda_1$", color="#A60628", density=True)
plt.legend(loc="upper left")
plt.title(r"""Posterior distributions of the variables
    $\lambda_1,\;\lambda_2,\;\tau$""")
plt.xlim([15, 30])
plt.xlabel("$\lambda_1$ value")

ax = plt.subplot(312)
ax.set_autoscaley_on(False)
plt.hist(lambda_2_samples, histtype='stepfilled', bins=30, alpha=0.85,
         label="posterior of $\lambda_2$", color="#7A68A6", density=True)
plt.legend(loc="upper left")
plt.xlim([15, 30])
plt.xlabel("$\lambda_2$ value")

plt.subplot(313)
w = 1.0 / tau_samples.shape[0] * np.ones_like(tau_samples)
plt.hist(tau_samples, bins=n_count_data, alpha=1,
         label=r"posterior of $\tau$",
         color="#467821", weights=w, rwidth=2.)
plt.xticks(np.arange(n_count_data))

plt.legend(loc="upper left")
plt.ylim([0, .75])
plt.xlim([35, len(count_data)-20])
plt.xlabel(r"$\tau$ (in days)")
plt.ylabel("probability");


# tau_samples, lambda_1_samples, lambda_2_samples contain
# N samples from the corresponding posterior distribution
N = tau_samples.shape[0]
expected_texts_per_day = np.zeros(n_count_data)
for day in range(0, n_count_data):
    # ix is a bool index of all tau samples corresponding to
    # the switchpoint occurring prior to value of 'day'
    ix = day < tau_samples
    # Each posterior sample corresponds to a value for tau.
    # for each day, that value of tau indicates whether we're "before"
    # (in the lambda1 "regime") or
    #  "after" (in the lambda2 "regime") the switchpoint.
    # by taking the posterior sample of lambda1/2 accordingly, we can average
    # over all samples to get an expected value for lambda on that day.
    # As explained, the "message count" random variable is Poisson distributed,
    # and therefore lambda (the poisson parameter) is the expected value of
    # "message count".
    expected_texts_per_day[day] = (lambda_1_samples[ix].sum()
                                   + lambda_2_samples[~ix].sum()) / N


plt.plot(range(n_count_data), expected_texts_per_day, lw=4, color="#E24A33",
         label="expected number of text-messages received")
plt.xlim(0, n_count_data)
plt.xlabel("Day")
plt.ylabel("Expected # text-messages")
plt.title("Expected number of text-messages received")
plt.ylim(0, 60)
plt.bar(np.arange(len(count_data)), count_data, color="#348ABD", alpha=0.65,
        label="observed texts per day")

plt.legend(loc="upper left");