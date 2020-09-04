import matplotlib.pyplot as plt
import numpy as np
import pymc3 as pm
import theano.tensor as tt


data = np.loadtxt("data/mixture_data.csv", delimiter=",")

plt.hist(data, bins=20, color="k", histtype="stepfilled", alpha=0.8)
plt.title("Histogram of the dataset")
plt.ylim([0, None])
print(data[:10], "...")

"""
It appears the data has a bimodal form, that is, it appears to have two peaks, 
one near 120 and the other near 200. Perhaps there are two clusters within this dataset.

1.For each data point, choose cluster 1 with probability p, else choose cluster 2.
2.Draw a random variate from a Normal distribution with parameters μi and σi 
  where i was chosen in step 1.
3.Repeat.
"""
# A specific data point can be from either N0 or N1,
# and we assume that the data point is assigned to N0 with probability p.
# A priori, we do not know what the probability of assignment to cluster 1 is,
# so we form a uniform variable on (0,1).
# We call call this p1, so the probability of belonging to cluster 2 is therefore p2=1−p1
with pm.Model() as model:
    p1 = pm.Uniform('p', 0, 1)
    p2 = 1 - p1
    p = tt.stack([p1, p2])
    assignment = pm.Categorical("assignment", p,
                                shape=data.shape[0],
                                testval=np.random.randint(0, 2, data.shape[0]))

print("prior assignment, with p = %.2f:" % p1.tag.test_value)
print(assignment.tag.test_value[:10])

"""
Looking at the above dataset, I would guess that the standard deviations of 
the two Normals are different. To maintain ignorance of what the standard deviations might be, 
we will initially model them as uniform on 0 to 100. 
We will include both standard deviations in our model using a single line of PyMC3 code:

sds = pm.Uniform("sds", 0, 100, shape=2)

Notice that we specified shape=2: we are modeling both σs as a single PyMC3 variable. 
Note that this does not induce a necessary relationship between the two σs, 
it is simply for succinctness.
"""

"""
We also need to specify priors on the centers of the clusters.
would guess somewhere around 120 and 190 respectively, 
though I am not very confident in these eyeballed estimates. 
Hence I will set μ0=120,μ1=190 and σ0=σ1=10.
"""
with model:
    sds = pm.Uniform("sds", 0, 100, shape=2)
    centers = pm.Normal("centers",
                        mu=np.array([120, 190]),
                        sd=np.array([10, 10]),
                        shape=2)

    center_i = pm.Deterministic('center_i', centers[assignment])
    sd_i = pm.Deterministic('sd_i', sds[assignment])

    # and to combine it with the observations:
    observations = pm.Normal("obs", mu=center_i, sd=sd_i, observed=data)

print("Random assignments: ", assignment.tag.test_value[:4], "...")
print("Assigned center: ", center_i.tag.test_value[:4], "...")
print("Assigned standard deviation: ", sd_i.tag.test_value[:4])

"""
Similarly, any sampling that we do within the context of Model() will be done only on the model 
whose context in which we are working. We will tell our model to explore the space 
that we have so far defined by defining the sampling methods, 
in this case Metropolis() for our continuous variables and 
ElemwiseCategorical() for our categorical variable. 
We will use these sampling methods together to explore the space by using sample( iterations, step ), 
where iterations is the number of steps you wish the algorithm to perform and 
step is the way in which you want to handle those steps. 
We use our combination of Metropolis() and ElemwiseCategorical() for the step and 
sample 25000 iterations below.
"""
with model:
    step1 = pm.Metropolis(vars=[p, sds, centers])
    # step2 = pm.ElemwiseCategorical(vars=[assignment])
    step2 = pm.CategoricalGibbsMetropolis(vars=[assignment])
    trace = pm.sample(25000, step=[step1, step2])


"""
We have stored the paths of all our variables, or "traces", in the trace variable. 
These paths are the routes the unknown parameters (centers, precisions, and p) have taken thus far. 
The individual path of each variable is indexed by the PyMC3 variable name 
that we gave that variable when defining it within our model. 
For example, trace["sds"] will return a numpy array object 
that we can then index and slice as we would any other numpy array object.
"""
plt.subplot(311)
lw = 1
center_trace = trace["centers"]

# for pretty colors later in the book.
colors = ["#348ABD", "#A60628"] if center_trace[-1, 0] > center_trace[-1, 1] \
    else ["#A60628", "#348ABD"]

plt.plot(center_trace[:, 0], label="trace of center 0", c=colors[0], lw=lw)
plt.plot(center_trace[:, 1], label="trace of center 1", c=colors[1], lw=lw)
plt.title("Traces of unknown parameters")
leg = plt.legend(loc="upper right")
leg.get_frame().set_alpha(0.7)

plt.subplot(312)
std_trace = trace["sds"]
plt.plot(std_trace[:, 0], label="trace of standard deviation of cluster 0",
     c=colors[0], lw=lw)
plt.plot(std_trace[:, 1], label="trace of standard deviation of cluster 1",
     c=colors[1], lw=lw)
plt.legend(loc="upper left")

plt.subplot(313)
p_trace = trace["p"]
plt.plot(p_trace, label="$p$: frequency of assignment to cluster 0",
     color=colors[0], lw=lw)
plt.xlabel("Steps")
plt.ylim(0, 1)
plt.legend()

"""
1. The traces converges, not to a single point, but to a distribution of possible points. 
This is convergence in an MCMC algorithm.
2. Inference using the first few thousand points is a bad idea, 
as they are unrelated to the final distribution we are interested in. 
Thus is it a good idea to discard those samples before using the samples for inference. 
We call this period before converge the "burn-in period".
3. The traces appear as a random "walk" around the space, that is, 
the paths exhibit correlation with previous positions. 
This is both good and bad. We will always have correlation between current positions 
and the previous positions, but too much of it means we are not exploring the space well. 
This will be detailed in the Diagnostics section later in this chapter.

To achieve further convergence, we will perform more MCMC steps. 
In the pseudo-code algorithm of MCMC above, the only position 
that matters is the current position (new positions are investigated near the current position), 
implicitly stored as part of the trace object. 
To continue where we left off, we pass the trace 
that we have already stored into the sample() function with the same step value. 
The values(trace) that we have already calculated will not be overwritten. 
This ensures that our sampling continues where it left off in the same way that it left off.
"""

with model:
    trace = pm.sample(50000, step=[step1, step2], trace=trace)

center_trace = trace["centers"][25000:]
prev_center_trace = trace["centers"][:25000]

x = np.arange(25000)
plt.plot(x, prev_center_trace[:, 0], label="previous trace of center 0",
     lw=lw, alpha=0.4, c=colors[1])
plt.plot(x, prev_center_trace[:, 1], label="previous trace of center 1",
     lw=lw, alpha=0.4, c=colors[0])

x = np.arange(25000, 75000)
plt.plot(x, center_trace[:, 0], label="new trace of center 0", lw=lw, c="#348ABD")
plt.plot(x, center_trace[:, 1], label="new trace of center 1", lw=lw, c="#A60628")

plt.title("Traces of unknown center parameters")
leg = plt.legend(loc="upper right")
leg.get_frame().set_alpha(0.8)
plt.xlabel("Steps")