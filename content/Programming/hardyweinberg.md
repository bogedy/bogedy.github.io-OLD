---
title: "Using MCMC to find allele frequencies under Hardy-Weinberg equilibrium"
date: 2022-05-27
katex: true
---

I would like to work through this example to show how Bayesian methods can be useful for finding estimates and expressing uncertainty. I also compare using different tools for the job.

## Hardy-Weinberg equilibrium

I'm going to walk through a problem in genetics where we're interested in finding the rate of a certain gene in a population.

Imagine a population of bi-allelic organisms, each carrying $A$ or $a$ at each loci. So they are each an $AA$, $Aa$ or $aa$ genotype. For modeling purposes we'll assume that the population is randomly mating. From this it follows that if $r$ is the frequency of the $A$ allele, then the proportions of each genotype are $r^2$, $2r(1-r)$ and $(1-r)^2$ respectively. These sum to 1 and we say that the population is in *[Hardy-Weinberg equillibrium](https://en.wikipedia.org/wiki/Hardy%E2%80%93Weinberg_principle)*.

In practice we might also want to relax our assumption of random mating. It's posible that similar organisms will mate together more frequently. This will raise the number of homozygotes in the population. So we're also interested in modeling $f$, the rate of *inbred by descent* (IBD) in the population. An example of an IBD pedigree chart: 

![image](https://github.com/bogedy/mcmc_thesis/blob/master/latex_src/IBD.jpg?raw=true)

With $f$ involved we calculate new proportions for each genotype:

$$p(AA) = p(AA|\text{IBD})p(\text{IBD}) + p(AA|\text{not IBD})p(\text{not IBD})$$

$$ = rf + r^2(1-f) $$

etc.

## The problem

We'd like to esitmate likely values of $f$ and $r$ by observing a sample from the population. First we
observe counts of genotypes $X = (n_{AA}, n_{Aa}, n_{aa})$. We want to find the joint posterior distribution $p(f,r | X)$.

There is no closed form expression of this posterior distribution for any $x$. We could plug in a specific $x$ and find the maximum likelihood point estimate of $f, r$ by optimizing the likelihood function

$$ L(f,r;x) = p(x|f,r) = p(AA)^{n_{AA}}p(Aa)^{n_{Aa}}p(aa)^{n_{aa}}$$

but this is a less useful solution than what we can get with MCMC, as we'll see in a minute. Also, analytical methods fail when dealing with more than two alleles ([more info](https://www.nature.com/articles/6883600#Sec2)).

And we can't describe the whole posterior distribution either, because the denominator is a really difficult integral:

$$p(f,r | X) = \frac{p(x|f,r)p(f,r)}{p(x)} = \frac{p(x|f,r)p(f,r)}{\int{p(x|f,r)p(f,r) dfdr}}$$

How can we describe the posterior distribution if we can't write down the posterior? With MCMC, we can sample from it without knowing it completely.

## Review of MCMC simulation for this example

Here I describe a specific type of MCMC, the Metropolis algorithm.
The MC- means that we're constructing a Markov Chain. The -MC means we are doing a Monte Carlo simulation: a random simulation meant to solve something.

Each step in the Markov Chain starts with a proposal value of $(f,r)$. We accept or reject the proposal by examining the ratio $R$ of the likelihood of the proposal to the likelihood of the last step in the chain. If $R>1$ then we accept the proposal and move on. If $R<1$ then we accept the proposal with probability $R$ or else repeat the same $(f,r)$ value from the last step.

(In the limit, this Markov Chain *is* the posterior distribution. See [my thesis](https://github.com/bogedy/mcmc_thesis/blob/master/thesisIsaiahKriegman2020Final.pdf) for a summary of why this is true!)

We sample $(f,r)$ from a prior distribution and at each step $t$ we calculate the ratio

$$R=\frac{L(f_{t}, r_{t};x)}{L(f_{t-1}, r_{t-1};x)}.$$

Let's say that we observe $x=(50, 21, 29)$. Let's look at how we might sample from the posterior:

## Programming an MCMC simulation

I'll use the following in Python:


```python
import numpy as np
from collections import defaultdict
from matplotlib import pyplot as plt
import pymc3 as pm
import arviz as az
import time
```

### Using PyMC3

PyMC is pretty cool because in just a few lines I can express the reltionship between $f$, $r$ and $x$ and the package will automatically define the likelihood function and start sampling from the posterior using MCMC, complete with a simulation tuning algorithm.


```python
# Our data, x:
x=np.array([50, 21, 29]) #nAA, nAa, naa

data = np.array([0]*x[0] + [1]*x[1] + [2]*x[2])

with pm.Model() as hardy_weinberg:
    
    f = pm.Beta('f', alpha=1, beta=1) # uniform prior for f
    r = pm.Beta('r', alpha=1, beta=1) # uniform prior for r
    param1 = f*r+(1-f)*(r**2)
    param2 = 2*(1-f)*r*(1-r)
    param3 = f*(1-r)+(1-f)*(1-r)
    genotype = pm.Categorical('genotype', p=pm.math.stack(param1, param2, param3), 
                              observed=data)
```

#### Metropolis


```python
with hardy_weinberg:
    breeding_samples=pm.sample(10000, step=pm.Metropolis(), return_inferencedata=True)
```

    Sequential sampling (1 chains in 1 job)
    CompoundStep
    >Metropolis: [r]
    >Metropolis: [f]




<style>
    /* Turns off some styling */
    progress {
        /* gets rid of default border in Firefox and Opera. */
        border: none;
        /* Needs to be in here for Safari polyfill so background images work as expected. */
        background-size: auto;
    }
    .progress-bar-interrupted, .progress-bar-interrupted::-webkit-progress-bar {
        background: #F44336;
    }
</style>





<div>
  <progress value='11000' class='' max='11000' style='width:300px; height:20px; vertical-align: middle;'></progress>
  100.00% [11000/11000 00:02<00:00 Sampling chain 0, 0 divergences]
</div>



    Sampling 1 chain for 1_000 tune and 10_000 draw iterations (1_000 + 10_000 draws total) took 3 seconds.
    /usr/local/Caskroom/miniconda/base/lib/python3.9/site-packages/arviz/data/base.py:220: UserWarning: More chains (10000) than draws (2). Passed array should have shape (chains, draws, *shape)
      warnings.warn(
    Only one chain was sampled, this makes it impossible to run some convergence checks



```python
az.plot_posterior(breeding_samples)
plt.show()
```

<img src="/assets/hardyweinberg/output_17_0.png" style="background-color:white;padding:20px;">
    


Here we samples from the posterior distributions of $f$ and $r$. The HDI is just a "high density interval" where 94% of the data lies. There is not one unique HDI. Below I show a histogram of their joint distribution. 

(Why does it show the 94% HDI instead of 95%? The authors of this graphics package said it's to [keep you on your toes](https://arviz-devs.github.io/arviz/getting_started/Introduction.html#arviz-rcparams).)

Below is a histogram of the joint distribution:


```python
f_sample = az.extract_dataset(breeding_samples).f.values
r_sample = az.extract_dataset(breeding_samples).r.values

plt.hist2d(f_sample, r_sample, bins = 75)
plt.title('MCMC sample of $p(f,r|x)$ using Metropolis')
plt.show()
```

<img src="/assets/hardyweinberg/output_19_0.png" style="background-color:white;padding:20px;">

**This is cool!**

We can see the mode of the distribution in the center of the blob. That's a good estimate of $f$ and $r$. But what we also have is a measure of *uncertainty*. We can start answering questions like "in how many draws was $f$ between 0.35 and 0.50?"

The answer is:


```python
count = ((f_sample<0.50)&(f_sample>0.35)).sum()
total = f_sample.size

print(f"Total f between 0.35 and 0.50: {count}")
print(f"Total samples: {total}")
print(f"Percent: {count/total}")
```

    Total f between 0.35 and 0.50: 4137
    Total samples: 10000
    Percent: 0.4137


This says that $f$ was outside of that range more often that it was in it in our simulation. This is quantified uncertainty about $f$.

#### Now try with NUTS

NUTS is what is called a Hamiltonian MCMC sampler (blog post coming). It stands for "No U-Turns Sampler". This is the default sampler in PyMC. Basically instead of "proposing" values of $f$ and $r$ by sampling from their prior distributions, it takes the gradient of the likelihood function and explores the space of $(f,r)$ by running a physics simulation and sort of surfing over the likelihood function.


```python
with hardy_weinberg:
    breeding_samples_nuts=pm.sample(10000, chains=1, return_inferencedata=True)
```

    Auto-assigning NUTS sampler...
    Initializing NUTS using jitter+adapt_diag...
    Sequential sampling (1 chains in 1 job)
    NUTS: [r, f]




<style>
    /* Turns off some styling */
    progress {
        /* gets rid of default border in Firefox and Opera. */
        border: none;
        /* Needs to be in here for Safari polyfill so background images work as expected. */
        background-size: auto;
    }
    .progress-bar-interrupted, .progress-bar-interrupted::-webkit-progress-bar {
        background: #F44336;
    }
</style>





<div>
  <progress value='11000' class='' max='11000' style='width:300px; height:20px; vertical-align: middle;'></progress>
  100.00% [11000/11000 00:05<00:00 Sampling chain 0, 0 divergences]
</div>



    /usr/local/Caskroom/miniconda/base/lib/python3.9/site-packages/scipy/stats/_continuous_distns.py:624: RuntimeWarning: overflow encountered in _beta_ppf
      return _boost._beta_ppf(q, a, b)
    Sampling 1 chain for 1_000 tune and 10_000 draw iterations (1_000 + 10_000 draws total) took 5 seconds.
    Only one chain was sampled, this makes it impossible to run some convergence checks



```python
f_sample_nuts = az.extract_dataset(breeding_samples_nuts).f.values
r_sample_nuts = az.extract_dataset(breeding_samples_nuts).r.values

plt.hist2d(f_sample_nuts, r_sample_nuts, bins=75)
plt.xlabel('f')
plt.ylabel('r')
plt.title('MCMC sample of $p(f,r|x)$ using NUTS')
plt.show()
```

<img src="/assets/hardyweinberg/output_25_0.png" style="background-color:white;padding:20px;">    


### A hard-coded Gibbs sampler

Another method of MCMC sampling is Gibbs sampling, which works a little differently.

Instead of proposing points and evaluating their likelihood, we sample each paramter that we're interested in one at a time conditioned on the othe parameters and our data. For this example we do it like so:

We introduce a *latent variable* $Z$ representing if a locus is IBD or not. Then it's possible to sample $p(Z\|f, r, x)$ as follows

$$p(z)=f$$

$$p(z|\text{Aa})=0$$

$$p(z|\text{AA}) = \frac{p(\text{AA}|z)p(z)}{p(\text{AA})} = \frac{p(\text{AA}|z)p(z)}{p(\text{AA|z})p(z)+p(\text{AA|~z})p(\sim z)} = \frac{rf}{rf+r^2 (1-f)}$$

etc.

Then we sample $p(f,r\|z, x)$. We can do this because if we assign Beta priors to $f$ and $r$, then we can show that they're the conjugate priors to Binomial distirbutions.

(I don't go into the details here because I want to focus on the coding and sampling details. For a longer treatment of deriving these distributions, see page 19 [here](https://github.com/bogedy/mcmc_thesis/blob/master/thesisIsaiahKriegman2020Final.pdf))

Our Gibbs sampler:


```python
np.random.seed(123)

def gibbs_hw(niters, data, 
             prior_params=[1,1,1,1], 
             initial_values = {'f': 0.5, 'r': 0.5}
            ):
    
    # Turn counts into list of strings
    obs = np.array(['AA']*data[0] + ['Aa']*data[1] + ['aa']*data[2])
    
    f = np.zeros((niters))
    r = np.zeros_like(f)
    f[0] = initial_values['f']
    r[0] = initial_values['r']
    
    for i in range(1, niters):
        # Z_i is whether the ith observation is inbred or not.
        # g_i is the genotype of the ith individual
        # Calculate p(Z|f,r) for each case of g_i:
        zi_prob_map = {
            'AA': f[i-1] * r[i-1] / (f[i-1] * r[i-1] + (1 - f[i-1]) * r[i-1] ** 2),
            'Aa': 0,
            'aa': f[i-1] * (1 - r[i-1]) / (f[i-1] * (1 - r[i-1]) + (1 - f[i-1]) * (1 - r[i-1]) ** 2)
        }
        
        z_probs = np.array([zi_prob_map[key] for key in obs])        
        z = np.random.uniform(size = z_probs.size) < z_probs
        n_ibd = z.sum()
        n_not_ibd = (~z).sum()
        
        f[i] = np.random.beta(n_ibd + prior_params[0], n_not_ibd + prior_params[1], size=1)
        
        # Get counts of genotypes given NOT inbred. 
        types, not_idb_type_counts = np.unique(obs[~z], return_counts=True)
        not_ibd_counts = defaultdict(lambda :0, zip(types, not_idb_type_counts))
        nz_A = 2 * not_ibd_counts["AA"] + not_ibd_counts["Aa"] 
        nz_a = 2 * not_ibd_counts["aa"] + not_ibd_counts["Aa"]

        # Get counts of genotypes given  inbred.
        types, idb_type_counts = np.unique(obs[z], return_counts=True)
        ibd_counts = defaultdict(lambda :0, zip(types, idb_type_counts))
        z_A = ibd_counts["AA"]
        z_a = ibd_counts["aa"]

        r[i] = np.random.beta(prior_params[2] + nz_A + z_A, prior_params[3] + nz_a + z_a, size=1)
    
    return{
        'f': f,
        'r': r
    }
        
start = time.time()    
out = gibbs_hw(niters=10000, data=(50, 21, 29))
print(f"Ellapsed time: {time.time() - start}")
plt.hist2d(out['f'], out['r'], bins=75)
plt.show()
```

    Ellapsed time: 1.0994501113891602



    
<img src="/assets/hardyweinberg/output_30_1.png" style="background-color:white;padding:20px;">

    


## But are the samples any good?

In the limit, MCMC samples are samples from the true distribution. But our finite sample is just an estimate. We also want to quantify the quality of our estimate. 

We'd like our samples to be independent. But they aren't because they were constructed form a Markov chain, each sample is depednent on the last. But we can look at the autocorrelation of our sample with lags of itself to see how close to being independent it is.


```python
# For the Metropolis example:
az.plot_autocorr(breeding_samples, combined=True)
plt.show()
```


    
<img src="/assets/hardyweinberg/output_33_0.png" style="background-color:white;padding:20px;">

    


In this example the autocorrelation becomes negligible after about 15 or so lags. We can use this autocorrelation function to get our *effective sample size*:

$$ESS = N \left / \left(1+2\sum_{k=1}^{\infty}ACF(k)\right) \right.$$

(This definition comes from [Doing Bayesian Data Analysis](https://jkkweb.sitehost.iu.edu/DoingBayesianDataAnalysis/), ArViz uses a [different expression](https://arviz-devs.github.io/arviz/api/generated/arviz.ess.html).)


```python
f_ess = az.ess(breeding_samples).f.values
r_ess = az.ess(breeding_samples).r.values

print("Metropolis ESS:")
print(f"f ESS: {f_ess}\nr ESS: {r_ess}")
```

    Metropolis ESS:
    f ESS: 1134.3778293566886
    r ESS: 1107.0805390876833


We took around 40,000 samples, but only yielded around 4.5k effective samples. Let's compare that to the NUTS sampler:


```python
f_ess_nuts = az.ess(breeding_samples_nuts).f.values
r_ess_nuts = az.ess(breeding_samples_nuts).r.values

print("NUTS ESS:")
print(f"f ESS: {f_ess_nuts}\nr ESS: {r_ess_nuts}")
```

    NUTS ESS:
    f ESS: 4169.674233241658
    r ESS: 4164.192915822109



```python
gibbs_az = az.convert_to_inference_data(out)
f_ess_gibbs = az.ess(gibbs_az).f.values
r_ess_gibbs = az.ess(gibbs_az).r.values

print("Gibbs ESS:")
print(f"f ESS: {f_ess_gibbs}\nr ESS: {r_ess_gibbs}")
```

    Gibbs ESS:
    f ESS: 2070.7368432954504
    r ESS: 7592.132131705443


Our Gibbs sampler was a bit of a mixed bag here. Interestingly, the devleopers of PyMC say that Gibbs sampling tools are a [low priority](https://github.com/pymc-devs/pymc/issues/736) for them compared to Hamiltonian methods like NUTS. Typically, Gibbs yields fewer effective samples per second. Blog post coming soon on that too.

But notice that the NUTS sampler yielded a much higher ESS than the Metropolis sampler. Let's compare the effective samples *per second* for each of our samplers:


```python
print(f"Metropolis f ESS/sec: {f_ess/3}\nNuts f ESS/sec: {f_ess_gibbs/5}\nGibbs f ESS/sec: {f_ess_gibbs}")
```

    Metropolis f ESS/sec: 378.1259431188962
    Nuts f ESS/sec: 414.1473686590901
    Gibbs f ESS/sec: 2070.7368432954504


Our Gibbs sampler performed very well here, but it's interesting to note that even though the NUTS sampler is slower and more complicated than the Metropolis sampler, it yielded a higher number of effective samples per second.