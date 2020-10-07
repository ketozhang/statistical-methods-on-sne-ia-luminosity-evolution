import os
import numpy as np
import pandas as pd
import pymc3 as pm
from scipy import stats
from sklearn.mixture import GaussianMixture
from tqdm import tqdm, trange
from pathlib import Path


def get_gmm_params(x, k=3, cache=False):
    gmm_cache_path = Path(f"results/gmm_{k}_results.csv").resolve()

    if cache and gmm_cache_path.exists():
        results = pd.read_csv(gmm_cache_path).to_numpy()
        return results[:, 0], results[:, 1], results[:, 0]
        
    gmms = []
    for i in trange(x.shape[0], desc="GMM Fit"):
        gmm = GaussianMixture(n_components=k, covariance_type='spherical') \
              .fit(x[i].reshape(len(x[i]), -1))
        gmms.append(gmm)
        
    mu_x = np.array([gmm.means_ for gmm in gmms])
    sigma_x = np.array([gmm.covariances_ for gmm in gmms])
    weights_x = np.array([gmm.weights_ for gmm in gmms])
    
    if cache:
        results = np.stack((mu_x, sigma_x, weights_x))
        np.save_txt(results, delmiter=',')
    
    return mu_x, sigma_x, weights_x

def run_model(y, yerr, x, k=3, sample_kwargs={}):
    x = np.asarray(x)
    y = np.asarray(y)
    yerr = np.asarray(yerr)
    
    _sample_kwargs = dict(draws=1000, tune=1000, chains=2)
    _sample_kwargs.update(sample_kwargs)
    
    nsamples = y.shape[0]

    mu_x, sigma_x, weights_x = get_gmm_params(x, k)
    
    with pm.Model() as model:
        # Priors
        intercept = pm.Normal('intercept', mu=0, sigma=2) # [0, ~5]
        slope = pm.Uniform('slope', -0.1, 0)
        scatter = pm.HalfNormal('scatter', 2)

        components = []
        for i in range(k):
            component = pm.Normal.dist(mu=mu_x[:,i], sigma=sigma_x[:,i], shape=nsamples)
            components.append(component)
        x = pm.Mixture('x', w=weights_x, comp_dists=components, shape=nsamples)
        sigma = pm.Deterministic('sigma', pm.math.sqrt(yerr**2 + scatter))

    #     likelihood_x = pm.Normal('x', mu=x.mean(axis=1), sigma=x.std(axis=1), shape=y.shape[0])

        _y_true = pm.Deterministic('y_true', slope * x + intercept)
        sigma_y = pm.Uniform('yerr', 0, 1, observed=yerr, shape=nsamples)
        likelihood_y = pm.Normal('y', mu=_y_true, sigma=yerr, observed=y)

        trace = pm.sample(**_sample_kwargs)
        
    return trace, model
