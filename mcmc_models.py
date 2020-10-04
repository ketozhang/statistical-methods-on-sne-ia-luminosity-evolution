import pickle
import sys

import emcee
import numpy
import pandas as pd
from sklearn.neighbors import KernelDensity
from tqdm import tqdm

from dataloader import *
from linmix import LinMix
from multiprocessing import Pool
from scipy import integrate

np.random.seed(20200918)


def linear_model(x, slope, intercept):
    return slope * x + intercept


def log_lnl_chi2(theta, x, y, yerr):
    intercept, slope, scatter = theta
    ypred = linear_model(x, slope, intercept)
    var = (yerr ** 2) + (scatter ** 2)

    residual = y - ypred

    return -0.5 * np.sum(((residual ** 2) / var) + np.log(2 * np.pi * var))


def log_lnl_with_x_posterior(theta, x, y, yerr, x_kdes):
    """Similar to the chi-2 likelihood but we assume x is a random variable.

    Product of the joint probability of x and y which is an integral where P(y|x') is Gaussian
    with mean at y_hat(x') and variance yerr^2 + scatter^2 and P(x') is the posterior probability of x=x'
    integrated over dx'.
    """
    intercept, slope, scatter = theta

    def y_given_x_pdf(x, y, yerr):
        """Returns a single value, the log probability of y given x"""
        ypred = linear_model(x, slope, intercept)
        var = (yerr ** 2) + (scatter ** 2)
        return stats.norm(loc=ypred, scale=var).pdf(y)

    lnl = 0
    for i in range(x.shape[0]):
        x_pdf = lambda _x: np.exp(x_kdes[i].score(np.array([_x]).reshape(1, -1)))

        integrand = lambda _x: y_given_x_pdf(_x, y[i], yerr[i]) * x_pdf(_x)
        prob, _ = integrate.quad(integrand, 0, 20)
        lnl += np.log(prob)

    return lnl if np.isfinite(lnl) else -np.inf


def log_posterior(theta, x, y, yerr, log_lnl_func=log_lnl_chi2, log_prior_func=None):
    """Return the posterior probability.
    By default uniform prior is assumed on all free parameters
    """
    x_kdes = []
    for i in range(x.shape[0]):
        x_kde = KernelDensity().fit(x[i].reshape(len(x[i]), -1))
        x_kdes.append(x_kde)

    lnl = log_lnl_func(theta, x, y, yerr, x_kdes)
    if log_prior_func is not None:
        raise NotImplementedError

    return lnl


def run_x_posterior_mcmc(x, y, yerr, nwalkers=25, nsteps=5000, sampler_kwargs={}):
    """[summary]

    Args:
        x (array-like; n x m): Sample of the posterior distribution of sample size m
          for each of n rows.
        y (array-like; n x 1): Observed y values of size n.
    """
    if isinstance(x, pd.Series):
        x = x.values
    if isinstance(y, pd.Series):
        y = y.values
    if isinstance(yerr, pd.Series):
        yerr = yerr.values

    # Start with OLS solution as a guess
    xmean = np.mean(x, axis=1)
    slope_guess = stats.pearsonr(xmean, y)[0] * np.std(y) / np.std(xmean)
    intercept_guess = np.mean(y) - (slope_guess * np.mean(xmean))
    guess = np.array([intercept_guess, slope_guess, 1])
    ndim = len(guess)

    # Let the initial point sample from a Gaussian with standard deviation
    # that's 1/2 of the guess value such that about 95% of the initial position
    # starts within the range (-2*guess, 2*guess)
    pos = guess + ((guess / 2) * np.random.randn(nwalkers, ndim))
    sampler = emcee.EnsembleSampler(
        nwalkers,
        ndim,
        log_posterior,
        args=(x, y, yerr),
        kwargs=dict(log_lnl_func=log_lnl_with_x_posterior),
        **sampler_kwargs,
    )
    sampler.run_mcmc(pos, nsteps, progress=VERBOSE)

    theta_estimate = sampler.get_chain().reshape((nsteps * nwalkers, ndim)).mean(axis=0)
    if VERBOSE:
        print(
            f"ESTIMATED PARAMTERS\nSlope: {theta_estimate[1]}\nIntercept: {theta_estimate[0]}\nScatter: {theta_estimate[2]}"
        )
    return sampler


def run_chi2_mcmc(x, y, yerr, nwalkers=25, nsteps=5000, sampler_kwargs={}):
    # Start with OLS solution as a guess
    slope_guess = stats.pearsonr(x, y)[0] * np.std(y) / np.std(x)
    intercept_guess = np.mean(y) - (slope_guess * np.mean(x))
    guess = np.array([intercept_guess, slope_guess, 1])
    ndim = len(guess)

    # Let the initial point sample from a Gaussian with standard deviation
    # that's 1/2 of the guess value such that about 95% of the initial position
    # starts within the range (-2*guess, 2*guess)
    pos = guess + ((guess / 2) * np.random.randn(nwalkers, ndim))
    sampler = emcee.EnsembleSampler(
        nwalkers, ndim, log_posterior, args=(x, y, yerr), **sampler_kwargs
    )
    sampler.run_mcmc(pos, nsteps, progress=VERBOSE)

    theta_estimate = sampler.get_chain().reshape((nsteps * nwalkers, ndim)).mean(axis=0)
    if VERBOSE:
        print(
            f"ESTIMATED PARAMTERS\nSlope: {theta_estimate[1]}\nIntercept: {theta_estimate[0]}\nScatter: {theta_estimate[2]}"
        )
    return sampler


def run_linmix(age, age_err, hr, hr_err, linmix_kwargs={}, run_kwargs={}):
    """Estimated mock of Lee20 result."""

    lm = LinMix(x=age, y=hr, xsig=age_err, ysig=hr_err, **linmix_kwargs)
    lm.run_mcmc(silent=(not VERBOSE), **run_kwargs)

    results = pd.DataFrame({"slope": lm.chain["beta"], "intercept": lm.chain["alpha"]})
    return results


if __name__ == "__main__":
    VERBOSE = "-v" in sys.argv[1:]

    # # With summary statistics
    # data = pd.read_csv("data/HRvsAge_Median+STD+Bounds.csv")
    # age = data["Age_global"]
    # age_err = data["Age_global_err"]
    # hr = data["HR"]
    # hr_err = data["HR_err"]
    # results = run_linmix(age, age_err, hr, hr_err, linmix_kwargs=dict(K=2))
    # results.to_csv("results/linmix_1.csv", index=False)

    # # With real data
    # age_df = load_age_sample_from_mcmc_chains("campbell", mode="read")
    # hr_df = load_hr("campbell")
    # age_df, hr_df = clean_data(age_df, hr_df)

    # age = age_df["age"].groupby("snid").mean()
    # age_err = age_df["age"].groupby("snid").std()
    # hr = hr_df["hr"]
    # hr_err = hr_df["hr_err"]
    # results = run_linmix(age, age_err, hr, hr_err)
    # results.to_csv("results/linmix_2.csv", index=False)

    # # With real data but use the whole Age distribution
    # # Assume each age datapoint is a delta function with zero variance
    # age_df = load_age_sample_from_mcmc_chains("campbell", mode="read")
    # hr_df = load_hr("campbell")
    # age_df, hr_df = clean_data(age_df, hr_df)

    # df = age_df.merge(hr_df, on="snid", how="left")
    # age = df["age"]
    # age_err = np.zeros(len(age))
    # hr = df["hr"]
    # hr_err = df["hr_err"]

    # with Pool() as pool:
    #     sampler = run_chi2_mcmc(
    #         age, hr, hr_err, nsteps=1000, sampler_kwargs=dict(pool=pool)
    #     )
    # with open("results/chi2-mcmc-sampler.pkl", "wb") as f:
    #     pickle.dump(sampler.get_chain(), f)

    # results = run_linmix(age, age_err, hr, hr_err, linmix_kwargs=dict(K=1))
    # results.to_csv("results/linmix_3.csv", index=False)

    # Create all possible slope given each Age from its posterior sample
    # age_df = load_age_sample_from_mcmc_chains("campbell", mode="read")
    # hr_df = load_hr("campbell")
    # age_df, hr_df = clean_data(age_df, hr_df)

    # slopes = []
    # intercepts = []
    # for i in tqdm(range(1000), total=1000):
    #     age = age_df.groupby("snid").sample(1)["age"]
    #     hr = hr_df["hr"]
    #     hr_err = hr_df["hr_err"]

    #     with Pool() as pool:
    #         sampler = run_chi2_mcmc(
    #             age,
    #             hr,
    #             hr_err,
    #             nsteps=3000,
    #             nwalkers=10,
    #             sampler_kwargs=dict(pool=pool),
    #         )

    #     intercept, slope, _ = sampler.get_chain(flat=True, discard=1000).mean(axis=0)
    #     intercepts.append(intercept)
    #     slopes.append(slope)

    # pd.DataFrame({"slope": slopes, "intercept": intercepts}).to_csv(
    #     "results/chi2-mcmc-sampler.csv"
    # )

    # slopes = []
    # intercepts = []
    # for i in tqdm(range(1000)):
    #     age = age_df.groupby("snid").sample(1)["age"]
    #     age_err = np.zeros(len(age))
    #     hr = hr_df["hr"]
    #     hr_err = hr_df["hr_err"]

    #     results = run_linmix(age, age_err, hr, hr_err, linmix_kwargs=dict(K=2))

    #     slopes.append(results["slope"].mean())
    #     intercepts.append(results["intercept"].mean())
    # pd.DataFrame({"slope": slopes, "intercept": intercepts}).to_csv(
    #     "results/linmix-mcmc-sampler.csv"
    # )

    print(VERBOSE)

    age_df = load_age_sample_from_mcmc_chains("campbell", mode="read")
    hr_df = load_hr("campbell")
    age_df, hr_df = clean_data(age_df, hr_df)

    age_matrix = np.array(list(age_df["age"].groupby("snid").agg(list)))

    with Pool() as pool:
        sampler = run_x_posterior_mcmc(
            age_matrix,
            hr_df["hr"],
            hr_df["hr_err"],
            nwalkers=10,
            nsteps=100,
            sampler_kwargs=dict(pool=pool),
        )

    np.savetxt("results/x_posterior_mcmc.csv", sampler.get_chain(flat=True), delimiter=",")
