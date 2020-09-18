Gmport sys
import numpy
import pandas as pd
from dataloader import *
from linmix import LinMix

np.random.seed(20200918)


def run_linmix(age, age_err, hr, hr_err, linmix_kwargs={}, run_kwargs={}):
    """Estimated mock of Lee20 result."""

    lm = LinMix(x=age, y=hr, xsig=age_err, ysig=hr_err, **linmix_kwargs)
    lm.run_mcmc(silent=(not VERBOSE), **run_kwargs)

    results = pd.DataFrame({"slope": lm.chain["alpha"], "intercept": lm.chain["beta"]})
    return results


if __name__ == "__main__":
    VERBOSE = "-v" in sys.argv[1:]

    # With summary statistics
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

    # With real data but use the whole Age distribution
    # Assume each age datapoint is a delta function with zero variance
    age_df = load_age_sample_from_mcmc_chains("campbell", mode="read")
    hr_df = load_hr("campbell")
    age_df, hr_df = clean_data(age_df, hr_df)

    df = age_df.merge(hr_df, on="snid", how="left")
    age = df["age"]
    age_err = np.zeros(len(age))
    hr = df["hr"]
    hr_err = df["hr_err"]
    results = run_linmix(age, age_err, hr, hr_err, linmix_kwargs=dict(K=1))
    results.to_csv("results/linmix_3.csv", index=False)
