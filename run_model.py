import pandas as pd
import argparse

import numpy as np
from tqdm import tqdm

from data.dataloader import get_data
from models.gaussian_mixture import GaussianMixture
from models.pymc3_models import LinearModel

DATASETS = {
    "local": "campbell",
    "global": "campbellG"
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=["local", "global"])
    args = parser.parse_args()

    np.random.seed(11012020)
    snids, age_df, hr_df = get_data(DATASETS[args.dataset])
    age_matrix = np.array(age_df.groupby("snid").agg(list)["age"].tolist())

    # GMM fit on Age
    gmm = GaussianMixture(name=args.dataset, k=3)

    try:
        gmm_params_df = pd.read_csv(gmm.params_fpath, index_col="snid")
    except FileNotFoundError:
        gmm_params = gmm.fit_age_posteriors(age_df)
        gmm_params_df = pd.DataFrame(gmm_params).T
        gmm_params_df.to_csv(gmm.params_fpath, index_label="snid")

    # MCMC Linear Fit
    gmm_params_order = ["mean0", "mean1", "mean2", "sigma0", "sigma1", "sigma2", "weight0", "weight1", "weight2"]
    lm = LinearModel(name=args.dataset)
    trace, model = lm.fit(
        hr_df["hr"],
        hr_df["hr_err"],
        age_matrix,
        gmm_params=gmm_params_df[gmm_params_order].to_numpy(),
        sample_kwargs=dict(draws=25000, tune=1000, chains=4)
    )

    for varname in tqdm(trace.varnames, desc="Saving results"):
        result = []
        for i in range(trace.nchains):
            result.append(trace.get_values(varname, chains=[i]))

        result = np.array(result)
        if varname in ["slope", "intercept", "scatter_sigma"]:
            np.savez_compressed(
                f"results/pymc3/{varname}_{lm.name}.npz", result)
            print(f"Saved results/pymc3/{varname}_{lm.name}.npz")
        else:
            print(f"Skip saving {varname}...")
