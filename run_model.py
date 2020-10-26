import numpy as np
import argparse
from tqdm import tqdm

from data.dataloader import (clean_data, load_age_sample_from_mcmc_chains,
                             load_hr)
from models.pymc3_models import LinearModel
from models.gaussian_mixture import GaussianMixture
import pickle

DATASETS = {
    "local": "campbell",
    "global": "campbellG"
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=["local", "global"])
    args = parser.parse_args(["local"])

    np.random.seed(1032020)
    age_df = load_age_sample_from_mcmc_chains(
        DATASETS[args.dataset], mode="read")
    hr_df = load_hr()
    age_df, hr_df = clean_data(age_df, hr_df)
    snids = list(age_df.groupby("snid").groups.keys())
    age_matrix = np.array(age_df.groupby("snid").agg(list)["age"].tolist())

    # GMM fit on Age
    gmm = GaussianMixture(name=args.dataset)
    gmm.fit(age_matrix, n_components=3)
    gmm_params = gmm.get_params()
    gmm_params.to_csv(gmm.params_fpath)
    results = {snid: result for snid, result in zip(snids, gmm.get_results())}
    with gmm.results_fpath.open("wb") as f:
        pickle.dump(results, f)

    # MCMC Linear Fit
    lm = LinearModel(name=args.dataset)
    trace, model = lm.fit(
        hr_df["hr"],
        hr_df["hr_err"],
        age_matrix,
        gmm_params=gmm_params.to_numpy(),
        sample_kwargs=dict(
            draws=25000, tune=5000, discard_tuned_samples=True, chains=4
        )
    )

    for varname in tqdm(trace.varnames, desc="Saving results"):
        result = []
        for i in range(trace.nchains):
            result.append(trace.get_values(varname, chains=[i]))

        result = np.array(result)
        if varname in ["slope", "intercept", "scatter"]:
            print(f"Saving {varname}...")
            np.savez_compressed(
                f"results/pymc3/{varname}_{lm.name}.npz", result)
        else:
            print(f"Skip saving {varname}...")
