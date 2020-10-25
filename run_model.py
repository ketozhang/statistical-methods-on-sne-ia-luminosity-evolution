import numpy as np
import argparse
from tqdm import tqdm

from data.dataloader import (clean_data, load_age_sample_from_mcmc_chains,
                             load_hr)
from models.pymc3_models import LinearModel

DATASETS = {
    "local": "campbell",
    "global": "campbellG"
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=["local", "global"])
    args = parser.parse_args()

    np.random.seed(1032020)
    age_df = load_age_sample_from_mcmc_chains(
        DATASETS[args.dataset], mode="read")
    hr_df = load_hr(DATASETS[args.dataset])
    age_df, hr_df = clean_data(age_df, hr_df)
    age_matrix = np.array(age_df.groupby("snid").agg(list)["age"].tolist())

    lm = LinearModel(args.dataset)
    trace, model = lm.fit(
        hr_df["hr"],
        hr_df["hr_err"],
        age_matrix,
        sample_kwargs=dict(
            draws=25000, tune=5000, discard_tuned_samples=True, chains=1
        )
    )

    for varname in tqdm(trace.varnames, desc="Saving results"):
        result = []
        for i in range(trace.nchains):
            result.append(trace.get_values(varname, chains=[i]))

        result = np.array(result)
        if "_interval__" not in varname:
            np.savez(f"results/pymc3/{varname}_{lm.name}.npz", result)
