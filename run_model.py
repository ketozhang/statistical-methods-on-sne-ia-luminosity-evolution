import numpy as np
from tqdm import tqdm

from data.dataloader import (clean_data, load_age_sample_from_mcmc_chains,
                             load_hr)
from models.pymc3_models import LinearModel

if __name__ == "__main__":
    np.random.seed(1032020)
    age_df = load_age_sample_from_mcmc_chains("campbell", mode="read")
    hr_df = load_hr("campbell")
    age_df, hr_df = clean_data(age_df, hr_df)
    age_matrix = np.array(age_df.groupby("snid").agg(list)["age"].tolist())

    lm = LinearModel('local')
    trace, model = lm.fit(
        hr_df["hr"],
        hr_df["hr_err"],
        age_matrix,
        sample_kwargs=dict(
            draws=25000, tune=5000, discard_tuned_samples=True, chains=1
        ),
    )

    for varname in tqdm(trace.varnames, desc="Saving results"):
        result = []
        for i in range(trace.nchains):
            result.append(trace.get_values(varname, chains=[i]))

        result = np.array(result)
        if "_interval__" not in varname:
            np.savez(f"results/pymc3/{varname}_{lm.name}.npz", result)
