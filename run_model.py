import numpy
import pymc3 as pm 
from models.pymc3_models import run_model as run_model_pymc3
from tqdm import tqdm

from dataloader import *


np.random.seed(1032020)


if __name__ == "__main__":
    age_df = load_age_sample_from_mcmc_chains("campbell", mode="read")
    hr_df = load_hr("campbell")
    age_df, hr_df = clean_data(age_df, hr_df)
    age_matrix = np.array(age_df.groupby('snid').agg(list)['age'].tolist())
    
    trace, model = run_model_pymc3(
        hr_df['hr'], hr_df['hr_err'], age_matrix,
        sample_kwargs=dict(draws=25000, tune=10000, discard_tuned_samples=False, chains=4)
    )
    
    for varname in tqdm(trace.varnames, desc="Saving results"):
        
        result = []
        for i in range(trace.nchains):
            result.append(trace.get_values(varname, chains=[i]))
        
        result = np.array(result)   
        if result.ndim <= 2:
            np.savetxt(f"results/pymc3/{varname}.gz", np.array(result))
