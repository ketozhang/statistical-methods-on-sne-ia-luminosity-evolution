import re
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.io import ascii
from scipy import stats

RNG = np.random.RandomState(952020)
DATAPATH = Path("./data")
RESULTSPATH = Path("./results")
FIGURESPATH = Path("./figures")


def load_age_sample_from_mcmc_chains(dataset, mode="read", **kwargs):
    """Return a random sample of the mcmc chains dataset for all SNe available.

    dataset : str {"campbell", "campbellG", "gupta"}
        Dataset to load.
    mode : str {"read", "write"}
        If write, redo the creation of the mcmc chain sample and also write to file.
        WARNING: this may take a long time sampling from ~20GB of data.
        If read, the mcmc chain sample is read from cached output of the "write" mode.
    **kwargs : keyword arguments
        Used when mode="write" and passed onto the function `create_age_sample_from_mcmc_chains`.
    """
    dataset_path = DATAPATH / "mcmc_chains" / dataset

    if mode == "write":
        return _create_age_sample_from_mcmc_chains(dataset, dataset_path, **kwargs)
    if mode == "read":
        return pd.read_table(DATAPATH / f"{dataset}_samples.tsv")


def _create_age_sample_from_mcmc_chains(
    dataset, dataset_path, sample_size=10000, random_state=RNG
):
    dfs = []
    all_dataset_files = list(dataset_path.glob("*.tsv"))
    for i, sn_chain_path in enumerate(all_dataset_files):
        print(f"{i}/{len(all_dataset_files)}", end="\r")
        # Get number of rows in file
        # Header takes up 2 rows
        nheaders = 2
        nrows = sum(1 for line in open(sn_chain_path)) - nheaders
        skiprows = [1] + sorted(
            random_state.choice(
                range(nheaders, nrows + nheaders), nrows - sample_size, replace=False
            )
        )

        _df = pd.read_table(
            sn_chain_path, skiprows=skiprows, usecols=[7], index_col=False
        )

        # Set the index as the SNID parsed from its filename
        snid = re.findall("SN(\d+)_", sn_chain_path.name)[0]
        _df["snid"] = [snid] * len(_df)
        dfs.append(_df)
    print(f"{len(all_dataset_files)}/{len(all_dataset_files)}", end="\r")

    df = pd.concat(dfs)[["snid", "age"]]
    df.to_csv(DATAPATH / f"{dataset}_samples.tsv", sep="\t", index=False)
    return


def load_hr(dataset):
    """
    dataset : str {"campbell", "campbellG", "gupta"}
        Dataset to load.
    """
    dataset_path = DATAPATH / "campbell_local_r19t1.txt"

    return ascii.read(dataset_path).to_pandas()


def clean_data(age_df, hr_df):
    age_df = age_df.set_index("snid").sort_index()

    hr_df = load_hr("campbell")
    hr_df = hr_df
    hr_df.columns = hr_df.columns.str.lower()
    hr_df = (
        hr_df.rename(columns={"sdss": "snid", "e_hr": "hr_err"})
        .set_index("snid")
        .sort_index()[["hr", "hr_err"]]
    )

    in_age_not_hr = set(age_df.index) - set(hr_df.index)
    in_hr_not_age = set(hr_df.index) - set(age_df.index)
    print("Missing from R19 Table 1 of SNID:", in_age_not_hr)
    print("Missing from Campbell MCMC chains of SNID:", in_hr_not_age)

    print(
        "Resulting data will be an inner join of the two remove all SNID mentioned above"
    )
    age_df = age_df.drop(index=in_age_not_hr)
    hr_df = hr_df.drop(index=in_hr_not_age)
    age_snids = age_df.index.unique().tolist()
    hr_snids = age_df.index.unique().tolist()

    assert set(age_snids) == set(hr_snids)
    return age_df, hr_df
