import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn import mixture
from tqdm import trange
import textwrap


class GaussianMixture:
    def __init__(self, name):
        self.name = name
        self.params_fpath = Path(
            f"results/gmm_age_posterior_fit_params_{self.name}.csv")
        self.results_fpath = Path(
            f"results/gmm_age_posterior_fit_results_{self.name}.pkl")

        # Post-fit attribute
        self._k = None
        self.gmms = None

    def fit_age_posterior(self, age_df, **kwargs):
        # snids = age_df.index.unique()
        # results = {}
        data = age_df["age"].groupby("snid").apply(list)
        snids = data.index
        self.fit(
            np.array([_ for _ in data]),
            index=snids,
            **kwargs
        )
        # for i in trange(len(snids), desc="GMM Fit"):
        #     snid = snids[i]
        #     x = age_df.loc[snid, :].to_numpy()
        #     gmm = mixture.GaussianMixture(
        #         n_components=n_components, covariance_type="spherical"
        #     ).fit(x)
        #     results[snid] = gmm

        # df.to_csv(self.params_fpath, index_label="snid")
        # return results

    def fit(self, x, n_components=3):
        self._k = n_components

        self.gmms = []
        for i in trange(x.shape[0], desc="GMM Fit"):
            gmm = mixture.GaussianMixture(
                n_components=n_components, covariance_type="spherical"
            ).fit(x[i].reshape(len(x[i]), -1))
            self.gmms.append(gmm)

    def get_results(self):
        return self.gmms

    def get_params(self):
        mu_x = (np.array([gmm.means_ for gmm in self.gmms])
                .reshape(-1, self._k)
                )
        sigma_x = np.array([gmm.covariances_ for gmm in self.gmms])
        weights_x = np.array([gmm.weights_ for gmm in self.gmms])

        params = pd.DataFrame(
            np.hstack((mu_x, sigma_x, weights_x)),
            columns=[f"mean{i}" for i in range(1, self._k + 1)]
            + [f"sigma{i}" for i in range(1, self._k + 1)]
            + [f"weight{i}" for i in range(1, self._k + 1)]
        )
        return params
        # params = np.genfromtxt(self.params_fpath, delimiter=",")
        # if format == "numpy":
        #     return params
        # elif format == "dataframe":
        #     k = self._k
        #     # mu_x = params[:, :k]
        #     # sigma_x = params[:, k:2*k]
        #     # weights_x = params[:, 2*k:3*k]

        #     # Save the GMM parameters
        #     params_df = pd.DataFrame(params, )

        #     return params_df
