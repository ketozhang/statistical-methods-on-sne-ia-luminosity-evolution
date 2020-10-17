import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn import mixture
from tqdm import trange


class GaussianMixture:
    def __init__(self, name):
        self.name = name
        self.param_fpath = Path(
            f"results/gmm_age_posterior_fit_params_{self.name}.csv")
        self.results_fpath = Path(
            f"results/gmm_age_posterior_fit_results_{self.name}.pkl"
        )

        self._k = None

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

        # df.to_csv(self.param_fpath, index_label="snid")
        # return results

    def fit(self, x, index=None, n_components=3, save=True):
        self._k = n_components

        if self.results_fpath.exists() and self.param_fpath.exists():
            logging.warning(
                f"""{self.__class__.__name__} save found in files:
                    {self.param_fpath}
                    {self.results_fpath}

                skipping fit...
                """
            )
            return

        gmms = []
        for i in trange(x.shape[0], desc="GMM Fit"):
            gmm = mixture.GaussianMixture(
                n_components=n_components, covariance_type="spherical"
            ).fit(x[i].reshape(len(x[i]), -1))
            gmms.append(gmm)

        mu_x = (np.array([gmm.means_ for gmm in gmms])
                .reshape(-1, self._k)
                )
        sigma_x = np.array([gmm.covariances_ for gmm in gmms])
        weights_x = np.array([gmm.weights_ for gmm in gmms])

        results = pd.DataFrame(
            np.hstack((mu_x, sigma_x, weights_x)),
            index=index,
            columns=[f"mean{i}" for i in range(1, self._k + 1)]
            + [f"sigma{i}" for i in range(1, self._k + 1)]
            + [f"weight{i}" for i in range(1, self._k + 1)]
        )

        if save:
            # Save the GMM results object
            with self.results_fpath.open("wb") as f:
                pickle.dump({i: gmm for i, gmm in zip(index, gmms)}, f)

            # Save the GMM params
            results.to_csv(self.param_fpath)
        else:
            return results

    def get_params(self):
        return pd.read_csv(self.param_fpath, index_col=0)
        # params = np.genfromtxt(self.param_fpath, delimiter=",")
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

    def get_results(self):
        with self.results_fpath.open("rb") as f:
            return pickle.load(f)
