from pathlib import Path

import numpy as np
import pandas as pd
from sklearn import mixture
from tqdm import tqdm


class GaussianMixture:
    def __init__(self, name, k=3):
        self.name = name
        self.params_fpath = Path(
            f"results/gmm_age_posterior_fit_params_{self.name}.csv")
        self.results_fpath = Path(
            f"results/gmm_age_posterior_fit_results_{self.name}.pkl")
        self.k = k

    def fit_age_posteriors(self, age_df, **kwargs):
        # snids = age_df.index.unique()
        # results = {}
        series = age_df["age"].groupby("snid").apply(list)

        snid_gmm_params = {}
        for snid, age_posterior in tqdm(series.iteritems(), total=len(series), desc="Fitting age posteriors"):
            params = self.fit(age_posterior, **kwargs)
            snid_gmm_params[snid] = params

        return snid_gmm_params

    def fit(self, x, **kwargs):
        x = np.asarray(x)
        gmm = mixture.GaussianMixture(
            n_components=self.k, covariance_type="spherical", **kwargs)
        gmm.fit(x.reshape(len(x), -1))

        params = {}
        for i in range(self.k):
            params[f"mean{i}"] = gmm.means_.reshape(self.k)[i]
            params[f"sigma{i}"] = np.sqrt(gmm.covariances_[i])
            params[f"weight{i}"] = gmm.weights_[i]

        return params

    # def save(self):
    #     assert self.gmms is not None, "No results can be saved before fit is ran."

    #     # # Save GMM object
    #     # with self.results_fpath.open("wb") as f:
    #     #     pickle.dump(self.gmms, f)
    #     # print(f"Saved successful {self.results_fpath}")

    #     # Save GMM params
    #     params = self.get_params()
    #     params.to_csv(self.params_fpath, index=False)
    #     print(f"Saved successful {self.params_fpath}")

    # def load(self, fpath=None):
    #     # Load GMM object
    #     fpath = fpath or self.results_fpath
    #     with fpath.open("rb") as f:
    #         self.gmms = pickle.load(f)

    # def get_results(self):
    #     return self.gmms

    # def get_params(self):
    #     params = {
    #         "y": [],
    #         "mu_x":  [],
    #         "sigma_x":  [],
    #         "weights_x":  []
    #     }
    #     for y, gmm in self.gmms.items():
    #         params["y"].append(y)
    #         params["mu_x"].append(gmm.means_)
    #         params["sigma_x"].append(gmm.covariances_)
    #         params["weights_x"].append(gmm.weights_)

        # params["mu_x"] = np.array(params["mu_x"]).reshape(-1, self.k)
        # params["sigma_x"] = np.array(params["sigma_x"])
        # params["weights_x"] = np.array(params["sigma_x"])

    #     params = pd.DataFrame(
    #         np.hstack(
    #             (params["y"], params["mu_x"], params["sigma_x"], params["weights_x"])),
    #         columns=(
    #             "y",
    #             [f"mean{i}" for i in range(1, self.k + 1)] +
    #             [f"sigma{i}" for i in range(1, self.k + 1)] +
    #             [f"weight{i}" for i in range(1, self.k + 1)]
    #         ),
    #     )
    #     return params
        # params = np.genfromtxt(self.params_fpath, delimiter=",")
        # if format == "numpy":
        #     return params
        # elif format == "dataframe":
        #     k = self.k
        #     # mu_x = params[:, :k]
        #     # sigma_x = params[:, k:2*k]
        #     # weights_x = params[:, 2*k:3*k]

        #     # Save the GMM parameters
        #     params_df = pd.DataFrame(params, )

        #     return params_df
