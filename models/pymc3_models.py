import numpy as np
import pymc3 as pm


class LinearModel:
    def __init__(self, name):
        self.name = name
        self.gmm = None

    def fit(self, y, yerr, x, gmm_params, sample_kwargs={}):
        """
        yerr :
            The uncertainties (standard deviations) in measured y.
        """
        x = np.asarray(x)
        y = np.asarray(y)
        yerr = np.asarray(yerr)

        # Left merge dictionary
        default_sample_kwargs = dict(draws=1000, tune=1000, chains=2)
        sample_kwargs = {**default_sample_kwargs, **sample_kwargs}

        nsamples = y.shape[0]

        assert gmm_params.shape[1] % 3 == 0, "GMM params shape 1 must multiple of 3."
        k = gmm_params.shape[1] // 3
        mu_x = gmm_params[:, :k]
        sigma_x = gmm_params[:, k:2*k]
        weights_x = gmm_params[:, 2*k:3*k]

        with pm.Model() as model:  # noqa
            # Priors
            intercept = pm.Uniform("intercept", -1, 1)
            slope = pm.Uniform("slope", -1, 0)
            scatter_sigma = pm.HalfNormal("scatter", 0.2)

            components = []
            for i in range(k):
                component = pm.Normal.dist(mu=mu_x[:, i], sigma=sigma_x[:, i], shape=nsamples)
                components.append(component)
            x = pm.Mixture("x", w=weights_x, comp_dists=components, shape=nsamples)

            # Likelihoods
            # Constant value for true y which served as the mean value of the observed y
            y_true = pm.Deterministic("y_true", slope * x + intercept)
            # Standard deviation of observed y as Gaussian errors
            y_sigma = pm.Deterministic("sigma", pm.math.sqrt(yerr**2 + scatter_sigma**2))
            # Altogether, observed y is normally distributed
            y_ = pm.Normal("y", mu=y_true, sigma=y_sigma, observed=y)

            trace = pm.sample(**sample_kwargs)

        return trace, model

    def run_model(self, *args, **kwargs):
        return self.fit(*args, **kwargs)
