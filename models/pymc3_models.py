import numpy as np
import pymc3 as pm


class LinearModel:
    def __init__(self, name):
        self.name = name
        self.gmm = None

    def fit(self, y, yerr, x, gmm_params, sample_kwargs={}):
        x = np.asarray(x)
        y = np.asarray(y)
        yerr = np.asarray(yerr)

        _sample_kwargs = dict(draws=1000, tune=1000, chains=2)
        _sample_kwargs.update(sample_kwargs)

        nsamples = y.shape[0]

        assert gmm_params.shape[1] % 3 == 0, \
            "GMM params shape 1 must multiple of 3."
        k = gmm_params.shape[1] / 3
        mu_x = gmm_params[:, :k]
        sigma_x = gmm_params[:, k:2*k]
        weights_x = gmm_params[:, 2*k:3*k]

        with pm.Model() as model:
            # Priors
            intercept = pm.Uniform("intercept", -1, 1)
            slope = pm.Uniform("slope", -1, 0)
            scatter = pm.HalfNormal("scatter", 2)

            components = []
            for i in range(k):
                component = pm.Normal.dist(
                    mu=mu_x[:, i], sigma=sigma_x[:, i], shape=nsamples
                )
                components.append(component)
            x = pm.Mixture("x", w=weights_x,
                           comp_dists=components, shape=nsamples)
            sigma = pm.Deterministic("sigma", pm.math.sqrt(yerr ** 2 + scatter))  # noqa

            # likelihood_x = pm.Normal('x',
            #                          mu=x.mean(axis=1),
            #                          sigma=x.std(axis=1),
            #                          shape=y.shape[0])

            _y_true = pm.Deterministic("y_true", slope * x + intercept)
            sigma_y = pm.Uniform("yerr", 0, 1, observed=yerr, shape=nsamples)  # noqa
            likelihood_y = pm.Normal("y", mu=_y_true, sigma=yerr, observed=y)  # noqa

            trace = pm.sample(**_sample_kwargs)

        return trace, model

    def run_model(self, *args, **kwargs):
        return self.fit(*args, **kwargs)
