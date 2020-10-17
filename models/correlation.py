import numpy as np


# def estimate_correlation(age, hr):
#     age_sample_var = age["age"].groupby("snid").var(ddof=1)
#     age_sample_mean = age["age"].groupby("snid").mean()

#     age_variance = np.sum(
#         age_sample_var / len(snids)
#         + (np.mean(age_sample_mean) - age_sample_mean) ** 2 / (len(snids) - 1)
#     )

#     hr_sample_mean = hr["hr"].mean()
#     hr_sample_var = hr["hr"].var(ddof=1)
#     age_hr_prod = age.join(hr)
#     age_hr_prod["age*hr"] = age_hr_prod["age"] * age_hr_prod["hr"]
#     age_hr_prod_mean = age_hr_prod["age*hr"].groupby("snid").mean()

#     age_hr_covariance = np.mean(age_hr_prod_mean) + (
#         np.mean(age_sample_mean) * hr_sample_mean
#     )

#     bias_corr = age_hr_covariance / np.sqrt(
#         (hr_sample_var + np.mean(hr["hr_err"] ** 2)) * age_variance
#     )
#     bias_correction = 1 / np.sqrt(
#         (hr_sample_var - np.mean(hr["hr_err"] ** 2)) / hr_sample_var
#     )
#     return bias_corr, bias_corr * bias_correction


def estimate_correlation(age, hr):
    age_var = np.sum(
        age["var"] / len(age) + (np.mean(age["mean"]) - age["mean"]) / (len(age) - 1)
    )
    _corr = np.cov(age["mean"], hr["hr"])[0, 1] / np.sqrt(
        (hr["hr"].var(ddof=1) + np.mean(hr["hr_err"] ** 2)) * age_var
    )

    return _corr, _corr / np.sqrt(
        1 - (np.mean(hr["hr_err"] ** 2) / hr["hr"].var(ddof=1))
    )
