import numpy as np
from scipy.optimize import curve_fit
from uncertainties import ufloat
from uncertainties import unumpy as unp


def mean(x):
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        raise ValueError("x must not be empty.")
    return np.mean(x)


def std_sample(x):
    x = np.asarray(x, dtype=float)
    if x.size < 2:
        raise ValueError("Need at least two values.")
    return np.std(x, ddof=1)


def sem(x):
    x = np.asarray(x, dtype=float)
    if x.size < 2:
        raise ValueError("Need at least two values.")
    return std_sample(x) / np.sqrt(len(x))


def mean_result(x):
    x = np.asarray(x, dtype=float)
    return ufloat(mean(x), sem(x))


def combine_uncertainties(*u):
    u = np.asarray(u, dtype=float)
    return np.sqrt(np.sum(u**2))


def relative_uncertainty(value, uncertainty):
    value = float(value)
    uncertainty = float(uncertainty)
    if value == 0:
        raise ValueError("value must not be zero.")
    return abs(uncertainty / value)


def resolution_uncertainty(a):
    a = float(a)
    return a / np.sqrt(12)


def triangular_uncertainty(a):
    a = float(a)
    return a / np.sqrt(6)


def weighted_mean(values, sigmas):
    values = np.asarray(values, dtype=float)
    sigmas = np.asarray(sigmas, dtype=float)

    if values.shape != sigmas.shape:
        raise ValueError("values and sigmas must have the same shape.")
    if np.any(sigmas <= 0):
        raise ValueError("All sigmas must be positive.")

    w = 1 / sigmas**2
    m = np.sum(w * values) / np.sum(w)
    dm = np.sqrt(1 / np.sum(w))
    return ufloat(m, dm)


def covariance_empirical(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if x.shape != y.shape:
        raise ValueError("x and y must have the same shape.")
    if len(x) < 2:
        raise ValueError("Need at least two values.")

    return np.cov(x, y, ddof=1)[0, 1]


def correlation_coefficient(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if x.shape != y.shape:
        raise ValueError("x and y must have the same shape.")
    if len(x) < 2:
        raise ValueError("Need at least two values.")

    return np.corrcoef(x, y)[0, 1]


def chi2(y_obs, y_exp, sigmas):
    y_obs = np.asarray(y_obs, dtype=float)
    y_exp = np.asarray(y_exp, dtype=float)
    sigmas = np.asarray(sigmas, dtype=float)

    if y_obs.shape != y_exp.shape or y_obs.shape != sigmas.shape:
        raise ValueError("All inputs must have the same shape.")
    if np.any(sigmas <= 0):
        raise ValueError("All sigmas must be positive.")

    return np.sum(((y_obs - y_exp) / sigmas) ** 2)


def degrees_of_freedom(n_points, n_parameters):
    dof = int(n_points) - int(n_parameters)
    if dof <= 0:
        raise ValueError("Degrees of freedom must be positive.")
    return dof


def chi2_reduced(chi2_value, n_points, n_parameters):
    dof = degrees_of_freedom(n_points, n_parameters)
    return chi2_value / dof


def linear_model(x, m, c):
    return m * x + c


def linear_regression(x, y, sigma=None, absolute_sigma=True):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if x.shape != y.shape:
        raise ValueError("x and y must have the same shape.")
    if len(x) < 2:
        raise ValueError("Need at least two points.")

    if sigma is not None:
        sigma = np.asarray(sigma, dtype=float)
        if sigma.shape != x.shape:
            raise ValueError("sigma must have same shape as x and y.")
        if np.any(sigma <= 0):
            raise ValueError("All sigma values must be positive.")

    popt, pcov = curve_fit(
        linear_model,
        x,
        y,
        sigma=sigma,
        absolute_sigma=absolute_sigma
    )

    m, c = popt
    dm, dc = np.sqrt(np.diag(pcov))

    y_fit = linear_model(x, m, c)

    if sigma is None:
        residual = y - y_fit
        s_res = np.sqrt(np.sum(residual**2) / (len(x) - 2))
        chi2_val = np.sum((residual / s_res) ** 2)
    else:
        chi2_val = chi2(y, y_fit, sigma)

    chi2_red = chi2_reduced(chi2_val, len(x), 2)

    return {
        "m": ufloat(m, dm),
        "c": ufloat(c, dc),
        "popt": popt,
        "pcov": pcov,
        "y_fit": y_fit,
        "chi2": chi2_val,
        "chi2_red": chi2_red,
        "dof": degrees_of_freedom(len(x), 2)
    }


def fit_curve(model, x, y, sigma=None, p0=None, absolute_sigma=True):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if sigma is not None:
        sigma = np.asarray(sigma, dtype=float)
        if sigma.shape != x.shape:
            raise ValueError("sigma must have same shape as x and y.")
        if np.any(sigma <= 0):
            raise ValueError("All sigma values must be positive.")

    popt, pcov = curve_fit(
        model,
        x,
        y,
        sigma=sigma,
        p0=p0,
        absolute_sigma=absolute_sigma
    )

    perr = np.sqrt(np.diag(pcov))
    params_u = [ufloat(v, dv) for v, dv in zip(popt, perr)]

    y_fit = model(x, *popt)

    if sigma is None:
        npar = len(popt)
        residual = y - y_fit
        chi2_val = np.sum(residual**2)
        chi2_red = chi2_val / (len(x) - npar)
    else:
        chi2_val = chi2(y, y_fit, sigma)
        chi2_red = chi2_reduced(chi2_val, len(x), len(popt))

    return {
        "params": params_u,
        "popt": popt,
        "pcov": pcov,
        "y_fit": y_fit,
        "chi2": chi2_val,
        "chi2_red": chi2_red,
        "dof": degrees_of_freedom(len(x), len(popt))
    }


def propagate_product_quotient(value, exponents, values, uncertainties):
    exponents = np.asarray(exponents, dtype=float)
    values = np.asarray(values, dtype=float)
    uncertainties = np.asarray(uncertainties, dtype=float)

    if exponents.shape != values.shape or values.shape != uncertainties.shape:
        raise ValueError("All inputs must have the same shape.")
    if np.any(values == 0):
        raise ValueError("values must be non-zero.")

    rel = np.sqrt(np.sum((exponents * uncertainties / values) ** 2))
    return abs(value) * rel


def adc_uncertainty(delta, n_bits):
    return float(delta) / (2**int(n_bits) * np.sqrt(12))


def residuals(x, y, model, *params):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    return y - model(x, *params)


def make_ufloat(value, uncertainty):
    return ufloat(value, uncertainty)


def nominal(x):
    return x.n if hasattr(x, "n") else unp.nominal_values(x)


def stddev(x):
    return x.s if hasattr(x, "s") else unp.std_devs(x)