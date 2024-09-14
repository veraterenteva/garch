import pandas as pd
import numpy as np
import scipy.optimize as optimize


def calculate_log_returns(data: pd.DataFrame) -> pd.Series:
    """
    Function that adds returns percent change
    :param data: Dataframe from Yfinance
    :return: percent change
    """
    return 100 * np.log(data['Open'] / data['Open'].shift(1))


def calculate_rv(data: pd.DataFrame, timeframe) -> pd.Series:
    """
    Function that calculates rv
    :param data: Dataframe from Yfinance
    :param timeframe: timeframe to calculate rv
    :return: rv
    """
    return data['Log_returns'].rolling(timeframe).std()


def garch_likelihood(parameters: list, data: np.array) -> float:
    """
    Calculating garch_likelihood
    :param parameters: initial params
    :param data: data
    :return: log likelihood
    """
    omega, alpha1, beta1 = parameters
    t = len(data)
    sigma2 = np.zeros(t)
    sigma2[0] = np.var(data)

    for t in range(1, t):
        sigma2[t] = omega + alpha1 * (data[t-1])**2 + beta1 * sigma2[t-1]

    logliks = np.sum(0.5 * (np.log(sigma2) + data ** 2 / sigma2))
    return logliks


def fit_garch(data: np.array) -> list:
    """
    Fitting model
    :param data: data to fit
    :return: optimal params [omega, alpha, beta]
    """
    init_params = [0.5, 0.5, 0.5]
    results = optimize.minimize(garch_likelihood,
                                init_params,
                                args=(data.values),
                                bounds=[(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)],
                                method='Nelder-Mead')
    print(results)
    return results.x


def garch_apply(data, parameters):
    """
    Calculating variance in garch model
    :param data: data to calculate variance
    :param parameters: estimated model params
    :return: variance
    """
    omega, alpha1, beta1 = parameters
    t = len(data)
    sigma2 = np.zeros(t)
    sigma2[0] = np.var(data)
    for i in range(1, t):
        sigma2[i] = omega + alpha1 * data[i - 1] ** 2 + beta1 * sigma2[i - 1]

    return sigma2


def garch_forecast(horizon, data, parameters):
    """
    Forecasting with custom garch model
    :param horizon: steps to forecast
    :param data: data previous
    :param parameters: params of model
    :return: forecast with horizon len
    """
    omega, alpha1, beta1 = parameters
    t = len(data)
    sigma2 = np.zeros(t + horizon)
    sigma2[0:t] = garch_apply(data, parameters)
    for i in range(t, t + horizon):
        sigma2[i] = omega + alpha1 * sigma2[i - 1] + beta1 * sigma2[i - 1]

    return sigma2[t:]
