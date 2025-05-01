# Imports
import psycopg2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import seaborn as sns
import plotly.express as px
from scipy import stats
from scipy.stats import shapiro, boxcox, skew
from pymannkendall import original_test
from statsmodels.tsa.stattools import adfuller, acf, pacf  
from statsmodels.graphics.tsaplots import plot_pacf, plot_acf
from statsmodels.stats.diagnostic import acorr_ljungbox
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler  
from sklearn.metrics import median_absolute_error, mean_squared_error 
import statsmodels.api as sm
from statsmodels.tsa.statespace.sarimax import SARIMAX  
import torch
import torch.nn as nn 
import tensorflow as tf
from tensorflow import keras
from typing import Tuple
import joblib
from datetime import timedelta

import warnings
warnings.filterwarnings("ignore")

# Additional setup from the prompt
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


# Helper functions

def normality_shapiro(data: pd.DataFrame, column: str, alpha: float = 0.05) -> float:
    """
    Perform Shapiro-Wilk test for normality on a given column of a DataFrame.
    """
    stats_val, pval = shapiro(data[column])
    result = "Normal distribution" if pval > alpha else "Non-normal distribution"
    print(f"Shapiro-Wilk Test: Statistic={stats_val:.3f}, p={pval:.3f}")
    print(result)
    return pval


def identify_nulls(df: pd.DataFrame, column: str) -> Tuple[pd.DataFrame, str]:
    """
    Identify and classify missing values in a column as isolated, consecutive, or long gaps.
    """
    df_copy = df.copy()
    df_copy['is_na'] = df_copy[column].isna()
    df_copy['group'] = df_copy['is_na'].ne(df_copy['is_na'].shift()).cumsum()
    
    null_counts = df_copy[df_copy['is_na']].groupby('group').size()
    
    isolated_nulls = null_counts[null_counts == 1].index
    consecutive_nulls = null_counts[(null_counts > 1) & (null_counts <= 7)].index
    long_nulls = null_counts[null_counts > 7].index
    
    df_copy['isolated_null'] = df_copy['group'].isin(isolated_nulls)
    df_copy['consecutive_null'] = df_copy['group'].isin(consecutive_nulls)
    df_copy['long_null'] = df_copy['group'].isin(long_nulls)
    df_copy.drop(columns=['is_na', 'group'], inplace=True)
    
    count_summary = (f"Total isolated nulls: {df_copy['isolated_null'].sum()}\n"
                     f"Total consecutive nulls (2-7 days): {df_copy['consecutive_null'].sum()}\n"
                     f"Total long nulls (>7 days): {df_copy['long_null'].sum()}")
    return df_copy, count_summary


def analyze_trend(time_series: pd.Series) -> None:
    """
    Analyze trends in a time series using the Mann-Kendall test.
    """
    result = original_test(time_series)
    trend = "Increasing" if result.trend == "increasing" else "Decreasing" if result.trend == "decreasing" else "No significant trend"
    significance = "Significant" if result.p < 0.05 else "Not Significant"
    print(f"Identified Trend: {trend}\np-value: {round(result.p, 4)}\nStatistical Significance: {significance}")


def test_dickey_fuller(series: pd.Series, alpha: float = 0.05) -> None:
    """
    Perform the Augmented Dickey-Fuller test for stationarity.
    """
    result = adfuller(series.dropna())    
    print(f'ADF Statistic: {result[0]}\np-value: {result[1]}')
    print("The series is stationary" if result[1] < alpha else "The series is not stationary")


def log_transformation(df: pd.DataFrame, target_col: str) -> dict:
    """
    Determine whether a logarithmic transformation is needed based on heteroscedasticity and skewness.
    """
    results = {}
    residuals = sm.OLS(df[target_col], sm.add_constant(range(len(df)))).fit().resid
    bp_test = sm.stats.het_breuschpagan(residuals, sm.add_constant(range(len(df))))
    skewness = skew(df[target_col])
    adf_test = adfuller(df[target_col])
    results.update({
        'heteroscedasticity_p_value': bp_test[1],
        'skewness': skewness,
        'adf_p_value': adf_test[1],
        'adf_statistic': adf_test[0],
        'log_recommended': bp_test[1] < 0.05 or abs(skewness) > 1
    })
    print(f"ADF Test - p-value: {adf_test[1]}\nADF Test - ADF Statistic: {adf_test[0]}")
    plt.figure(figsize=(10, 5))
    plt.plot(df.index, df[target_col].rolling(window=30).mean(), label='Rolling Mean (30 days)')
    plt.plot(df.index, df[target_col].rolling(window=30).std(), label='Rolling Std Dev (30 days)')
    plt.legend()
    plt.title('Mean vs. Standard Deviation Over Time')
    plt.show()
    print("⚠️ Logarithmic transformation recommended." if results['log_recommended'] else "✅ Logarithmic transformation not necessary.")
    return results


def bds_test_acf(series: pd.Series, max_dim: int = 3, alpha: float = 0.05) -> str:
    """
    Approximate check for non-linearity in a time series using autocorrelation.
    """
    acf_values = acf(series, fft=True, nlags=max_dim)
    p_values = 1 - np.abs(acf_values[:max_dim])
    return "Non-linear patterns detected" if any(p < alpha for p in p_values) else "Series appears linear"


def evaluate_skewness(df, column):  
    """
    Calculates and interprets the skewness of a given column in a DataFrame.
    """
    data = df[column]
    skewness = skew(data)
    if skewness > 0:
        interpretation = "The distribution is right-skewed (long tail to the right)."
    elif skewness < 0:
        interpretation = "The distribution is left-skewed (long tail to the left)."
    else:
        interpretation = "The distribution is symmetrical."
    
    return {
        "skewness": skewness,
        "interpretation": interpretation}


def breusch_pagan_test(df, target_col):
    """
    Performs the Breusch-Pagan test for heteroscedasticity.
    """
    residuals = sm.OLS(df[target_col], sm.add_constant(range(len(df)))).fit().resid
    bp_test = sm.stats.het_breuschpagan(residuals, sm.add_constant(range(len(df))))
    p_value = bp_test[1]

    if p_value < 0.05:
        interpretation = "Heteroscedasticity present (null hypothesis of homoscedasticity rejected)."
    else:
        interpretation = "Heteroscedasticity not detected (null hypothesis of homoscedasticity not rejected)."

    return {'p_value': p_value, 'interpretation': interpretation}


def test_ljung_box(series, lags=[365], alpha=0.05):
    """
    Performs the Ljung-Box test for autocorrelation.
    """
    result = acorr_ljungbox(series, lags=lags, return_df=True)
    p_value = result['lb_pvalue'].iloc[0]

    if p_value < alpha:
        message = f"Evidence of autocorrelation found (p-value = {p_value:.4f})."
    else:
        message = f"No evidence of autocorrelation found (p-value = {p_value:.4f})."

    return message, result

