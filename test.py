from sklearn.linear_model import LinearRegression
from collections.abc import Iterable
from numpy import array, arange, sum, abs, issubdtype, number
from pandas import DataFrame, concat
from numpy.lib.stride_tricks import sliding_window_view


def test_split_serie(arr: Iterable):
    if not isinstance(arr, Iterable):
        raise TypeError("The data array must be an iterable.")


def test_build_regressors(arr: Iterable) -> array:
    if not (isinstance(arr, Iterable)):
        raise TypeError("The data array must be an iterable.")
    if not (
        sum([(len(X) == 2) | (issubdtype(X.dtype, number)) for X in arr])
        == arr.shape[0]
    ):
        raise ValueError(
            "Array passed in input must be an array of arrays of two elements."
        )


def test_extract_coefs(arr: Iterable) -> array:
    if not (isinstance(arr, Iterable)):
        raise TypeError("The data array must be an iterable.")
    if not (sum([(isinstance(X, LinearRegression)) for X in arr]) == arr.shape[0]):
        raise TypeError("The array must be an array of fitted LinearRegression")


def test_sum_absolute_coef_differences(
    y_true: Iterable, y_pred: Iterable, alpha: float = 0.5
):
    if array(y_true).shape != array(y_pred).shape:
        raise ValueError("Prediction and real values must have the same shapes.")
    if alpha > 1 or alpha < 0:
        raise ValueError("Alpha must be in [0, 1].")


def test_plot_multiple_joint_errors(y_true: array, preds: DataFrame):
    if y_true.shape[0] != preds.shape[0]:
        raise ValueError("Sizes must be the same between sets.")
    if preds.shape[1] != preds.select_dtypes(include=number).shape[1]:
        raise TypeError("Must be a dataset of numeric values.")


def test_plot_joint_betas(type_beta: str):
    if type_beta != "intercept" and type_beta != "slope":
        raise ValueError(
            "You must specify if you want to see the joint of the intercepts ('intercept') or the slopes ('slope')."
        )
