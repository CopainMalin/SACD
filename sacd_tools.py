from sklearn.linear_model import LinearRegression
from collections.abc import Iterable
from numpy import array, arange, sum, abs
from pandas import DataFrame, concat
from numpy.lib.stride_tricks import sliding_window_view

import matplotlib.pyplot as plt
import seaborn as sns

from test import (
    test_split_serie,
    test_build_regressors,
    test_extract_coefs,
    test_sum_absolute_coef_differences,
    test_plot_multiple_joint_errors,
    test_plot_joint_betas,
)


def __split_serie(data: Iterable) -> array:
    """Split an array into blocs of the form ((y_(i-1), y_i) for i in 2:n).

    Args:
        data (Iterable): The array to split.

    Returns:
        np.array: The array of blocs.
    """
    test_split_serie(data)
    return sliding_window_view(data, 2)


def __build_regressors(blocs: Iterable) -> array:
    """Build and fit a sklearn LinearRegression for each bloc composing the array.

    Args:
        blocs (Iterable): The array of blocs.

    Returns:
        np.array: The array of fitted LinearRegression.
    """
    test_build_regressors(blocs)
    sub_X = arange(2).reshape(-1, 1)
    return array([LinearRegression().fit(sub_X, bloc) for bloc in blocs])


def __extract_coefs(arr: Iterable) -> array:
    """Extract intercept and slope coefficients from an array of fitted linear regressions.

    Args:
        arr (Iterable): The fitted linear regression array.

    Returns:
        np.array, np.array: array of the intercepts, array of the slopes.
    """
    test_extract_coefs(arr)
    return array([X.intercept_ for X in arr]), array([X.coef_[0] for X in arr])


def get_coefs(arr: Iterable) -> array:
    """Given a array, returns the intercepts and slopes coefficients for each step.

    Args:
        arr (Iterable): The serie.

    Returns:
        np.array, np.array: array of the intercepts, array of the slopes.
    """
    blocs = __split_serie(arr)
    reg = __build_regressors(blocs)
    return __extract_coefs(reg)


def sum_absolute_coef_differences(
    y_true: Iterable, y_pred: Iterable, alpha: float = 0.5
) -> float:
    """Compute the sum of the absolute coefficient differences (SACD) between a value and a prediction.

    Args:
        y_true (Iterable): The vector of true values.
        y_pred (Iterable): The prediction vector.
        alpha (float, optional): Whether to penalize the slopes differences or the intercept. Defaults to 0.5.
            If set to 1, focus on the intercepts and the result is the sum of absolute error (at one point).
            If set to 0, focus on the slopes.

    Returns:
        float: The SACD.
    """
    test_sum_absolute_coef_differences(y_true, y_pred, alpha)
    b0, b1 = get_coefs(y_true)
    hb0, hb1 = get_coefs(y_pred)
    return sum(alpha * abs(b0 - hb0) + (1 - alpha) * abs(b1 - hb1))


def mean_absolute_coef_differences(
    y_true: Iterable, y_pred: Iterable, alpha: float = 0.5
):
    """Compute the mean of the absolute coefficient differences (MACD) between a value and a prediction.

    Args:
        y_true (Iterable): The vector of true values.
        y_pred (Iterable): The prediction vector.
        alpha (float, optional): Whether to penalize the slopes differences or the intercept. Defaults to 0.5.
            If set to 1, focus on the intercepts and the result is the mean absolute error (at one point).
            If set to 0, focus on the slopes.

    Returns:
        float: The MACD.
    """
    return 1 / (len(y_true) - 1) * sum_absolute_coef_differences(y_true, y_pred, alpha)


### Plotting tools ###
def plot_joint_errors(y_true: Iterable, y_pred: Iterable, color: str = None) -> None:
    """Plot the joint distribution between the slopes errors and the intercepts errors.
    Also plot both marginals.

    Args:
        y_true: (Iterable) : The vector of true values.
        y_pred (Iterable): The prediction vector.
        color (str, optional): The plot color. Defaults to None.
    """
    if color is None:
        color = "#7209b7"

    b0, b1 = get_coefs(y_true)
    hb0, hb1 = get_coefs(y_pred)
    b0_diff = b0 - hb0
    b1_diff = b1 - hb1

    if y_true.shape[0] < 50:
        g = sns.jointplot(
            x=b0_diff,
            y=b1_diff,
            kind="scatter",
            s=500,
            alpha=0.7,
            color=color,
            height=7,
        )
    else:
        g = sns.jointplot(
            x=b0_diff,
            y=b1_diff,
            kind="hex",
            color=color,
            height=7,
        )
    # sns.scatterplot(x=b0_diff, y=b1_diff, s=50, alpha=1, color=color)
    g.set_axis_labels(
        xlabel=r"$\epsilon_{intercept}$", ylabel=r"$\epsilon_{slope}$", color="black"
    )
    return g


def plot_multiple_joint_errors(y_true: Iterable, preds: DataFrame) -> None:
    """Plot the joint distribution between the slopes errors and the intercepts errors.
    Also plot both marginals.

    Args:
        y_true: (Iterable) : The vector of true values.
        preds (Iterable): The dataset containting predictions.
        color (str, optional): The plot color. Defaults to None.
    """
    test_plot_multiple_joint_errors(y_true, preds)
    coef_df = DataFrame(columns=["b0 diff", "b1 diff", "Model"])

    b0, b1 = get_coefs(y_true)

    for key in preds.columns:
        macd = mean_absolute_coef_differences(y_true, preds[key])
        hb0, hb1 = get_coefs(preds[key])
        labels = [key + f" | MACD : {macd:.0f}"] * (y_true.shape[0] - 1)
        b0_diff = b0 - hb0
        b1_diff = b1 - hb1
        new_df = DataFrame(data=[b0_diff, b1_diff, labels]).T

        new_df.columns = ["b0 diff", "b1 diff", "Model"]
        coef_df = concat([coef_df, new_df], axis=0)

    g = sns.jointplot(
        data=coef_df,
        x="b0 diff",
        y="b1 diff",
        hue="Model",
        kind="kde",
        fill=True,
        palette="tab10",
        alpha=0.7,
        height=7,
        levels=7,
    )
    sns.scatterplot(
        data=coef_df,
        x="b0 diff",
        y="b1 diff",
        hue="Model",
        s=50,
        alpha=1,
        palette="tab10",
    )
    g.set_axis_labels(
        xlabel=r"$\epsilon_{intercept}$", ylabel=r"$\epsilon_{slope}$", color="black"
    )
    return g


def plot_joint_betas(
    y_true: Iterable, y_pred: Iterable, type_coef: str = "intercept", color: str = None
) -> None:
    """Plot the joint distribution between the true coefficients and the predicted ones.
    Also plot both marginals.

    Args:
        y_true: (Iterable) : The vector of true values.
        y_pred (Iterable): The prediction vector.
        type_beta (str, optional): The type of coefficient to plot. Must be "intercept" for the intercept or "slope" for the slope. Defaults to "intercept".
        color (str, optional): The plot color. Defaults to None.
    """
    test_plot_joint_betas(type_coef)
    b0, b1 = get_coefs(y_true)
    hb0, hb1 = get_coefs(y_pred)

    if type_coef == "intercept":
        b, _ = get_coefs(y_true)
        hb, _ = get_coefs(y_pred)
        if color is None:
            color = "#43aa8b"
        title = r"Intercept"
    else:
        _, b = get_coefs(y_true)
        _, hb = get_coefs(y_pred)
        if color is None:
            color = "#14213d"
        title = r"Slope"

    g = sns.jointplot(x=hb, y=b, kind="kde", fill=True, color=color, height=6)
    sns.scatterplot(x=hb, y=b, edgecolor="black", color=color, s=50, alpha=1)
    g.set_axis_labels(
        xlabel=title + " prévision", ylabel=title + " réel", color="black"
    )
    g.fig.suptitle(title + " analysis", fontweight="bold", color="black")
    return g
