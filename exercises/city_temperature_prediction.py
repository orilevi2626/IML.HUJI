import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio

from IMLearn.learners.regressors import PolynomialFitting
from IMLearn.utils import split_train_test
pio.templates.default = "simple_white"


def load_data(filename: str) -> pd.DataFrame:
    """
    Load city daily temperature dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (Temp)
    """
    df: pd.DataFrame = pd.read_csv(filename, parse_dates=["Date"])
    df.dropna(inplace=True)
    df.drop_duplicates(inplace=True)
    df["DayOfYear"] = df["Date"].dt.dayofyear
    df = df[(df["Temp"] > -70)]
    return df
    # raise NotImplementedError()


if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of city temperature dataset
    # raise NotImplementedError()
    df = load_data("../datasets/City_Temperature.csv")
    # Question 2 - Exploring data for specific country
    # df_israel = df[df["Country"] == "Israel"]
    # # raise NotImplementedError()
    df["Year"] = df["Year"].astype(str)
    fig = px.scatter(df[df["Country"] == "Israel"], x="DayOfYear", y="Temp", color="Year",
                     title="relation between day of year and measured temperature")
    fig.show()
    df_month = df[df["Country"] == "Israel"].groupby("Month").Temp.agg("std")
    fig_bar = px.bar(df_month, x=df_month.index.get_level_values("Month"), y="Temp", text_auto=True,
                     title="Standard deviation of temperature each month in Israel")
    fig_bar.show()
    # Question 3 - Exploring differences between countries
    # raise NotImplementedError()
    df_group = df.groupby(["Country", "Month"]).Temp.agg(["mean", "std"])
    fig_line = px.line(df_group, x=df_group.index.get_level_values("Month"),
                       y="mean", color=df_group.index.get_level_values("Country"),
                       error_y="std", labels={"x": "Month"},
                       title="Average temp every month in all countries with standard deviation")
    fig_line.show()

    # Question 4 - Fitting model for different values of `k`
    df_israel = df[df["Country"] == "Israel"]
    israel_temps = df_israel.pop("Temp")
    x_train: pd.DataFrame
    y_train: pd.DataFrame
    x_test: pd.DataFrame
    y_test: pd.DataFrame
    x_train, y_train, x_test, y_test = split_train_test(df_israel, israel_temps)
    k_loss = []
    for k in np.arange(1, 11):
        poly_fit = PolynomialFitting(k=k)
        poly_fit.fit(x_train["DayOfYear"].to_numpy(), y_train.to_numpy())
        k_loss.append(np.round(poly_fit.loss(x_test["DayOfYear"].to_numpy(), y_test.to_numpy()), 2))
        print(f"The loss of the model fitted on a polynomial of degree {k} is {k_loss[-1]}")
    loss_df = pd.DataFrame(k_loss, index=np.arange(1, 11),
                           columns=['Loss'])
    fig_bar2 = px.bar(loss_df, x=np.arange(1, 11), y=k_loss, text_auto=True,
                      title="Loss of prediction of a temperature in israel as a function of k'th degree polynomial")
    fig_bar2.show()

    # raise NotImplementedError()
    # Question 5 - Evaluating fitted model on different countries
    poly_fit = PolynomialFitting(k=5)
    df_israel = df[df["Country"] == "Israel"]

    poly_fit.fit(df_israel["DayOfYear"], df_israel["Temp"])
    country_losses = []
    countries= ["Jordan", "The Netherlands", "South Africa"]
    for country in countries:
        curr_country_df = df[df["Country"] == country]
        day_of_year = curr_country_df["DayOfYear"]
        country_temp = curr_country_df["Temp"]
        country_losses.append(poly_fit.loss(day_of_year, country_temp))
    country_loss_df = pd.DataFrame(country_losses, index=countries, columns=['Loss'])
    fig_bar3 = px.bar(loss_df, x=country_loss_df.index, y=country_losses, text_auto=True,
                      title="Loss of prediction of a temperature in all countries as "
                            "fitted with polynomial of degree 5")
    fig_bar3.show()
    # raise NotImplementedError()
