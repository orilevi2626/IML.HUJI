from typing import NoReturn

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio

from IMLearn.learners.regressors.linear_regression import LinearRegression
from IMLearn.utils import split_train_test

pio.templates.default = "simple_white"


def load_data(filename: str):
    """
    Load house prices dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (prices) - either as a single
    DataFrame or a Tuple[DataFrame, Series]
    """
    # raise NotImplementedError()
    df_data = pd.read_csv(filename)
    # remove null values
    df_data.dropna(inplace=True)
    df_data.drop(['date', 'long', 'lat', 'id'], axis=1, inplace=True)
    df_data = df_data[
        (df_data.price > 0) & (df_data.sqft_lot15 > 0) & (df_data.sqft_living > 0) & (df_data.sqft_living15 > 0) & (
                df_data.yr_built > 0) & (df_data.sqft_lot > 0)]
    df_data['yr_built'] = (df_data['yr_built'] // 10) * 10  # look at data in 10 year intervals
    df_data = pd.get_dummies(df_data, columns=['zipcode'])
    return df_data


def feature_evaluation(X: pd.DataFrame, y: pd.Series, output_path: str = ".") -> NoReturn:
    """
    Create scatter plot between each feature and the response.
        - Plot title specifies feature name
        - Plot title specifies Pearson Correlation between feature and response
        - Plot saved under given folder with file name including feature name
    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Design matrix of regression problem

    y : array-like of shape (n_samples, )
        Response vector to evaluate against

    output_path: str (default ".")
        Path to folder in which plots are saved
    """
    # raise NotImplementedError()
    for col in X:
        if col == 'id' or 'zipcode' in col:
            continue
        cov_mat = np.cov(X[col], y)
        corr = cov_mat[0, 1] / (np.std(X[col]) * np.std(y))
        fig = px.scatter({'Feature Values': X[col], 'Response Values': y}, x='Feature Values', y='Response Values',
                         trendline='ols', trendline_color_override='darkblue')
        fig.update_layout(title=f'Plot of Pearson correlation {corr} of feature {col} and price')
        import os
        fig.write_image(os.path.join(output_path, f'{col}.png'))


if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of housing prices dataset
    # raise NotImplementedError()
    df = load_data("../datasets/house_prices.csv")
    # # Question 2 - Feature evaluation with respect to response
    # # raise NotImplementedError()
    y = df.pop('price')
    feature_evaluation(df, y)
    # Question 3 - Split samples into training- and testing sets.
    # raise NotImplementedError()
    train_x: pd.DataFrame
    train_y: pd.DataFrame
    test_x: pd.DataFrame
    test_y: pd.DataFrame
    train_x, train_y, test_x, test_y = split_train_test(df, y)
    lin_model = LinearRegression(include_intercept=True)
    # Question 4 - Fit model over increasing percentages of the overall training data
    mse, std = [], []
    p_range = np.arange(10, 101)
    for percentage in p_range:
        curr_percentage_loss = []
        for i in range(10):
            train_samples: pd.DataFrame = train_x.sample(n=(train_x.shape[0] * percentage) // 100)
            train_labels: pd.DataFrame = train_y.reindex(train_samples.index)
            lin_model.fit(train_samples.to_numpy(), train_labels.to_numpy())
            curr_percentage_loss.append(lin_model.loss(test_x.to_numpy(), test_y.to_numpy()))
        mse.append(np.mean(curr_percentage_loss))
        std.append(np.std(curr_percentage_loss))
    mse = np.array(mse)
    std = np.array(std)
    fig = go.Figure(data=[
        go.Scatter(x=p_range, y=mse, mode="markers+lines", name="Prediction of Mean", line=dict(dash="dash"),
                   marker=dict(color="green"), opacity=0.7),
        go.Scatter(x=p_range, y=(mse - 2 * std), fill=None, mode="lines", line=dict(color="lightgrey"),
                   showlegend=False),
        go.Scatter(x=p_range, y=(mse + 2 * std), fill="tonexty", mode="lines", line=dict(color="lightgrey"),
                   showlegend=False)],
        layout=go.Layout(title=r"$\text{Mean of loss as a function of % of training data}$",
                         xaxis={"title": r"$\text{percentage of training data}$"},
                         yaxis={"title": r"$\text{Mean loss}$"}))
    fig.show()
    # For every percentage p in 10%, 11%, ..., 100%, repeat the following 10 times:
    #   1) Sample p% of the overall training data
    #   2) Fit linear model (including intercept) over sampled set
    #   3) Test fitted model over test set
    #   4) Store average and variance of loss over test set
    # Then plot average loss as function of training size with error ribbon of size (mean-2*std, mean+2*std)
    # raise NotImplementedError()
