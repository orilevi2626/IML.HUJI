import numpy as np
import plotly.graph_objects as go
import plotly.io as pio

from IMLearn.learners import UnivariateGaussian, MultivariateGaussian

pio.templates.default = "simple_white"


def test_univariate_gaussian():
    # Question 1 - Draw samples and print fitted model
    array = np.random.normal(10, 1, 1000)
    uni_gaus = UnivariateGaussian().fit(array)

    print(uni_gaus.mu_, uni_gaus.var_)
    # Question 2 - Empirically showing sample mean is consistent
    gaus_list = []
    sample_size = [i for i in range(10, 1001, 10)]
    for i in sample_size:
        gaus_list.append(abs(uni_gaus.fit(array[:i]).mu_ - 10))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=sample_size, y=gaus_list,
                             name="Distance between expected value and sample mean"))
    fig.update_layout(title="Plotting distance from expected value by number of samples taken",
                      xaxis_title="Number of samples",
                      yaxis_title="Distance",
                      font=dict(family="Courier New, monospace",
                                size=18, color="RebeccaPurple"))
    fig.show()

    # Question 3 - Plotting Empirical PDF of fitted model
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=array, y=uni_gaus.pdf(array), mode="markers"))
    fig2.update_layout(title="Samples vs PDF of fitted samples",
                       xaxis_title="Sample value",
                       yaxis_title="PDF of fitted sample",
                       font=dict(family="Courier New, monospace",
                                 size=18, color="RebeccaPurple"))

    fig2.show()


def test_multivariate_gaussian():
    # Question 4 - Draw samples and print fitted model
    sigma_cov = np.array([[1, 0.2, 0, 0.5],
                          [0.2, 2, 0, 0],
                          [0, 0, 1, 0],
                          [0.5, 0, 0, 1]])
    mu = np.array([0, 0, 4, 0])
    samples = np.random.multivariate_normal(mu, sigma_cov, size=1000)
    gaus = MultivariateGaussian().fit(samples)
    print(gaus.mu_)
    print(gaus.cov_)
    # raise NotImplementedError()
    # Question 5 - Likelihood evaluation
    f1, f3 = np.linspace(-10, 10, 200), np.linspace(-10, 10, 200)
    log_array = []
    max_val_arg = [-np.inf, (0, 0)]
    for sample_1 in f1:
        log_row = []
        for sample_3 in f3:
            mu = np.array([sample_1, 0, sample_3, 0])
            log_row.append(MultivariateGaussian.log_likelihood(mu, sigma_cov, samples))
            if max_val_arg[0] < log_row[-1]:
                max_val_arg = [log_row[-1], (sample_1, sample_3)]

        log_array.append(log_row)
    heatmap = go.Figure(data=go.Heatmap(z=log_array, x=f1, y=f3, hoverongaps=False))
    heatmap.update_layout(
        title="Heatmap representing log likelihood of [f1,0,f3,0] as samples from -10,10 linear space ",
        xaxis_title="feature 1", yaxis_title="feature 3")
    heatmap.show()
    # raise NotImplementedError()

    # Question 6 - Maximum likelihood
    print(max_val_arg[::-1])
    # raise NotImplementedError()


if __name__ == '__main__':
    np.random.seed(0)
    test_univariate_gaussian()
    test_multivariate_gaussian()
