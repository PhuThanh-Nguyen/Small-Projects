# Gibbs Sampling: 1D Bayesian Regression example
Consider the following model

$$
  Y_j = ax_j + b + \epsilon_j, \epsilon_j \sim \mathcal{N}(0, \sigma^2) \ \forall j = 1,2,\ldots, N
$$

Put priors on the slope and intercept coefficients and $\sigma^2$:

$$
  \begin{align*}
    a &\sim \mathcal{N}(0, \tau^2)\\
    b &\sim \mathcal{N}(0, \tau^2)\\
    \sigma^2 &\sim \mathcal{IG}(\alpha, \beta)
  \end{align*}
$$

where $\tau, \alpha, \beta$ are fixed suitable constants. With these priors, the horizontal line is our assumption before seeing data

To use Gibbs sampling, calculate full conditional on these parameters gives:

$$
  \begin{align*}
    a \vert y, b, \sigma^2 &\sim \mathcal{N}\left(\dfrac{\tau^2 \sum_i x_i(y_i - b)}{\sigma^2 + \tau^2 \sum_i x_i^2}, \dfrac{\sigma^2 \tau^2}{\sigma^2 + \tau^2 \sum_i x_i^2}\right)\\
    b \vert y, a, \sigma^2 &\sim \mathcal{N}\left(\dfrac{\tau^2 \sum_i (y_i - ax_i)}{\sigma^2 + N \tau^2}, \dfrac{\sigma^2 \tau^2}{\sigma^2 + N \tau^2}\right)\\
    \sigma^2 \vert y, a, b &\sim \mathcal{IG}\left(\alpha + \dfrac{N}{2}, \beta + \dfrac{1}{2}\sum_i e_i^2\right)
  \end{align*}
$$

where $e_i = y_i - (ax_i + b)$

In the file Gibbs_sampling_BayesRegression.R is the implementation of Gibbs sampling scheme for this scenario with $a = 2, b = 1, \sigma^2 = 0.75^2$

The histograms of the parameters by the Gibbs sampling are given below:
<p align="center"> <img src=https://github.com/PhuThanh-Nguyen/Small-Projects/blob/main/Gibbs%20Sampling:%201D%20Bayesian%20Regression%20example/Miscellaneous/Histogram%20of%20slope.png width=700 height=400> </p>
<p align="center"> <img src=https://github.com/PhuThanh-Nguyen/Small-Projects/blob/main/Gibbs%20Sampling:%201D%20Bayesian%20Regression%20example/Miscellaneous/Histogram%20of%20intercept.png width=700 height=400> </p>
<p align="center"> <img src=https://github.com/PhuThanh-Nguyen/Small-Projects/blob/main/Gibbs%20Sampling:%201D%20Bayesian%20Regression%20example/Miscellaneous/Histogram%20of%20variance%20of%20errors.png width=700 height=400> </p>

One could use the means of these samples as the estimated slope and intercept to get estimated line
<p align="center"> <img src=https://github.com/PhuThanh-Nguyen/Small-Projects/blob/main/Gibbs%20Sampling:%201D%20Bayesian%20Regression%20example/Miscellaneous/Estimated%20line.png width=700 height=400> </p>

