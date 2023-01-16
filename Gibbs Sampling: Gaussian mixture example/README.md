# Gibbs sampling: Gaussian mixture example
Assume the data $X = (X_1, X_2, \ldots, X_N)$ comes from a Gaussian mixture of the form

$$
  f_X(x) = \sum_{j = 1}^K \omega_j f_{\mathcal{N}}(x \vert \mu_j, \sigma)
$$

where $\omega_j \in [0, 1]$ and $\displaystyle\sum_{j = 1}^K \omega_j = 1$. Moreover, we'll define a latent vector variable $c = (c_1, \ldots, c_N)$ with a
prior:

$$
  \mathbb{P}(c_i = j \vert \omega) = \omega_j \ \forall j \in \{1,2,\ldots, K\}
$$

The latent variable $c_i$ denotes where the sample $X_i$ comes from, i.e $X_i \sim \mathcal{N}(\mu_j, \sigma^2)$ if $c_i = j$

Finally, we put priors on $\mu, \sigma^2, \omega$:

$$
\begin{align*}
  \mu_j &\sim \mathcal{N}(0, \tau^2)\\
  \sigma^2 &\sim \mathcal{IG}(\gamma, \delta)\\
  \omega &\sim \text{Dir}(\alpha, \alpha, \ldots, \alpha)
\end{align*}
$$

The full conditional on each parameter is given by

$$
\begin{align*}
  \mu_j \vert x, \omega, \mu, c &\sim \mathcal{N}\left(\dfrac{N_j \bar{x}_j \tau^2}{N_j \tau^2 + \sigma^2}, \dfrac{\sigma^2 \tau^2}{N_j \tau^2 + \sigma^2}\right)\\
  \sigma^2 \vert x, \omega, \mu, c &\sim \mathcal{IG}\left(\dfrac{N}{2} + \gamma, \dfrac{1}{2}\sum_i (x_i - \mu_{c_i})^2 + \delta\right)\\
  \omega \vert x, \mu, \sigma^2, c &\sim \text{Dir}(\alpha + N_1, \alpha + N_2, \ldots, \alpha + N_K)\\
  \mathbb{P}(c_i = j \vert x, \omega, \mu, \sigma^2) &\propto \omega_j\ \text{exp}\left(-\dfrac{1}{2\sigma^2}(x_i - \mu_j)^2\right)
\end{align*}
$$

where $N_j$ (resp. $\bar{x}_j$) is the number (resp. the mean) of samples $X_i$ such that $c_i = j$

In the file Gibbs_sampler_mixture.R is my implementation (not very efficient) of Gibbs sampling for Gaussian mixture when $K = 2$ in R language. 
In this file, the data is generated from mixture of $\mathcal{N}(-1, 0.7^2)$ and $\mathcal{N}(3, 0.7^2)$ with $\omega_1 = 2/3, \omega_2 = 1/3$

The histogram of the marginal posterior for each parameter from Gibbs sampling is given below:
<p align="center"> <img src=https://github.com/PhuThanh-Nguyen/Small-Projects/blob/main/Gibbs%20Sampling:%20Gaussian%20mixture%20example/Miscellaneous/Histogram%20mu1.png width=700 height=400> </p>
<p align="center"> <img src=https://github.com/PhuThanh-Nguyen/Small-Projects/blob/main/Gibbs%20Sampling:%20Gaussian%20mixture%20example/Miscellaneous/Histogram%20mu2.png width=700 height=400> </p>
<p align="center"> <img src=https://github.com/PhuThanh-Nguyen/Small-Projects/blob/main/Gibbs%20Sampling:%20Gaussian%20mixture%20example/Miscellaneous/Histogram%20sigma2.png width=700 height=400> </p>
<p align="center"> <img src=https://github.com/PhuThanh-Nguyen/Small-Projects/blob/main/Gibbs%20Sampling:%20Gaussian%20mixture%20example/Miscellaneous/Histogram%20sqrt(sigma2).png width=700 height=400> </p>

The results of the Gibbs sampling are quite good, even it gives a good estimate on which sample is from which distribution and the weights $\omega = (\omega_1, \omega_2)$

**Update:** Upload the MATLAB's version of this implementation. Note: The MATLAB code to generate Dirichlet distribution comes from: https://cxwangyi.wordpress.com/2009/03/18/to-generate-random-numbers-from-a-dirichlet-distribution/
