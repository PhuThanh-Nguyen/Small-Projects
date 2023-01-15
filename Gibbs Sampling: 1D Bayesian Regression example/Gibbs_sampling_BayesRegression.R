# --- Generate dataset ---
x = seq(from = -3, to = 3, by = 1e-2); N = length(x)
a = 2; b = 1; sigma = 0.75
y = a*x + b + rnorm(N, 0, sigma)
plot(x, y, type = 'p')

# --- Precalculating coeffs ---
xy_sum = sum(x * y); x_sum = sum(x); x2_sum = sum(x^2)
y_sum = sum(y)

# --- Initialize priors ---
tau = 0.5; alpha = 1; beta = 1; burnin = 500; Nsim = 10^5 + burnin;
a_gibbs = rep(rnorm(1, 0, tau), Nsim)
b_gibbs = rep(rnorm(1, 0, tau), Nsim)
sigma_gibbs = rep(1/rgamma(1, alpha, beta), Nsim)

# --- Perform Gibbs sampling
for(i in 2:Nsim){
    a_gibbs[i] = rnorm(
        1, 
        tau^2 * (xy_sum - b_gibbs[i - 1] * x_sum)/(sigma_gibbs[i - 1] + tau^2 * x2_sum),
        sqrt(sigma_gibbs[i - 1] * tau^2/(sigma_gibbs[i - 1] + tau^2 * x2_sum))
    )
    b_gibbs[i] = rnorm(
        1,
        tau^2 * (y_sum - a_gibbs[i]*x_sum)/(sigma_gibbs[i - 1] + N * tau^2),
        sqrt(sigma_gibbs[i - 1]*tau^2/(sigma_gibbs[i - 1] + N*tau^2))
    )
    errs = y - a_gibbs[i] * x - b_gibbs[i]
    errs2_sum = sum(errs^2)
    sigma_gibbs[i] = 1/rgamma(1, alpha + N/2, beta + 1/2 * errs2_sum)
}

a_gibbs = tail(a_gibbs, Nsim - burnin)
b_gibbs = tail(b_gibbs, Nsim - burnin)
sigma_gibbs = tail(sigma_gibbs, Nsim - burnin)

hist(a_gibbs, breaks = 100, freq = F); 
abline(v = a, col = 'red', lwd = 2, lty = 2)
hist(b_gibbs, breaks = 100, freq = F);
abline(v = b, col = 'red', lwd = 2, lty = 2)
hist(sigma_gibbs, breaks = 100, freq = F);
abline(v = sigma^2, col = 'red', lwd = 2, lty = 2)

nSample = 100
aEst = mean(a_gibbs); bEst = mean(b_gibbs)
a_sample = sample(a_gibbs, nSample)
b_sample = sample(b_gibbs, nSample)

plot(x, y, type = 'p'); 
lines(x, aEst * x + bEst, col = 'red', lwd = 3)
title(main = 'Estimated line')