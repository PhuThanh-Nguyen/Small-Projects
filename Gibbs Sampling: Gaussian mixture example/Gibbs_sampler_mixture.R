library(igraph)
# --- Generate data sample ---
p = 2/3; N = 1000; sigma = 0.7
trueMu1 = -1; trueMu2 = 3;
trueC = rbinom(N, 1, p)
Xs = trueC * rnorm(
    N, mean = trueMu1, sd = sigma) + (1 - trueC) * rnorm(
        N, mean = trueMu2, sd = sigma)
hist(Xs, freq = F)

# --- Sample latent variable ---
sample_latent = function(w, x, mu, sigma2, K = 2){
    N = length(x)
    probs_table = matrix(data = rep(0, N*K), nrow = K, ncol = N)
    for(i in 1:K){
        probs_table[i,] = dnorm(x, mu[i], sqrt(sigma2))*w[i]
    }
    cs = apply(
        probs_table, 2, 
        function(p){sample(1:K, 1, replace = T, prob = p)})
    return(cs)
}

# --- Initialize Gibbs sampler ---
K = 2; # Number of class
tau = 1; # Prior for mus
gamma = 1; delta = 1 # Prior for sigma^2
alpha = 1.5; # Prior for w
burnin = 500; Nsim = 5000 + burnin;
mu1 = rep(rnorm(1, 0, tau), Nsim)
mu2 = rep(rnorm(1, 0, tau), Nsim)
sigma2 = rep(1/rgamma(1,1,1), Nsim)
w = sample_dirichlet(1, rep(alpha, K))
cs = sample(1:K, size = N, replace = T, prob = w)

for(i in 2:Nsim){
    X1 = Xs[cs == 1]; X2 = Xs[cs == 2];
    N1 = length(X1); N2 = length(X2);
    mean_1 = ifelse(N1 == 0, 0, mean(X1));
    mean_2 = ifelse(N2 == 0, 0, mean(X2));
    #print(c(N0, N1))
    mu1[i] = rnorm(1, 
                   N1*tau^2*mean_1/(N1 * tau^2 + sigma2[i - 1]),
                   sqrt(sigma2[i - 1] * tau^2/(N1 * tau^2 + sigma2[i - 1]))
    )
    mu2[i] = rnorm(1, 
                   N2*tau^2*mean_2/(N2 * tau^2 + sigma2[i - 1]),
                   sqrt(sigma2[i - 1] * tau^2/(N2 * tau^2 + sigma2[i - 1]))
    )
    sigma2[i] = 1/rgamma(
        1, N/2 + gamma, 
        1/2*sum((X1 - mu1[i])^2) + 1/2*sum((X2 - mu2[i])^2) + delta
    )
    w = sample_dirichlet(1, c(alpha + N1, alpha + N2))
    cs = sample_latent(w, Xs, c(mu1[i], mu2[i]), sigma2[i], K)
}
# Extract sample after burn-in
mu1 = tail(mu1, Nsim - burnin)
mu2 = tail(mu2, Nsim - burnin)
sigma2 = tail(sigma2, Nsim - burnin)

hist(sigma2, freq = F); abline(v = sigma^2, col = 'red', lty = 2)
hist(sqrt(sigma2), freq = F); abline(v = sigma, col = 'red', lty = 2)

if(abs(mean(mu1) - trueMu2) < abs(mean(mu1) - trueMu1)){
    hist(mu1, freq = F); abline(v = trueMu2, col = 'red', lty = 2)
    hist(mu2, freq = F); abline(v = trueMu1, col = 'red', lty = 2)
    print(sum(ifelse(cs == 1, 0, 1) == trueC)/N)
}else{
    hist(mu1, freq = F); abline(v = trueMu1, col = 'red', lty = 2)
    hist(mu2, freq = F); abline(v = trueMu2, col = 'red', lty = 2)
    print(sum(ifelse(cs == 1, 1, 0) == trueC)/N)
}
print(w)