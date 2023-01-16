%% Generate data
p = 2/3; N = 1000; sigma = 0.7;
trueMu1 = -1; trueMu2 = 3;
trueC = binornd(1, p, N, 1); % trueC is a vector of size N x 1
Xs = trueC .* (trueMu1 + sigma*randn(N, 1)) + (1 - trueC) .*(trueMu2 + sigma*randn(N, 1));

%% Initialize Gibbs sampler
K = 2; % Number of mixtures
tau = 1; % Standard deviation of normal prior for each mu
gamma = 1; delta = 1; % Prior parameters for sigma^2
alpha = 1.5 * ones(K, 1); % Prior parameter for w
burnin = 500; Nsim = 5000 + burnin;
mus = zeros(K, Nsim);
for i = 1:K
    mus(i, 1) = tau * randn(); % mu_j^{(1)} ~ N(0, tau^2)
end
sigma2 = zeros(Nsim, 1); sigma2(1) = 1/gamrnd(gamma, 1/delta);
w = drchrnd(alpha, 1); 
cs = randsample(1:K, N, true, w)';

%% Perform Gibbs sampling
for i = 2:Nsim
    beta_i = 0; alphai = alpha;
    for j = 1:K
        Xj = Xs(cs == j); Nj = length(Xj);
        if Nj == 0 meanj = 0; else meanj = mean(Xj); end
        mus(j, i) = Nj*meanj * tau^2/(Nj * tau^2 + sigma2(i - 1))...
            + tau * sqrt(sigma2(i - 1)/(Nj * tau^2 + sigma2(i - 1))) * randn();
        beta_i = beta_i + 1/2*sum((Xj - mus(j, i)).^2);
        alphai(j) = alphai(j) + Nj;
    end
    sigma2(i) = 1/gamrnd(N/2 + gamma, 1/(beta_i + delta));
    w = drchrnd(alphai, 1);
    cs = sample_latent(w, Xs, mus(:, i), sigma2(i), K);
end

mus = mus(:,(burnin + 1):Nsim);
sigma2 = sigma2((burnin + 1):Nsim);

for i = 1:K
    figure
    histogram(mus(i, :), 100);
end
figure
histogram(sigma2, 100);