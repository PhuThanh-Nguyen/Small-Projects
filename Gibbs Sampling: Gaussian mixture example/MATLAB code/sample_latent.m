function cs = sample_latent(w, x, mu, sigma2, K)
    N = length(x); prob_table = zeros(K, N);
    cs = zeros(N, 1);
    for i = 1:K
        prob_table(i, :) = w(i) * exp(-1/(2*sigma2) * (x - mu(i)).^2)'; 
    end
    
    function s = sample_prob(p)
        s = randsample(1:K, 1, true, p);
    end

    C = num2cell(prob_table, 1);
    cs = cellfun(@sample_prob, C)';
    %cs = randsample(1:K, 1, true, prob_table);
    %cs = splitapply(randsample, 1: K, 1, true, prob_table, prob_table, 1:size(prob_table,2));
    
    %for i = 1:N
    %    cs(i) = randsample(1:K, 1, true, prob_table(:, i));
    %end
    
end