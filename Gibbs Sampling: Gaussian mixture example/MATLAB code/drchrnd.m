% Obtained from:
% https://cxwangyi.wordpress.com/2009/03/18/to-generate-random-numbers-from-a-dirichlet-distribution/
function r = drchrnd(a,n)
p = length(a');
r = gamrnd(repmat(a',n,1),1,n,p);
r = r ./ repmat(sum(r,2),1,p);
r = r';
end