// adapted from https://mc-stan.org/docs/2_29/stan-users-guide/multivariate-outcomes.html
data {
  int<lower=1> K; // number of response variables
  int<lower=1> J; // number of fixed effects
  int<lower=1> P; // number of random effects
  int<lower=0> N; // number of samples
  matrix[N,J] x;
  matrix[N,K] y;
  array[P] matrix[N,N] R; // correlation matrices for random effects
}
transformed data {
  int JP1 = J+P+1;
  int NK = N*K;
}
parameters {
  matrix[J,K] beta_raw;
  cholesky_factor_corr[K] L_k;
  vector<lower=0>[K] sigma_k;
  real<lower=1> rho;
  simplex[JP1] var_jp_mu;
  array[K] simplex[JP1] var_jp;
}
transformed parameters {
  matrix[K,K] L_k_s = diag_pre_multiply(sigma_k, L_k);
  matrix[J,K] beta = beta_raw * L_k_s';
  for(k in 1:K) {
    beta[,k] = sqrt(var_jp[k][1:J]) .* beta[,k]; // assume effects are correlated same as residuals
  }
}
model {
  matrix[K,K] s_k = tcrossprod(L_k_s);
  matrix[NK,NK] sigma;
  
  for(k1 in 1:K) {
    int s1 = N * (k1 - 1) + 1;
    int e1 = s1 + N - 1;
    matrix[N,N] temp = rep_matrix(0,N,N);
    for(p in 1:P) {
      temp = temp + var_jp[k1][J+p] * R[p];
    }
    temp = add_diag(temp, var_jp[k1][JP1] + 1e-10); // marginal correlation for response variable k1
    for(k2 in k1:K) {
      int s2 = N * (k2 - 1) + 1;
      int e2 = s2 + N - 1;
      sigma[s2:e2, s1:e1] = s_k[k2,k1] * temp; // part of matrix analogous to kronecker product; both residuals and random effects are correlated across response variables
    }
  }
  
  sigma_k ~ std_normal();
  rho ~ pareto(1,2);
  var_jp ~ dirichlet(JP1 * rho * var_jp_mu);

  to_vector(beta_raw) ~ std_normal();
  L_k ~ lkj_corr_cholesky(1);

  to_vector(y) ~ multi_normal(to_vector(x * beta), symmetrize_from_lower_tri(sigma));
}
