data {
  int<lower=1> K; // number of response variables
  int<lower=1> J; // number of fixed effects
  int<lower=1> P; // number of random effects
  int<lower=0> N; // number of samples
  int<lower=0> M; // number of missing values
  matrix[N,J] x; // fixed effect model matrix
  matrix[N,K] y; // response data
  array[P] matrix[N,N] R; // correlation matrices for random effects
  array[M,2] int m; // indices for missing values
}
transformed data {
  int JP1 = J+P+1;
  int NK = N*K;
}
parameters {
  matrix[J,K] beta_fixed_raw;
  cholesky_factor_corr[K] L_k;
  vector<lower=0>[K] sigma_k;
  array[K] simplex[JP1] var_jp;
  matrix[N,K] beta_random_raw;
  vector[M] missing;
}
transformed parameters {
  matrix[J,K] beta;
  matrix[N,K] yy = y;
  for(k in 1:K) {
    beta[,k] = sigma_k[k] * sqrt(var_jp[k][1:J]) .* beta_fixed_raw[,k];
  }
  for(i in 1:M) {
    yy[m[i,1],m[i,2]] = missing[i];
  }
}
model {
  matrix[K,K] L_k_s = diag_pre_multiply(sigma_k, L_k);
  matrix[N,K] mu = x * beta;
  for(k in 1:K) {
    matrix[N,N] sigma_n = rep_matrix(0,N,N);
    for(p in 1:P) {
      sigma_n = sigma_n + var_jp[k][J+p] * R[p];
    }
    mu[,k] += cholesky_decompose(add_diag(sigma_n, 1e-8)) * beta_random_raw[,k]; // inefficient way to incorporate random effects allowing for low rank. ideally should just deal with full-rank matrices and indices
  } // correlation of samples is structured by variance partitioning of random effects

  // priors
  sigma_k ~ std_normal(); // if all variables are pre-standardized, this is a strong prior but in the right order of magnitude
  to_vector(beta_fixed_raw) ~ std_normal(); // 'fixed effects' are not really distinct from 'random effects', having the same implicit priors, but we want these coefficients because they're interesting 
  to_vector(beta_random_raw) ~ std_normal();
  L_k ~ lkj_corr_cholesky(1); // correlation of response variables is estimated without constraints, with uniform prior

  // likelihood
  (yy - mu) ~ multi_gp_cholesky(L_k_s, ones_vector(N));
}
