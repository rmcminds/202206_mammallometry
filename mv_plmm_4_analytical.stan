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
  cholesky_factor_corr[K] L_k;
  vector<lower=0>[K] sd_k;
  array[K] simplex[JP1] var_jp;
  matrix[N,K] mu_raw;
  vector[M] missing;
}
transformed parameters {
  matrix[N,K] yy = y;
  array[K] matrix[N,N] sigma_random = rep_array(diag_matrix(rep_vector(1e-8,N)),K);
  matrix[N,K] mu;
  for(i in 1:M) {
    yy[m[i,1],m[i,2]] = missing[i];
  }
  for(k in 1:K) {
    matrix[N,N] sigma_fixed = tcrossprod(diag_post_multiply(x, sqrt(var_jp[k][1:J])));
    for(p in 1:P) {
      sigma_random[k] += var_jp[k][J+p] * R[p];
    }
    mu[,k] = cholesky_decompose(sigma_fixed + sigma_random[k]) * mu_raw[,k]; // inefficient way to incorporate random effects allowing for low rank. ideally should just deal with full-rank matrices and indices
  }
}
model {
  // priors
  sd_k ~ std_normal(); // if all variables are pre-standardized, this is a strong prior but in the right order of magnitude
  to_vector(mu_raw) ~ std_normal();
  L_k ~ lkj_corr_cholesky(1); // correlation of response variables is estimated without constraints, with uniform prior

  // likelihood
  (yy - mu) ~ multi_gp_cholesky(diag_pre_multiply(sd_k, L_k), ones_vector(N));
}
generated quantities {
  matrix[J,K] beta;
  for(k in 1:K) {
    matrix[N,J+N] xx
    = append_col(diag_post_multiply(x, sqrt(var_jp[k][1:J])),
                 cholesky_decompose(sigma_random[k]));
    matrix[N,J] mm_ginv = generalized_inverse(xx)[1:J,]';
    beta[,k]
      = mu[,k]' * mm_ginv
        + normal_rng(zeros_vector(J+N), 1)
          * (identity_matrix(J+N)[,1:J] - xx' * mm_ginv);
  }
}
