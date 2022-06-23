// adapted from https://mc-stan.org/docs/2_29/stan-users-guide/multivariate-outcomes.html
functions {
  real matrix_normal_cholesky_lpdf(matrix Y, matrix X, matrix UL, matrix VL) {
    int n = rows(X);
    int p = cols(X);
    real log_num = -dot_self(to_vector(mdivide_left_tri_low(VL, mdivide_left_tri_low(UL, Y-X)')));
    real log_denom = (n * p) * log(2 * pi()) + p * sum(log(diagonal(UL))) + n * sum(log(diagonal(VL)));
    return(0.5 * (log_num - log_denom));
  }
}
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
  matrix[J,K] beta;
  cholesky_factor_corr[K] L_k;
  vector<lower=0>[K] sigma_k;
  simplex[JP1] var_jp;
  vector[M] missing;
}
transformed parameters {
  matrix[N,K] yy = y;
  for(i in 1:M) {
    yy[m[i,1],m[i,2]] = missing[i];
  }
}
model {
  matrix[K,K] L_k_s = diag_pre_multiply(sigma_k, L_k);
  matrix[N,N] L_n = rep_matrix(0,N,N);
  for(p in 1:P) {
    L_n = L_n + var_jp[J+p] * R[p];
  }
  L_n = cholesky_decompose(add_diag(L_n, var_jp[JP1] + 1e-10)); // correlation of samples is structured by variance partitioning of random effects
  
  // priors
  sigma_k ~ std_normal(); // if all variables are pre-standardized, this is a strong prior but in the right order of magnitude
  beta ~ multi_gp_cholesky(L_k_s, inv(var_jp[1:J])); // 'fixed effects' are not really distinct from 'random effects', having the same implicit priors, but we want these coefficients because they're interesting 
  L_k ~ lkj_corr_cholesky(1); // correlation of response variables is estimated without constraints, with uniform prior

  // likelihood
  yy ~ matrix_normal_cholesky(x * beta, L_n, L_k_s); // all response variables use the same correlation structure
}
