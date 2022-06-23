// adapted from https://mc-stan.org/docs/2_29/stan-users-guide/multivariate-outcomes.html
functions {
  real matrix_normal_cholesky_lpdf(matrix Y, matrix UL, matrix VL) {
    int n = rows(Y);
    int p = cols(Y);
    real log_num = -dot_self(to_vector(mdivide_left_tri_low(VL, mdivide_left_tri_low(UL, Y)')));
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
}
parameters {
  vector<lower=0>[K] sd_k;
  cholesky_factor_corr[K] L_k;
  simplex[JP1] var_jp;
  vector[M] missing;
}
transformed parameters {
  matrix[N,K] yy = y;
  matrix[N,N] sigma_random = diag_matrix(rep_vector(var_jp[JP1] + 1e-10, N));
  for(i in 1:M) {
    yy[m[i,1],m[i,2]] = missing[i];
  }
  for(p in 1:P) {
    sigma_random += var_jp[J+p] * R[p];
  } // correlation of samples is structured by variance partitioning of random effects
}
model {
  matrix[N,N] sigma_fixed = tcrossprod(diag_post_multiply(x, sqrt(var_jp[1:J])));

  // priors
  sd_k ~ std_normal(); // if all variables are pre-standardized, this is a strong prior but in the right order of magnitude
  L_k ~ lkj_corr_cholesky(1); // correlation of response variables is estimated without constraints, with uniform prior

  // likelihood
  yy ~ matrix_normal_cholesky(cholesky_decompose(sigma_fixed + sigma_random), diag_pre_multiply(sd_k, L_k)); // all response variables use the same correlation structure
}
generated quantities {
  matrix[J,K] beta;
  {
    matrix[N,J+N] xx
    = append_col(diag_post_multiply(x, sqrt(var_jp[1:J])),
                 cholesky_decompose(sigma_random));
    matrix[N,J] mm_ginv = generalized_inverse(xx)[1:J,]';
    beta
      = yy' * mm_ginv
        + diag_pre_multiply(sd_k, L_k)
          * to_matrix(normal_rng(zeros_vector(K*(J+N)), 1),K,J+N)
          * (identity_matrix(J+N)[,1:J] - xx' * mm_ginv);
  }
}
