// adapted from https://mc-stan.org/docs/2_29/stan-users-guide/multivariate-outcomes.html
functions {
  real matrix_normal_cholesky_lpdf(matrix Y, matrix X, array[] matrix UL, matrix VL) {
    int n = rows(X);
    int p = cols(X);
    matrix[n,p] part1 = Y-X;
    real part2 = 0;
    real log_num; 
    real log_denom;
    for(k in 1:p) {
      part1[,p] = mdivide_left_tri_low(UL[p], part1[,p]);
      part2 += sum(log(diagonal(UL[p])));
    }
    log_num = -dot_self(to_vector(mdivide_left_tri_low(VL, part1')));
    log_denom = (n * p) * log(2 * pi()) + part2 + n * sum(log(diagonal(VL)));
    return(0.5 * (log_num - log_denom));
    // the determinant of a block triangular matrix is the product of the determinants of its diagonal blocks https://djalil.chafai.net/blog/2012/10/14/determinant-of-block-matrices/
    // matrix normal is special case of kronecker; cholesky is block triangular
    // this is slightly more complicated; cholesky of almost-kronecker is not kroneker of cholesks, so maybe this construction doesn't work
  }
}
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
  array[K] matrix[N,N] L_n = rep_array(rep_matrix(0,N,N),K);
  for(k in 1:K) {
    for(p in 1:P) {
      L_n[k] = L_n[k] + var_jp[k][J+p] * R[p];
    }
    L_n[k] = cholesky_decompose(add_diag(L_n[k], var_jp[k][JP1] + 1e-10)); // correlation of samples is structured by variance partitioning of random effects
  }
  
  sigma_k ~ std_normal();
  rho ~ pareto(1,2);
  var_jp ~ dirichlet(JP1 * rho * var_jp_mu);

  to_vector(beta_raw) ~ std_normal();
  L_k ~ lkj_corr_cholesky(1);

  y ~ matrix_normal_cholesky(x * beta, L_n, L_k_s); // all response variables use the same correlation structure
}
