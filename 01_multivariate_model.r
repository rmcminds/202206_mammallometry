print(sessionInfo())

args <- commandArgs(TRUE)

indir <- args[[1]]
model_dir <- args[[2]]
model_name <- args[[3]]
output_prefix <- args[[4]]

dir.create(output_prefix,showWarnings=F)
setwd(output_prefix)

####import data and tree

#data
ECdata <- read.csv(file.path(indir, 'EC_22Mar2021.csv')) 

#tree
OGtree <- ape::read.tree(file.path(indir, 'fullnewmammal.tre')) #read in tree with branch lengths
#this tree is from Josef's work
#This tree is written as a newick file.
#need to prune tree so that the tips match the species list in the data file after cleaning the data file.
#consider finding a new tree at http://vertlife.org/phylosubsets/

#change species names in the data to match those in the tree
changenames <- c(Aonyx_cinereus           = 'Aonyx_cinerea', 
                 Capra_aegagrus           = 'Capra_hircus',
                 Equus_ferus              = 'Equus_caballus',
                 Lagothrix_poeppigii      = 'Lagothrix_lagotricha',
                 Notamacropus_eugenii     = 'Macropus_eugenii',
                 Notamacropus_parma       = 'Macropus_parma',
                 Notamacropus_rufogriseus = 'Macropus_rufogriseus',
                 Osphranter_robustus      = 'Macropus_robustus',
                 Osphranter_rufus         = 'Macropus_rufus',
                 Panthera_uncia           = 'Uncia_uncia',
                 Papio_papio              = 'Papio_hamadryas',
                 Propithecus_coquereli    = 'Propithecus_verreauxi',
                 Rucervus_eldii           = 'Rucervus_eldi',
                 Varecia_rubra            = 'Varecia_variegata')

ECdata2 <- ECdata
for(name in names(changenames)) {
  ECdata2$Scientific.name[ECdata2$Scientific.name == name] <- changenames[[name]]
}

#filter species not in tree
ECdata2 <- ECdata2[ECdata2$Scientific.name %in% OGtree$tip.label,]
rownames(ECdata2) <- NULL

#filter species not in data
tree <- ape::drop.tip(OGtree, OGtree$tip.label[!OGtree$tip.label %in% ECdata2$Scientific.name])

#correct tree if not ultrametric
tree$edge.length[tree$edge.length==0] <- 1e-10
nH <- phytools::nodeHeights(tree)
tips <- which(tree$edge[,2] <= length(tree$tip.label))
tree$edge.length[tips] <- tree$edge.length[tips] + (max(nH[tips, 2])-nH[tips,2])
tree$node.label <- NULL

#calculate correlation matrix for Brownian evolution random effect
phylo_cov <- ape::vcv(tree,"Brownian",TRUE)

#create model matrix for 'fixed effects'
logmass_mean <- mean(ECdata2$logmass)
logmass_sd <- sd(ECdata2$logmass)
ECdata2$logmass_std <- (ECdata2$logmass - logmass_mean) / logmass_sd / 2
ECdata2$logmass_means <- sapply(seq(nrow(ECdata2)), \(x) mean(ECdata2[ECdata2$Scientific.name == ECdata2[x,'Scientific.name'],'logmass']))
logmass_means_mean <- mean(ECdata2$logmass_means)
logmass_means_sd <- sd(ECdata2$logmass_means)
ECdata2$logmass_means_std <- (ECdata2$logmass_means - logmass_means_mean) / logmass_means_sd / 2
ECdata2$logmass_diff <- sapply(seq(nrow(ECdata2)), \(x) ECdata2[x,'logmass'] - ECdata2[x,'logmass_means'])
logmass_diff_mean <- mean(ECdata2$logmass_diff)
logmass_diff_sd <- sd(ECdata2$logmass_diff)
ECdata2$logmass_diff_std <- (ECdata2$logmass_diff - logmass_diff_mean) / logmass_diff_sd / 2

x <- model.matrix(~ logmass_std + logmass_diff_std, data = ECdata2)

#create response matrix
y <- as.matrix(ECdata2[,c('logProp.TopAsym','Log.Slope','Infl.logX','logAsym.Coef','logInfl.propY','logProp.BottomAsym')]) ## ask Cynthia if proportions would be better modeled with logistic transform, and why some have negative values. were some based on counts which should be modeled directly?
quants <- vector("list", ncol(y))
for(col in 1:ncol(y)) {
  quants[[col]] <- list(mean = mean(y[,col], na.rm=TRUE), sd = sd(y[,col], na.rm=TRUE), NAs = is.na(y[,col]))
  y[,col] <- (y[,col] - quants[[col]][['mean']]) / quants[[col]][['sd']] / 2
  y[quants[[col]][['NAs']],col] <- 0
} ## standardize variables so default priors make sense. use quants to reconstruct natural scale after model fitting

#get indices for missing values
m <- which(do.call(cbind, lapply(quants, \(x) x$NAs)), arr.ind=TRUE)

#get count variables
N <- nrow(y)
K <- ncol(y)
J <- ncol(x)
M <- nrow(m)

#create random effect correlation matrices
P <- 2
R <- array(dim=c(P,N,N))
R[1,,] <- phylo_cov[ECdata2$Scientific.name,ECdata2$Scientific.name]
R[2,,] <- matrix(diag(diag(phylo_cov)), nrow(phylo_cov), dimnames=dimnames(phylo_cov))[ECdata2$Scientific.name,ECdata2$Scientific.name]

#logmass_cov <- ECdata2$logmass_std %*% t(ECdata2$logmass_std)
#R[3,,] <- logmass_cov * R[1,,] # control for ontogenetic or intraspecific effects so fixed effect represents long-term trends (does not resolve simpson's paradox)
#R[4,,] <- logmass_cov * R[2,,] # control for ontogenetic or intraspecific effects so fixed effect represents long-term trends (does not resolve simpson's paradox)

# bundle data for stan
standat <- list(x=x,
                y=y,
                N=N,
                K=K,
                J=J,
                P=P,
                R=R,
                M=M,
                m=m)

cmdstanr::write_stan_json(standat, file.path(output_prefix, 'mv_plmm_data.json'))

save.image(file.path(output_prefix, 'mv_plmm_setup.RData'))

nchains <- 4
opencl <- TRUE
nthreads <- nchains

sampling_command <- paste(paste0('./', model_name),
                         'data file=./mv_plmm_data.json',
                         'init=0.1',
                         'output',
                         paste0('file=', paste0('./', paste0(model_name, '_samples.csv'))),
                         paste0('refresh=', 1),
                         'method=sample',
                         paste0('num_chains=', nchains),
                         'algorithm=hmc',
                         #'stepsize=0.01',
                         'engine=nuts',
                         #'max_depth=14',
                         #'adapt t0=10',
                         #'delta=0.95',
                         #'kappa=0.75',
                         'num_warmup=1000',
                         'num_samples=1000',
                         #'thin=1',
                         paste0('num_threads=', nthreads),
                         ('opencl platform=0 device=2')[opencl],
                         sep=' ')

file.copy(file.path(model_dir, model_name), output_prefix)
setwd(cmdstanr::cmdstan_path())
system(paste0(c('make ', 'make STAN_OPENCL=true ')[opencl+1], file.path(output_prefix, model_name)))

setwd(output_prefix)
print(sampling_command)
print(date())
system(sampling_command)

stan.fit.var <- cmdstanr::read_cmdstan_csv(Sys.glob(file.path(output_prefix, paste0(model_name, '_samples*'))),
                                           format = 'draws_array')

nuts_params_fit <- reshape2::melt(stan.fit.var$post_warmup_sampler_diagnostics)
colnames(nuts_params_fit) <- c('Iteration','Chain','Parameter','Value')
summary(stan.fit.var$post_warmup_sampler_diagnostics)

posterior::summarise_draws(stan.fit.var$post_warmup_draws)
#posterior::summarise_draws(stan.fit.var$post_warmup_draws[,,grep('^beta$', dimnames(stan.fit.var$post_warmup_draws)[[3]])])
bayesplot::mcmc_trace(stan.fit.var$post_warmup_draws, regex_pars='^var_jp*')
bayesplot::mcmc_pairs(stan.fit.var$post_warmup_draws, regex_pars='^var_jp*', np=nuts_params_fit)
bayesplot::mcmc_pairs(stan.fit.var$post_warmup_draws, regex_pars='sigma_k', np=nuts_params_fit)



