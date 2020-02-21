# SKFR
Sparse k-means clustering with feature selection, available in Julia and R. Reference: https://arxiv.org/abs/2002.08541

## Julia Files
The Julia functions are compatible with Julia V1.1.1.  The main functions, SKFR1 and SKFR2 are in sparse.jl, and the extension of the algorithm that is suitable for partially observed data is in sparsekpod.jl. The permutation test to select the sparsity level of SKFR is given in sparsepermute.jl. k_generalized_source.jl contains helper functions including k-means++ initial seeding. Finally, a demo file is provided. 


## R Files
kmeansClustering.R contains SKFR1 and SKFR2. sparsekpod.R contains the partially observed data extension function. sparsepermute.R has the permutation test. 
