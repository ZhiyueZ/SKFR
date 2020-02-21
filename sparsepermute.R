source("kmeansClustering.R")
library(foreach)
library(doParallel)


sparsekmeans1.permute<-function (x, class, K, sbounds, nperms=25, silent=FALSE) {
  permx <- list()

  for (i in 1:nperms) {
    permx[[i]] <- matrix(NA, nrow = nrow(x), ncol = ncol(x))
    for (j in 1:ncol(x)) permx[[i]][, j] <- sample(x[, j])
  }
  
  tots <- NULL
  out<-list()
  iternum<-1
  for(s in sbounds){
    out[[iternum]] <- sparsekmeans1(t(x),class, K, s)
    iternum<-iternum+1
    }
  for (i in 1:length(out)) {
    tots <- c(tots, (out[[i]]$TSS-sum(out[[i]]$WSS)))
  }
  permtots <- matrix(NA, nrow = length(sbounds), ncol = nperms)


  for (k in 1:nperms) {
    if (!silent){
      print(k)
    }
    perm.out<-list()
    iternum2<-1
    for(s in sbounds){
      perm.out[[iternum2]] <- sparsekmeans1(t(permx[[k]]), class, K, s)
      iternum2<-iternum2+1 
    }
    for (i in 1:length(perm.out)) {
      permtots[i, k] <- (perm.out[[i]]$TSS-sum(perm.out[[i]]$WSS))
    }
  }

  
  gaps <- (log(tots) - apply(log(permtots), 1, mean))
  out <- list(tots = tots, permtots = permtots,
              gaps = gaps, sdgaps = apply(log(permtots), 1, sd), sbounds = sbounds, 
              bests = sbounds[which.max(gaps)])
  return(out)
}

sparsekmeans2.permute<-function (x, class, K, sbounds, nperms=25, silent=FALSE) {
  permx <- list()
  
  for (i in 1:nperms) {
    permx[[i]] <- matrix(NA, nrow = nrow(x), ncol = ncol(x))
    for (j in 1:ncol(x)) permx[[i]][, j] <- sample(x[, j])
  }
  
  tots <- NULL
  out<-list()
  iternum<-1
  for(s in sbounds){
    out[[iternum]] <- sparsekmeans2(t(x),class, K, s)
    iternum<-iternum+1
  }
  for (i in 1:length(out)) {
    tots <- c(tots, (out[[i]]$TSS-sum(out[[i]]$WSS)))
  }
  permtots <- matrix(NA, nrow = length(sbounds), ncol = nperms)
  
  
  for (k in 1:nperms) {
    if (!silent){
      print(k)
    }
    perm.out<-list()
    iternum2<-1
    for(s in sbounds){
      perm.out[[iternum2]] <- sparsekmeans2(t(permx[[k]]), class, K, s)
      iternum2<-iternum2+1 
    }
    for (i in 1:length(perm.out)) {
      permtots[i, k] <- (perm.out[[i]]$TSS-sum(perm.out[[i]]$WSS))
    }
  }
  
  
  gaps <- (log(tots) - apply(log(permtots), 1, mean))
  out <- list(tots = tots, permtots = permtots,
              gaps = gaps, sdgaps = apply(log(permtots), 1, sd), sbounds = sbounds, 
              bests = sbounds[which.max(gaps)])
  return(out)
}
