source("kmeansClustering.R")
require(kpodclustr)

assign_clustppSparse <-function (X, k, init_centers,sparsity, kmpp_flag = TRUE, max_iter = 20) 
{
  init_classes<- get_classes(t(X),t(init_centers))
  res <- sparsekmeans1(t(X), init_classes, k, sparsity)
  clusts <- res$m1
  obj <- res$TSS
  fit <- 1 - (sum(res$WSS)/res$TSS)
  centers <- t(res$m2)
  if (kmpp_flag == TRUE) {
    for (iter in 1:max_iter) {
      #centers_kmpp <- kmpp(X, length(res$size))
      centers_kmpp <- kmpp(X, k)
      #sol <- kmeans(X, centers_kmpp)
      classes_kmpp <- get_classes(t(X),t(centers_kmpp))
      sol <- sparsekmeans1(t(X), classes_kmpp, k, sparsity)
      if (sol$TSS < obj) {
        obj <- sol$TSS
        clusts <- sol$m1
        fit <- 1 - (sum(sol$WSS)/sol$TSS)
        centers <- t(sol$m2)
        break
      }
    }
  }
  return(list(clusts = clusts, obj = obj, centers = centers, 
              fit = fit))
}

sparsekpod<-function (X, k, sparsity, kmpp_flag = TRUE, maxiter = 20) 
{
  n <- nrow(X)
  p <- ncol(X)
  cluster_vals <- vector(mode = "list", length = maxiter)
  obj_vals <- double(maxiter)
  fit <- double(maxiter)
  missing <- findMissing(X)
  X_copy <- initialImpute(X)
  init_centers <- kmpp(X_copy, k)
  init_classes<- get_classes(t(X_copy),t(init_centers))
  temp <- sparsekmeans1(t(X_copy), init_classes, k, sparsity)
  clusts <- temp$m1
  centers <- t(temp$m2)
  fit[1] <- 1 - (sum(temp$WSS)/temp$TSS)
  clustMat <- centers[clusts, ]
  X_copy[missing] <- clustMat[missing]
  obj_vals[1] <- sum((X[-missing] - clustMat[-missing])^2)
  cluster_vals[[1]] <- clusts
  for (i in 2:maxiter) {
    temp <- assign_clustppSparse(X_copy, k, centers, sparsity, kmpp_flag)
       clusts <- temp$clusts
    centers <- temp$centers
    fit[i] <- temp$fit
    clustMat <- centers[clusts, ]
    X_copy[missing] <- clustMat[missing]
    obj_vals[i] <- sum((X[-missing] - clustMat[-missing])^2)
    cluster_vals[[i]] <- clusts
    if (all(cluster_vals[[i]] == cluster_vals[[i - 1]])) {
      noquote("Clusters have converged.")
      return(list(cluster = clusts, cluster_list = cluster_vals[1:i], 
                  obj_vals = obj_vals[1:i], fit = fit[i], fit_list = fit[1:i]))
      break
    }
  }
  return(list(cluster = clusts, cluster_list = cluster_vals[1:i], 
              obj_vals = obj_vals[1:i], fit = fit[i], fit_list = fit[1:i]))
}




kpodprint<-function (X, k, kmpp_flag = TRUE, maxiter = 100) 
{
  n <- nrow(X)
  p <- ncol(X)
  cluster_vals <- vector(mode = "list", length = maxiter)
  obj_vals <- double(maxiter)
  fit <- double(maxiter)
  missing <- findMissing(X)
  X_copy <- initialImpute(X)
  init_centers <- kmpp(X_copy, k)
  temp <- kmeans(X_copy, init_centers)
  clusts <- temp$cluster
  centers <- temp$centers
  fit[1] <- 1 - (sum(temp$withinss)/temp$totss)
  clustMat <- centers[clusts, ]
  X_copy[missing] <- clustMat[missing]
  obj_vals[1] <- sum((X[-missing] - clustMat[-missing])^2)
  cluster_vals[[1]] <- clusts
  for (i in 2:maxiter) {
    print("this is i")
    print(i)
    temp <- assign_clustppprint(X_copy, centers, kmpp_flag)
    clusts <- temp$clusts
    centers <- temp$centers
    fit[i] <- temp$fit
    clustMat <- centers[clusts, ]
    X_copy[missing] <- clustMat[missing]
    obj_vals[i] <- sum((X[-missing] - clustMat[-missing])^2)
    cluster_vals[[i]] <- clusts
    if (all(cluster_vals[[i]] == cluster_vals[[i - 1]])) {
      noquote("Clusters have converged.")
      return(list(cluster = clusts, cluster_list = cluster_vals[1:i], 
                  obj_vals = obj_vals[1:i], fit = fit[i], fit_list = fit[1:i]))
      break
    }
  }
  return(list(cluster = clusts, cluster_list = cluster_vals[1:i], 
              obj_vals = obj_vals[1:i], fit = fit[i], fit_list = fit[1:i]))
}

assign_clustppprint<-function (X, init_centers, kmpp_flag = TRUE, max_iter = 20) 
{
  res <- kmeans(X, init_centers)
  clusts <- res$cluster
  obj <- res$totss
  fit <- 1 - (sum(res$withinss)/res$totss)
  centers <- res$centers
  if (kmpp_flag == TRUE) {
    for (iter in 1:max_iter) {
      print(iter)
      centers_kmpp <- kmpp(X, length(res$size))
      sol <- kmeans(X, centers_kmpp)
      if (sol$totss < obj) {
        obj <- sol$totss
        clusts <- sol$cluster
        fit <- 1 - (sum(sol$withinss)/sol$toss)
        centers <- sol$centers
        break
      }
    }
  }
  return(list(clusts = clusts, obj = obj, centers = centers, 
              fit = fit))
}
