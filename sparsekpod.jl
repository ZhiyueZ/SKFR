include("sparse.jl")
include("k_generalized_source.jl")
using StatsBase, Distances,DelimitedFiles

function assign_clustppSparse(X, k, init_centers,sparsity,
 kmpp_flag= true, max_iter= 20)
  init_classes= get_classes(copy(X'),copy(init_centers'))
  (clusts, centerout,selectedvec,WSS,obj) = sparsekmeans1(X', init_classes, k, sparsity)
  fit = 1 - (sum(WSS)/obj)
  centers = copy(centerout')
  if kmpp_flag == true
    for iter = 1:max_iter
      classes_kmpp = initclass(copy(X'), k)
      (newclusts, newcenterout,selectedvec,newWSS,newobj) = sparsekmeans1(X', classes_kmpp, k, sparsity)
      if newobj < obj
        obj = newobj
        clusts = newclusts
        fit = 1 - (sum(newWSS)/newobj)
        centers = copy(newcenterout')
        break
    end
  end
end
  return(clusts, obj, centers,fit)
end

function findMissing(X)
missing_all=findall(ismissing.(X))
return(missing_all)
end

function initialImpute(X)
    avg = mean(skipmissing(vec(X)))
    X[findall(ismissing.(X))] .= avg
    return(X)
end

# sparsekpod function, doing SKFR1 on partially observed data
# Input:
# data   n by p
# number of classes k
# sparsity level s
# kmpp_flag  boolean for using k means plus plus seeding. default true
# max iteration, default= 20
#
#
# Output:
# class labels
# class labels of each iteration
# TSS of each iteration
# fit= 1 - (sum(WSS)/obj)  (refer to Chi's paper or our manuscript)
# fit of each iteration


function sparsekpod(X, k, sparsity, kmpp_flag::Bool = true,
     maxiter::Int = 20)

  n = size(X,1)
  p = size(X,2)
  cluster_vals = zeros(maxiter,n)
  obj_vals = []
  fit = []
  missingindices = findMissing(X)
  nonmissingindices=setdiff(CartesianIndices(X)[1:end],missingindices)
  X_copy = initialImpute(X)
  X_copy=convert(Array{Float64,2}, X_copy)
  init_classes = initclass(copy(X_copy'), k)
  (clusts, centerout,selectedvec,WSS,obj)= sparsekmeans1(X_copy', init_classes, k, sparsity)
  centers = copy(centerout')
  append!(fit,1 - (sum(WSS)/obj))
  clustMat = centers[clusts, :]
  X_copy[missingindices] = clustMat[missingindices]
  append!(obj_vals,sum((X_copy[nonmissingindices] .- clustMat[nonmissingindices]).^2))
  cluster_vals[1,:]=clusts

  for i = 2:maxiter
    (tempclusts, tempobj, tempcenters,tempfit)= assign_clustppSparse(X_copy, k, centers, sparsity, kmpp_flag)
    clusts = tempclusts
    centers =tempcenters
    append!(fit,tempfit)
    clustMat = centers[clusts,:]
    X_copy[missingindices] =clustMat[missingindices]
    append!(obj_vals,sum((X_copy[nonmissingindices] .- clustMat[nonmissingindices]).^2))
    cluster_vals[i,:] = clusts
    if (all(cluster_vals[i,:] == cluster_vals[i - 1,:]))
      println("Clusters have converged.")
      return(clusts, cluster_vals[1:i,:],obj_vals[1:i],fit[i],fit[1:i])
      break
  end
end
  return(clusts, cluster_vals[1:i,:],obj_vals[1:i],fit[i],fit[1:i])
end
