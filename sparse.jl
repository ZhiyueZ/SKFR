using StatsBase, Distances,DelimitedFiles

"""Implements sparse kmeans clustering. The variable class should
enter with an initial guess of the classifications."""

#The first version of SKFR
# Input:
# data p by n
# initial class, a vector
# number of classes k
# sparsity level
# Output:
# class labels
# class centers p by k
# informative feature indices, a vector
# Within cluster sum of squares WSS, a vector
# total sum of squares TSS

function sparsekmeans1(X, class::Vector{Int}, classes::Int,
  sparsity::Int)

  (features, cases) = size(X)
  center = zeros( features, classes)
  members = zeros(Int, classes)
  criterion = zeros( features)
  distance = zeros( classes, cases)
  wholevec=1:features
  for i = 1:features # normalize each feature
    X[i, :] = zscore(X[i, :])
  end
  switched = true
  selectedvec=zeros(sparsity)
  while switched # iterate until class assignments stabilize
    center=zeros( features, classes)
    members = zeros(Int, classes)
    for case = 1:cases
      i = class[case]
      center[:, i] = center[:, i] + X[:, case]
      members[i] = members[i] + 1
    end
    for j = 1:classes
      if members[j] > 0
        center[:, j] = center[:, j] ./ members[j]
      end
    end
    for i = 1:features # compute the sparsity criterion
      criterion[i] = 0
      for j = 1:classes
        criterion[i] = criterion[i] + members[j] * center[i, j]^2
      end
    end
    #J = partialsortperm(criterion, 1:sparsity, rev = true)
    # find the (p-s) least informative features and setting them to 0
J=partialsortperm(criterion,1:(features-sparsity),rev=false)
center[J,:]=zeros(length(J),classes)
selectedvec=setdiff(wholevec,J)
    distance = pairwise(Euclidean(), center, X, dims = 2)
    switched = false # update classes
    for case = 1:cases
      j = argmin(distance[:, case]) # class of closest center
      if j != class[case]
        switched = true
        class[case] = j
      end
    end
end
    # now calculating the WSS and TSS; used in the permutation test and sparse kpod
    WSSval=zeros(classes)
    for k=1:classes
        tempIndex=findall(class.==k)
        tempX=X[:,tempIndex]
        tempX=transpose(tempX)
        for r=1:size(tempX,2)
            tempX[:,r]=tempX[:,r].-mean(tempX[:,r])
        end
        WSSval[k]=sum(tempX.^2)
  end
  Xtran=transpose(X)
  for r=1:size(Xtran,2)
      Xtran[:,r]=Xtran[:,r].-mean(Xtran[:,r])
  end
  TSSval=sum(Xtran.^2)
  return (class, center,selectedvec,WSSval,TSSval)
end

"""Implements sparse kmeans clustering and feature selection within
each class. The variable class should enter with an initial guess of
the classifications."""
# The second version of SKFR
# Input:
# data p by n
# initial class, a vector
# number of classes k
# sparsity level
#
#
# Output:
# class labels
# class centers p by k
# informative feature indices, a vector
# Within cluster sum of squares WSS, a vector
# total sum of squares TSS

function sparsekmeans2(X::Matrix{T}, class::Vector{Int}, classes::Int,
  sparsity::Int) where T <: Real

  (features, cases) = size(X)
  center = zeros(T, features, classes)
  members = zeros(Int, classes)
  criterion = zeros(T, features)
  distance = zeros(T, classes, cases)
  selectedvec=zeros(T,classes,sparsity)
  wholevec=1:features
  for i = 1:features # normalize each feature
    X[i, :] = zscore(X[i, :])
  end
  switched = true
  while switched # iterate until class assignments stabilize
    center = zeros(T, features, classes)
    members = zeros(Int, classes)
    for case = 1:cases
      i = class[case]
      center[:, i] = center[:, i] + X[:, case]
      members[i] = members[i] + 1
    end
    for i = 1:classes # revise class centers
      if members[i] > 0 # set the smallest center components to 0
        center[:, i] = center[:, i] ./ members[i]
        J = partialsortperm(center[:, i].^2 .*members[i], 1: features - sparsity, by = abs)
        center[J,i]=zeros(T,length(J))
        selectedvec[i,:]=setdiff(wholevec,J)
      end
    end
    switched = false # update classes
    (j, k) = (features - sparsity + 1, features)
    distance = pairwise(Euclidean(), center, X, dims = 2)
    for case = 1:cases
      j = argmin(distance[:, case]) # class of closest center
      if j != class[case]
        switched = true
        class[case] = j
      end
    end
  end
  WSSval=zeros(T,classes)
  for k=1:classes
      tempIndex=findall(class.==k)
      tempX=X[:,tempIndex]
      tempX=transpose(tempX)
      for r=1:size(tempX,2)
          tempX[:,r]=tempX[:,r].-mean(tempX[:,r])
      end
      WSSval[k]=sum(tempX.^2)
end
Xtran=transpose(X)
for r=1:size(Xtran,2)
    Xtran[:,r]=Xtran[:,r].-mean(Xtran[:,r])
end
TSSval=sum(Xtran.^2)
return (class, center,selectedvec,WSSval,TSSval)
end
