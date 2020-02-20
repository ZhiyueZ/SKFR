include("sparse.jl")
using StatsBase, Distances,DelimitedFiles

# permutation test for SKFR1
# Input:
# data  n by p (the more common format of real data.)
# initial class, a vector
# number of classes k
# the range of sparsity levels, an Integer vector
# number of permuted datasets
# boolean for reading out the process
#
#
# Output:
# O(X,s)=BSS, refer to line 225 of the manuscript
# O(X_b,s)
# gap statistics for each sparsity level
# standard deviation of the gap statistics
# sparsity levels
# best sparsity level (corresponding to the highest gap statistic)



function sparsekmeans1permute(x::Matrix{T}, class::Vector{Int}, K::Int, sbounds,
    nperms::Int=15, silent::Bool=false) where T <: Real
#println("test")
  permx=zeros(T,size(x,1),size(x,2),nperms)
  for i = 1:nperms
        tempX=zeros(T, size(x,1), size(x,2))
        for j=1:size(x,2)
            tempX[:,j] = sample(x[:, j],length(x[:,j]),replace=false)
        end
        permx[:,:,i]=tempX
end

  tots =[]
  for s in sbounds
#      println(s)
    (classout, centerout,selectedvec,WSSval,TSSval)= sparsekmeans1(copy(x'),class, K, s)
    append!(tots,TSSval-sum(WSSval))
end
#println(tots)
  permtots = zeros(T, length(sbounds), nperms)


  for k = 1:nperms
    if !silent
      println(k)
  end
    iternum=1
    for s in sbounds
      (classout, centerout,selectedvec,WSSval,TSSval)= sparsekmeans1(copy(transpose(permx[:,:,k])), class, K, s)
      permtots[iternum,k]=TSSval-sum(WSSval)
      iternum=iternum+1
  end
end

gaps=zeros(T,length(sbounds))
for i=1:length(tots)
gaps[i]=log.(tots[i])-mean(log.(permtots[i,:]))
end

sdgaps=zeros(T,length(sbounds))
for i=1:length(sdgaps)
    sdgaps[i]=std(log.(permtots[i,:]))
end
bests=sbounds[argmax(gaps)]
  return(tots,permtots,gaps,sdgaps,sbounds,bests)
end


# permutation test for SKFR2
# Input:
# data  n by p (the more common format of real data.)
# initial class, a vector
# number of classes k
# the range of sparsity levels, an Integer vector
# number of permuted datasets
# boolean for reading out the process
#
#
# Output:
# O(X,s)=BSS, refer to line 225 of the manuscript
# O(X_b,s)
# gap statistics for each sparsity level
# standard deviation of the gap statistics
# sparsity levels
# best sparsity level (corresponding to the highest gap statistic)


function sparsekmeans2permute(x::Matrix{T}, class::Vector{Int}, K::Int, sbounds,
    nperms::Int=15, silent::Bool=false) where T <: Real
println("test")
  permx=zeros(T,size(x,1),size(x,2),nperms)
  for i = 1:nperms
        tempX=zeros(T, size(x,1), size(x,2))
        for j=1:size(x,2)
            tempX[:,j] = sample(x[:, j],length(x[:,j]),replace=false)
        end
        permx[:,:,i]=tempX
end

  tots =[]
  for s in sbounds
      println(s)
    (classout, centerout,selectedvec,WSSval,TSSval)= sparsekmeans2(copy(x'),class, K, s)
    append!(tots,TSSval-sum(WSSval))
end
println(tots)
  permtots = zeros(T, length(sbounds), nperms)


  for k = 1:nperms
    if !silent
      println(k)
  end
    iternum=1
    for s in sbounds
      (classout, centerout,selectedvec,WSSval,TSSval)= sparsekmeans2(copy(transpose(permx[:,:,k])), class, K, s)
      permtots[iternum,k]=TSSval-sum(WSSval)
      iternum=iternum+1
  end
end

gaps=zeros(T,length(sbounds))
for i=1:length(tots)
gaps[i]=log.(tots[i])-mean(log.(permtots[i,:]))
end

sdgaps=zeros(T,length(sbounds))
for i=1:length(sdgaps)
    sdgaps[i]=std(log.(permtots[i,:]))
end
bests=sbounds[argmax(gaps)]
  return(tots,permtots,gaps,sdgaps,sbounds,bests)
end
