
# power mean for any s
# to prevent numerical underflow, return minimum when mean is rounded to zero
function powmean(m::Array{T}, s::T) where T <: Union{Float32, Float64}
  res = mean(m.^s)^(1/s)
  if res > 0
    return( res )
  else
    return( minimum(m))
  end
end


"""Implements generalized k-means clustering. The variable 'class' should enter
with an initial guess of the classifications."""
function power_kmeans(X::Matrix{T}, s::T,
  k::Integer, center::Array{T}) where T <: Union{Float32, Float64}

  (m_features, N_points) = size(X)
  #(center, weights) = (zeros(T, m_features,k), zeros(T, k, N_points))
  weights =  zeros(T, k, N_points)
  #center = mean(X,2).+5*randn(m_features,k)  #initialization
  obj_old = 1e300; obj = 1e200
  iter = 0
  while (obj_old - obj)/(obj_old*sqrt(m_features)) > 1e-7 || s > -sqrt(m_features)
    iter +=1
    dist = pairwise(Euclidean(), center, X) #rows: centers; columns: distance from center to point i 1 thru N
    #println(dist[:,1])
    #dist[isnan.(dist)] = eps()

    coef = sum( dist.^(2*s), 1).^(1/s-1)
    if minimum(coef) < 1e-280 #check for underflow
      println("coef vector small")
      break
    end
    #println(coef[:,1])
    #update weights
    weights = dist.^(2*(s-1)).*coef
    #println(minimum(weights))
    if minimum(weights) < 1e-280 #check for underflow
      println("weight vector small")
      break
    end
    #weights[weights.< eps()] = 3*eps() #make sure non zero
    #weights[isnan.(weights)] = 3*eps()

    #update centers
    center = (X*weights')./ sum(weights,2)'

    obj_old = obj
    obj_temp = 0
    for j in 1:N_points
      obj_temp += powmean(dist[:,j].^2,s)
    end
    obj = obj_temp
    #println(obj)
    #anneal the s value
    if iter % 2 == 0
      if s > -1.0
        s += -.2
      elseif s > -120.0
        s *= 1.06
        #println( (obj_old - obj)/obj_old )
      end
    end
  end

  #print(center)
  #assign labels:
  class = rand(1:k, N_points)
  dist = pairwise(Euclidean(), center, X) # fetch distances
  for point = 1:N_points
    class[point] = argmin(dist[:, point]) # closest center
  end
  println("powermeans final s: $s number of iters: $iter")
  #println(iter)
  return( class, center)
end

"""Implements regular k means"""
function kmeans(X::Matrix{T}, class::Vector{Int},
  k::Integer) where T <: Real
  (features, points) = size(X)
  (center, members) = (zeros(T, features, k), zeros(Int, k))
  switched = true
  iters = 0
  while switched # iterate until membership stabilizes
    iters += 1
    fill!(center, zero(T)) # update centers
    fill!(members, 0)
    for point = 1:points
      i = class[point]
      center[:, i] = center[:, i] + X[:, point]
      members[i] = members[i] + 1
    end
    for j = 1:k
      center[:, j] = center[:, j] / max(members[j], 1)
    end
    switched = false # update classes
    dist = pairwise(Euclidean(), center, X) # fetch distances
    for point = 1:points
      j = argmin(dist[:, point]) # closest center
      if class[point] != j
        class[point] = j
        switched = true
      end
    end
  end
  println("Lloyd iters: $iters")
  return (class, center)
end

#function get_classes(X::Matrix{T}, center::Array{T}) where T <: Union{Float32, Float64}
function get_classes(X, center)
  points = size(X,2)
  class = zeros(Int,points)
  dist = pairwise(Euclidean(), center, X) # fetch distances
  for point = 1:points
    class[point] = argmin(dist[:, point]) # closest center
  end
  return class
end

# kmeans plusplus initialization for classes, modified from Clustering.jl
function initclass(X, k::Int) 
  points = size(X, 2)
  iseeds = zeros(Int, k)
  class = zeros(Int, points)
  p = rand(1:points)
  iseeds[1] = p
  if k > 1
    mincosts = Distances.colwise(Euclidean(), X, view(X,:,p))
    mincosts[p] = 0
#
# Pick remaining seeds with a chance proportional to mincosts.
    tmpcosts = zeros(points)
    for j = 2:k
      p = wsample(1:points, mincosts)
      iseeds[j] = p
      c = view(X,:,p)
      Distances.colwise!(tmpcosts, Euclidean(), X, view(X,:,p))
      updatemin!(mincosts, tmpcosts)
      mincosts[p] = 0
    end
  end
  dist = pairwise(Euclidean(), X[:, iseeds], X) # fetch distances
  for point = 1:points
    class[point] = argmin(dist[:, point]) # closest center
  end
  return class
end


#taken from Clustering.jl; returns the centers
function initcenters(X::Matrix{T}, k::Int) where T <: Real
    n = size(X, 2)
    iseeds = zeros(Int, k)
    # randomly pick the first center
    p = rand(1:n)
    iseeds[1] = p

    if k > 1
        mincosts = Distances.colwise(Euclidean(), X, view(X,:,p))
        mincosts[p] = 0

        # pick remaining (with a chance proportional to mincosts)
        tmpcosts = zeros(n)
        for j = 2:k
            p = wsample(1:n, mincosts)
            iseeds[j] = p
            # update mincosts
            c = view(X,:,p)
            Distances.colwise!(tmpcosts, Euclidean(), X, view(X,:,p))
            updatemin!(mincosts, tmpcosts)
            mincosts[p] = 0
        end
    end
    return X[:, iseeds]
end

#taken from Clustering.jl; returns the centers
function initseeds(X::Matrix{T}, k::Int) where T <: Real
    n = size(X, 2)
    iseeds = zeros(Int, k)
    # randomly pick the first center
    p = rand(1:n)
    iseeds[1] = p

    if k > 1
        mincosts = Distances.colwise(Euclidean(), X, view(X,:,p))
        mincosts[p] = 0

        # pick remaining (with a chance proportional to mincosts)
        tmpcosts = zeros(n)
        for j = 2:k
            p = wsample(1:n, mincosts)
            iseeds[j] = p
            # update mincosts
            c = view(X,:,p)
            Distances.colwise!(tmpcosts, Euclidean(), X, view(X,:,p))
            updatemin!(mincosts, tmpcosts)
            mincosts[p] = 0
        end
    end
    return iseeds
end

function seed2center(iseeds::Array{Int},X::Matrix{T}) where T <: Real
  return(X[:,iseeds])
end

function seed2class(iseeds::Array{Int},X::Matrix{T}) where T <: Real
  points = size(X,2)
  class = zeros(Int, points)
  dist = pairwise(Euclidean(), X[:, iseeds], X) # fetch distances
  for point = 1:points
    class[point] = argmin(dist[:, point]) # closest center
  end
  return class
end


#helper function
function updatemin!(r::AbstractArray, x::AbstractArray)
  n = length(r)
  length(x) == n || throw(DimensionMismatch("Inconsistent array lengths."))
  @inbounds for i = 1:n
    xi = x[i]
    if xi < r[i]
      r[i] = xi
    end
  end
  return r
end


#taken from clustering.jl
function ARI(c1,c2)
    # rand_index - calculates Rand Indices to compare two partitions
    # (AR, RI, MI, HI) = rand(c1,c2), where c1,c2 are vectors listing the
    # class membership, returns the "Hubert & Arabie adjusted Rand index".
    # (AR, RI, MI, HI) = rand(c1,c2) returns the adjusted Rand index,
    # the unadjusted Rand index, "Mirkin's" index and "Hubert's" index.
    #
    # See L. Hubert and P. Arabie (1985) "Comparing Partitions" Journal of
    # Classification 2:193-218

    c = counts(c1,c2,(1:maximum(c1),1:maximum(c2))) # form contingency matrix

    n = round(Int,sum(c))
    nis = sum(sum(c,2).^2)        # sum of squares of sums of rows
    njs = sum(sum(c,1).^2)        # sum of squares of sums of columns

    t1 = binomial(n,2)            # total number of pairs of entities
    t2 = sum(c.^2)                # sum over rows & columnns of nij^2
    t3 = .5*(nis+njs)

    # Expected index (for adjustment)
    nc = (n*(n^2+1)-(n+1)*nis-(n+1)*njs+2*(nis*njs)/n)/(2*(n-1))

    A = t1+t2-t3;        # no. agreements
    D = -t2+t3;          # no. disagreements

    if t1 == nc
        # avoid division by zero; if k=1, define Rand = 0
        ARI = 0
    else
        # adjusted Rand - Hubert & Arabie 1985
        ARI = (A-nc)/(t1-nc)
    end

    #RI = A/t1            # Rand 1971      # Probability of agreement
    #MI = D/t1            # Mirkin 1970    # p(disagreement)
    #HI = (A-D)/t1        # Hubert 1977    # p(agree)-p(disagree)

    return ARI
end


function VI(k1::Int, a1::AbstractVector{Int},
                 k2::Int, a2::AbstractVector{Int})
    # check input arguments
    n = length(a1)
    length(a2) == n || throw(DimensionMismatch("Inconsistent array length."))

    # count & compute probabilities
    p1 = zeros(k1)
    p2 = zeros(k2)
    P = zeros(k1, k2)

    for i = 1:n
        @inbounds l1 = a1[i]
        @inbounds l2 = a2[i]
        p1[l1] += 1.0
        p2[l2] += 1.0
        P[l1, l2] += 1.0
    end

    for i = 1:k1
        @inbounds p1[i] /= n
    end
    for i = 1:k2
        @inbounds p2[i] /= n
    end
    for i = 1:(k1*k2)
        @inbounds P[i] /= n
    end

    # compute variation of information
    H1 = entropy(p1)
    H2 = entropy(p2)

    I = 0.0
    for j = 1:k2, i = 1:k1
        pi = p1[i]
        pj = p2[j]
        pij = P[i,j]
        if pij > 0.0
            I += pij * log(pij / (pi * pj))
        end
    end

    return H1 + H2 - I * 2.0
end

# X is the d x n matrix of data; center is the d x k centers
function kmeans_obj(center,X)
  sum( minimum(pairwise(Euclidean(), center, X),1) )
end

function kgen_obj(center,X,s)
  temp = pairwise(Euclidean(), center,X)
  res = 0.0
  for i = 1:size(X,2)
    res += powmean(temp[:,i],s)
  end
  return res
end

"""
    randindex(a, b) -> NTuple{4, Float64}
Compute the tuple of Rand-related indices between the clusterings `c1` and `c2`.
`a` and `b` can be either [`ClusteringResult`](@ref) instances or
assignments vectors (`AbstractVector{<:Integer}`).
Returns a tuple of indices:
  - Hubert & Arabie Adjusted Rand index
  - Rand index (agreement probability)
  - Mirkin's index (disagreement probability)
  - Hubert's index (``P(\\mathrm{agree}) - P(\\mathrm{disagree})``)
# References
> Lawrence Hubert and Phipps Arabie (1985). *Comparing partitions.*
> Journal of Classification 2 (1): 193–218
> Meila, Marina (2003). *Comparing Clusterings by the Variation of
> Information.* Learning Theory and Kernel Machines: 173–187.
"""
function randindex(a, b)
    c = counts(a, b)

    n = sum(c)
    nis = sum(abs2, sum(c, dims=2))        # sum of squares of sums of rows
    njs = sum(abs2, sum(c, dims=1))        # sum of squares of sums of columns

    t1 = binomial(n, 2)                    # total number of pairs of entities
    t2 = sum(abs2, c)                      # sum over rows & columnns of nij^2
    t3 = .5*(nis+njs)

    # Expected index (for adjustment)
    nc = (n*(n^2+1)-(n+1)*nis-(n+1)*njs+2*(nis*njs)/n)/(2*(n-1))

    A = t1+t2-t3;        # agreements count
    D = -t2+t3;          # disagreements count

    if t1 == nc
        # avoid division by zero; if k=1, define Rand = 0
        ARI = 0
    else
        # adjusted Rand - Hubert & Arabie 1985
        ARI = (A-nc)/(t1-nc)
    end

    RI = A/t1            # Rand 1971      # Probability of agreement
    MI = D/t1            # Mirkin 1970    # p(disagreement)
    HI = (A-D)/t1        # Hubert 1977    # p(agree)-p(disagree)

    return (ARI, RI, MI, HI)
end
