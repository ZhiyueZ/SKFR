using Distances;
using StatsBase;
using Plots;
using DataFrames;
using Statistics;
using CSV;
include("k_generalized_source.jl")
include("sparse.jl")
include("sparsepermute.jl")
include("sparsekpod.jl")

## Starting with the core algorithm
# a simple experiment with n=300 p=100 sparsity level s=33. X is a simple 3 cluster data
(features, cases) = (100, 300);
(classes, sparsity)  = (3, 33);
X = randn(features, cases);
(m, n) = (div(features, 3), 2 * div(features, 3));
(r, s) = (div(cases, 3) + 1, 2 * div(cases, 3));
X[1:m, r:s] = X[1:m, r:s] .+ 1.0;
X[1:m, s + 1:end] = X[1:m, s + 1:end] .+ 2.0;
# getting true labels
truelabels=[];
class1labels=ones(100);
append!(truelabels,class1labels);
class2labels=ones(100)*2;
append!(truelabels,class2labels);
class3labels=ones(100)*3;
append!(truelabels,class3labels);
# k means++ initial seeding
class =initclass(X,classes);
X1=copy(X);
X2=copy(X);

class1=copy(class);
class2=copy(class);
# classout is the output cluster labels, center is the output cluster centers,
# selectedvec contains the top s most informative features.
#WSSval= within cluster sum of squares; TSSval=total sum of squares
@time (classout1, center,selectedvec,WSSval,TSSval) = sparsekmeans1(X1, class1, classes,m);
@time (classout2, center,selectedvec,WSSval,TSSval) = sparsekmeans2(X2, class2, classes,m);

arisparse1=randindex(classout1, convert(Array{Int64,1},truelabels))
println("ARI of SKFR1: ",arisparse1[1])
arisparse2=randindex(classout2, convert(Array{Int64,1},truelabels))
println("ARI of SKFR2: ",arisparse2[1])






## Now replacing 10% of the entries with missing values to test sparsekpod function

# replacing 10% of entries at random
missingix=sample(1:features*cases,Int(features*cases*0.1),replace=false)
 y = convert(Array{Union{Missing,Float64},2}, X)
# y is the partially observed version of X above
y[ CartesianIndices(y)[missingix]]=missings(Float64, length(missingix))

## The first output argument is the cluster labels, and the rest are not of importance in this example.
(classout3,aa,bb,cc,dd)=sparsekpod(copy(y'),classes,m)
arisparse3=randindex(classout3, convert(Array{Int64,1},truelabels))
println("ARI of sparsekpod: ",arisparse3[1])


## real data  the wine data
#reading the data
dataorig = CSV.read("wine.dat");

# taking out the class labels
nolabel=dataorig[:,1:13];
class=initclass(copy(transpose(convert(Matrix,nolabel))),3);
#without the knowledge of the sparsity level, we use the permutation test and find the best sparsity level
# that corresponds to the highest gap statistic
# tots is the BSS of the dataset, and permtots are the BSS of the permuted sets (O(X_b,s))
#gaps are the gap statistics
(tots,permtots,gaps,sdgaps,sbounds,bests)=sparsekmeans1permute(convert(Matrix,nolabel),class,3,1:13,25,false);
xs = range(1,stop=13)

# plotting the gap statistic of each sparsity level, with the standard deviations being the error bars
plot(xs,gaps,yerror=sdgaps,legend=false,xlabel="sparsity level s", ylabel="gap statistic",title="permutation test")

println("best sparsity level: ",bests)
class=initclass(copy(transpose(convert(Matrix,nolabel))),3);
class1=copy(class);
class2=copy(class);
# now we use the sparsity level determined by the permutation test to do sparse kmeans clustering
(classout, centerout,selectedvec,WSSval,TSSval)=sparsekmeans1(copy(transpose(convert(Matrix,nolabel))),class1,3,bests);
(class_kmean, center_kmean) = kmeans(copy(transpose(convert(Matrix,nolabel))), class2, 3);
arisparse=randindex(classout, dataorig[:,14])
arikmean=randindex(class_kmean, dataorig[:,14])
println("ARI of SKFR1: ",arisparse[1])
println("ARI of kmeans: ",arikmean[1])



#### a more complicated data generator, similar to the experiment described in section 5.3 and power k kmeans
# In this example, k means fails for the high dimensional data
# we perform 10 independent experiments, each seeded with kmeans++

trial=10;
arisparsevec=zeros(10);
arikmeansvec=zeros(10);
for i=1:trial
K = 5; N = 250; D = 2000;
r = 6;
true_centers = r*rand(K,D);
true_centers[:,21:D].=0.0;
dat = reshape(convert(Array{Float64,1},[]),D,0);
sizes = 2*rand(K,1).+1;
sizes=round.(N*sizes/sum(sizes));
N1 = sum(sizes); diff = abs(N-N1);
sizes[1:Int(diff)] = sizes[1:Int(diff)].+ sign(N-N1); #adjust so sum to N
sz = convert(Array{Int,2},sizes);
true_labels=[];
for k = 1:K
  cluster = randn(sz[k],D);
  mean_temp = sum(cluster)/(sz[k]*D);
  Sk= repeat(true_centers[k,:].-mean_temp,1,sz[k]) + cluster';
  noisemat=zeros(sz[k],D);
  tempmat=randn(sz[k],D-20);
  noisemat[:,21:D]=tempmat;
  Sk=Sk+noisemat';
   dat = [dat Sk];
  append!(true_labels,ones(sz[k])*k)
end

class0=initclass(dat,K)
dat1=copy(dat)
dat2=copy(dat)
class1=copy(class0)
class2=copy(class0)

(classout, center,selectedvec,WSSval,TSSval)=sparsekmeans1(dat1,class1,K,20);

(class_kmean, center_kmean) = kmeans(dat2, class2, K);
arisparse=randindex(classout, convert(Array{Int64,1},true_labels))
arikmeans=randindex(class_kmean, convert(Array{Int64,1},true_labels))
arisparsevec[i]=arisparse[1];
arikmeansvec[i]=arikmeans[1];
end

# plotting the performance in terms of ARI of the 10 experiments
xaxis=1:trial
plot(xaxis,[arisparsevec,arikmeansvec],lw=3,title="ARI performance of SKFR and kmeans",label=["SKFR" "kmeans"])
xlabel!("Trial number")
ylabel!("ARI")
