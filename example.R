source("kmeansClustering.R")
## an example

features <- 100
cases <- 300
classes <- 3
sparsity <- 33
X<- matrix(rnorm(features*cases),features,cases)
m <- features%/%3
n <- 2*features%/%3
r <- cases%/%3+1
s <- 2*cases%/%3
X[1:m,r:s] <- X[1:m,r:s]+1.0
X[1:m,(s+1):cases] <- X[1:m,(s+1):cases]+2.0
class<-sample(1:3,300,replace = TRUE)
resultsSKFR<-sparsekmeans1(X,class,classes,33)