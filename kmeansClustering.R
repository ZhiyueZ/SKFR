#rm(list=ls())
require(stats)
require(rdist)


# implementing z score to normalize each feature
zscore <- function(vec){
  m = mean(vec)
  s = sd(vec)
  if(s!=0){
    zs = (vec - m)/s
  } else{
    zs=vec-m
  }
  
    return(zs)
}

sparsekmeans1 <- function(X,class, classes, sparsity) {
  temp <- dim(X)
  features <- temp[1]
  cases <- temp[2]
  center <- matrix(0,features,classes)
  members <- array(0, classes)
  criterion <- array(0, features)
  distance <- matrix(0, classes, cases)
  wholevec<-1:features
  for (i in 1 : features){
    X[i,] <- zscore(X[i,])
  }
  switched <- TRUE
  iternum<-1
  importantvec<-array(0,sparsity)
  while (switched) {
    center <- matrix(0,features,classes) # update centers
    members <- array(0, classes)
    for (case in 1 : cases){ 
      i <- class[case]
      center[,i] <- center[,i] + X[,case]
      #print(center)
      members[i] <- members[i]+1
    }
    for (j in 1 : classes){ 
      if (members[j]>0){
        center[,j] <- center[,j]/members[j] 
      }
    }
    for (i in 1 : features){ #compute the sparsity criterion
      criterion[i] <- 0
      for (j in 1 : classes){
        criterion[i] <- criterion[i] + members[j]*center[i,j]^2
      }
    }
    
    
    ###
    #J <- rev(sort.list(criterion))[1:sparsity]
    J <- sort.list(criterion)[1:(features-sparsity)]
    importantvec<-setdiff(wholevec,J)
    center[J,]<-matrix(0,length(J),classes)
    distance <- cdist(t(center),t(X),metric = "euclidean") #cdist takes obersvations in rows
    switched <- FALSE
    for (case in 1:cases){
      #j <- argmin(distance[,case])
      j<-which.min(distance[,case])
      if(j!=class[case]){
        switched <- TRUE
        class[case] <- j  #updating the classes
      }
    }
    WSStemp<-array(0,classes)
    for(k in 1:classes){
      tempIndex<-which(class %in% k)
      tempX<-X[,tempIndex]
      WSStemp[k]<-sum((tempX-center[,k])^2)
    }
    iternum<-iternum+1
  }
  
  WSSval<-array(0,classes)
  for(k in 1:classes){
    tempIndex <- which(class %in% k)
    tempX <- X[,tempIndex]
    WSSval[k]<-sum(scale(t(tempX),scale=FALSE,center=TRUE)^2)
  }
  TSSval <-sum(scale(t(X), scale = FALSE,center=TRUE)^2)
  return(list(m1=class,m2=center,WSS=WSSval,TSS=TSSval,selected=importantvec))
}
# sparse kmeans and feature selection within each class
sparsekmeans2<- function(X,class, classes, sparsity) {
  temp <- dim(X)
  features <- temp[1]
  cases <- temp[2]
  center <- matrix(0,features,classes)
  members <- array(0, classes)
  criterion <- array(0, features)
  distance <- matrix(0, classes, cases)
  importantvec<-matrix(0,classes,sparsity)
  wholevec<-1:features
  for (i in 1 : features){
    X[i,] <- zscore(X[i,])
  }
  switched <- TRUE
  while (switched) {
    #print("one iter")
    center <- matrix(0,features,classes) # update centers
    members <- array(0, classes)
    for (case in 1 : cases){ 
      i <- class[case]
      center[,i] <- center[,i] + X[,case]
      members[i] <- members[i]+1
    }
    for (i in 1 : classes){ 
      if (members[i]>0){
        center[,i] <- center[,i]/members[i] 
        J <- (sort.list(center[,i]^2*members[i]))[1:(features-sparsity)]
        #J<-rev(sort.list(center[,i]^2*members[i]))[1:sparsity]
        center[J,i]<-t(array(0,length(J)))
        importantvec[i,]<-setdiff(wholevec,J)
        #importantvec[i,]<-J
      }
    }
    
    switched <- FALSE

    distance <- cdist(t(center),t(X),metric = "euclidean")
    #print(distance)
    for (case in 1:cases){
      j <- which.min(distance[,case])
      if(j!=class[case]){
        switched <- TRUE
        class[case] <- j
      }
     # print(switched)
    }
  }
  WSSval<-array(0,classes)
  for(k in 1:classes){
    tempIndex <- which(class %in% k)
    tempX <- X[,tempIndex]
    WSSval[k]<-sum(scale(t(tempX),scale=FALSE,center=TRUE)^2)
  }
  TSSval <-sum(scale(t(X), scale = FALSE,center=TRUE)^2)
  return(list(m1=class,m2=center,WSS=WSSval,TSS=TSSval,selected=importantvec))
}

get_classes<- function(X, center){
points <- ncol(X)
class <- array(0,points)
dist <- cdist(t(center),t(X),metric = "euclidean")
for (point in 1:points){
class[point] <- which.min(dist[, point]) # closest center
}
return(class)
}

