library(RLT)
library(randomForest)
library(randomForestSRC)
library(ranger)

set.seed(1)
n = 100

p = 6
x = matrix(runif(n*p), n, p)
xlink <- function(x) exp(x[, 1] + x[, p]) 
FT = rexp(n, rate = xlink(x) )
CT = rexp(n, rate = 1)

y = pmin(FT, CT)
censor = as.numeric(FT <= CT)
mean(censor)

testn <- 2
testx = matrix(runif(testn*p), testn, p)

timepoints = sort(unique(y[censor==1]))
yloc = rep(NA, length(timepoints))
for (i in 1:length(timepoints)) yloc[i] = sum( timepoints[i] >= y )

SurvMat = matrix(NA, testn, length(timepoints))

for (j in 1:length(timepoints))
{
  SurvMat[, j] = 1 - pexp(timepoints[j], rate = xlink(testx) )
}
ntrees = 100
mtry = 3
nmin = 20
split.gen = "best" 
nsplit = 1
ncores = 1
alpha = 0
k = nrow(x) / 2
importance = FALSE
verbose = 0

myfit = surv_var_est(x, y, censor, testx, ntrees = ntrees, 
                     mtry = mtry, nmin = nmin, #k = 600,
                    split.gen = "best", 
                    nsplit = nsplit, ncores = ncores)

plot(cumsum(myfit$pred[1,]), type="l")
lines(-log(SurvMat[1,]), type="l", col="red")
lines(cumsum(myfit$pred[1,] + myfit$var[1,]),type="l",col="green")
lines(cumsum(myfit$pred[1,] - myfit$var[1,]),type="l",col="green")
lines(cumsum(myfit$pred[1,] + diag(myfit$cov[,,1])),type="l",col="blue")
lines(cumsum(myfit$pred[1,] - diag(myfit$cov[,,1])),type="l",col="blue")
plot(cumsum(myfit$pred[2,]), type="l")
lines(-log(SurvMat[2,]), type="l", col="red")
lines(cumsum(myfit$pred[2,] + myfit$var[2,]),type="l",col="green")
lines(cumsum(myfit$pred[2,] - myfit$var[2,]),type="l",col="green")
lines(cumsum(myfit$pred[2,] + diag(myfit$cov[,,2])),type="l",col="blue")
lines(cumsum(myfit$pred[2,] - diag(myfit$cov[,,2])),type="l",col="blue")

myfit$var

plot(myfit$estimation[1,])

rowSums(sweep(myfit$estimation, 2, myfit$allc, FUN = "*"))/sum(myfit$allc)

myfit$var

myfit$allc


myfit$sd



n = 1000
k = 500
x = seq(1:k)

dhyper(x, k, n - k, k)

qhyper(0.05, k, n - k, k)
qhyper(0.95, k, n - k, k)


