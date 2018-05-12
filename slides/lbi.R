library(MASS)
library(lars)
library(Libra)

# Simulation of a Linear Regression Model
n = 80; 	# sample size
p = 100; 	# dimension
k = 30;	# sparsity
sigma = 1;# noise level
Sigma = 1/(3*p)*matrix(rep(1,p^2),p,p)	# covariance of random design/feature matrix
diag(Sigma) = 1
A = mvrnorm(n, rep(0, p), Sigma)		# random design/feature matrix
u_ref = rep(0,p)						# ground truth
supp_ref = 1:k						# k-sparse
u_ref[supp_ref] = rnorm(k)
u_ref[supp_ref] = u_ref[supp_ref]+sign(u_ref[supp_ref])

b = as.vector(A%*%u_ref + sigma*rnorm(n))	# noisy linear measurement

# Lasso regularization path
lasso = lars(A,b,normalize=FALSE,intercept=FALSE,max.steps=100)
par(mfrow=c(2,2))
matplot(n/lasso$lambda, lasso$beta[1:100,], xlab = bquote(n/lambda),
ylab = "Coefficients", xlim=c(0,3),ylim=c(range(lasso$beta)),type='l', main="Lasso")

# ISS regularization/solution path
object = iss(A,b,intercept=FALSE,normalize=FALSE)
plot(object,xlim=c(0,3),main=bquote("ISS kappa=Inf"))

# Four choices of kappa and alpha
kappa_list = c(4,256) #c(4,16,64,256)
alpha_list = 1/10/kappa_list

# Linearized Bregman Iteration (LB) regularization paths
for (i in 1:4){
object <- lb(A,b,kappa_list[i],alpha_list[i],family="gaussian",group=FALSE,
trate=20,intercept=FALSE,normalize=FALSE)
plot(object,xlim=c(0,3),main=bquote(paste("LB ",kappa,"=",.(kappa_list[i]))))
}

# Journey to the West
# Logistic regression path of Sunwukong ~  9 main characters
data(west10)
y<-2*west10[,1]-1;
X<-as.matrix(2*west10[,2:10]-1);

path <- lb(X,y,kappa = 1,family="binomial",trate=100,normalize = FALSE)
plot(path,xtype="norm",omit.zeros=FALSE)
title(main=paste("Logistic",attributes(west10)$names[1],"~."),line=3)
legend("bottomleft", legend=attributes(west10)$names[-1], col=c(1:6,1:3),lty=c(1:5,1:4))

#Diabetes, comparison with lars

