library(Libra)
data(west10)
y<-2*west10[,1]-1;
X<-as.matrix(2*west10[,2:10]-1);

path <- lb(X,y,kappa = 1,family="binomial",trate=100,normalize = FALSE)
plot(path,xtype="norm",omit.zeros=FALSE)
title(main=paste("Logistic",attributes(west10)$names[1],"~."),line=3)
legend("bottomleft", legend=attributes(west10)$names[-1], col=c(1:6,1:3),lty=c(1:5,1:4))