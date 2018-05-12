library(lars)
library(Libra)
data(diabetes)
attach(diabetes)

lasso <- lars(x,y)
par(mfrow=c(2,2))
plot(lasso)

issobject <- iss(x,y)
plot(issobject,xtype="norm")  #plot.lb
title("ISS",line = 2.5)

kappa <- c(100,500)
for (i in 1:2){
    object <- lb(x,y,kappa[i],family="gaussian",trate=100)
    plot(object,xtype="norm")
    title(paste("LBI:kappa =",kappa[i]),line = 2.5)
}
detach(lasso)
