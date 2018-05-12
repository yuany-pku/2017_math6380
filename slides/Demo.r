load("demo.RData")

#contruct in-sample and out-sample datasets
dt<-data.frame(pseudo_log_return,pseudo_log_return_lag_1,pseudo_log_return_lag_2,pseudo_log_return_lag_3,STD_lag_1,STD_lag_2,STD_lag_3,HOUR,MINUTE)
#take first 70 days as in-sample data.
#remaining 30 days as out-sample data.
IN<-dt[4:16800,]  #n=3, leave out first 3 observations 
OUT<-dt[16801:24000,]

#training DNN on in-sample dataset
library(h2o)
h2o.init()
h2o.no_progress()
h2oIN<-as.h2o(IN)
h2oOUT<-as.h2o(OUT)
y<-"pseudo_log_return"

dl_fit <- h2o.deeplearning(y = y,training_frame = h2oIN,hidden = c(8,6,4,3,1),activation='Tanh')

#check out-sample performance
dl_fit_perf<- h2o.performance(model = dl_fit, newdata = h2oOUT)
mse<-h2o.mse(dl_fit_perf) #mean square error
predict<-h2o.predict(dl_fit, h2oOUT)

predictreturn<-c(as.matrix(predict)) #it's predicted return on test set
actualreturn<-pseudo_log_return[16801:24000] #it's actual return on test set

mean(actualreturn[predictreturn>0]>0,na.rm=T)

h2o.shutdown(prompt = TRUE)

