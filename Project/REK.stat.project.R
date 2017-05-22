######### Robert Kramer #########
######### Project 12-08 #########

######### Initial #########
setwd("/Users/kramerPro/Google Drive/Stat Learning Fall 2016/Project/")
library(data.table)
library(caret)
library(MASS)
library(rpart)
library(stepPlr)
library(kernlab)
library(klaR)
library(ROCR)
library(nnet)
library(devtools)
library(mRMRe)
source_url('https://gist.githubusercontent.com/fawda123/7471137/raw/466c1474d0a505ff044412703516c34f1a4684a5/nnet_plot_update.r')

# library(doParallel) # got an error with QDA
# registerDoParallel(cores = 2)

library(devtools)
source_url('https://gist.githubusercontent.com/fawda123/7471137/raw/466c1474d0a505ff044412703516c34f1a4684a5/nnet_plot_update.r')

# Original.Data <- fread("./arrhythmia-data/arrhythmia.data.txt",
#               header = F, stringsAsFactors = F, data.table = F)
# Data <- Original.Data

########## PreProcessing #########
## Presaved -- uncomment to run own version
# # Assigning the missing values
# Data[Data == '?'] <- NA
# # Setting all modes to numeric
# for(i in 1:length(Data)){
#   Data[,i] <- as.numeric(Data[,i])
# }
# 
# # Simplifying the classification task to 0 normal (class=1) vs 1 (not normal) class != 1
# colnames(Data)[280] <- 'class'
# Data$class <- lapply(Data$class, function (x) ifelse(x>1,1,0))
# 
# x.data <- Data[,-length(Data)]
# y.data <- factor(Data$class)
# # 90 / 10 split training and test data low n/p ratio 
# # test set may not be representative
# train.index <- createDataPartition(Data$class, p = .9, list = F)
# 
# train.x <- x.data[train.index,]
# train.y <- y.data[train.index]
# 
# # eliminate near zero variation using training set
# # could compare to overall
# # could use for informative prior (figure out how)
# nzv.info <- nearZeroVar(train.x, saveMetrics = T)
# nzv <- nearZeroVar(train.x)
# # eliminate near zero var columns
# train.x <- x.data[train.index, - nzv]
# train.y <- y.data[train.index]
# 
# test.x <- x.data[- train.index, - nzv]
# test.y <- y.data[- train.index]
# 
# #### Saving data vars to ensure I use the same settings for every model
# saveRDS(train.x, "train.x.rds")
# saveRDS(train.y, "train.y.rds")
# saveRDS(test.x, "test.x.rds")
# saveRDS(test.y, "test.y.rds")

train.x = readRDS('train.x.rds')
train.y = readRDS('train.y.rds')
test.x = readRDS('test.x.rds')
test.y = readRDS('test.y.rds')

print.me <- list()

### L2 based classifiers
######### L2 Penalized Logistic Regression ##########

i=1
print.me$method[i]="penalized logistic regression"
# modelplr=train(train.x,train.y,method="plr",preProcess=c("center","scale","knnImpute"),
#              trControl=trainControl("cv", number = 10)) # sample too small for cross validation
# saveRDS(modelplr, "modelplr.rds") # saved to save time when re-running
modelplr <- readRDS("modelplr.rds")
#print(summary(modelplr))
train.predict=predict(modelplr,train.x)
print.me$train.error[i]=sum(train.y != train.predict)/length(train.y)
test.predict=predict(modelplr,test.x)
print.me$test.error[i]=sum(test.y != test.predict)/length(test.y)
print.me

############# QDA #############

i=2
print.me$method[i]="QDA"
modelQDA=train(train.x,train.y,method="qda",preProcess=c("center","scale","knnImpute"),
               trControl=trainControl("cv", number = 10))

train.predict=predict(modelQDA,train.x)
print.me$train.error[i]=sum(train.y != train.predict)/length(train.y)
test.predict=predict(modelQDA,test.x)
print.me$test.error[i]=sum(test.y != test.predict)/length(test.y)
print.me

############### CART ###############

i=3
print.me$method[i]="CART"
# selection via complexity parameter
modelCART=train(train.x,train.y,method="rpart",preProcess=c("center","scale","knnImpute"),
               trControl=trainControl("cv", number = 10))

train.predict=predict(modelCART,train.x)
print.me$train.error[i]=sum(train.y != train.predict)/length(train.y)
test.predict=predict(modelCART,test.x)
print.me$test.error[i]=sum(test.y != test.predict)/length(test.y)
print.me

################# Gausian Process #############

i=4
print.me$method[i]="Gaussian Process"

# modelGauss=train(train.x,train.y,method="gaussprPoly",preProcess=c("center","scale","knnImpute"),
#                 trControl=trainControl("cv", number = 10))
# # takes a while to run - saved the model
# saveRDS(modelGauss, "modelGauss.rds") # saved to save time when re-running
modelGauss <- readRDS("modelGauss.rds")
train.predict=predict(modelGauss,train.x)
print.me$train.error[i]=sum(train.y != train.predict)/length(train.y)
test.predict=predict(modelGauss,test.x)
print.me$test.error[i]=sum(test.y != test.predict)/length(test.y)
print.me

################ KNN #############

i=5
print.me$method[i]="KNN"

modelKNN=train(train.x,train.y,method="knn",preProcess=c("center","scale","knnImpute"),
                trControl=trainControl("cv", number = 10))

train.predict=predict(modelKNN,train.x)
print.me$train.error[i]=sum(train.y != train.predict)/length(train.y)
test.predict=predict(modelKNN,test.x)
print.me$test.error[i]=sum(test.y != test.predict)/length(test.y)
print.me

################ Naive Bayes #################
i=6
print.me$method[i]="Naive Bayes"

modelNB=train(train.x,train.y,method="nb",preProcess=c("center","scale","knnImpute"),
                trControl=trainControl("cv", number = 10))

train.predict=predict(modelNB,train.x)
print.me$train.error[i]=sum(train.y != train.predict)/length(train.y)
test.predict=predict(modelNB,test.x)
print.me$test.error[i]=sum(test.y != test.predict)/length(test.y)
print.me
################ Neural Net  ##########

i=7
print.me$method[i]="NN net"

# gbmGrid <-  expand.grid(size = c(4, 16, 32, 64), 
#                         decay = c(0, .1, .2)) # experimented a bit

# modelNNet=train(train.x,train.y,method="nnet",preProcess=c("center","scale","knnImpute"),
#                 trControl=trainControl("cv", number = 10),
#                 tuneGrid = gbmGrid)
# # takes a while to run - saved the model
# saveRDS(modelNNet, "modelNNet.rds") # saved to save time when re-running
modelNNet <- readRDS("modelNNet.rds")
train.predict=predict(modelNNet,train.x)
print.me$train.error[i]=sum(train.y != train.predict)/length(train.y)
test.predict=predict(modelNNet,test.x)
print.me$test.error[i]=sum(test.y != test.predict)/length(test.y)
print.me

############ Evaluating Classifier performance ########
# L2 Logistic regression
# need a type = response for the ROC curve

plr.prob = predict(modelplr, test.x, type = "prob")
test.predict = predict(modelplr, test.x)
pred = prediction(plr.prob$`1`, test.y)
perf <- performance(pred,'tpr', 'fpr')

rocDataFrame <- data.frame(tpr=perf@x.values[[1]],fpr=perf@y.values[[1]],
                           type=as.factor('Logistic Regression'))

plr.confusion = confusionMatrix(data = test.predict,
                                reference = test.y, mode = "prec_recall")


## QDA
qda.prob = predict(modelQDA, test.x, type = "prob")
test.predict = predict(modelQDA, test.x)
pred = prediction(qda.prob$`1`, test.y)
perf <- performance(pred,'tpr', 'fpr')

new <- data.frame(tpr=perf@x.values[[1]],fpr=perf@y.values[[1]],
                  type=as.factor('QDA'))
rocDataFrame = rbind(rocDataFrame,new)

qda.confusion = confusionMatrix(data = test.predict,
                                reference = test.y, mode = "prec_recall")
## CART
cart.prob = predict(modelCART, test.x, type = "prob")
test.predict = predict(modelCART, test.x)
pred = prediction(cart.prob$`1`, test.y)
perf <- performance(pred,'tpr', 'fpr')

new <- data.frame(tpr=perf@x.values[[1]],fpr=perf@y.values[[1]],
                  type=as.factor('CART'))
rocDataFrame = rbind(rocDataFrame,new)

cart.confusion = confusionMatrix(data = test.predict,
                                reference = test.y, mode = "prec_recall")


## Gauss
gauss.prob = predict(modelGauss, test.x, type = "prob")
test.predict = predict(modelGauss, test.x)
pred = prediction(gauss.prob$`1`, test.y)
perf <- performance(pred,'tpr', 'fpr')

new <- data.frame(tpr=perf@x.values[[1]],fpr=perf@y.values[[1]],
                  type=as.factor('Gauss'))
rocDataFrame = rbind(rocDataFrame,new)

gauss.confusion = confusionMatrix(data = test.predict,
                                 reference = test.y, mode = "prec_recall")

## KNN
knn.prob = predict(modelKNN, test.x, type = "prob")
test.predict = predict(modelKNN, test.x)
pred = prediction(knn.prob$`1`, test.y)
perf <- performance(pred,'tpr', 'fpr')

new <- data.frame(tpr=perf@x.values[[1]],fpr=perf@y.values[[1]],
                  type=as.factor('KNN'))
rocDataFrame = rbind(rocDataFrame,new)

knn.confusion = confusionMatrix(data = test.predict,
                                  reference = test.y, mode = "prec_recall")

## Naive Bayes
nb.prob = predict(modelNB, test.x, type = "prob")
test.predict = predict(modelNB, test.x)
pred = prediction(nb.prob$`1`, test.y)
perf <- performance(pred,'tpr', 'fpr')

new <- data.frame(tpr=perf@x.values[[1]],fpr=perf@y.values[[1]],
                  type=as.factor('Naive Bayes'))
rocDataFrame = rbind(rocDataFrame,new)

nb.confusion = confusionMatrix(data = test.predict,
                                reference = test.y, mode = "prec_recall")

## Neural Net
nn.prob = predict(modelNNet, test.x, type = "prob")
test.predict = predict(modelNNet, test.x)
pred = prediction(nn.prob$`1`, test.y)
perf <- performance(pred,'tpr', 'fpr')

new <- data.frame(tpr=perf@x.values[[1]],fpr=perf@y.values[[1]],
                  type=as.factor('Neural Net'))
rocDataFrame = rbind(rocDataFrame,new)

nn.confusion = confusionMatrix(data = test.predict,
                               reference = test.y, mode = "prec_recall")



qplot(tpr, fpr, data = rocDataFrame, geom = c("point","line"),
      main = "ROC Comparison") + facet_grid(type ~ .)

qplot(tpr, fpr, data = rocDataFrame, geom = c("point","line"),
      fill = type, color = type)



# ############### scratch ##########
# 
# data(ROCR.simple)
# pred <- prediction( plr.prob$`1`, test.y)
# perf <- performance(pred,"sens","spec")
# plot(perf)
# 
# data(ROCR.hiv)
# attach(ROCR.hiv)
# pred.svm <- prediction(hiv.svm$predictions, hiv.svm$labels)
# perf.svm <- performance(pred.svm, 'tpr', 'fpr')
# pred.nn <- prediction(hiv.nn$predictions, hiv.svm$labels)
# perf.nn <- performance(pred.nn, 'tpr', 'fpr')
# plot(perf.svm, lty=3, col="red",main="SVMs and NNs for prediction of
#      HIV-1 coreceptor usage")
# plot(perf.nn, lty=3, col="blue",add=TRUE)
# plot(perf.svm, avg="vertical", lwd=3, col="red",
#      spread.estimate="stderror",plotCI.lwd=2,add=TRUE)
# plot(perf.nn, avg="vertical", lwd=3, col="blue",
#      spread.estimate="stderror",plotCI.lwd=2,add=TRUE)
# legend(0.6,0.6,c('SVM','NN'),col=c('red','blue'),lwd=3)
# 
# data(ROCR.xval)
# pred <- prediction(ROCR.xval$predictions, ROCR.xval$labels)
# perf <- performance(pred,"tpr","fpr")
# plot(perf,col="grey82",lty=3)
# plot(perf,lwd=3,avg="vertical",spread.estimate="boxplot",add=TRUE)
# 
# library(plyr)
# library(ggplot2)
# 
# quakes$level <- cut(quakes$depth, 5, 
#                     labels=c("Very Shallow", "Shallow", "Medium", "Deep", "Very Deep"))
# 
# quakes.summary <- ddply(quakes, .(level), summarise, mag=round(mean(mag), 1))
# 
# ggplot(quakes, aes(x=long, y=lat)) + 
#   geom_point(aes(colour=mag)) + 
#   geom_text(aes(label=mag), data=quakes.summary, x=185, y=-35) +
#   facet_grid(~level) + 
#   coord_map()
# 
# 
# roc.info$type <-  as.factor(rep("Logistic Regression",
#                                 length(perf@x.values[[1]])))
# 
# rocr.plot <- ggplot(data=rocDataFrame, aes(x=tpr, y=fpr)) + geom_path(size=1)
# rocr.plot <- rocr.plot + geom_text(aes(x=1, y= 0,
#                                        hjust=1, vjust=0,
#                                        label=paste(sep = "",
#                                                    "AUC = ",round(auc,4))),
#                                    colour="black",size=4)


gbmGrid <-  expand.grid(size = c(4, 16, 32, 64), 
                        decay = c(0, .1, .2))


