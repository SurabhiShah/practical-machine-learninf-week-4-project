rm(list = ls())
install.packages("caret")
library(caret)
i
#set the working directory and get the datasets
setwd("C:/Users/user/Documents")
training <- read.csv("pml-training.csv")
testing <- read.csv("pml-testing.csv")

#removing the unused variables

training1 = training[,-c(1:5,12:36, 50:59, 69:83,87:101,103:112,125:139,141:150)]
testing1 = testing[,-c(1:5,12:36, 50:59, 69:83,87:101,103:112,125:139,141:150)]

#creating 20 folds and appliny different models
set.seed(2233)
folds<-createFolds(y=training1$classe,k=20,list=TRUE,returnTrain=FALSE)
#training set 1
training1_1<-training1[folds$Fold01,]
#modelling randomforest in fold number 1
modFit1_1_rf<- train(classe~., data=training1_1, method = "rf", prox = TRUE)
#modelling gbm in fold number 1
modFit1_1_gbm<- train(classe~., data=training1_1, method = "gbm",  verbose = FALSE)
#modelling rpart in fold number 1
modFit1_1_rpart<-train(classe~., data=training1_1, method = "rpart")
#modelling lda in fold number 1
modFit1_1_lda<- train(classe~., data=training1_1, method = "lda")
#trainging set 2
training1_2<-training1[folds$Fold02,]
#modelling randomforest in fold number 2
modFit1_2_rf<- train(classe~., data=training1_2, method = "rf", prox = TRUE)
#modelling gbm in fold number 1
modFit1_2_gbm<- train(classe~., data=training1_2, method = "gbm",  verbose = FALSE)
# training set 2
training1_3<-training1[folds$Fold03,]
#predicting for rf
confusionMatrix(training1_3$classe,predict(modFit1_1_rf,training1_3))  
#predicting for gbm
confusionMatrix(training1_3$classe,predict(modFit1_1_gbm,training1_3))  
#traing set3
training1_4<-training1[folds$Fold04,]
#predicting for random forest
confusionMatrix(training1_4$classe,predict(modFit1_1_rf,training1_4))  
#predicting for gbm
confusionMatrix(training1_4$classe,predict(modFit1_1_gbm,training1_4)) 
#predicting for rf
predict(modFit1_1_rf,testing1) 
