# practical-machine-learning-week-4-project

Executive Summary
From the dataset, it is first observed from basic exploratory analysis that only 1/3 of the data is informative. After screening out the uninformative data, the author has tried 4 different machines learning models: random forest, boosting, linear discriminant, and classification trees on subsets of the training data. After a few trials, the random forest model was chosen to generate the answers for the quiz, which achieved 17 correct answers out of 20 questions.

Data Loading
The first step is to load the training and testing data to R. There are 160 variables on 19622 observations

library(caret)
## Loading required package: lattice
## Loading required package: ggplot2
setwd("/Users/Brandon/datasciencecoursera/")
training = read.csv("/Users/Brandon/datasciencecoursera/data/cp_training.csv")
testing = read.csv("/Users/Brandon/datasciencecoursera/data/cp_testing.csv")
Cleaning the data
training1 = training[,-c(1:5,12:36, 50:59, 69:83,87:101,103:112,125:139,141:150)]
testing1 = testing[,-c(1:5,12:36, 50:59, 69:83,87:101,103:112,125:139,141:150)]
On closer inspection, the original dataset has 160 variables (including the response variable). However, upon scanning through the summary of the dataset, the first 5 variables are unlikely to be explanatory since they are data identifiers for individual and time, and an additional 100 variables has 98% of their observations in NAs so they are highly unlikely to contain much value as well so they are also dropped.

Model selection
Due to the size of the dataset, it is diffcult to employ algorithms like random forest on all of the data. So the training dataset is further divided into 20 random subsets. The first subset is first used to train 4 types of models: random forest, boosting, classification trees, and linear discriminant.

The accuracy performance of classification trees (54%) and linear discriminant (67%) is far below that of random forest (88.7%) and boosting (87%). As such, only the latter two are tried on a second subset of the training dataset.

set.seed(2233)
folds<-createFolds(y=training1$classe,k=20,list=TRUE,returnTrain=FALSE)
training1_1<-training1[folds$Fold01,]
modFit1_1_rf<- train(classe~., data=training1_1, method = "rf", prox = TRUE)
## Loading required package: randomForest
## randomForest 4.6-10
## Type rfNews() to see new features/changes/bug fixes.
modFit1_1_gbm<- train(classe~., data=training1_1, method = "gbm",  verbose = FALSE)
## Loading required package: gbm
## Loading required package: survival
## Loading required package: splines
## 
## Attaching package: 'survival'
## 
## The following object is masked from 'package:caret':
## 
##     cluster
## 
## Loading required package: parallel
## Loaded gbm 2.1.1
## Loading required package: plyr
modFit1_1_rpart<-train(classe~., data=training1_1, method = "rpart")
modFit1_1_lda<- train(classe~., data=training1_1, method = "lda")

training1_2<-training1[folds$Fold02,]
modFit1_2_rf<- train(classe~., data=training1_2, method = "rf", prox = TRUE)
modFit1_2_gbm<- train(classe~., data=training1_2, method = "gbm",  verbose = FALSE)
On the second subset, accuracy of random forest is 88% and that of boosting is 87.2 %. Both still have very high accuracy. We will then deploy the model to test out-of-sample error with a third subset of data

training1_3<-training1[folds$Fold03,]
confusionMatrix(training1_3$classe,predict(modFit1_1_rf,training1_3))  
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction   A   B   C   D   E
##          A 267   4   2   6   0
##          B   8 169  10   1   1
##          C   0   3 166   2   0
##          D   1   2  16 138   3
##          E   0   3   4   1 172
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9316          
##                  95% CI : (0.9139, 0.9466)
##     No Information Rate : 0.2819          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9135          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9674   0.9337   0.8384   0.9324   0.9773
## Specificity            0.9829   0.9749   0.9936   0.9735   0.9900
## Pos Pred Value         0.9570   0.8942   0.9708   0.8625   0.9556
## Neg Pred Value         0.9871   0.9848   0.9604   0.9878   0.9950
## Prevalence             0.2819   0.1849   0.2022   0.1512   0.1798
## Detection Rate         0.2727   0.1726   0.1696   0.1410   0.1757
## Detection Prevalence   0.2850   0.1931   0.1747   0.1634   0.1839
## Balanced Accuracy      0.9752   0.9543   0.9160   0.9530   0.9837
confusionMatrix(training1_3$classe,predict(modFit1_1_gbm,training1_3)) 
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction   A   B   C   D   E
##          A 267   4   4   4   0
##          B   9 165   9   1   5
##          C   0   7 162   2   0
##          D   2   2  15 137   4
##          E   0   4   4   5 167
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9173          
##                  95% CI : (0.8982, 0.9338)
##     No Information Rate : 0.284           
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.8954          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9604   0.9066   0.8351   0.9195   0.9489
## Specificity            0.9829   0.9699   0.9885   0.9723   0.9838
## Pos Pred Value         0.9570   0.8730   0.9474   0.8563   0.9278
## Neg Pred Value         0.9843   0.9785   0.9604   0.9853   0.9887
## Prevalence             0.2840   0.1859   0.1982   0.1522   0.1798
## Detection Rate         0.2727   0.1685   0.1655   0.1399   0.1706
## Detection Prevalence   0.2850   0.1931   0.1747   0.1634   0.1839
## Balanced Accuracy      0.9717   0.9382   0.9118   0.9459   0.9663
training1_4<-training1[folds$Fold04,]
confusionMatrix(training1_4$classe,predict(modFit1_1_rf,training1_4))  
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction   A   B   C   D   E
##          A 275   0   2   2   0
##          B   6 160  15   4   5
##          C   0  11 159   1   0
##          D   1   1  14 144   1
##          E   1   3   6   4 166
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9215          
##                  95% CI : (0.9029, 0.9376)
##     No Information Rate : 0.2885          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9007          
##  Mcnemar's Test P-Value : 0.0005203       
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9717   0.9143   0.8112   0.9290   0.9651
## Specificity            0.9943   0.9628   0.9847   0.9794   0.9827
## Pos Pred Value         0.9857   0.8421   0.9298   0.8944   0.9222
## Neg Pred Value         0.9886   0.9810   0.9543   0.9866   0.9925
## Prevalence             0.2885   0.1784   0.1998   0.1580   0.1753
## Detection Rate         0.2803   0.1631   0.1621   0.1468   0.1692
## Detection Prevalence   0.2844   0.1937   0.1743   0.1641   0.1835
## Balanced Accuracy      0.9830   0.9385   0.8980   0.9542   0.9739
confusionMatrix(training1_4$classe,predict(modFit1_1_gbm,training1_4)) 
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction   A   B   C   D   E
##          A 270   3   4   2   0
##          B   4 171   8   1   6
##          C   0   7 162   2   0
##          D   2   1  13 142   3
##          E   3   5   8   4 160
## 
## Overall Statistics
##                                          
##                Accuracy : 0.9225         
##                  95% CI : (0.904, 0.9385)
##     No Information Rate : 0.2844         
##     P-Value [Acc > NIR] : < 2.2e-16      
##                                          
##                   Kappa : 0.902          
##  Mcnemar's Test P-Value : 0.009013       
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9677   0.9144   0.8308   0.9404   0.9467
## Specificity            0.9872   0.9761   0.9885   0.9771   0.9754
## Pos Pred Value         0.9677   0.9000   0.9474   0.8820   0.8889
## Neg Pred Value         0.9872   0.9798   0.9593   0.9890   0.9888
## Prevalence             0.2844   0.1906   0.1988   0.1539   0.1723
## Detection Rate         0.2752   0.1743   0.1651   0.1448   0.1631
## Detection Prevalence   0.2844   0.1937   0.1743   0.1641   0.1835
## Balanced Accuracy      0.9775   0.9453   0.9097   0.9588   0.9611
From the out-of-sample testing for subset 3 and 4, random forest model still outperforms boosting, with an accuracy rate of over 94%. As such, the random forest model 1 is chosen.
