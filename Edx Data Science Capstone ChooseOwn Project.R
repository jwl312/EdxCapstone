
#############################################################
# Edx Data Science Capstone project Choose Your Own
#############################################################


#Load in the required package
library(tidyverse)
library(caret)
library(lubridate)
library(kableExtra)
library(knitr)
library(summarytools)
library(broom)
library(e1071)
library(pROC)
   
## Read in the dataset
# This is fall detection data from: https://www.kaggle.com/pitasr/falldata

#Falls among the elderly is an important health issue. Fall detection and movement tracking are therefore
# instrumental in addressing this issue. This paper responds to the challenge of classifying different 
#movements as a part of a system designed to fulfill the need for a wearable device to collect data for 
#fall and near-fall analysis. Four different fall trajectories (forward, backward, left and right), 
#three normal activities (standing, walking and lying down) and near-fall situations are identified 
#and detected.

#Falls are a serious public health problem and possibly life threatening for people in fall risk groups. 
#We develop an automated fall detection system with wearable motion sensor units fitted to the subjects' 
#body at six different positions. Each unit comprises three tri-axial devices (accelerometer, gyroscope, 
#and magnetometer/compass). Fourteen volunteers perform a standardized set of movements including 20 
#voluntary falls and 16 activities of daily living (ADLs), resulting in a large dataset with 2520 trials. 
#To reduce the computational complexity of training and testing the classifiers, we focus on the raw data 
#for each sensor in a 4 s time window around the point of peak total acceleration of the waist sensor, and 
#then perform feature extraction and reduction.

## Read in the falls data
falls <- read.csv("C:/Users/jerem/Documents/GitHub/EdxCapstone/falldeteciton.csv")

head(falls)

#Variables in the file
#ACTIVITYactivity classification
#TIMEmonitoring time
#SLsugar level
#EEGEEG monitoring rate
#BPBlood pressure
#HRHeart beat rate
#CIRCLUATIONBlood circulation

#The outcome of interest is ACTIVITY:
#0- Standing 1- Walking 2- Sitting 3- Falling 4- Cramps 5- Running

#Activity is factor, so change it to that
falls$ACTIVITY <- factor(falls$ACTIVITY)

#Create a dummy variable for falls
falls$fall <- ifelse(falls$ACTIVITY == "3",1,0)

#Print as percentage
paste(round(mean(falls$fall)*100, 1), "%", sep="")

#All other variable are numeric, so don't need to change

#Plot of the different activity
falls %>% ggplot(aes(x = ACTIVITY)) +
  geom_bar(fill = "mediumblue") + 
  ylab("Number of Occurances") + xlab("Activity") + scale_y_continuous(expand = c(0,0)) +
  scale_x_discrete(labels=c("0:Standing","1:Walking","2:Sitting","3:Falling","4:Cramps","5:Running")) +
  theme(panel.background = element_blank(), panel.grid.major.y = element_line(colour = "grey", size = 0.2))

# Check for near zero variance.
nzv <- nearZeroVar(falls)
nzv
#No variables with no zero variance

###  Check for missing valueS
sum( is.na(falls$ACTIVITY) )
sum( is.na(falls$TIME) )
sum( is.na(falls$SL) )
sum( is.na(falls$EEG) )
sum( is.na(falls$BP) )
sum( is.na(falls$HR) )
sum( is.na(falls$CIRCULATION) )
#No missing values

#######################################################################################
### Data Exploration and Cleaning

## Box plot of Time and Activity
falls %>% ggplot(aes(x = ACTIVITY, y = TIME)) + geom_boxplot(utlier.shape = NA) + 
  scale_x_discrete(labels=c("0:Standing","1:Walking","2:Sitting","3:Falling","4:Cramps","5:Running")) +
  geom_jitter(size = 0.5, position=position_jitter(0.2), alpha = 0.5, colour = "#002266", fill = "#002266") +
  theme(panel.background = element_blank(), panel.grid.major.y = element_line(colour = "grey", size = 0.2))

## Histogram of time
falls %>% ggplot(aes(x = TIME)) +
  geom_histogram(binwidth = 10, colour = "#002266") + ggtitle("Histogram of time") +
  scale_y_continuous(expand = c(0,0)) + ylab("Count") + xlab("Time (in seconds)") +
  theme(panel.background = element_blank(), panel.grid.major.y = element_line(colour = "grey", size = 0.2))

#Since there doesn't appear to be a strong relationship and it is unclear why there would be, drop the time variable.
falls <- falls %>% select(-TIME)

#Something looks off with sugar level, some are very very high
summary(falls$SL)
quantile(falls$SL,c(0.01,0.1,0.5,0.9,0.95,0.99))
skewness(falls$SL)

#Try to use the BoxConTrans function in caret to check for skewness
BoxCoxTrans(falls$SL)
falls_pre <- preProcess(falls, method = c("BoxCox") )

#Now transforms the falls data
falls2 <- predict(falls_pre, falls)

#Now check skewness after
skewness(falls2$SL)

#Top code at the 99th percentile
falls$SL <- ifelse(falls$SL > 521414.680, 521414.680,falls$SL)
#Check the top code
summary(falls$SL)

## Box plot of Sugar Level and Activity
falls %>% ggplot(aes(x = ACTIVITY, y = SL)) + geom_boxplot(outlier.shape = NA) + 
  scale_x_discrete(labels=c("0:Standing","1:Walking","2:Sitting","3:Falling","4:Cramps","5:Running")) +
  geom_jitter(size = 0.5, position=position_jitter(0.2), alpha = 0.5, colour = "#002266", fill = "#002266") +
  theme(panel.background = element_blank(), panel.grid.major.y = element_line(colour = "grey", size = 0.2))

## Histogram of Sugar Level
falls %>% ggplot(aes(x = SL)) +
  geom_histogram(binwidth = 1000, colour = "#002266") + ggtitle("Histogram of Sugar Level") +
  scale_y_continuous(expand = c(0,0)) + ylab("Count") + xlab("Sugar Level") +
  theme(panel.background = element_blank(), panel.grid.major.y = element_line(colour = "grey", size = 0.2))

summary(falls$EEG)
quantile(falls$EEG,c(0.01,0.1,0.5,0.9,0.95,0.99))
skewness(falls$EEG)

#Bottom code at the 1st percentile
falls$EEG <- ifelse(falls$EEG < -15043.3800, -15043.3800,falls$EEG)
#Top code at the 1st percentile
falls$EEG <- ifelse(falls$EEG > 17147.6, 17147.6,falls$EEG)
#Check the top code
summary(falls$EEG)

## Histogram of EEG
falls %>% ggplot(aes(x = EEG)) +
  geom_histogram(binwidth = 1000, colour = "#002266", fill = "#002266") + ggtitle("Histogram of EEG Monitoring Rate") +
  scale_y_continuous(expand = c(0,0)) + ylab("Count") + xlab("EEG Monitoring Rate") +
  theme(panel.background = element_blank(), panel.grid.major.y = element_line(colour = "grey", size = 0.2))

#Summary statistics for Blood Pressue
summary(falls$BP)
quantile(falls$BP,c(0.01,0.1,0.5,0.9,0.95,0.99))
skewness(falls$BP)
#BP is not too skewed

## Box plot of EEG and Activity
falls %>% ggplot(aes(x = ACTIVITY, y = EEG)) + geom_boxplot(outlier.shape = NA) + 
  scale_x_discrete(labels=c("0:Standing","1:Walking","2:Sitting","3:Falling","4:Cramps","5:Running")) +
  theme(panel.background = element_blank(), panel.grid.major.y = element_line(colour = "grey", size = 0.2)) +
  geom_jitter(size = 0.5, position=position_jitter(0.2), alpha = 0.5, colour = "#002266", fill = "#002266")
  
## Histogram of Blood Pressure
falls %>% ggplot(aes(x = BP)) +
  geom_histogram(colour = "#002266", fill = "#002266") + ggtitle("Histogram of Blood Pressure") +
  scale_y_continuous(expand = c(0,0)) + ylab("Count") + xlab("Blood Pressure") +
  theme(panel.background = element_blank(), panel.grid.major.y = element_line(colour = "grey", size = 0.2))

#Summary statistics for Heart Rate
summary(falls$HR)
quantile(falls$HR,c(0.001,0.01,0.1,0.9,0.95,0.99,0.999))
skewness(falls$HR)
#HR is not too skewed

## Box plot of BP and Activity
falls %>% ggplot(aes(x = ACTIVITY, y = BP)) + geom_boxplot(outlier.shape = NA) + 
  scale_x_discrete(labels=c("0:Standing","1:Walking","2:Sitting","3:Falling","4:Cramps","5:Running")) +
  theme(panel.background = element_blank(), panel.grid.major.y = element_line(colour = "grey", size = 0.2)) +
  geom_jitter(size = 0.5, position=position_jitter(0.2), alpha = 0.5, colour = "#002266", fill = "#002266")

## Histogram of BP
falls %>% ggplot(aes(x = HR)) +
  geom_histogram(colour = "#002266", fill = "#002266") + ggtitle("Histogram of Heart Rate") +
  scale_y_continuous(expand = c(0,0)) + ylab("Count") + xlab("Heart Rate") +
  theme(panel.background = element_blank(), panel.grid.major.y = element_line(colour = "grey", size = 0.2))

#Summary statistics for Blood Circulation
summary(falls$CIRCLUATION)
quantile(falls$CIRCLUATION,c(0.001,0.01,0.1,0.9,0.95,0.99,0.999))
skewness(falls$CIRCLUATION)

## Box plot of HR and Activity
falls %>% ggplot(aes(x = ACTIVITY, y = HR)) + geom_boxplot(outlier.shape = NA) + 
  scale_x_discrete(labels=c("0:Standing","1:Walking","2:Sitting","3:Falling","4:Cramps","5:Running")) +
  theme(panel.background = element_blank(), panel.grid.major.y = element_line(colour = "grey", size = 0.2)) +
  geom_jitter(size = 0.5, position=position_jitter(0.2), alpha = 0.5, colour = "#002266", fill = "#002266")

## Histogram of Circulation
falls %>% ggplot(aes(x = CIRCLUATION)) +
  geom_histogram(colour = "#002266", fill = "#002266") + ggtitle("Histogram of Blood Circulation") +
  scale_y_continuous(expand = c(0,0)) + ylab("Count") + xlab("Blood Circulation") +
  theme(panel.background = element_blank(), panel.grid.major.y = element_line(colour = "grey", size = 0.2))

#####################################################################################################
########### Divide into Train and Validation Set

#Delete Activity since only interested in predicting falls
#Delete Time since there is no reason it should be predictive.
falls  <- falls  %>% select(-ACTIVITY)

#Convert fall into a factor, but make it an ordered factor to calculate AUC
falls$fall <- ordered(falls$fall, levels = c("0","1"))

#The full dataset isn't that big, so don't need to take a sample
#Use that to create an index of the sampled rows, just 20% of the sample
test_index <- createDataPartition(y = falls$fall, times = 1, p = 0.2, list = FALSE)
#Create a train dataframe with 80% of the rows
train_set  <- falls[-test_index,]
#Create a test dataset with the other 20%
test_set   <- falls[test_index,]

#################################################################################
## Try a logistic  regression

#First fit the model
fit <- train(fall ~ ., data = train_set, method = "glm")
#Use the fitted model to predict ratings values in test_set
y_hat <- predict(fit, newdata = test_set)

#Use a confusion matrix to examine the accuracy
acc_glm <- confusionMatrix(data = y_hat, reference = test_set$fall, mode ="everything")$overall["Accuracy"]

#Calculate the ROC
roc <- roc(test_set$fall ~ y_hat)
#Calculate the area under the curve
auc_glm <- auc(roc)

#Create a table of RMSEs for the different models 
acc_results <- data_frame(method = "Logistic Regression", Accuracy = acc_glm, AUC = auc_glm)

#################################################################################
### Try a Random Forest Model
#Tune the number of nodes
train_rf_2 <- train(fall ~ ., method = "Rborist",
                    tuneGrid = data.frame(predFixed = 2, minNode = seq(1, 10, 1)),
                    data = train_set)

##The RMSE was lowest at 0.8528138 for m nNode = 1
train_rf_2
#minNode  Accuracy   Kappa    
#1       0.8528138  0.5503004
#2       0.8525185  0.5494173
#3       0.8524207  0.5444607
#4       0.8520864  0.5428276
#5       0.8519333  0.5408785
#6       0.8521212  0.5405008
#7       0.8515634  0.5376950
#8       0.8514050  0.5366938
#9       0.8508457  0.5340993
#10      0.8507733  0.5335633

#Try to tune the number of trees
#mtry: Number of variables randomly sampled as candidates at each split, equivalent to predFixed
#Good rule of thumb for mtry is the number of predictors divided by 3, or 2 in this case.

#Keep the number of cross validations low to reduce run time, set it to 5
control <- trainControl(method = "cv", number = 5, p = .9)
train_rf <- train(train_set[,1:5], train_set$fall,
                 method = "rf", 
                 tuneGrid = data.frame(mtry = c(1,2,3) ), 
                 trControl = control)
          
train_rf

#13105 samples
#5 predictor
#2 classes: '0', '1' 
#No pre-processing
#Resampling: Cross-Validated (5 fold) 
#Summary of sample sizes: 10484, 10484, 10484, 10484, 10484 
#Resampling results across tuning parameters:
#  mtry  Accuracy   Kappa    
#1     0.8600534  0.5669536
#2     0.8582984  0.5661891
#3     0.8547882  0.5543570
#Accuracy was used to select the optimal model using the largest value.
#The final value used for the model was mtry = 1.

varImp(train_rf)
#Overall
#SL          100.000
#EEG          52.010
#BP           20.754
#CIRCLUATION   7.011
#HR            0.000

## Now create the predicted Ys for the random forest model
y_hat_rf <- predict(train_rf, test_set)

#Use a confusion matrix to examine the accuracy
acc_rf <- confusionMatrix(data = y_hat_rf, reference = test_set$fall, mode ="everything")$overall["Accuracy"]

#Calculate the ROC
roc <- roc(test_set$fall ~ y_hat_rf)
#Calculate the area under the curve
auc_rf <- auc(roc)

#Create a table of RMSEs for the different models 
acc_results <- bind_rows(acc_results,
                         data_frame(method = "Random Forest", Accuracy = acc_rf, AUC = auc_rf))

#################################################################################
### Try KNN function

#This is for cross validation, number is the number of folds
control <- trainControl(method = "cv", number = 2, p = .9)
#First try this large range of Ks
train_knnh <- train(train_set[,1:5],train_set$fall, 
                                          method = "knn",
                                          tuneGrid = data.frame(k = seq(5, 35, 10)),
                                          trControl = control)
train_knnh

#13105 samples
#5 predictor
#2 classes: '0', '1' 

#No pre-processing
#Resampling: Cross-Validated (2 fold) 
#Summary of sample sizes: 6553, 6552 
#Resampling results across tuning parameters:
  
#  k   Accuracy   Kappa    
#5  0.8009918  0.3923604
#15  0.8104540  0.3864828
#25  0.8095380  0.3651377
#35  0.8062569  0.3408582

#Accuracy was used to select the optimal model using the largest value.
#The final value used for the model was k = 15.

#Try k = 40, since it would be hard to optimize further without taking a lot of time
y_hat_knn <- predict(train_knnh, test_set[,1:5])

#Use a confusion matrix to examine the accuracy
acc_knn <- confusionMatrix(data = y_hat_knn, reference = test_set$fall, mode ="everything")$overall["Accuracy"]

#Calculate the ROC
roc <- roc(test_set$fall ~ y_hat_knn)
#Calculate the area under the curve
auc_knn <- auc(roc)

#Create a table of RMSEs for the different models 
acc_results <- bind_rows(acc_results, 
                         data_frame(method = "KNN", Accuracy = acc_knn, AUC = auc_knn) )
## Save the acc_results table
save(acc_results, file = "C:/Users/jerem/Documents/GitHub/EdxCapstone/acc_results.Rda")

