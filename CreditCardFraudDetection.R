## Importing the data from the csv
card_data <- read.csv("creditcard.csv")
library(caret)
library(ranger)
library(data.table)

## Exploring the data
summary(card_data)
head(card_data,6)
tail(card_data,6)
dim(card_data)
summary(card_data$Amount)
table(card_data$Class)
names(card_data)
sd(card_data$Amount)
var(card_data$Amount)


## Using scale function to standardization features of our data. With scaling we avoid all exyreme values in our dataset to ensure smooth functioning of our model.
card_data$Amount <- scale(card_data$Amount)
NewData <- card_data[,-c(1)]
head(NewData)


## Spliting the data into train and test
library(caTools)
set.seed(123)
sample_data <- sample.split(NewData$Class, SplitRatio = 0.8)
trainingdata <- subset(NewData, sample_data == T)
testdata <- subset(NewData, sample_data == F)
dim(trainingdata)
dim(testdata)


## Fitting the logistic regression model
LogisticModel <- glm(Class ~ ., trainingdata, family = binomial())
summary(LogisticModel)
plot(LogisticModel)


## To access the performance  of our model using the ROC curve to analyze its performance
library(pROC)
LogiReg.Prediction <- predict(LogisticModel, testdata, probability = T)
auc.gbm <- roc(testdata$Class, LogiReg.Prediction, plot = T, col = "cyan")


## Fitting the decision tree model
library(rpart)
library(rpart.plot)
DecisionTreeModel <- rpart(Class ~ ., card_data, method = 'class')
predictedValue <- predict(DecisionTreeModel, card_data, type = 'class')
prob <- predict(DecisionTreeModel, card_data, type = 'prob')
rpart.plot(DecisionTreeModel)
mean(predictedValue==NewData$Class)

## Using Artificial Neural Network (Setting the threshold as 0.5)

library(neuralnet)
NNModel <- neuralnet(Class ~ ., trainingdata, linear.output = F)
plot(NNModel)
NNPrediction <- compute(NNModel, testdata)
NNResult <- NNPrediction$net.result
NNResult <- ifelse(NNResult>0.5, 1, 0)
mean(NNResult==testdata$Class)
table(NNResult, testdata$Class)


## Fitting a Gradiant Boosting Model
library(gbm, quietly = T)
# Setting Time to train the GBM Model
system.time(
  model_gbm <- gbm(Class ~ .,
                   distribution = "bernoulli",
                   data = rbind(trainingdata, testdata),
                   n.trees = 500,
                   interaction.depth = 3,
                   n.minobsinnode = 100,
                   shrinkage = 0.01,
                   bag.fraction = 0.5,
                   train.fraction = nrow(trainingdata) / 
                     (nrow(trainingdata) + nrow(testdata))
                   )
)
# Determining the bet iteration based on test data
gbmiteration <- gbm.perf(model_gbm, method = "test")
modelinfluence <- relative.influence(model_gbm, n.trees = gbmiteration, sort. = T)
# Plot the gbm model
plot(model_gbm)


## Ploting and calculating AUC on test data
gbm_test <- predict(model_gbm, newdata = testdata, n.trees = gbmiteration)
gbm_auc <- roc(testdata$Class, gbm_test, plot = T, col = "maroon")
print(gbm_auc)

###+++===============================================================+++###
# In conclusion, we developed credit card fraud detection model using machine learning.
# We used a variety of ML algorithms to implement this model and also plotted the respective performance curves for the models.