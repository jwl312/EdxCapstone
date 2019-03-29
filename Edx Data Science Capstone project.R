
# Edx Data Science Capstone project

#############################################################
# Create edx set, validation set, and submission file
#############################################################

# Note: this process could take a couple of minutes

library(tidyverse)
library(caret)
library(lubridate)

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- read.table(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                      col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(levels(movieId))[movieId],
                                           title = as.character(title),
                                           genres = as.character(genres))

movielens <- left_join(ratings, movies, by = "movieId")

# Validation set will be 10% of MovieLens data

set.seed(1)
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in validation set are also in edx set

validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set

removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed)

## Save a copy to be loaded so don't need to run above code
save(edx, file = "C:/Users/jerem/Documents/GitHub/EdxCapstone/edx.Rda")
save(validation, file = "C:/Users/jerem/Documents/GitHub/EdxCapstone/validation.Rda")

#Load in the Edx and Validation datasets
load("C:/Users/jerem/Documents/GitHub/EdxCapstone/edx.Rda")
load("C:/Users/jerem/Documents/GitHub/EdxCapstone/validation.Rda")

#Create a table that has the number of movies with certain ratings
ratings <- edx %>% group_by(rating) %>% tally() %>% mutate(Percent = round(((n/sum(n)) * 100),1)  ) %>% select(rating, Percent)

#Plot of ratings
edx %>% ggplot(aes(x = rating)) +
  geom_histogram(binwidth = 0.5) + ggtitle("Number of Ratings") + 
  ylab("Number of Ratings") + xlab("Rating") + scale_y_continuous(expand = c(0,0)) + 
  scale_x_continuous(expand = c(0,0), limits = c(0,5))

#How many different movies? - 10,677
nrow(distinct(edx, movieId))
#Plot the number of ratings per movie
edx %>% group_by(movieId) %>% tally() %>% ggplot(aes(x = n)) +
  geom_histogram(binwidth = 10) + ggtitle("Number of Ratings Per Movie") + 
  ylab("Number of Movies") + xlab("Number of Ratings") + scale_y_continuous(expand = c(0,0)) + 
  scale_x_continuous(expand = c(0,0), limits = c(1,750))

#How many unique users? - 69,787
nrow(distinct(edx, userId))
#Plot the number of ratings per user
edx %>% group_by(userId) %>% tally() %>% ggplot(aes(x = n)) +
  geom_histogram(binwidth = 10) + ggtitle("Number of Ratings Per User") + 
  ylab("Number of Users") + xlab("Number of Ratings") + scale_y_continuous(expand = c(0,0)) + 
  scale_x_continuous(expand = c(0,0), limits = c(0,2000))

#Need to find the number for different genres. Genres are sometimes combined
# so use str_detect to find the genres within longer strings
table(str_detect(edx$genres, "Drama"))
table(str_detect(edx$genres, "Comedy"))
table(str_detect(edx$genres, "Thriller"))
table(str_detect(edx$genres, "Romance"))

# Which movie has the greatest number of ratings?
# Using tally look at counts and then the top ten with the most ratings
edx %>% group_by(title) %>% tally %>% top_n(10, n) %>% arrange(desc(n))

### Instructions:: You will use the following code to generate your datasets. 
# Develop your algorithm using the edx set. For a final test of your algorithm, 
# predict movie ratings in the validation set as if they were unknown. 
# RMSE will be used to evaluate how close your predictions are to the true values in the validation set.

#So train the model on the EdX set then test on validation

### Cleaning

#Check for missing values - none found
sum( is.na(edx$rating) )
sum( is.na(edx$genres) )
sum( is.na(edx$date) )

#The movie title won't be used in the model, remove that and
edx <- edx %>% select(-title)

#See if there is a relationship between timestamp and number of ratings
edx <- mutate(edx, date = as_datetime(timestamp))
edx %>% mutate(date = round_date(date, unit = "week")) %>%
  group_by(date) %>%
  summarize(rating = mean(rating)) %>%
  ggplot(aes(date, rating)) +
  geom_point() +
  geom_smooth(span = 0.15)

#No relationship so drop timestamp
edx <- edx %>% select(-timestamp, -date)

#### Create new variabless

#Examine impact of genres on rating
edx %>% group_by(genres) %>%
  summarize(n = n(), avg = mean(rating), se = sd(rating)/sqrt(n())) %>%
  filter(n >= 40000) %>% 
  mutate(genres = reorder(genres, avg)) %>%
  ggplot(aes(x = genres, y = avg, ymin = avg - 2*se, ymax = avg + 2*se)) + 
  geom_point() +
  geom_errorbar() + 
  theme(axis.text.x = element_text(angle = 90, hjust = 1))

##Convert the genres variables to dummies
# What are the individual genres, not combined, create table to find them
#Filter this table so only genres without the | in them are included
genres <- edx %>% mutate(nc = nchar(genres)) %>% group_by(genres,nc) %>% 
  tally %>% arrange(nc) %>% mutate(s = str_detect(genres, "[|]")) %>% filter(s == FALSE)

#Look at the unique individual genres
genres$genres

edx$Drama       <- str_detect(edx$genres,"Drama")
edx$War         <- str_detect(edx$genres,"War")
edx$IMAX        <- str_detect(edx$genres,"IMAX")
edx$Crime       <- str_detect(edx$genres,"Crime")
edx$Action      <- str_detect(edx$genres,"Action")
edx$Comedy      <- str_detect(edx$genres,"Comedy")
edx$Horror      <- str_detect(edx$genres,"Horror")
edx$SciFi       <- str_detect(edx$genres,"Sci-Fi")
edx$Fantasy     <- str_detect(edx$genres,"Fantasy")
edx$Musical     <- str_detect(edx$genres,"Musical")
edx$Mystery     <- str_detect(edx$genres,"Mystery")
edx$Romance     <- str_detect(edx$genres,"Romance")
edx$Western     <- str_detect(edx$genres,"Western")
edx$Children    <- str_detect(edx$genres,"Children")
edx$Thriller    <- str_detect(edx$genres,"Thriller")
edx$Adventure   <- str_detect(edx$genres,"Adventure")
edx$Animation   <- str_detect(edx$genres,"Animation")
edx$FilmNoir    <- str_detect(edx$genres,"Film-Noir")
edx$Documentary <- str_detect(edx$genres,"Documentary")

#Delete the original genres variable
edx <- edx %>% select(-genres)

### Create regularized user and movie means

## First create graphics of the distribution of average ratings for movies and users
edx %>% group_by(movieId) %>% summarize(mu_m = mean(rating)) %>% ggplot(aes(x = mu_m)) +
  geom_histogram(binwidth = 0.1) + ggtitle("Average Ratings by Movie") +
  scale_y_continuous(expand = c(0,0)) + ylab("Number of Movies") + xlab("Average Movie Rating")

edx %>% group_by(userId) %>% summarize(mu_u = mean(rating)) %>% ggplot(aes(x = mu_u)) +
  geom_histogram(binwidth = 0.1) + ggtitle("Average Ratings by User") +
  scale_y_continuous(expand = c(0,0)) + ylab("Number of Users") + xlab("Average User Rating")

#Create an RMSE function
RMSE <- function(true_ratings, predicted_ratings){ sqrt(mean((true_ratings - predicted_ratings)^2 )) }

#Penalized Least Squares
#Lambda is a tuning parameter, determine the best one
#Create a sequence of lambdas
lambdas <- seq(0, 0.75, 0.05)

#Apply this function which calculates the movie and then user effects using the different lambdas
rmses <- sapply(lambdas, function(l){
  
  mu <- mean(edx$rating)
  
  #Create the movie effect by calculating the mean rating per movie but subtracting the overall mean
  b_i <- edx %>% 
    group_by(movieId) %>%
    summarize(b_i = sum(rating - mu)/(n()+l))
  
  #Create the user effect by calculating the mean rating for user but subtracting the overall mean
  # and the movie effect
  b_u <- edx %>% 
    left_join(b_i, by="movieId") %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - b_i - mu)/(n()+l))
  
  #Create predicted ratings using the movie and user effects and the overall mean
  predicted_ratings <- 
    edx %>% 
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    mutate(pred = mu + b_i + b_u) %>%
    pull(pred)
  
  #Calculate the RMSE between the prediced ratings and the true rating and store that in a vector
  return(RMSE(edx$rating, predicted_ratings))
})

#Plot the RMSE using different lambdas
qplot(lambdas, rmses)  

#Return the lambda with the lowest RMSE
lambdas[which.min(rmses)] 

## Lowest RMSE is when lamda is 0.5, so use that to create the movie and user effect variables
l <- lambdas[which.min(rmses)]
l
#Create a measure of the overall mean
mu <- mean(edx$rating)
#Create the movie effect by calculating the mean rating per movie but subtracting the overall mean
b_i <- edx %>%  group_by(movieId) %>%summarize(b_i = sum(rating - mu)/(n()+l))
#Create the user effect by calculating the mean rating for user but subtracting the overall mean
# and the movie effect
b_u <- edx %>%  left_join(b_i, by="movieId") %>% group_by(userId) %>%
  summarize(b_u = sum(rating - b_i - mu)/(n()+l))

#Combine the movie and user effects back into the overall dataset but rename it so old version kept
train <-  edx %>% left_join(b_i, by = "movieId") %>% left_join(b_u, by = "userId")

#Remove these dataframes as they are no longer needed
rm(b_i, b_u)

# Check for near zero variance.
nzv <- nearZeroVar(train)
nzv
##: Columns with zero variance:6,13,16,21,22
#IMAX, Musical, Western, FilmNoir, Documentary

# Remove the columns with zero variance, these are all genre dummies
train <- train[ -c(6,13,16,21,22) ]

#Save a version of this train dataset
save(train, file = "C:/Users/jerem/Documents/GitHub/EdxCapstone/train.Rda")


#Create a smaller version of the train ds to speed up the models
set.seed(1996)
#Sample 100,000 rows
train_sub  <- train[sample(nrow(train), 100000),]
#Use that to create an index of the sampled rows, just 20% of the sample
test_index <- createDataPartition(y = train_sub$rating, times = 1, p = 0.2, list = FALSE)
#Create a train dataframe with 80% of the rows
train_set  <- train_sub[-test_index,]
#Create a test dataset with the other 20%
test_set   <- train_sub[test_index,]

#Do a semi_join to ensure that all movies and users in test set are in training set
test_set <- test_set %>% semi_join(train_set, by = "movieId") %>% semi_join(train_set, by = "userId")

#Remove the ID columns
test_set  <- test_set  %>% select(-userId, -movieId)
train_set <- train_set %>% select(-userId, -movieId)

## Try a linear regression
#First fit the model
fit <- lm(rating ~ ., data = train_set)
#Use the fitted model to predict ratings values in test_set
y_hat <- predict(fit, newdata = test_set)

#Now see the root mean square error: 0.8486
rmse_lm <- RMSE(test_set$rating, y_hat)

#Doublecheck the RMSE value
sqrt( mean((y_hat - test_set$rating)^2 ) )

#Create a table of RMSEs for the different models 
rmse_results <- data_frame(method = "Linear Regression", RMSE = rmse_lm)

#################################################################################
### Try a Penalized Ridge Regression Model

#Create movie and user means for ridge regression, not regularized
b_i <- edx %>% group_by(movieId) %>% summarize(b_i = mean(rating) )
b_u <- edx %>% group_by(userId) %>% summarize(b_u = mean(rating) )

edx_r <-  edx %>%  left_join(b_i, by = "movieId") %>% left_join(b_u, by = "userId")

#Create a smaller version of ridge regression datasets
set.seed(2001)
train_sub2 <- edx_r[sample(nrow(train), 200000),]
train_set2 <- train_sub2[-test_index,]
test_set2  <- train_sub2[test_index,]

#Do a semi_join to ensure that all movies in test set are in training set
test_set2 <- test_set2 %>% semi_join(train_set, by = "movieId") %>% 
    semi_join(train_set, by = "userId")

#Remove the ID columns
test_set2  <- test_set2 %>% select(-userId, -movieId)
train_set2 <- train_set2 %>% select(-userId, -movieId)

#Create a vector of the ratings for the ridge regression
edx_y <- train_set2$rating
testedx_y <- test_set2$rating

#Remove the ratings from the dataframe with just predictors
train_set2 <- train_set2 %>% select(-rating)
test_set2 <- test_set2 %>% select(-rating)

#Tune the lamda for the ridge regression
ridgeGrid <- data.frame(.lambda = seq(0, 0.7, length = 10) )

#Set the number of cross validation folds to 10
ctrl <- trainControl(method = "cv", number = 10)

ridgeRegFit <- train(train_set2, edx_y,
                     method = "ridge",
                     tuneGrid = ridgeGrid,
                     trControl = ctrl,
                     #Scale the predictors
                     preProc = c("center","scale"))

#Lowest RMSE is when Lambda is 0.0
ridgeRegFit

train_ridge <- train(train_set2, edx_y,
                     method = "ridge",
                     lamda = 0.0,
                     trControl = ctrl,
                     #Scale the predictors
                     preProc = c("center","scale"))

## Now create the predicted Ys for the ridge regression
y_hat_ridge <- predict(train_ridge, test_set2, type = "raw")

#RMSE  for regression trees 0.7526076
rmse_r <- RMSE(testedx_y,y_hat_ridge)

#Add this to the table of RMSE
rmse_results <- bind_rows(rmse_results,
                          data_frame(method = "Penalized Ridge Regression", RMSE = rmse_r ))

#################################################################################
### Regression trees
# Try with different levels of CP in the tuning grid
train_rpart <- train(rating ~ ., 
                     method = "rpart",
                     tuneGrid = data.frame(cp = seq(0, 0.01, len = 10)),
                     data = train_set)
#Plot the tuning parameters
ggplot(train_rpart)

#Try smaller calues of cp since those have lower RMSE
train_rpart <- train(rating ~ ., 
                     method = "rpart",
                     tuneGrid = data.frame(cp = seq(0, 0.001, len = 10)),
                     data = train_set)
ggplot(train_rpart)

train_rpart <- train(rating ~ ., 
                     method = "rpart",
                     tuneGrid = data.frame(cp = seq(0, 0.0005, len = 20)),
                     data = train_set)
ggplot(train_rpart)

#Plot the regression trees
plot(train_rpart$finalModel, margin = 0.1)
text(train_rpart$finalModel, cex = 0.75)
#The plot is not very useful

train_rpart$finalModel

## cp at 0.00024 produces the lowest error
train_rpart <- train(rating ~ ., method = "rpart", cp = 0.00024, data = train_set)

## Now create the predicted Ys for the regression trees
y_hat_rpart <- predict(train_rpart, test_set, type = "raw")

#RMSE  for regression trees
rmse_rt <- RMSE(test_set$rating, y_hat_rpart)
rmse_results <- bind_rows(rmse_results,
                          data_frame(method = "Regression Trees Model", RMSE = rmse_rt ))


#################################################################################
### Try a Random Forest Model
#Tune the number of nodes
train_rf_2 <- train(rating ~ ., method = "Rborist",
                    tuneGrid = data.frame(predFixed = 2, minNode = seq(1, 10, 1)),
                    data = train_set)

##The RMSE was high, lowest was 0.9164772 for minNode = 4
train_rf_2
##minNode  RMSE       Rsquared   MAE      
#1       0.9192283  0.2987140  0.7212660
#2       0.9183750  0.2993035  0.7204278
#3       0.9189227  0.2993747  0.7209477
#4       0.9182957  0.3001221  0.7204291
#5       0.9187032  0.2999568  0.7207166
#6       0.9187269  0.3005963  0.7209528
#7       0.9186815  0.3006666  0.7207485
#8       0.9186136  0.3008872  0.7206874
#9       0.9187092  0.3015820  0.7209318
#10      0.9184179  0.3019828  0.7209016

#Try to tune the number of trees
#mtry: Number of variables randomly sampled as candidates at each split, equivalent to predFixed
#Good rule of thumb for mtry is the number of predictors divided by 3, or 5 in this case.

#Keep the number of cross validations low to reduce run time
#verboseIter should track how long it takes
control <- trainControl(method = "cv", number = 2, p = .9)
train_rf <- train(train_set[,2:17], train_set$rating,
                 method = "rf", 
                 tuneGrid = data.frame(mtry = c(3,5) ), 
                 trControl = control,
                 verboseIter = TRUE)
          
train_rf

#No pre-processing
#Resampling: Cross-Validated (2 fold) 
#Summary of sample sizes: 39998, 40001 
#Resampling results across tuning parameters:
  
#  mtry  RMSE       Rsquared   MAE      
#3     0.8695489  0.3370193  0.6763025
#5     0.8750665  0.3228978  0.6787266
#RMSE was used to select the optimal model using the smallest value.
#The final value used for the model was mtry = 3.

## Now create the predicted Ys for the random forest model
y_hat_rf <- predict(train_rf, test_set)

#RMSE  for random forest where mtry = 3:
rmse_rf_mtry3 <- RMSE(test_set$rating,y_hat_rf)
rmse_results <- bind_rows(rmse_results,
                          data_frame(method = "Random Forest", RMSE = rmse_rf_mtry3 ))

#################################################################################
### Try KNN function

#This is for cross validation, number is the number of folds
control <- trainControl(method = "cv", number = 2, p = .9)
#First try this large range of Ks
train_knnh <- train(train_set[,2:17],train_set$rating, 
                                          method = "knn",
                                          tuneGrid = data.frame(k = seq(20, 50, 10)),
                                          trControl = control)
train_knnh

#No pre-processing
#Resampling: Cross-Validated (2 fold) 
#Summary of sample sizes: 39998, 40001 
#Resampling results across tuning parameters:
  
#  k   RMSE       Rsquared   MAE      
#20  0.8879322  0.3019723  0.6896326
#30  0.8843840  0.3079554  0.6864500
#40  0.8841499  0.3095794  0.6865460
#50  0.8850363  0.3094689  0.6874967

#RMSE was used to select the optimal model using the smallest value.
#The final value used for the model was k = 40.

#Try k = 40, since it would be hard to optimize further without taking a lot of time
y_hat_knn <- predict(train_knnh, test_set[,2:17])

#Look at RMSE 
rmse_knn <- RMSE(test_set$rating, y_hat_knn)

#Add the KNN rmse to the table
rmse_results <- bind_rows(rmse_results,
                          data_frame(method = "KNN", RMSE = rmse_knn ))

## Save the rmse_results table
save(rmse_results, file = "C:/Users/jerem/Documents/GitHub/EdxCapstone/RMSE_results.Rda")

##########################################################################################

## Since the Linear Regression Model had the lowest RMSE run that on the full dataset
# to get the final RMSE

### Examine RMSE for Full EdX and Validation datasets for linear regression

#First need to process the validation dataset so it has the same predictors as EdX
validation$Drama       <- str_detect(validation$genres,"Drama")
validation$War         <- str_detect(validation$genres,"War")
validation$IMAX        <- str_detect(validation$genres,"IMAX")
validation$Crime       <- str_detect(validation$genres,"Crime")
validation$Action      <- str_detect(validation$genres,"Action")
validation$Comedy      <- str_detect(validation$genres,"Comedy")
validation$Horror      <- str_detect(validation$genres,"Horror")
validation$SciFi       <- str_detect(validation$genres,"Sci-Fi")
validation$Fantasy     <- str_detect(validation$genres,"Fantasy")
validation$Musical     <- str_detect(validation$genres,"Musical")
validation$Mystery     <- str_detect(validation$genres,"Mystery")
validation$Romance     <- str_detect(validation$genres,"Romance")
validation$Western     <- str_detect(validation$genres,"Western")
validation$Children    <- str_detect(validation$genres,"Children")
validation$Thriller    <- str_detect(validation$genres,"Thriller")
validation$Adventure   <- str_detect(validation$genres,"Adventure")
validation$Animation   <- str_detect(validation$genres,"Animation")
validation$FilmNoir    <- str_detect(validation$genres,"Film-Noir")
validation$Documentary <- str_detect(validation$genres,"Documentary")

#Now delete the original genres variable
val <- validation %>% select(-genres)

#Regularize and scale the movie and user means as done on the training dataset above
#Use the lamda found above

l <- 0.5
mu <- mean(val$rating)

#Movie effect
b_i <- val %>%  group_by(movieId) %>%summarize(b_i = sum(rating - mu)/(n()+l))
  
#User effect
b_u <- val %>%  left_join(b_i, by="movieId") %>% group_by(userId) %>%
  summarize(b_u = sum(rating - b_i - mu)/(n()+l))

#Add the movie and user effects to the validation dataset
val <-  val %>% left_join(b_i, by = "movieId") %>% left_join(b_u, by = "userId")

#Drop the variables not used in the final model
val <- val %>% select(-timestamp, -title)

# Remove the columns with zero variance and the ID columns
##: Columns with zero variance:6,13,16,21,22
#IMAX, Musical, Western, FilmNoir, Documentary
val <- val[ -c(6,13,16,21,22) ]
val <- val %>% select(-userId, -movieId)
edx2 <- train %>% select(-userId, -movieId)

## Try a linear regression on full EdX dataset
fit2 <- lm(rating ~ ., data = edx2)
## Use the betas to predict ratings in the validation dataset
y_hat_final <- predict(fit2, newdata = val)

#Now see the root mean square error: 0.6816817
final_rmse <- RMSE(val$rating, y_hat_final)

print(final_rmse)
