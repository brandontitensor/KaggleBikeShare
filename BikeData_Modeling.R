#############
##LIBRARIES##
#############

library(tidymodels) #For the recipes
library(tidyverse) #Given for EDA
library(poissonreg) #For Poisson Regression
library(vroom) #For reading in data
library(DataExplorer)

#################
##BIKE DATASETS##
#################

bike <- vroom::vroom("./train.csv") #reading in training data
bike <- bike %>% select(-c(casual, registered)) #Takes out these variables because they are not in test data

##########
##RECIPE##
##########

 my_recipe <- recipe(count~., data = bike) %>% 
  step_mutate(weather=ifelse(weather==4, 3, weather)) %>% #Changes any Weather events 4 into Weather event 3
  step_num2factor(season, levels = c("Winter","Spring","Summer","Fall")) %>% #Changes the labels of the Seasons from numbers
  step_num2factor(weather, levels = c("Sunny","Misty","Rainy")) %>%
  step_num2factor(workingday, levels = c("No","Yes"),transform = function(x) x+1) %>% #Changes the labels of workingday from numbers
  step_num2factor(holiday, levels = c("No", "Yes"),transform = function(x) x+1)  %>%  #Changes the labesl of Holiday from numbers
  step_date(datetime, features="dow") %>% #Creates a seperate variable called time of the week
  step_time(datetime, features="hour") %>% #Creates seperate variable with the hours and minutes
 # step_poly(windspeed, degree = 2) %>% #Adds a polynomial value of windspped
 # step_poly(atemp, degree = 2) %>%  #Adds a polynomial value of atemp
 # step_poly(temp, degree = 2) %>%  #Adds a polynomial value of temp
 # step_poly(humidity, degree = 2) %>% #Adds a polynomial value of humidity
  step_rm(datetime)  %>% #Removes the datetime and minute variables
  step_log(all_numeric_predictors(), signed = T)

prepped_recipe <- prep(my_recipe, verbose = T) #Prepping the recipe, Verbose helps me see if there are any errors

bake_1 <- bake(prepped_recipe, new_data = bike) #Testing to see if it recipe will work


#################
##TEST DATASETS##
#################

test <- vroom::vroom("./test.csv") #Read in the test dataset

bake(prepped_recipe, new_data = test) #Testing to see if recipe will work

#####################
##LINEAR REGRESSION##
#####################

lm_mod <- linear_reg() %>% #Applies Linear Model
  set_engine("lm")

bike_lm_workflow <- workflow() %>%  #Creates Linear Regression Workflow
  add_recipe(my_recipe) %>% 
  add_model(lm_mod) %>% 
  fit(data = bike)

extract_fit_engine(bike_lm_workflow) %>% #Extracts model details from workflow
  summary()

#################################
##LINEAR REGRESSION PREDICTIONS##
#################################

bike_lm_predictions <- predict(bike_lm_workflow,new_data = test) #linear Regression extracted ddetails

#######################################
##FORMAT AND WRITE LINEAR PREDICTIONS##
#######################################

bike_lm_predictions[bike_lm_predictions < 0] <- 0
bike_lm_predictions <- cbind(test$datetime,bike_lm_predictions)
colnames(bike_lm_predictions) <- c("datetime","count")
bike_lm_predictions$datetime <- as.character(format(bike_lm_predictions$datetime))

vroom::vroom_write(bike_lm_predictions,"bike_lm_predictions.csv",',')

############################
##POISSON REGRESSION MODEL##
############################

 pois_mod <- poisson_reg() %>% #Type of model
  set_engine("glm") # GLM = generalized linear model

bike_pois_workflow <- workflow() %>% #Creates a workflow
add_recipe(my_recipe) %>% #Adds in my recipe
add_model(pois_mod) %>% #Adds in which model we are using
fit(data = bike) # Fits the workflow to the data

extract_fit_engine(bike_pois_workflow) %>% #Extracts model details from workflow
  summary()

###################################
##POISSON REGRESSION PREDICTIONS##
###################################

bike_predictions_pois<- predict(bike_pois_workflow, new_data = test) #Creates preds for Possion Regression

########################################
##FORMAT AND WRITE POISSON PREDICTIONS##
########################################

bike_predictions_pois <- cbind(test$datetime,bike_predictions_pois) #Adds back in the dattime variable for submission
colnames(bike_predictions_pois) <- c("datetime","count") #Changes the labels for submission
bike_predictions_pois$datetime <- as.character(format(bike_predictions_pois$datetime)) #Formats the datetime variable to get rid of T's and Z's

vroom::vroom_write(bike_predictions_pois,"bike_predictions_pois.csv",',') #Writes a csv file for my preds to submit


