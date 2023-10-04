#############
##LIBRARIES##
#############

library(tidymodels) #For the recipes
library(tidyverse) #Given for EDA
library(poissonreg) #For Poisson Regression
library(vroom) #For reading in data
library(DataExplorer)
library(glmnet)
library(mltools)
library(randomForest)
library(doParallel)
library(xgboost)
tidymodels_prefer()
conflicted::conflicts_prefer(yardstick::rmse)
library(rpart)
library(stacks) #For stacking
####################
##WORK IN PARALLEL##
####################

#library(doParallel)
all_cores <- parallel::detectCores(logical = FALSE)
registerDoParallel(cores = all_cores)

#################
##BIKE DATASETS##
#################

bike <- vroom::vroom("./train.csv") #reading in training data
bike <- bike %>% select(-c(casual, registered)) #Takes out these variables because they are not in test data

log_bike <- bike %>%
  mutate(count=log(count))

##########
##RECIPE##
##########

 my_recipe <- recipe(count~., data = log_bike) %>% 
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
  step_mutate(datetime_hour=factor(datetime_hour, levels=0:23, labels=c(0:23))) %>%
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

#######################
##LOG TRANSFORM COUNT##
#######################

########
##POIS##
########

bike_log_workflow <- workflow() %>% #Creates a workflow
  add_recipe(my_recipe) %>% #Adds in my recipe
  add_model(pois_mod) %>% #Adds in which model we are using
  fit(data = log_bike) # Fits the workflow to the data

extract_fit_engine(bike_log_workflow) %>% #Extracts model details from workflow
  summary()

bike_predictions_log<- predict(bike_log_workflow, new_data = test)

bike_predictions_log <- cbind(test$datetime,bike_predictions_log) #Adds back in the dattime variable for submission
bike_predictions_log <- bike_predictions_log %>% mutate(.pred=exp(.pred))
colnames(bike_predictions_log) <- c("datetime","count") #Changes the labels for submission
bike_predictions_log$datetime <- as.character(format(bike_predictions_log$datetime)) 

vroom_write(bike_predictions_log,"bike_predictions_log.csv",',')

######
##LM##
######

bike_log_lm_workflow <- workflow() %>% #Creates a workflow
  add_recipe(my_recipe) %>% #Adds in my recipe
  add_model(lm_mod) %>% #Adds in which model we are using
  fit(data = log_bike) # Fits the workflow to the data

extract_fit_engine(bike_log_lm_workflow) %>% #Extracts model details from workflow
  summary()

bike_predictions_log_lm<- predict(bike_log_lm_workflow, new_data = test)

bike_predictions_log_lm[bike_predictions_log_lm < 0] <- 0
bike_predictions_log_lm <- cbind(test$datetime,bike_predictions_log_lm) #Adds back in the dattime variable for submission
bike_predictions_log_lm <- bike_predictions_log_lm %>% mutate(.pred=exp(.pred))
colnames(bike_predictions_log_lm) <- c("datetime","count") #Changes the labels for submission
bike_predictions_log_lm$datetime <- as.character(format(bike_predictions_log_lm$datetime)) 

vroom_write(bike_predictions_log_lm,"bike_predictions_log_lm.csv",',')


########################
##PENALIZED REGRESSION##
########################

##########
##RECIPE##
##########

my_recipe_penreg <- recipe(count~., data = log_bike) %>% 
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
  step_mutate(datetime_hour=factor(datetime_hour, levels=0:23, labels=c(0:23))) %>%
  step_log(all_numeric_predictors(), signed = T) %>% 
  step_dummy(all_nominal_predictors()) %>% 
  step_normalize(all_numeric_predictors())

prepped_recipe_penreg <- prep(my_recipe_penreg, verbose = T) #Prepping the recipe, Verbose helps me see if there are any errors

bake_2 <- bake(prepped_recipe_penreg, new_data = bike) #Testing to see if it recipe will work

log_bike <- bike %>%
  mutate(count=log(count))

penreg_model <- linear_reg(penalty = .05, mixture = .05) %>% 
  set_engine("glmnet")

bike_penreg_workflow <- workflow() %>% #Creates a workflow
  add_recipe(my_recipe_penreg) %>% #Adds in my recipe
  add_model(penreg_model) %>% #Adds in which model we are using
  fit(data = log_bike) # Fits the workflow to the data

extract_fit_engine(bike_penreg_workflow) %>% #Extracts model details from workflow
  summary()

bike_predictions_penreg<- predict(bike_penreg_workflow, new_data = test)

bike_predictions_penreg[bike_predictions_penreg < 0] <- 0
bike_predictions_penreg <- cbind(test$datetime,bike_predictions_penreg) #Adds back in the dattime variable for submission
bike_predictions_penreg <- bike_predictions_penreg %>% mutate(.pred=exp(.pred))
colnames(bike_predictions_penreg) <- c("datetime","count") #Changes the labels for submission
bike_predictions_penreg$datetime <- as.character(format(bike_predictions_penreg$datetime)) 

vroom_write(bike_predictions_penreg,"bike_predictions_penreg.csv",',')

########
##POIS##
########

penreg_model1 <- poisson_reg(penalty = .05, mixture = .05) %>% 
  set_engine("glmnet")

bike_penreg_workflow1 <- workflow() %>% #Creates a workflow
  add_recipe(my_recipe_penreg) %>% #Adds in my recipe
  add_model(penreg_model1) %>% #Adds in which model we are using
  fit(data = log_bike) # Fits the workflow to the data

extract_fit_engine(bike_penreg_workflow1) %>% #Extracts model details from workflow
  summary()

bike_predictions_penreg1<- predict(bike_penreg_workflow1, new_data = test)

bike_predictions_penreg1[bike_predictions_penreg1 < 0] <- 0
bike_predictions_penreg1 <- cbind(test$datetime,bike_predictions_penreg1) #Adds back in the dattime variable for submission
bike_predictions_penreg1 <- bike_predictions_penreg1 %>% mutate(.pred=exp(.pred))
colnames(bike_predictions_penreg1) <- c("datetime","count") #Changes the labels for submission
bike_predictions_penreg1$datetime <- as.character(format(bike_predictions_penreg1$datetime)) 

vroom_write(bike_predictions_penreg1,"bike_predictions_penreg1.csv",',')


#########
##RLMSE##
#########

##Linear Regression
prediction1 <- predict(bike_lm_workflow,new_data = log_bike)
prediction1[prediction1 < 0] <- 0
rmsle(prediction1,log_bike$count)

#################
##RANDOM FOREST##
#################

RF_mod <- rand_forest(mode = "regression", trees = 500) %>% #Applies Linear Model
  set_engine("randomForest")

bike_RF_workflow <- workflow() %>% #Creates a workflow
  add_recipe(my_recipe) %>% #Adds in my recipe
  add_model(RF_mod) %>% #Adds in which model we are using
  fit(data = log_bike) # Fits the workflow to the data

extract_fit_engine(bike_RF_workflow) %>% #Extracts model details from workflow
  summary()

bike_predictions_RF<- predict(bike_RF_workflow, new_data = test)

bike_predictions_RF[bike_predictions_RF < 0] <- 0
bike_predictions_RF <- cbind(test$datetime,bike_predictions_RF) #Adds back in the dattime variable for submission
bike_predictions_RF <- bike_predictions_RF %>% mutate(.pred=exp(.pred))
colnames(bike_predictions_RF) <- c("datetime","count") #Changes the labels for submission
bike_predictions_RF$datetime <- as.character(format(bike_predictions_RF$datetime)) 

vroom_write(bike_predictions_RF,"bike_predictions_RF.csv",',')

############
##XG BOOST##
############

##############
##data split##
##############

bike_split <- vfold_cv(
  log_bike, 
  v = 5,
  repeats = 1
)

#########
##model##
#########

bike_cv_folds <- 
  recipes::bake(
    prepped_recipe, 
    new_data = log_bike
  ) %>%  
  rsample::vfold_cv(v = 20)

xgboost_model <- 
  parsnip::boost_tree(
    mode = "regression",
    trees = 500,
    min_n = tune(),
    tree_depth = tune(),
    learn_rate = tune(),
    loss_reduction = tune()
  ) %>%
  set_engine("xgboost", objective = "reg:squaredlogerror")

# grid specification
xgboost_params <- 
  dials::parameters(
    min_n(),
    tree_depth(),
    learn_rate(),
    loss_reduction()
  )

xgboost_grid <- 
  dials::grid_max_entropy(
    xgboost_params, 
    size = 60
  )

knitr::kable(head(xgboost_grid))

xgboost_wf <- 
  workflows::workflow() %>%
  add_model(xgboost_model) %>% 
  add_formula(count ~ .) 

# hyperparameter tuning
xgboost_tuned <- tune::tune_grid(
  object = xgboost_wf,
  resamples = bike_cv_folds,
  grid = xgboost_grid,
  metrics = yardstick::metric_set( yardstick::rmse, rsq, mae),
  control = tune::control_grid(verbose = TRUE)
)

xgboost_tuned %>%
  tune::show_best() %>%
  knitr::kable()

xgboost_best_params <- xgboost_tuned %>%
  tune::select_best()

knitr::kable(xgboost_best_params)

xgboost_model_final <- xgboost_model %>% 
  finalize_model(xgboost_best_params)

bike_xg_workflow <- workflows::workflow() %>%
  add_model(xgboost_model_final) %>% 
  add_formula(count ~ .) %>% 
  fit(data = log_bike)

bike_predictions_XG<- predict(bike_xg_workflow, new_data = test)
bike_predictions_XG[bike_predictions_XG < 0] <- 0
bike_predictions_XG <- cbind(test$datetime,bike_predictions_XG) #Adds back in the dattime variable for submission
bike_predictions_XG <- bike_predictions_XG %>% mutate(.pred=exp(.pred))
colnames(bike_predictions_XG) <- c("datetime","count") #Changes the labels for submission
bike_predictions_XG$datetime <- as.character(format(bike_predictions_XG$datetime)) 

vroom_write(bike_predictions_XG,"bike_predictions_XG.csv",',')


#############
##CV PREGLM##
#############

preg_model <- linear_reg(penalty=tune(),
                         mixture=tune()) %>% #Set model and tuning
  set_engine("glmnet") 

preg_wf <- workflow() %>%
add_recipe(my_recipe_penreg) %>%
add_model(preg_model)

tuning_grid <- grid_regular(penalty(),
                            mixture(),
                            levels = 10)
folds <- vfold_cv(log_bike, v = 20, repeats=1)

CV_results <- preg_wf %>%
tune_grid(resamples=folds,
          grid=tuning_grid,
          metrics=metric_set(rmse, mae, rsq))
bestTune <- CV_results %>%
select_best("rmse")

final_penreg_wf <- preg_wf %>% 
  finalize_workflow(bestTune) %>% 
  fit(log_bike)

###penreg_model1 <- linear_reg(penalty = 0.005994843, mixture = 0.2222222) %>% 
 # set_engine("glmnet")

#bike_penreg_workflow1 <- workflow() %>% #Creates a workflow
 # add_recipe(my_recipe_penreg) %>% #Adds in my recipe
#  add_model(penreg_model1) %>% #Adds in which model we are using
#  fit(data = log_bike) # Fits the workflow to the data


#bike_predictions_penreg1<- predict(bike_penreg_workflow1, new_data = test)

bike_predictions_penreg1 <- final_penreg_wf %>% 
  predict(new_data = test)

bike_predictions_penreg1[bike_predictions_penreg1 < 0] <- 0
bike_predictions_penreg1 <- cbind(test$datetime,bike_predictions_penreg1) #Adds back in the dattime variable for submission
bike_predictions_penreg1 <- bike_predictions_penreg1 %>% mutate(.pred=exp(.pred))
colnames(bike_predictions_penreg1) <- c("datetime","count") #Changes the labels for submission
bike_predictions_penreg1$datetime <- as.character(format(bike_predictions_penreg1$datetime)) 

vroom_write(bike_predictions_penreg1,"bike_predictions_penreg1.csv",',')

#########
##CV RF##
#########


RF_model <- rand_forest(mode = "regression",
                        mtry = tune(),
                        trees = 500,
                        min_n = tune()) %>% #Applies Linear Model
  set_engine("randomForest")

RF_workflow <- workflow() %>% #Creates a workflow
  add_recipe(my_recipe) %>% #Adds in my recipe
  add_model(RF_model) 

tuning_grid_rf <- grid_regular(mtry(range = c(1,10)),
                               min_n(),
                               levels = 5)
folds_rf <- vfold_cv(log_bike, v = 10, repeats=1)

CV_results_rf <- RF_workflow %>%
  tune_grid(resamples=folds_rf,
            grid=tuning_grid_rf,
            metrics=metric_set(rmse, mae, rsq))
bestTune_rf <- CV_results_rf %>%
  select_best("rmse")

##.4317 Trees = 500 minn = 11 mtry = 7

#Trees = 1333 minn = 2

#RF_mod <- rand_forest(mode = "regression", trees = 500) %>% #Applies Linear Model
#  set_engine("randomForest")

#bike_RF_workflow <- workflow() %>% #Creates a workflow
#  add_recipe(my_recipe) %>% #Adds in my recipe
#  add_model(RF_mod) %>% #Adds in which model we are using
#  fit(data = log_bike) # Fits the workflow to the data

#extract_fit_engine(bike_RF_workflow) %>% #Extracts model details from workflow
#  summary()

#bike_predictions_RF<- predict(bike_RF_workflow, new_data = test)

final_rf_wf1 <- RF_workflow %>% 
  finalize_workflow(bestTune_rf) %>% 
  fit(data = log_bike)


bike_predictions_RF<- final_rf_wf1 %>% 
  predict(new_data = test)

bike_predictions_RF[bike_predictions_RF < 0] <- 0
bike_predictions_RF <- cbind(test$datetime,bike_predictions_RF) #Adds back in the dattime variable for submission
bike_predictions_RF <- bike_predictions_RF %>% mutate(.pred=exp(.pred))
colnames(bike_predictions_RF) <- c("datetime","count") #Changes the labels for submission
bike_predictions_RF$datetime <- as.character(format(bike_predictions_RF$datetime)) 

vroom_write(bike_predictions_RF,"bike_predictions_RF.csv",',')

##########################
##CLASS REGRESSION TREES##. **DIFFERENT THAN RANDOM FORESTS**
##########################

RF_mod1 <- decision_tree(tree_depth = tune(),
                         cost_complexity = tune(),
                         min_n = tune()) %>% 
  set_engine("rpart") %>% 
  set_mode("regression")

RF_workflow2 <- workflow() %>% #Creates a workflow
  add_recipe(my_recipe) %>% #Adds in my recipe
  add_model(RF_mod1)

tuning_grid_rf2 <- grid_regular(tree_depth(),
                               cost_complexity(),
                               min_n(),
                               levels = 10)
folds_rf2 <- vfold_cv(log_bike, v = 5, repeats=1)

CV_results_rf2 <- RF_workflow2 %>%
  tune_grid(resamples=folds_rf2,
            grid=tuning_grid_rf2,
            metrics=metric_set(rmse, mae, rsq))
bestTune_rf2 <- CV_results_rf2 %>%
  select_best("rmse")

final_rf_wf <- RF_workflow2 %>% 
  finalize_workflow(bestTune_rf2) %>% 
  fit(data = log_bike)


bike_predictions_RF2<- final_rf_wf %>% 
  predict(new_data = test)

bike_predictions_RF2[bike_predictions_RF2 < 0] <- 0
bike_predictions_RF2 <- cbind(test$datetime,bike_predictions_RF2) #Adds back in the dattime variable for submission
bike_predictions_RF2 <- bike_predictions_RF2 %>% mutate(.pred=exp(.pred))
colnames(bike_predictions_RF2) <- c("datetime","count") #Changes the labels for submission
bike_predictions_RF2$datetime <- as.character(format(bike_predictions_RF2$datetime)) 

vroom_write(bike_predictions_RF2,"bike_predictions_RF2.csv",',')

###################
##STACKING MODELS##
###################

folds_stack <- vfold_cv(log_bike, v = 3, repeats=1)
metric_stack <- metric_set(rmse, mae, rsq)

untuned_model <- control_stack_grid()
tuned_model <- control_stack_resamples()

#PENALIZED REGRESSION

preg_model_stack <- linear_reg(penalty = tune(),
                            mixture = tune()) %>% 
  set_engine("glmnet")

preg_wf_stack <- workflow() %>% 
  add_recipe(my_recipe_penreg) %>% 
  add_model(preg_model_stack)

stack_tuning_grid <- grid_regular(penalty(),
                                  mixture(),
                                  levels = 5)

preg_stack_models <- preg_wf_stack %>%
  tune_grid(resamples=folds_stack,
            grid=stack_tuning_grid,
            metrics=metric_stack,
            control = untuned_model)   


#LINEAR REGRESSION

lm_stack_model <- 
  linear_reg() %>% 
  set_engine("lm")

lin_reg_wf <- workflow()  %>% #Creates a workflow
  add_recipe(my_recipe) %>% #Adds in my recipe
  add_model(lm_stack_model)


lin_reg_models <-
  fit_resamples(lin_reg_wf,
                 resamples = folds_stack,
                 metrics = metric_stack,
                 control = tuned_model)


#RANDOM FOREST

RF_model_stack <- rand_forest(mode = "regression",
                        mtry = tune(),
                        trees = 500,
                        min_n = tune()) %>% #Applies Linear Model
  set_engine("randomForest")

RF_stack_workflow <- workflow() %>% #Creates a workflow
  add_recipe(my_recipe) %>% #Adds in my recipe
  add_model(RF_model_stack) 

tuning_grid_rf_stack <- grid_regular(mtry(range = c(1,10)),
                               min_n(),
                               levels = 3)

rf_stack_models <- RF_stack_workflow %>%
  tune_grid(resamples=folds_stack,
            grid=tuning_grid_rf_stack,
            metrics=metric_stack,
            control = untuned_model)

autoplot(rf_stack_models)


#STACKING MODELS

my_stack <- stacks() %>%
  add_candidates(preg_stack_models) %>%
  add_candidates(lin_reg_models) %>%
  add_candidates(rf_stack_models) 

stack_mod <- my_stack %>%
blend_predictions() %>% 
  fit_members() 

stack_pred <- stack_mod %>% predict(new_data=test)

stack_pred[stack_pred < 0] <- 0
stack_pred <- cbind(test$datetime,stack_pred) #Adds back in the dattime variable for submission
stack_pred <- stack_pred %>% mutate(.pred=exp(.pred))
colnames(stack_pred) <- c("datetime","count") #Changes the labels for submission
stack_pred$datetime <- as.character(format(stack_pred$datetime)) 

vroom_write(stack_pred,"stack_pred.csv",',')

autoplot(stack_mod)

########
##BART##
########

bart_model <- bart(mode = "regression",
                   engine = "dbarts",
                   trees = 1113,
                   prior_terminal_node_coef = tune(),
                   prior_terminal_node_expo = tune(),
                   prior_outcome_range = tune())


bart_workflow <- workflow() %>% #Creates a workflow
  add_recipe(registered_recipe) %>% #Adds in my recipe
  add_model(bart_model) 

tuning_grid_bart <- grid_regular(prior_terminal_node_coef(),
                                 prior_terminal_node_expo(),
                                 prior_outcome_range(),
                                 levels = 3)
folds_bart <- vfold_cv(log_bike, v = 5, repeats=1)

CV_results_bart <- bart_workflow %>%
  tune_grid(resamples=folds_bart,
            grid=tuning_grid_bart,
            metrics=metric_set(rmse, mae, rsq))
bestTune_bart <- CV_results_bart %>%
  select_best("rmse")

final_bart_wf1 <- bart_workflow %>% 
  finalize_workflow(bestTune_bart) %>% 
  fit(data = log_new_bike)


bike_predictions_bart<- final_bart_wf1 %>% 
  predict(new_data = test)

bike_predictions_bart[bike_predictions_bart < 0] <- 0
bike_predictions_bart <- cbind(test$datetime,bike_predictions_bart) #Adds back in the dattime variable for submission
bike_predictions_bart <- bike_predictions_bart %>% mutate(.pred=exp(.pred))
colnames(bike_predictions_bart) <- c("datetime","count") #Changes the labels for submission
bike_predictions_bart$datetime <- as.character(format(bike_predictions_bart$datetime)) 

vroom_write(bike_predictions_bart,"bike_predictions_bart.csv",',')

##First Result = .42233, prior_terminal_node_coef = .5, prior_terminal_node_expo = 1, prior_outcome_range = 2.5
## new_recipe Second Result =.36834 ,same things 
## registered_recipe Third Result = 

##############
##NEW RECIPE##
###############

new_recipe <- recipe(count~., data = log_bike) %>% 
  step_mutate(weather=ifelse(weather==4, 3, weather)) %>% #Changes any Weather events 4 into Weather event 3
  step_num2factor(season, levels = c("Winter","Spring","Summer","Fall")) %>% #Changes the labels of the Seasons from numbers
  step_num2factor(weather, levels = c("Sunny","Misty","Rainy")) %>%
  step_num2factor(workingday, levels = c("No","Yes"),transform = function(x) x+1) %>% #Changes the labels of workingday from numbers
  step_num2factor(holiday, levels = c("No", "Yes"),transform = function(x) x+1)  %>%  #Changes the labesl of Holiday from numbers
  step_date(datetime, features="dow") %>% #Creates a seperate variable called time of the week
  step_time(datetime, features="hour") %>% #Creates seperate variable with the hours and minutes
  step_date(datetime, features = "year") %>% 
  step_rm(datetime)  %>% #Removes the datetime and minute variables
  step_mutate(datetime_hour=factor(datetime_hour, levels=0:23, labels=c(0:23))) %>%
  step_mutate(datetime_year = factor(datetime_year, levels = c(2011,2012), labels = c(2011,2012))) %>% 
  step_log(all_numeric_predictors(), signed = T)


prepped_recipe <- prep(new_recipe, verbose = T)

bake_3 <- bake(prepped_recipe, new_data = bike)

##REgistered and Casual Data##

new_bike <- vroom::vroom("./train.csv") #reading in training data
new_bike <- new_bike %>% select(-c(count))

registered_bike <- new_bike %>%
  mutate(registered=log(registered)) %>% 
  select(-c(casual))

registered_bike$registered[registered_bike$registered < 0] <- 0

registered_recipe <- recipe(registered~., data = registered_bike) %>% 
  step_mutate(weather=ifelse(weather==4, 3, weather)) %>% #Changes any Weather events 4 into Weather event 3
  step_num2factor(season, levels = c("Winter","Spring","Summer","Fall")) %>% #Changes the labels of the Seasons from numbers
  step_num2factor(weather, levels = c("Sunny","Misty","Rainy")) %>%
  step_num2factor(workingday, levels = c("No","Yes"),transform = function(x) x+1) %>% #Changes the labels of workingday from numbers
  step_num2factor(holiday, levels = c("No", "Yes"),transform = function(x) x+1)  %>%  #Changes the labesl of Holiday from numbers
  step_date(datetime, features="dow") %>% #Creates a seperate variable called time of the week
  step_time(datetime, features="hour") %>% #Creates seperate variable with the hours and minutes
  step_date(datetime, features = "year") %>% 
  step_rm(datetime)  %>% #Removes the datetime and minute variables
  step_mutate(datetime_hour=factor(datetime_hour, levels=0:23, labels=c(0:23))) %>%
  step_mutate(datetime_year = factor(datetime_year, levels = c(2011,2012), labels = c(2011,2012))) %>% 
  step_log(all_numeric_predictors(), signed = T) 

prepped_recipe <- prep(registered_recipe, verbose = T)

bake_4 <- bake(prepped_recipe, new_data = registered_bike)

casual_bike <- new_bike %>%
  mutate(casual=log(casual)) %>% 
  select(-c(registered))

casual_bike$casual[casual_bike$casual < 0] <- 0

casual_recipe <- recipe(casual~., data = casual_bike) %>% 
  step_mutate(weather=ifelse(weather==4, 3, weather)) %>% #Changes any Weather events 4 into Weather event 3
  step_num2factor(season, levels = c("Winter","Spring","Summer","Fall")) %>% #Changes the labels of the Seasons from numbers
  step_num2factor(weather, levels = c("Sunny","Misty","Rainy")) %>%
  step_num2factor(workingday, levels = c("No","Yes"),transform = function(x) x+1) %>% #Changes the labels of workingday from numbers
  step_num2factor(holiday, levels = c("No", "Yes"),transform = function(x) x+1)  %>%  #Changes the labesl of Holiday from numbers
  step_date(datetime, features="dow") %>% #Creates a seperate variable called time of the week
  step_time(datetime, features="hour") %>% #Creates seperate variable with the hours and minutes
  step_date(datetime, features = "year") %>% 
  step_rm(datetime)  %>% #Removes the datetime and minute variables
  step_mutate(datetime_hour=factor(datetime_hour, levels=0:23, labels=c(0:23))) %>%
  step_mutate(datetime_year = factor(datetime_year, levels = c(2011,2012), labels = c(2011,2012))) %>% 
  step_log(all_numeric_predictors(), signed = T)

prepped_recipe <- prep(casual_recipe, verbose = T)

bake_5 <- bake(prepped_recipe, new_data = casual_bike)

bart_model <- bart(mode = "regression",
                   engine = "dbarts",
                   trees = 1113,
                   prior_terminal_node_coef = tune(),
                   prior_terminal_node_expo = tune(),
                   prior_outcome_range = tune())


bart_workflow <- workflow() %>% #Creates a workflow
  add_recipe(registered_recipe) %>% #Adds in my recipe
  add_model(bart_model) 

casual_workflow <- workflow() %>% #Creates a workflow
  add_recipe(casual_recipe) %>% #Adds in my recipe
  add_model(bart_model) 


tuning_grid_bart1 <- grid_regular(prior_terminal_node_coef(),
                                 prior_terminal_node_expo(),
                                 prior_outcome_range(),
                                 levels = 3)
folds_bart <- vfold_cv(registered_bike, v = 5, repeats=1)

CV_results_bart <- bart_workflow %>%
  tune_grid(resamples=folds_bart,
            grid=tuning_grid_bart1,
            metrics=metric_set(rmse, mae, rsq))
bestTune_bart <- CV_results_bart %>%
  select_best("rmse")



final_registered_wf1 <- bart_workflow %>% 
  finalize_workflow(bestTune_bart) %>% 
  fit(data = registered_bike)

final_casual_wf1 <- casual_workflow %>% 
  finalize_workflow(bestTune_bart) %>% 
  fit(data = casual_bike)


registered_predictions_bart<- final_registered_wf1 %>% 
  predict(new_data = test)

casual_predictions_bart<- final_casual_wf1 %>% 
  predict(new_data = test)

casual_predictions_bart$.pred <- (casual_predictions_bart$.pred + registered_predictions_bart$.pred)/2

split_predictions <- casual_predictions_bart

split_predictions[split_predictions < 0] <- 0
split_predictions <- cbind(test$datetime,split_predictions) #Adds back in the dattime variable for submission
split_predictions <- split_predictions %>% mutate(.pred=exp(.pred))
colnames(split_predictions) <- c("datetime","count") #Changes the labels for submission
split_predictions$datetime <- as.character(format(split_predictions$datetime)) 

vroom_write(split_predictions,"split_predictions.csv",',')
