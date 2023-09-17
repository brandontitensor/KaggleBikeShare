library(tidymodels)

test <- vroom::vroom("./test.csv")

change_test <- test %>%   
  filter(weather == 4) %>% 
  select(-weather) %>% 
  mutate(weather = 3) 

change_test <- change_test[,c(1,2,3,4,12,5,6,7,8,9,10,11)]

clean_test <- test %>% 
  filter(weather != 4) %>% 
  rbind(.,change_test)

my_mod <- linear_reg() %>% 
  set_engine("lm")

bike_workflow <- workflow() %>% 
  add_recipe(my_recipe) %>% 
  add_model(my_mod) %>% 
  fit(data = clean_bike)

bike_predictions <- predict(bike_workflow,
                           new_data = clean_test)
bike_predictions[bike_predictions < 0] <- 0
bike_predictions <- cbind(clean_test$datetime,bike_predictions)
colnames(bike_predictions) <- c("datetime","count")
bike_predictions$datetime <- as.character(format(bike_predictions$datetime))

#bike_predictions<- bike_predictions %>% mutate(datetime = as.character(datetime))

vroom::vroom_write(bike_predictions,"bike_predictions.csv",',')
write.csv(bike_predictions, "Bike_predictions.csv")
