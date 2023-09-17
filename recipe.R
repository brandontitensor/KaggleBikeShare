 
bike <- vroom::vroom("./train.csv") #reading in training data
library(tidyverse)


change_bike <- bike %>%   
  filter(weather == 4) %>% 
  select(-weather) %>% 
  mutate(weather = 3) 

change_bike <- change_bike[,c(1,2,3,4,12,5,6,7,8,9,10,11)]

clean_bike <- bike %>% 
  filter(weather != 4) %>% 
  rbind(.,change_bike) %>% 
  select(-c(casual,registered))





library(tidymodels)

my_recipe <- recipe(count~., data = clean_bike) %>% 
  step_num2factor(season, levels = c("Winter","Spring","Summer","Fall")) %>% 
  step_num2factor(workingday, levels = c("No","Yes"),transform = function(x) x+1) %>% 
  step_num2factor(holiday, levels = c("No", "Yes"),transform = function(x) x+1)  %>% 
  step_date(datetime, features="dow") %>% 
  step_time(datetime, features=c("hour", "minute")) %>% 
  step_poly(windspeed, degree = 2) %>% 
  step_poly(atemp, degree = 2) %>% 
  step_poly(temp, degree = 2) %>% 
  step_poly(humidity, degree = 2)

###%>% 
 ### step_rm(casual, registered, datetime)
  
  
  

prepped_recipe <- prep(my_recipe, verbose = T)

clean_data <- bake(prepped_recipe, new_data = clean_bike)


clean_data <- clean_data %>% 
  select(-c(datetime, datetime_minute))

clean_data <- clean_data[,c(1,2,3,4,6,7,8,9,10,11,12,13,14,15,5)]

lm_bike <- lm(`count` ~., data = clean_data)
summary(lm_bike)





