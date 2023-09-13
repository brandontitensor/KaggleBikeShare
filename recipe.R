bike <- vroom::vroom("./train.csv") #reading in training data
library(tidyverse)

change_bike <- bike %>%   
  filter(weather == 4) %>% 
  select(-weather) %>% 
  mutate(weather = 3) 

change_bike <- change_bike[,c(1,2,3,4,12,5,6,7,8,9,10,11)]

clean_bike <- bike %>% 
  filter(weather != 4) %>% 
  rbind(.,change_bike)

library(tidymodels)

my_recipe <- recipe(count~., data = clean_bike) %>% 
  step_num2factor(season, levels = c("Winter","Spring","Summer","Fall")) %>% 
  step_num2factor(workingday, levels = c("No","Yes"),transform = function(x) x+1) %>% 
  step_num2factor(holiday, levels = c("No", "Yes"),transform = function(x) x+1)  %>% 
  step_date(datetime, features="dow") %>% 
  step_time(datetime, features=c("hour", "minute"))%>% 
  step_rm(casual, registered, datetime)
  
  
  

prepped_recipe <- prep(my_recipe, verbose = T)

clean_data <- bake(prepped_recipe, new_data = clean_bike)
