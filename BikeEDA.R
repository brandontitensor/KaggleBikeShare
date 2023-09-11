bike <- vroom::vroom("./train.csv") #reading in training data
library(tidyverse)

bike %>%  as_factor(bike$season) +
  as.factor(bike$holiday)+
  as.factor(bike$workingday) +
  as.factor(bike$weather) 


dplyr::glimpse(bike)
DataExplorer::plot_intro(bike)
DataExplorer::plot_correlation(bike)
DataExplorer::plot_bar(bike)
DataExplorer::plot_histogram(bike)
DataExplorer::plot_missing(bike)
GGally::ggpairs(bike)
library(patchwork)
ggplot(data=bike)
plot1 <- ggplot(data=bike) + 
  geom_point(mapping=aes(x=datetime,y=windspeed))
plot2 <- DataExplorer::plot_histogram(bike)
plot3 <- DataExplorer::plot_correlation(bike)
plot4 <- DataExplorer::plot_intro(bike)
plot1
EDA_plot <- (plot1 + plot2) / (plot3 + plot4)

ggsave("EDA_plot.png")
