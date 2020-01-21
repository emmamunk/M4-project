#Indlæsning af pakker
if (!require("pacman")) install.packages("pacman") # package for loading and checking packages :)
pacman::p_load(readxl,
               tidyverse,
               ggplot2,
               ggthemes,
               keras,
               broom,
               drat,
               reticulate,
               EBImage,
               gtools,
               gridExtra
)


#Data-preprocess
DATA_rå <- read_excel("C:/Users/Andreas/Desktop/SDS - M4/Skadespoint.xlsx")
#Lav tidsstempel til år
DATA_rå$TIDSTEMPEL <- as.Date(DATA_rå$TIDSTEMPEL)
DATA_rå$ÅR <- format(as.Date(DATA_rå$TIDSTEMPEL, format="%Y/%m/%d"),"%Y")
#Sæt året til 2018
DATA_rå <- DATA_rå %>% filter(ÅR==2018)

DATA_rå$Nummer <- seq.int(nrow(DATA_rå))

DATA_udgå <- read_excel("C:/Users/Andreas/Desktop/SDS - M4/DATA18 - Ikke brugbare.xlsx")
DATA_udgå <- DATA_udgå %>% select(Nummer) 
#Samle datasæts gennem antijoin funktionen
Data <- anti_join(DATA_rå,DATA_udgå, by = "Nummer")
#Udvælgelse af variabler  
Data <- Data %>% select(SKADESPOINT, VEJNAVN, GRUPPE)
#Omdøb skadespoint til numeriske værdier
Data$SKADESPOINT <- as.numeric(Data$SKADESPOINT)

#Antal af unikke observationer
summary(Data)
Skadespoint <- unique(Data$SKADESPOINT)
Vejnavn <- unique(Data$VEJNAVN)
Gruppe <- unique(Data$GRUPPE)

#Plot af fordeling af skadespoint og gruppe
Data %>% ggplot(aes(x=SKADESPOINT)) +
  geom_bar() +
  labs(x=NULL, y="Antal") +
  theme_economist_white()

Data %>% select(GRUPPE) %>% group_by(GRUPPE) %>% count() %>% ungroup() %>% 
  ggplot(aes(x=GRUPPE, y=n)) +
  geom_col() +
  labs(x=NULL, y="Antal") +
  theme_economist_white()

#Model-data
setwd("...") #indsæt sti til train billeder

#Indlæsning af training datasæt
train_image_navn <- list.files(path=getwd(), pattern = "*.png")
train_image_navn <- mixedsort(sort(train_image_navn)) #Sorter billede efter navn (1, 2, 3 etc.)
train_image <- readImage(train_image_navn)
train_image <- as.array(train_image)
dim(train_image)

#Transpose array
train_data <- aperm(train_image, c(4,1,2,3))
dim(train_data)

#Indlæs training label og transponere vektor
train_label <- read.csv2("...")#"indsæt csv-fil"
train_label <- as.numeric(t(test_label))

setwd("...") #indsæt sti til test billeder

#Indlæsning af test datasæt
test_image_navn <- list.files(path=getwd(), pattern = "*.png")
test_image_navn <- mixedsort(sort(test_image_navn)) #Sorter billede efter navn (1, 2, 3 etc.)
test_image <- readImage(test_image_navn)
test_image <- as.array(test_image)
dim(test_image)
#Transpose array
test_data <- aperm(test_image, c(4,1,2,3))
dim(test_data)
#Indlæs test label og transponere vektor
test_label <- read.csv2("...")#"indsæt csv-fil"
test_label <- as.numeric(t(test_label))


#Model - Baseline med MAE
baseline1 <- keras_model_sequential()
#Setup af baseline model
baseline1 %>%
  layer_conv_2d(filter = 32, kernel_size = c(3,3), padding = "same", input_shape = c(124,124, 3), activation = "relu") %>%
  layer_conv_2d(filter = 16, kernel_size = c(3,3), padding = "same", activation = "relu") %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_flatten() %>%
  layer_dense(100, activation = "relu") %>%
  layer_dropout(0.5) %>%
  layer_dense(1)
#Summary af model
summary(baseline1)
#Angiv compiler
baseline1 %>% compile(
  loss = "mae",
  optimizer = "adam"
)
#Træningsforløb plot
history1 <- baseline1 %>% fit(train_data, train_label,
                               epochs = 30,
                               validation_split = 0.2,
                               verbose = 1)
#Prædiktions resultat
baseline1_resultat <- predict(baseline1, test_data)


#Model - Baseline med MSE
baseline2 <- keras_model_sequential()
#Setup af baseline model
baseline2 %>%
  layer_conv_2d(filter = 32, kernel_size = c(3,3), padding = "same", input_shape = c(124,124, 3), activation = "relu") %>%
  layer_conv_2d(filter = 16, kernel_size = c(3,3), padding = "same", activation = "relu") %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_flatten() %>%
  layer_dense(100, activation = "relu") %>%
  layer_dropout(0.5) %>%
  layer_dense(1)
#Summary af model
summary(baseline2)
#Angiv compiler
baseline2 %>% compile(
  loss = "mse",
  optimizer = "adam"
)
#Træningsforløb plot
history2 <- baseline2 %>% fit(train_data, train_label,
                               epochs = 30,
                               validation_split = 0.2,
                               verbose = 1)
#Prædiktions resultat
baseline2_resultat <- predict(baseline2, test_data)


#Model - VGG16 med MAE
vgg16 <- keras::application_vgg16(include_top = FALSE,
                                  weights = 'imagenet',
                                  input_shape = c(124, 124 ,3))
#Frys de første 14 lag af modellen
vgg16 %>% freeze_weights(from = 1, to = 14)
#Setup af VGG16 model
vgg16_1 <- keras_model_sequential(vgg16)
vgg16_1 %>% 
  layer_flatten() %>% 
  layer_dense(100, activation = "relu") %>%
  layer_dropout(0.5) %>%
  layer_dense(1)
#Summary VGG16
summary(vgg16_1)
#Angiv compiler
vgg16_1 %>% compile(
  loss = "mae",
  optimizer = "adam"
)
#Træningsforløb plot
history3 <- vgg16_1 %>% fit(train_data, train_label,
                           epochs = 30,
                           validation_split = 0.2,
                           verbose = 1)
#Prædiktions resultat
vgg16_1_resultat <- predict(vgg16_1, test_data)


#Model - VGG16 med MSE
vgg16_2 <- keras_model_sequential(vgg16)
vgg16_2 %>% 
  layer_flatten() %>% 
  layer_dense(100, activation = "relu") %>%
  layer_dropout(0.5) %>%
  layer_dense(1)
#Summary VGG16
summary(vgg16_2)
#Angiv compiler
vgg16_2 %>% compile(
  loss = "mse",
  optimizer = "adam"
)
#Træningsforløb plot
history4 <- vgg16_2 %>% fit(train_data, train_label,
                           epochs = 30,
                           validation_split = 0.2,
                           verbose = 1)
#Prædikations plot
vgg16_2_resultat <- predict(vgg16_2, test_data)


#Sekvens data -> Definerer længde af sekvens og batch size
train_data_sekvens <- timeseries_generator(train_data,train_label,length = 5, batch_size = 15)
test_data_sekvens <- timeseries_generator(test_data,test_label,length = 5, batch_size = 15)

#Sekvens-model - med Baseline og MAE
sekvens_base1 <- keras_model_sequential()
#Model setup
sekvens_base1 %>% 
  time_distributed(layer_conv_2d(filter = 32, kernel_size = c(3,3), padding = "same", activation = "relu"),input_shape=c(5,124,124,3)) %>% 
  time_distributed(layer_conv_2d(filter = 16, kernel_size = c(3,3), padding = "same", activation = "relu")) %>%
  time_distributed(layer_max_pooling_2d(pool_size = c(2,2))) %>% 
  time_distributed(layer_flatten()) %>% 
  layer_lstm(units=16,activation = "relu") %>% 
  layer_dense(100, activation = "relu") %>%
  layer_dropout(0.5) %>%
  layer_dense(1)
#Summary af baseline sekvens model
summary(sekvens_base1)
#Angiv compiler
sekvens_base1 %>% compile(
  loss = "mae",
  optimizer = "adam"
)
#Træningsforløb plot
history5 <- sekvens_base1 %>% fit_generator(train_data_sekvens,
                                                 steps_per_epoch = 4,
                                                 epochs = 2,
                                                 verbose = 1)
#Prædiktions resultat
sekvens_base1_resultat <- predict_generator(sekvens_base1, test_data_sekvens, 2)


#Sekvens-model - med Baseline og MSE
sekvens_base2 <- keras_model_sequential()
#Model setup
sekvens_base2 %>% 
  time_distributed(layer_conv_2d(filter = 32, kernel_size = c(3,3), padding = "same", activation = "relu"),input_shape=c(5,124,124,3)) %>% 
  time_distributed(layer_conv_2d(filter = 16, kernel_size = c(3,3), padding = "same", activation = "relu")) %>%
  time_distributed(layer_max_pooling_2d(pool_size = c(2,2))) %>% 
  time_distributed(layer_flatten()) %>% 
  layer_lstm(units=16,activation = "relu") %>% 
  layer_dense(100, activation = "relu") %>%
  layer_dropout(0.5) %>%
  layer_dense(1)
#Summary af model
summary(sekvens_base2)
#Angiv compiler
sekvens_base2 %>% compile(
  loss = "mse",
  optimizer = "adam"
)
#Træningsforløb plot
history6 <- sekvens_base2 %>% fit_generator(train_data_sekvens,
                                                 steps_per_epoch = 4,
                                                 epochs = 2,
                                                 verbose = 1)
#Prædiktions resultat
sekvens_base2_resultat <- predict_generator(sekvens_base2, test_data_sekvens, 2)


#Sekvens-model - med VGG16 og MAE
vgg16_sek <- keras_model_sequential(vgg16)
vgg16_sek <- vgg16_sek %>% layer_flatten()

sekvens_vgg16_1 <- keras_model_sequential()
#Model setup
sekvens_vgg16_1 %>% 
  time_distributed(vgg16_sek,input_shape=c(5,124,124,3)) %>% 
  layer_lstm(units=16,activation = "relu") %>% 
  layer_dense(100, activation = "relu") %>%
  layer_dropout(0.5) %>%
  layer_dense(1)
#Summary af model
summary(sekvens_vgg16_1)
#Angiv compiler
sekvens_vgg16_1 %>% compile(
  loss = "mae",
  optimizer = "adam"
)
#Træningsforløb plot
history7 <- sekvens_vgg16_1 %>% fit_generator(train_data_sekvens,
                                                     steps_per_epoch = 4,
                                                     epochs = 2,
                                                     verbose = 1)
#Prædiktions resultat
sekvens_vgg16_1_resultat <- predict_generator(sekvens_vgg16_1, test_data_sekvens, 2) 


#Sekvens-model - med VGG16 og MSE
sekvens_vgg16_2 <- keras_model_sequential()
#Model setup
sekvens_vgg16_2 %>% 
  time_distributed(vgg16_sek,input_shape=c(5,124,124,3)) %>% 
  layer_lstm(units=16,activation = "relu") %>% 
  layer_dense(100, activation = "relu") %>%
  layer_dropout(0.5) %>%
  layer_dense(1)
#Summary af model
summary(sekvens_vgg16_2)
#Angiv compiler
sekvens_vgg16_2 %>% compile(
  loss = "mse",
  optimizer = "adam"
)
#Træningsforløb plot
history8 <- sekvens_vgg16_2 %>% fit_generator(train_data_sekvens,
                                                     steps_per_epoch = 4,
                                                     epochs = 2,
                                                     verbose = 1)
#Prædiktions resultat
sekvens_vgg16_2_resultat <- predict_generator(sekvens_vgg16_2, test_data_sekvens, 2) 


#Samlet træningsforløbs plot for alle modeller
par(mfrow=c(1,2))
plot(history1)
plot(history2)
plot(history3)
plot(history4)
plot(history5)
plot(history6)
plot(history7)
plot(history8)

#Resultat plots
RMSE = function(m, o){
  sqrt(mean((m - o)^2))
}
MAE = function(m, o){
  mean(abs(m - o))
}
#Prædiktion af MAE for de fire modeller
mae_base1 <- MAE(baseline1_resultat, test_label)
mae_vgg16_1 <- MAE(vgg16_1_resultat, test_label)
mae_base_sek1 <- MAE(sekvens_base1_resultat, test_label30)
mae_VGG16_sek1 <- MAE(sekvens_vgg16_1_resultat, test_label30)
#Samle alle MAE
plot_mae <- cbind(mae_base1,mae_vgg16_1,mae_base_sek1,mae_VGG16_sek1)
#Plot performance evne graf for MAE
p0a <- plot_mae %>% 
  as_tibble %>% 
  gather(variable, mae) %>% 
  mutate(Model = c("Baseline", "VGG16", "Baseline sekvens", "VGG16 sekvens")) %>% 
  ggplot(aes(x=reorder(Model, desc(mae)),y=mae)) + 
  geom_col() +
  geom_text(aes(y=(mae)+0.025, label=format(round(mae,2))), color="black") +
  labs(title="Model performance: Mean Absolute Error", subtitle="Alle modellerne er kørt med samme train/test split",
       y="MAE", x=NULL) + 
  coord_flip()
#Prædiktion af RMSE for de fire modeller
rmse_base2 <- RMSE(baseline2_resultat, test_label)
rmse_vgg16_2 <- RMSE(vgg16_2_resultat, test_label)
rmse_base_sek2 <- RMSE(sekvens_base2_resultat, test_label30)
rmse_VGG16_sek2 <- RMSE(sekvens_vgg16_2_resultat, test_label30)
#Samle alle RMSE
plot_rmse <- cbind(rmse_base2,rmse_vgg16_2,rmse_base_sek2,rmse_VGG16_sek2)
#Plot performance evne graf for RMSE
p0b <- plot_rmse %>% 
  as_tibble %>% 
  gather(variable, rmse) %>% 
  mutate(Model = c("Baseline", "VGG16", "Baseline sekvens", "VGG16 sekvens")) %>% 
  ggplot(aes(x=reorder(Model, desc(rmse)),y=rmse)) + 
  geom_col() +
  geom_text(aes(y=(rmse)+0.03, label=format(round(rmse,2))), color="black") +
  labs(title="Model performance: Root Mean Squared Error", subtitle="Alle modellerne er kørt med samme train/test split",
       y="RMSE", x=NULL) + 
  coord_flip()
grid.arrange(p0a,p0b,nrow=2)

#Scatterplot for de fire model typer
#Baseline
scatter1a <- as.data.frame(baseline1_resultat)
scatter1a$label <- test_label
scatter1b <- as.data.frame(baseline2_resultat)
scatter1b$label <- test_label
#VGG16
scatter2a <- as.data.frame(vgg16_1_resultat)
scatter2a$label <- test_label
scatter2b <- as.data.frame(vgg16_2_resultat)
scatter2b$label <- as.data.frame(test_label)
#Baseline sekvens
scatter3a <- as.data.frame(sekvens_base1_resultat)
scatter3a$label <- test_label30
scatter3b <- as.data.frame(sekvens_base2_resultat)
scatter3b$label <- test_label30
#VGG16 sekvens
scatter4a <- as.data.frame(sekvens_vgg16_1_resultat)
scatter4a$label <- test_label30
scatter4b <- as.data.frame(sekvens_vgg16_2_resultat)
scatter4b$label <- test_label30
#Scatter plot baseline MAE
p1 <-scatter1a %>% rename(predicion=baseline1_resultat) %>% 
  ggplot(aes(x = predicion, y = test_label)) +
  geom_point() +
  stat_smooth(method = "lm",
              col = "#C42126",
              se = T,
              size = 1) + 
  labs(title="Scatterplot af baseline-model med MAE",y="Sande værdi", x="Prædikteret værdi")
#Scatterplot for baseline MSE
p2 <-scatter1b %>% rename(predicion=baseline2_resultat) %>% 
  ggplot(aes(x = predicion, y = test_label)) +
  geom_point() +
  stat_smooth(method = "lm",
              col = "#C42126",
              se = T,
              size = 1) + 
  labs(title="Scatterplot af baseline-model med MSE",y="Sande værdi", x="Prædikteret værdi")

grid.arrange(p1,p2, nrow=2)
#Scatter plot VGG16 MAE
p3 <-scatter2a %>% rename(predicion=vgg16_1_resultat) %>% 
  ggplot(aes(x = predicion, y = test_label)) +
  geom_point() +
  stat_smooth(method = "lm",
              col = "#C42126",
              se = T,
              size = 1) + 
  labs(title="Scatterplot af VGG16-model med MAE",y="Sande værdi", x="Prædikteret værdi")
#Scatter plot VGG16 MSE
p4 <-scatter2b %>% rename(predicion=vgg16_2_resultat) %>% 
  ggplot(aes(x = predicion, y = test_label)) +
  geom_point() +
  stat_smooth(method = "lm",
              col = "#C42126",
              se = T,
              size = 1) + 
  labs(title="Scatterplot af VGG16-model med MSE",y="Sande værdi", x="Prædikteret værdi")

grid.arrange(p3,p4, nrow=2)
#Scatter plot baseline sekvens MAE
p5 <-scatter3a %>% rename(predicion=sekvens_base1_resultat) %>% 
  ggplot(aes(x = predicion, y = label)) +
  geom_point() +
  stat_smooth(method = "lm",
              col = "#C42126",
              se = T,
              size = 1) + 
  labs(title="Scatterplot af sekvens baseline-model med MAE",y="Sande værdi", x="Prædikteret værdi")
#Scatter plot baseline sekvens MSE
p6 <-scatter3b %>% rename(predicion=sekvens_base2_resultat) %>% 
  ggplot(aes(x = predicion, y = label)) +
  geom_point() +
  stat_smooth(method = "lm",
              col = "#C42126",
              se = T,
              size = 1) + 
  labs(title="Scatterplot af sekvens baseline-model med MSE",y="Sande værdi", x="Prædikteret værdi")

grid.arrange(p5,p6, nrow=2)
#Scatter plot VGG16 sekvens MAE
p7 <-scatter4a %>% rename(predicion=sekvens_vgg16_1_resultat) %>% 
  ggplot(aes(x = predicion, y = label)) +
  geom_point() +
  stat_smooth(method = "lm",
              col = "#C42126",
              se = T,
              size = 1) + 
  labs(title="Scatterplot af sekvens VGG16-model med MAE",y="Sande værdi", x="Prædikteret værdi")
#Scatter plot VGG16 sekvens MSE
p8 <-scatter4b %>% rename(predicion=sekvens_vgg16_2_resultat) %>% 
  ggplot(aes(x = predicion, y = label)) +
  geom_point() +
  stat_smooth(method = "lm",
              col = "#C42126",
              se = T,
              size = 1) + 
  labs(title="Scatterplot af sekvens VGG16-model med MSE",y="Sande værdi", x="Prædikteret værdi")

grid.arrange(p7,p8, nrow=2)


