library(caret)
library(ggplot2)
library(dplyr)
library(data.table)
library(xgboost)
library(Matrix)
library(visNetwork)

options(scipen = 999)

train <- read.csv('train.csv')
test <- read.csv('test.csv')

train_df <- as.data.frame(train)
test_df <- as.data.frame(test)

head(train_df[,1:25],1)
head(train_df[,c(5,6,10,18,23,24)])

x_train <- train[,c(5:25)]
y_train <- train[,26]
head(x_train)
x_test <- test[,c(3,5,6,10,18,23,24,25)]
y_test <- test[,26]


col_means_train <- attr(x_train, "scaled:center") 
col_stddevs_train <- attr(x_train, "scaled:scale")
x_test <- scale(x_train, center = col_means_train, scale = col_stddevs_train)

x_train_p <- x_train[1:80000,]
y_train_p <- y_train[1:80000]
x_train_p <- (as.matrix.data.frame(x_train_p))
x_train_p <- scale(x_train_p)
head(x_train_p)


x_train_p1 <- pad_sequences(x_train_p)
head(x_train_p1)


build_model <- function() {
  model <- keras_model_sequential() %>%
    layer_embedding(input_dim = 256, output_dim = 256,input_length = 21) %>%
    bidirectional(layer_cudnn_lstm(units = 500, return_sequences = TRUE)) %>%
    layer_dropout(0.25) %>%
    bidirectional(layer_cudnn_lstm(units = 250, return_sequences = TRUE)) %>%
    layer_dropout(0.25) %>%
    layer_conv_1d(filters = 128, kernel_size = 4, activation = 'relu') %>%
    layer_max_pooling_1d(pool_size=2) %>%
    layer_spatial_dropout_1d(0.5) %>%
    layer_flatten() %>%
    layer_dense(units=128, activation = 'relu') %>%
    layer_dropout(0.2) %>%
    layer_dense(units = 1)
  
  model %>% compile(
    loss = "mse",
    optimizer = optimizer_rmsprop(),
    metrics = list("mean_absolute_error")
  )
  
  model
}

model <- build_model()
model %>% summary()

epochs <- 100
# The patience parameter is the amount of epochs to check for improvement.
early_stop <- callback_early_stopping(monitor = "val_mean_absolute_error", patience = 10)

model <- build_model()
history <- model %>% fit(
  x_train_p,
  y_train_p,
  epochs = epochs,
  validation_split = 0.2,
  callbacks = list(early_stop)
)

