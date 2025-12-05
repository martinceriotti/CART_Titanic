library(readr)
library(dplyr)
library(rpart)
library(rpart.plot)
library(caret)


# Lectura de los datos
data  <- read_csv("datos/titanic.csv")

# Analisis del dataset.

names(data)
str(data)

# Quitamos PassengerId porque no lo usamos para árbol.

data$PassengerId <- NULL

table(data$Survived, data$Sex)
table(data$Sex)

# Para que se entienda mejor, cambiamos 1 y 0 por si / no.
# Convertimos a factor las variables que nos interesen para el análisis y se puedan usar como tales.

new_Survived <- factor(data$Survived)
levels(new_Survived) <- c("Murio", "Sobrevivio")
table(new_Survived)

data$Pclass <- ordered(data$Pclass, levels = c("3", "2", "1"))

table(data$Cabin)

# El dato de cabina es muy variado. Vamos a tomar una letra como para tener grupos de cabina.

na_logical <- is.na(data$Cabin)
data$Cabin <- ifelse(na_logical, 'X', data$Cabin)

char_cabin <- as.character(data$Cabin) # Convert to character


new_Cabin <- ifelse(char_cabin == "", "", substr(char_cabin, 1, 1))

new_Cabin <- factor(new_Cabin)                # Convert back to a factor

table(new_Cabin)                             # Inspect the result as a table
data$Cabin <- new_Cabin

data$Survived <-  new_Survived

# Tratamiento de los nulos y outliers.

summary(data$Age)
hist(data$Age, breaks = 20)

# Como tenemos muchos nulos, podemos usar el valor mediano. (Median: 28 )

na_logical <- is.na(data$Age)

new_age <- ifelse(na_logical, 28, data$Age)
data$Age <- new_age

#Tarifa

boxplot(data$Fare)

# Buscamos el valor mas alto.

high_roller_index <- which.max(data$Fare)

high_roller_index

data[high_roller_index, ]


# ARBOLES

# Dividimos 70% entrenamiento / 30% prueba.

set.seed(42) # Fija la semilla para que la división sea reproducible
train_index <- createDataPartition(data$Survived, p = 0.8, list = FALSE)

# Creamos los subconjuntos.
train_data <- data[train_index, ]
test_data  <- data[-train_index, ]



# Configuramos las opciones de arbol.

options(repr.plot.width = 6, repr.plot.height = 5)

gender_tree <- rpart(Survived ~  Sex, data = train_data)
# Dibujo del Arbol

prp(gender_tree,
    extra = 102,
    type = 2,
    xsep = "/")

# Tenemos un total de 625 pasajeros, de los cuales 385 murieron. De los pasajerons que murieron
# 324 eran del género masculino y 61 eran del género femenino.
# Del total de pasajeros, el 63% eran hombres y 37% mujeres.

complex_tree <- rpart(
  Survived ~ Sex + Pclass + Fare  + Age   ,
  cp = 0.015,
  # Set complexity parameter*
  data = train_data
)       # Use the titanic training data

options(repr.plot.width = 4, repr.plot.height = 4)

prp(complex_tree,
    extra = 102,
    type = 2,
    xsep = "/")

printcp(complex_tree)
plotcp(complex_tree, upper = "splits")

#Esto te muestra:
#El error de cross-validation vs. tamaño del árbol.
#El punto donde el árbol deja de mejorar.

# Predicciones.

train_preds <- predict(complex_tree, newdata = test_data, type = "class")

confusionMatrix(factor(train_preds), factor(test_data$Survived))


#CROSS VALIDATION

# Create a trainControl object to control how the train function creates the model
train_control <- trainControl(
  method = "repeatedcv",
  # Use cross validation
  number = 10,
  # Use 10 partitions
  repeats = 2
)             # Repeat 2 times

# Set required parameters for the model type we are using**
tune_grid = expand.grid(cp = c(0.015))


# Use the train() function to create the model
validated_tree <- train(
  Survived ~ Sex + Pclass + Fare  + Age   ,
  data = train_data,
  # Data set
  method = "rpart",
  # Model type(decision tree)
  trControl = train_control,
  # Model control options
  tuneGrid = tune_grid,
  # Required model parameters
  maxdepth = 5,
  # Additional parameters***
  minbucket = 5
)

validated_tree         # View a summary of the model
