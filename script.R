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


new_Cabin <- ifelse(
  char_cabin == "",
  "",
  substr(char_cabin, 1, 1)
)    

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

# Configuramos las opciones de arbol.

options(repr.plot.width = 6, repr.plot.height = 5)

gender_tree <- rpart(Survived ~  Sex,              
                     data = data)       
# Dibujo del Arbol

prp(gender_tree,      
    extra=102, type=2,  xsep="/")

# Tenemos un total de 891 pasajeros, de los cuales 549 murieron. 468 de los pasajerons que murieron 
# eran del género masculino y 81 eran del género femenino.
# Del total de pasajeros, el 65% eran hombres y 35% mujeres.


complex_tree <- rpart(Survived ~ Sex + Pclass + Fare  + Age   ,
                      cp = 0.012,                 # Set complexity parameter*
                      data = data)       # Use the titanic training data

options(repr.plot.width = 4, repr.plot.height = 4)

prp(complex_tree, 
    extra=102, type=2,  xsep="/")

#printcp(complex_tree)
#plotcp(complex_tree, upper = "splits")  

#Esto te muestra:
#El error de cross-validation vs. tamaño del árbol.
#El punto donde el árbol deja de mejorar.

# Predicciones.

train_preds <- predict(complex_tree, 
                       newdata = data, 
                       type = "class")               # Return class predictions

confusionMatrix(factor(train_preds), factor(data$Survived))
