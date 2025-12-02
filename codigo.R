library(readr)
library(dplyr)
library(rpart)
library(rpart.plot)
library(car)

titanic <- read_csv("titanic.csv")

datos <-mutate(titanic, 
               Survived = factor(Survived),
               Pclass = factor(Pclass, levels = c(1, 2, 3), ordered = TRUE), #Ticket class
               Sex = factor(Sex, levels=c("male", "female")), #Sexo
               Embarked = factor(Embarked))

formula <- formula(Survived ~ Pclass + Sex + Age + SibSp + Parch + Fare + Embarked)

arbol.titanic <- rpart(formula, data = datos)
arbol.titanic

prp(arbol.titanic, extra=101, type=2,  xsep="/")

arbol.titanic2 <- rpart(formula, data = datos, control = rpart.control(minbucket = 3))
arbol.titanic2
prp(arbol.titanic2, extra=101, type=2,  xsep="/")

arbol.titanic3 <- rpart(formula, data = datos, control = rpart.control(minbucket = 3))
rpart.plot(arbol.titanic3)

printcp(arbol.titanic3)
predict(arbol.titanic3)
predict(arbol.titanic2)

arbol.titanic4 <- rpart(formula, 
                      data = datos, 
                      control = rpart.control(minbucket = 3),
                      parms=list(loss=matrix(c(0,2,1,0), #TP, FN, FP, TN: siempre el TP y TN son 0
                                             byrow=TRUE,
                                             nrow=2)))
rpart.plot(arbol.titanic4)

printcp(arbol.titanic4)

scatterplotMatrix(~ Pclass + Sex + Age + SibSp + Parch + Fare + Embarked, data = datos,
                  diagonal = "density",
                  regLine = list(col = "green", # Linear regression line color
                                 lwd = 3),      # Linear regression line width
                  smooth = list(col.smooth = "red",   # Non-parametric mean color
                                col.spread = "black")) # Non-parametric variance color
