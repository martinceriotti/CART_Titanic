library(readr)
library(dplyr)

data  <- read_csv("datos/titanic.csv")
names(data)
str(data)

data$PassengerId <- NULL

table(data$Survived)

new_Survived <- factor(data$Survived)
levels(new_Survived) <- c("Murio", "Sobrevivio")
table(new_Survived)

data$Pclass <- ordered(data$Pclass, levels = c("3", "2", "1"))

table(data$Cabin)

char_cabin <- as.character(data$Cabin) # Convert to character

new_Cabin <- ifelse(
  char_cabin == "",
  # If the value is ""
  "",
  # Keep it
  substr(char_cabin, 1, 1)
)    # Else transform it to a substring*

new_Cabin <- factor(new_Cabin)                # Convert back to a factor

table(new_Cabin)                             # Inspect the result as a table
data$Cabin <- new_Cabin
data$Survived <-  new_Survived


#Are there NA Values, Outliers or Other Strange Values?
summary(data$Age)
hist(data$Age, breaks = 20)

na_logical <- is.na(data$Age)
new_age <- ifelse(na_logical, 28, data$Age)
data$Age <- new_age


boxplot(data$Fare)

high_roller_index <- which.max(data$Fare)

high_roller_index                   # Check the index

data[high_roller_index, ]   # Use the index to check the record


# ARBOLES

library(rpart)
library(rpart.plot)
#install.packages("caret")
library(caret)    

options(repr.plot.width = 6, repr.plot.height = 5)

gender_tree <- rpart(Survived ~ Pclass + Sex,              # Predict survival based on gender
                     data = data)        # Use the titanic training data

prp(gender_tree,      # Plot the decision tree
    space=4,          # (Formatting options chosen for notebook)
    split.cex = 1.5,
    nn.border.col=0)

class_tree <- rpart(Survived ~ Sex - Pclass,    # Predict survival based on gender
                    data = data)       # Use the titanic training data

prp(class_tree,      # Plot the decision tree
    space=4,          # (Formatting options chosen for notebook)
    split.cex = 1.2,
    nn.border.col=0)


complex_tree <- rpart(Survived ~ Sex + Pclass + Age + SibSp + Fare + Embarked,
                      cp = 0.001,                 # Set complexity parameter*
                      data = data)       # Use the titanic training data

options(repr.plot.width = 8, repr.plot.height = 8)

prp(complex_tree, 
    type = 1,
    nn.border.col=0, 
    border.col=1, 
    cex=0.4)

limited_complexity_tree <- rpart(Survived ~ Sex + Pclass + Age + SibSp +Fare+Embarked,
                                 cp = 0.001,              # Set complexity parameter
                                 maxdepth = 5,            # Set maximum tree depth
                                 minbucket = 5,           # Set min number of obs in leaf nodes
                                 method = "class",        # Return classifications instead of probs
                                 data = data)    # Use the titanic training data

options(repr.plot.width = 6, repr.plot.height = 6)

prp(limited_complexity_tree,
    space=4,          
    split.cex = 1.2,
    nn.border.col=0)



train_preds <- predict(limited_complexity_tree, 
                       newdata = data, 
                       type = "class")               # Return class predictions

confusionMatrix(factor(train_preds), factor(data$Survived))
