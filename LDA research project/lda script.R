library(RWeka)
library(dplyr)
library(MASS)
library(DiscriMiner)
library(pracma)

# Loading the data
Year1 = read.arff('1year.arff')
Year2 = read.arff('2year.arff')
Year3 = read.arff('3year.arff')
Year4 = read.arff('4year.arff')
Year5 = read.arff('5year.arff')

# Removing any entries that have a NA anywhere in the data
Year1 <- Year1[complete.cases(Year1),]
Year2 <- Year2[complete.cases(Year2),]
Year3 <- Year3[complete.cases(Year3),]
Year4 <- Year4[complete.cases(Year4),]
Year5 <- Year5[complete.cases(Year5),]

# Changing class values to be 5 for Year1, 4 for Year2 etc, and 0 for all non-bankrupt. 
Year1$class <- as.numeric(ifelse(as.numeric(as.character(Year1$class)) == 1,5,0))
Year2$class <- as.numeric(ifelse(as.numeric(as.character(Year2$class)) == 1,4,0))
Year3$class <- as.numeric(ifelse(as.numeric(as.character(Year3$class)) == 1,3,0))
Year4$class <- as.numeric(ifelse(as.numeric(as.character(Year4$class)) == 1,2,0))
Year5$class <- as.numeric(as.character(Year5$class))

# Randomly Shuffling the rows
set.seed(42)
Year1 <- Year1[sample(nrow(Year1)), ]
Year2 <- Year2[sample(nrow(Year2)), ]
Year3 <- Year3[sample(nrow(Year3)), ]
Year4 <- Year4[sample(nrow(Year4)), ]
Year5 <- Year5[sample(nrow(Year5)), ]

Year1Train_index <- sample(1:nrow(Year1), 0.8 * nrow(Year1))
Year1Test_index <- setdiff(1:nrow(Year1), Year1Train_index)
Year1_train <- Year1[Year1Train_index, ]
Year1_Test <- Year1[Year1Test_index, ]

Year2Train_index <- sample(1:nrow(Year2), 0.8 * nrow(Year2))
Year2Test_index <- setdiff(1:nrow(Year2), Year2Train_index)
Year2_train <- Year2[Year2Train_index, ]
Year2_Test <- Year2[Year2Test_index, ]

Year3Train_index <- sample(1:nrow(Year3), 0.8 * nrow(Year3))
Year3Test_index <- setdiff(1:nrow(Year3), Year3Train_index)
Year3_train <- Year3[Year3Train_index, ]
Year3_Test <- Year3[Year3Test_index, ]

Year4Train_index <- sample(1:nrow(Year4), 0.8 * nrow(Year4))
Year4Test_index <- setdiff(1:nrow(Year4), Year4Train_index)
Year4_train <- Year4[Year4Train_index, ]
Year4_Test <- Year4[Year4Test_index, ]

Year5Train_index <- sample(1:nrow(Year5), 0.8 * nrow(Year5))
Year5Test_index <- setdiff(1:nrow(Year5), Year5Train_index)
Year5_train <- Year5[Year5Train_index, ]
Year5_Test <- Year5[Year5Test_index, ]

#producing final training and test data
train_data <- rbind(Year1_train, Year2_train, Year3_train, Year4_train, Year5_train)
test_data <- rbind(Year1_Test, Year2_Test, Year3_Test, Year4_Test, Year5_Test)

#lda on data
model.lda <- lda(class ~ ., data=train_data)
model.lda
plot(model.lda)
model.results = predict(model.lda, test_data)
model.results
plot(x, y, model.results)

library(caret)
t = table(model.results$class, test_data$class)
print(t)
print(confusionMatrix(t))

# histograms
par(mar=rep(2,4))
# histogram for LDA1, as [,1] is the 1st column 
hist1 = ldahist(data = model.results$x[,1], g = train_data$class, xlim=c(-10,10))
# histogram for LDA2
hist2 = ldahist(data = model.results$x[,2], g = train_data$class, xlim=c(-10,10))
# histogram for LDA3
hist3 = ldahist(data = model.results$x[,3], g = train_data$class, xlim=c(-10,10))
# histogram for LDA4
hist4 = ldahist(data = model.results$x[,4], g = train_data$class, xlim=c(-10,10))
# histogram for LDA5
hist5 = ldahist(data = model.results$x[,5], g = train_data$class, xlim=c(-10,10))

# scatterplot of two Discriminant Functions (all combinations)
library(ggplot2)
lda.data <- cbind(train_data, predict(model.lda)$x)
ggplot(lda.data, aes(LD1, LD2, color = class)) + geom_point(aes(color = as.character(class)))
ggplot(lda.data, aes(LD1, LD3, color = class)) + geom_point(aes(color = as.character(class)))
ggplot(lda.data, aes(LD1, LD4, color = class)) + geom_point(aes(color = as.character(class)))
ggplot(lda.data, aes(LD1, LD5, color = class)) + geom_point(aes(color = as.character(class)))
ggplot(lda.data, aes(LD2, LD3, color = class)) + geom_point(aes(color = as.character(class)))
ggplot(lda.data, aes(LD2, LD4, color = class)) + geom_point(aes(color = as.character(class)))
ggplot(lda.data, aes(LD2, LD5, color = class)) + geom_point(aes(color = as.character(class)))
ggplot(lda.data, aes(LD3, LD4, color = class)) + geom_point(aes(color = as.character(class)))
ggplot(lda.data, aes(LD3, LD5, color = class)) + geom_point(aes(color = as.character(class)))
ggplot(lda.data, aes(LD4, LD5, color = class)) + geom_point(aes(color = as.character(class)))

#data to reduce and its class membership
to_reduce <- train_data[, 1:64]
class_membership <- train_data[, 65]

#scatter matrices
within_scatter <- withinSS(to_reduce, class_membership)
between_scatter <- betweenSS(to_reduce, class_membership)
within_scatter_inv <- pinv(within_scatter)

#projection to 5 dimensions
decomp <- eigen(within_scatter_inv %*% between_scatter)
projection <- Re(as.matrix(decomp$vectors[, 1:5]))

#projected training and test data
train_reduced <- as.data.frame(t(t(projection) %*% t(as.matrix(to_reduce))))
train_reduced$class = class_membership
test_reduced <- as.data.frame(t(t(projection) %*% t(as.matrix(test_data[, 1:64]))))
test_reduced$class = test_data[, 65]

#lda on reduced data
model_reduced.lda <- lda(class ~ ., data=train_reduced)
model_reduced.lda
model_reduced.results = predict(model_reduced.lda, test_reduced)
model_reduced.results

# tables, confusion matrix
t_reduced = table(model_reduced.results$class, test_reduced$class)
t_reduced
print(confusionMatrix(t_reduced))

# histogram for LDA1 reduced

par(mar=rep(2,4))
hist1_reduced = ldahist(data = model_reduced.results$x[,1], g = class_membership,  xlim=c(-10,10))
# histogram for LDA2
hist2_reduced = ldahist(data = model_reduced.results$x[,2], g = class_membership, xlim=c(-10,10))
# histogram for LDA3
hist3_reduced = ldahist(data = model_reduced.results$x[,3], g = class_membership, xlim=c(-10,10))
# histogram for LDA4
hist4_reduced = ldahist(data = model_reduced.results$x[,4], g = class_membership, xlim=c(-10,10))
# histogram for LDA5
hist5_reduced = ldahist(data = model_reduced.results$x[,5], g = class_membership, xlim=c(-10,10))

# scatterplot of  two Discriminant Function 
library(ggplot2)
lda_reduced.data <- cbind(train_reduced, predict(model_reduced.lda)$x)
ggplot(lda_reduced.data , aes(LD1, LD2, color = class)) + geom_point(aes(color = as.character(class)))
ggplot(lda_reduced.data, aes(LD1, LD3, color = class)) + geom_point(aes(color = as.character(class)))
ggplot(lda_reduced.data, aes(LD1, LD4, color = class)) + geom_point(aes(color = as.character(class)))
ggplot(lda_reduced.data, aes(LD1, LD5, color = class)) + geom_point(aes(color = as.character(class)))
ggplot(lda_reduced.data, aes(LD2, LD3, color = class)) + geom_point(aes(color = as.character(class)))
ggplot(lda_reduced.data, aes(LD2, LD4, color = class)) + geom_point(aes(color = as.character(class)))
ggplot(lda_reduced.data, aes(LD2, LD5, color = class)) + geom_point(aes(color = as.character(class)))
ggplot(lda_reduced.data, aes(LD3, LD4, color = class)) + geom_point(aes(color = as.character(class)))
ggplot(lda_reduced.data, aes(LD3, LD5, color = class)) + geom_point(aes(color = as.character(class)))
ggplot(lda_reduced.data, aes(LD4, LD5, color = class)) + geom_point(aes(color = as.character(class)))

print(confusionMatrix(t_reduced))