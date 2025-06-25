# from StatQuest

library(ggplot2)

# create fake data
data.matrix <- matrix(nrow=100, ncol = 10)
colnames(data.matrix) <- c(
  paste("wt", 1:5, sep=" "),
  paste("ko", 1:5, sep=" ")
)

# set the row names of the data
rownames(data.matrix) <- paste("gene", 1:100, sep=" ")

for(i in 1:100) {
  wt.values <- rpois(5, lambda = sample(x=10:1000, size=1))
  ko.values <- rpois(5, lambda = sample(x=10:1000, size=1))
  
  data.matrix[i,] <- c(wt.values, ko.values)
}

# calculate the principle components
pca <- prcomp(t(data.matrix), scale. = TRUE)
pca.var <- pca$sdev^2
pca.var.per <- round(pca.var / sum(pca.var) * 100, 2)
pca.data <- data.frame(
  Sample=rownames(pca$x),
  X=pca$x[,1],
  Y=pca$x[,2]
)

ggplot(data=pca.data, aes(x=X, y=Y, label=Sample)) +
  geom_text() +
  xlab(paste("PC1 - ", pca.var.per[1], "%", sep = "")) +
  ylab(paste("PC2 - ", pca.var.per[2], "%", sep = "")) +
  theme_bw() +
  ggtitle("PCA Graph")
