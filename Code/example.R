# Load embedding functions
source("joint_embedding.R")

# Load useful libraries
library(docstring)
library(irlba)
library(lattice)
library(Matrix)

# See documentation of joint embedding functions

# docstring(multidembed)
# docstring(multidembed_random_parallel)
# docstring(multidembed_test)


# Simulate data ###################
# Number of vertices
n = 100
h1 <- rep(1, n) / sqrt(n)
h2 <- kronecker(c(1, -1), rep(1, n/2)) /sqrt(n)
h3 <- kronecker(c(1, -1, 1, -1), rep(1, n/4)) / sqrt(n)

# number of graphs
m <- 2^10
lambda1 <- runif(m, 8,16)
lambda2 <- runif(m,  0,2)
lambda3 <- runif(m, 0,1)

H1 <- tcrossprod(h1)
H2 <- tcrossprod(h2)
H3 <- tcrossprod(h3)

l1 = lambda1[1]
l2 = lambda2[1]
l3 = lambda3[1]
# Function to generate a graph from the MREG model
generate_graph <- function(l1,l2,l3) {
  P <- l1*H1 + l2*H2 + l3*H3
  P <- ifelse(P>1, 1, ifelse(P<0, 0, P))
  P.updi <- P[upper.tri(P)]
  A.updi <- 1*(runif(length(P.updi))<P.updi)
  A <- matrix(0, n, n)
  A[upper.tri(A)] <- A.updi
  A <- A + t(A)
  A
}

A.list <- mapply(generate_graph, lambda1, lambda2, lambda3,   SIMPLIFY = FALSE)

#### joint embedding -------------------------------------------------
je.A <- multidembed(A = A.list, d = 3, Innitialize = 1, maxiter = 100, large.and.sparse = F)

# Estimated H = h h'
png("../results/Hhat1.png")
print(levelplot(tcrossprod(je.A$h[,1])))
dev.off()
png("../results/Hhat2.png")
levelplot(tcrossprod(je.A$h[,2]))
dev.off()
png("../results/Hhat3.png")
levelplot(tcrossprod(je.A$h[,3]))
dev.off()

# generate test data
m <- 2^10
lambda1.t <- runif(n = m, 8,16)
lambda2.t <- runif(n = m,  0,2)
lambda3.t <- runif(n = m, 0,1)
A.list.test <- mapply(generate_graph, lambda1.t, lambda2.t, lambda3.t,   SIMPLIFY = FALSE)

# out-of-sample embedding
L.test <- multidembed_test(A.list.test, je.A$h)

# each row contains the coordinate of each graph, printing only first 10 graphs
print(L.test[1:10, ])