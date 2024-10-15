library(activegp)
library(hetGP)

args = commandArgs(trailingOnly=TRUE)
if (args[1] == "Cvar") {
    C_func = C_var
} else if (args[1] == "Cvar2") {
    C_func = C_var2
} else if (args[1] == "Ctr") {
    C_func = C_tr
} else {
    stop("Unknown method")
}

n_cands <- 100

X <- as.matrix(read.table('X.txt'))
Y <- scan('Y.txt')
lower <- scan('lower.txt')
upper <- scan('upper.txt')

# Normalize
X_norm <- t((t(X) - lower) / (upper - lower))

nvar <- ncol(X)

model <- mleHomGP(
    X_norm,
    Y,
    covtype = 'Gaussian',
    known = list(g = 1e-6)
)
C_hat <- C_GP(model = model)
af <- function(x, C) C_var(C, x, grad = FALSE)
af_gr <- function(x, C) C_var(C, x, grad = TRUE)

candidates <- matrix(runif(n_cands * nvar), ncol = nvar)
cvals <- apply(candidates, 1, function(x) af(x, C_hat))
opt_cand <- matrix(candidates[which.max(cvals),], 1)

# Refine with gradient based optimization
opt <-  optim(
    opt_cand,
    af,
    af_gr,
    method = 'L-BFGS-B',
    lower = rep(0, length(lower)),
    C = C_hat,
    upper = rep(1, length(upper)),
    hessian = TRUE,
    control = list(fnscale=-1, trace = 0)
)

x_opt <- lower + opt$par * (upper - lower)

write.table(x_opt, "x_opt.txt", quote=FALSE, row.names=FALSE, col.names=FALSE)
