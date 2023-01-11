Rcpp::sourceCpp("src/btbart.cpp")

x <- matrix(seq(-pi,pi,length.out = 100))
y <- sin(x)
colnames(x) <- "x"

# Testing the GP-BART
bart_test <- bart(x_train = x,y_train = y,x_test = x,n_tree = 200,n_mcmc = 5000,
                  n_burn = 0,tau = 1,mu = 1,
                  tau_mu = 4*4*200,naive_sigma = 1,alpha = 0.95,
                  beta = 2,a_tau = 1,d_tau = 1,nsigma = 1)


# unit_test_grow(x_train = x,y_train = y)

# dim(bart_test[[2]])


plot(x,y)
points(x,apply(bart_test[[1]],1,median),pch=20)


microbenchmark::microbenchmark(bart(x_train = x,y_train = y,x_test = x,n_tree = 200,n_mcmc = 5000,
                                    n_burn = 0,tau = 1,mu = 1,
                                    tau_mu = 4*4*200,naive_sigma = 1,alpha = 0.95,
                                    beta = 2,a_tau = 1,d_tau = 1,nsigma = 1),
                               dbarts::bart(x.train = x,y.train = y,x.test = ,ntree = 200,ndpost = 5000,nskip = 0),times = 10)
