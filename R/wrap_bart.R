# Getting the BART wrapped function
rbart <- function(x_train,
                 y,
                 x_test,
                 n_tree = 200,
                 n_mcmc = 2000,
                 n_burn = 500,
                 alpha = 0.95,
                 beta = 2,
                 df = 3,
                 sigquant = 0.9,
                 kappa = 2) {

     # Verifying if x_train and x_test are matrices
     if(!is.matrix(x_train) || !is.matrix(x_test)){
          x_train <- as.matrix(x_train)
          x_test <- as.matrix(x_test)
     }

     if(is.null(colnames(x_train)) || is.null(colnames(x_test))) {
          stop("Insert valid NAMED matrix for x_train or x_test")
     }

     # Scaling x
     x_min <- apply(x_train,2,min)
     x_max <- apply(x_train,2,max)

     # Storing the original
     x_train_original <- x_train
     x_test_original <- x_test

     # Creating the scaled vesion
     x_train_scale <- x_train
     x_test_scale <- x_test

     # Normalising all the columns
     for(i in 1:ncol(x_train)){
          x_train_scale[,i] <- normalize_covariates_bart(y = x_train[,i],a = x_min[i], b = x_max[i])
          x_test_scale[,i] <- normalize_covariates_bart(y = x_train[,i],a = x_min[i], b = x_max[i])
     }

     # Scaling the y
     min_y <- min(y)
     max_y <- max(y)
     y_scale <- normalize_bart(y = y,a = min_y,b = max_y)

     # Calculating \tau_{mu}
     tau_mu <- (4*n_tree*(kappa^2))


     # Getting the naive sigma value
     nsigma <- naive_sigma(x = x_train_scale,y = y_scale)

     # Calculating tau hyperparam
     a_tau <- df/2
     # Calculating lambda
     qchi <- stats::qchisq(p = 1-sigquant,df = df,lower.tail = 1,ncp = 0)
     lambda <- (nsigma*nsigma*qchi)/df
     d_tau <- (lambda*df)/2


     # Call the bart function
     tau_init <- nsigma^(-2)
     mu_init <- mean(y_scale)


     # Generating the BART obj
     bart_obj <- bart(x_train_scale,
          y_scale,
          x_test_scale,
          n_tree,
          n_mcmc,
          n_burn,
          tau_init,
          mu_init,
          tau_mu,
          alpha,
          beta,
          a_tau,d_tau)


     # Tidying up the posterior elements
     y_train_post <- unnormalize_bart(z = bart_obj[[1]],a = min_y,b = max_y)
     y_test_post <- unnormalize_bart(z = bart_obj[[2]],a = min_y,b = max_y)
     tau_post <- bart_obj[[3]]/((max_y-min_y)^2)

     # Return the list with all objects and parameters
     return(list(y_hat = y_train_post,
                 y_hat_test = y_test_post,
                 tau_post = tau_post,
                 prior = list(n_tree = n_tree,
                              alpha = alpha,
                              beta = beta,
                              tau_mu = tau_mu,
                              a_tau = a_tau,
                              d_tau = d_tau),
                 mcmc = list(n_mcmc = n_mcmc,
                             n_burn = n_burn),
                 data = list(x_train = x_train,
                             y = y,
                             x_test = x_test)))
}

colnames(x) <- "x"
bartmodel <- rbart(x_train = x,y = y,x_test = x,n_tree = 200)

plot(bartmodel$tau_post, type = "l")

# Comparing with dbarts
dbartmodel <- dbarts::bart(x.train = x,y.train = y,x.test = x,nskip = 500,ndpost = 1500)
plot(x,y)
points(x,rowMeans(bartmodel$y_hat),pch=20)
points(x,colMeans(dbartmodel$yhat.train), col = "red", pch=20)
plot(bartmodel$tau_post, type = "l")
lines(dbartmodel$sigma^-2,type = "l", col = "blue")
#
