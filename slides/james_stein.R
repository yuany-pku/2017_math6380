# The following codes are from website:
# https://cran.r-project.org/web/packages/simulator/vignettes/james-stein.html

library(simulator)

make_normal_model <- function(theta_norm, p) {
  new_model(name = "norm",
            label = sprintf("p = %s, theta_norm = %s", p, theta_norm),
            params = list(theta_norm = theta_norm, p = p,
                          theta = c(theta_norm, rep(0, p - 1))),
            simulate = function(theta, p, nsim) {
              Y <- theta + matrix(rnorm(nsim * p), p, nsim)
              return(split(Y, col(Y))) # make each col its own list element
            })
}


# we create two functions, one for each method:

mle <- new_method(name = "mle", label = "MLE",
                  method = function(model, draw) return(list(est = draw)))

js <- new_method(name = "jse", label = "James-Stein",
                 method = function(model, draw) {
                   l2 <- sum(draw^2)
                   return(list(est = (1 - (model$p - 2) / l2) * draw))
                 })

# we create a function for the squared-error metric

sqr_err <- new_metric(name = "sqrerr", label = "Squared Error Loss",
                      metric = function(model, out) {
                        mean((out$est - model$theta)^2)
                      })

# Let’s start by simulating from two models, one in which p=2 and one in which p=6.

sim1 <- new_simulation(name = "js-v-mle",
                       label = "Investigating the James-Stein Estimator") %>%
  generate_model(make_normal_model, theta_norm = 1, p = list(2, 6),
                 vary_along = "p", seed = 123) %>%
  simulate_from_model(nsim = 20) %>%
  run_method(list(js, mle)) %>%
  evaluate(sqr_err)

# As expected, the James-Stein estimator does better than the MLE when p=6, whereas for p=2 they perform the same (as should be the case since they are in fact identical!). We see that the individual plots’ titles come from each model’s label. Likewise, each boxplot is labeled with the corresponding method’s label. And the y-axis is labeled with the label of the metric used. In the simulator, each label is part of the corresponding simulation component and used when needed. 

plot_eval(sim1, metric_name = "sqrerr")

# if instead of a plot we wished to view this as a table, we could do the following:

tabulate_eval(sim1, metric_name = "sqrerr", output_type = "markdown")

# If this document were in latex, we would instead use output_type="latex". Since reporting so many digits is not very meaningful, we may wish to adjust the number of digits shown:

tabulate_eval(sim1, metric_name = "sqrerr", output_type = "markdown",
              format_args = list(nsmall = 1, digits = 0))

# Rather than looking at just two models, we might wish to generate a sequence of models, indexed by p.

sim2 <- new_simulation(name = "js-v-mle2",
                       label = "Investigating James-Stein Estimator") %>%
  generate_model(make_normal_model, vary_along = "p",
                 theta_norm = 1, p = as.list(seq(1, 30, by = 5))) %>%
  simulate_from_model(nsim = 20) %>%
  run_method(list(js, mle)) %>%
  evaluate(sqr_err)

plot_eval(sim2, metric_name = "sqrerr")

tabulate_eval(sim2, metric_name = "sqrerr", output_type = "markdown",
              format_args = list(nsmall = 2, digits = 1))

# We can also use base plot functions rather than ggplot2:
plot_eval_by(sim2, metric_name = "sqrerr", varying = "p", use_ggplot2 = FALSE)

# in cases where one wishes to work directly with the generated results, one may output the evaluated metrics as a data.frame:
df <- as.data.frame(evals(sim2))
head(df)

# One can also extract more specific slices of the evaluated metrics. For example:
evals(sim2, p == 6, methods = "jse") %>% as.data.frame %>% head


# we wish to vary both the dimension p and the norm of the mean vector theta_norm. The following generates a simulation with 30 models, corresponding to all pairs between 3 values of p and 10 values of theta_norm. For each of these 30 models, we generate 20 simulations on which we run two methods and then evaluate one metric:

sim3 <- new_simulation(name = "js-v-mle3",
                       label = "Investigating the James-Stein Estimator") %>%
  generate_model(make_normal_model, vary_along = c("p", "theta_norm"),
                 theta_norm = as.list(round(seq(0, 5, length = 10), 2)),
                 p = as.list(seq(1, 30, by = 10))) %>%
  simulate_from_model(nsim = 20) %>%
  run_method(list(js, mle)) %>%
  evaluate(sqr_err)

# we can make plots that vary ‖θ‖2 for fixed values of p. To do so, we use the subset_simulation function, that allows us to select (out of all 30 models) those ones meeting a certain criterion such as p having a certain value. 

subset_simulation(sim3, p == 11) %>% 
  plot_eval_by(metric_name = "sqrerr", varying = "theta_norm", main = "p = 11")

