DEBUG <- FALSE
N_CPU <- 5L
N_SIMULATIONS <- 100L
if (DEBUG)
  N_SIMULATIONS <- 5L

suppressMessages({
  require(data.table)
  if (!DEBUG) {
    require(snowfall)
    sfInit(parallel = TRUE, cpus = N_CPU, type = "SOCK")
    sfLibrary(snowfall)
    sfLibrary(doParallel)
    sfLibrary(grf)
    sfLibrary(moreparty)
    sfLibrary(ranger)
    sfLibrary(randomForest)
    sfLibrary(reticulate)
    sfLibrary(vita)
    sfLibrary(gtools)
    sfLibrary(deepTL)
    sfLibrary(MASS)
    sfSource("data/data_gen.R")
    sfSource("utils/compute_methods.R")
  } else {
    library(doParallel)
    library(grf)
    library(gtools)
    library(moreparty)
    library(ranger)
    library(randomForest)
    library(reticulate)
    library(vita)
    library(MASS)
    library(deepTL)
    source("data/data_gen.R")
    source("utils/compute_methods.R")
  }
})

my_apply <- lapply
if (!DEBUG)
  my_apply <- sfLapply

##### Running Methods #####

methods <- c(
  "marginal",
  "knockoff",
  "shap",
  "mdi",
  "d0crt",
  "bart",
  "dnn",
  # "ale",
  # "strobl",
  # "janitza",
  # "altmann",
  # "bartpy",
  # "grf"
)

##### Configuration #####

param_grid <- expand.grid(
  # The running methods implemented
  method = methods,
  # The task (computation of the response vector)
  prob_sim_data = c(
                    "classification",
                    "regression",
                    "regression_combine",
                    "regression_product",
                    "regression_relu"
                    ),
  # The type of problem
  prob_type = c("classification", "regression"),
  # The number of samples
  n_samples = `if`(!DEBUG, c(100L, 1000L), 10L),
  # The number of covariates
  n_features = ifelse(!DEBUG, 50L, 5L),
  # The number of relevant covariates
  n_signal = ifelse(!DEBUG, 20L, 2L),
  # The correlation coefficient used if the covariates are correlated
  corr_coeff = c(0, 0.8),
  # The d0crt method'statistic tests scaled or not
  # scaled_statistics = c(TRUE, FALSE),
  scaled_statistics = FALSE,
  refit = FALSE,
  # The holdout importance's implementation (ranger or original)
  with_ranger = FALSE,
  # The holdout importance measure to use (impurity corrected vs MDA)
  with_impurity = FALSE,
  # The holdout importance in python
  with_python = FALSE,
  # The statistic to use with the knockoff method
  # stat_knockoff = c("lasso_cv", "l1_regu_path", "bart"),
  stat_knockoff = c("l1_regu_path", "bart"),
  # Type of forest for grf package
  type_forest = c("regression", "quantile")
)

# Now we need to filter out unnecessary or non-sensical combinations.
# e.g. regression / classification and prob type cannot be combined.

matches <- mapply(grepl,  # Test if ...
                  param_grid$prob_type, # ... prob type is in
                  param_grid$prob_sim_data  # prob sim data.
)

param_grid <- param_grid[matches, ]

param_grid <- param_grid[
  ((!param_grid$scaled_statistics) & # if scaled stats
   (param_grid$stat_knockoff == "bart") & # and defaults
   (!param_grid$refit) & # and refit
   (param_grid$type_forest == "regression") & # and type_forest
   (!param_grid$method %in% c("d0crt", # but not ...
                              "knockoff",
                              "grf")))
   | ((!param_grid$scaled_statistics) & # or scaled
      (!param_grid$refit) &
      (param_grid$type_forest == "regression") &
      (param_grid$method == "knockoff"))
   | ((param_grid$stat_knockoff == "bart") &
      (param_grid$type_forest == "regression") &
      (param_grid$method == "d0crt"))
   | ((!param_grid$scaled_statistics) &
      (!param_grid$refit) &
      (param_grid$stat_knockoff == "bart") &
      (param_grid$method == "grf")),
]
param_grid$index_i <- 1:nrow(param_grid)
cat(sprintf("Number of rows: %i \n", nrow(param_grid)))

if (!DEBUG)
  sfExport("param_grid")

compute_method <- function(method,
                           index_i,
                           n_simulations = 100L, ...) {
  print("Begin")
  cat(sprintf("%s: %i \n", method, index_i))

  compute_fun <- function(seed, ...) {
    sim_data <- generate_data(seed,
                              ...)

    timing <- system.time(
      out <- switch(as.character(method),
        marginal = compute_marginal(sim_data,
                                    ...),
        ale = compute_ale(sim_data,
                          ntree = 100L,
                          ...),
        knockoff = compute_knockoff(sim_data,
                                    verbose = TRUE,
                                    ...),
        bart = compute_bart(sim_data,
                            ntree = 100L,
                            ...),
        mdi = compute_mdi(sim_data,
                          ntree = 500L,
                          ...),
        shap = compute_shap(sim_data,
                            ntree = 100L,
                            ...),
        strobl = compute_strobl(sim_data,
                                ncores = 3L,
                                conditional = TRUE,
                                ...),
        d0crt = compute_d0crt(sim_data,
                              loss = "least_square",
                              statistic = "randomforest",
                              ntree = 100L,
                              verbose = TRUE,
                              ...),
        janitza = compute_janitza(sim_data,
                                  cv = 2L,
                                  ncores = 3L,
                                  ...),
        altmann = compute_altmann(sim_data,
                                  nper = 100L,
                                  ...),
        dnn = compute_dnn(sim_data,
                          ...),
        grf = compute_grf(sim_data,
                          ...),
        bartpy = compute_bart_py(sim_data,
                                 ...))
    )
    out <- data.frame(out)
    if (!"p_value" %in% names(out)) {
      out <- data.frame(out, p_value = NA)
    }
    out$elapsed <- timing[[3]]
    out$independence <- ifelse(list(...)$rho > 0,
                               "Correlated",
                               "Independent")
    out$n_samples <- list(...)$n_samples
    out$prob_data <- list(...)$prob_sim_data

    return(out)
  }
  sim_range <- 1L:n_simulations
  # compute results
  result <- my_apply(sim_range, compute_fun, ...)
  # postprocess and package outputs
  result <- do.call(rbind, lapply(sim_range, function(ii) {
    out <- result[[ii]]
    out$iteration <- ii
    out
  }))
  res <- data.table(result)[,
                            elapsed[1],
                            by = .(
                                  n_samples,
                                  independence,
                                  method,
                                  iteration,
                                  prob_data)]
  res <- res[,
             sum(V1) / N_CPU / 60,
             by = .(
                    n_samples,
                    method,
                    independence,
                    prob_data)]
  print(res)
  print("Finish")
  print(method)
  return(result)
}


if (DEBUG) {
  set.seed(42)
  param_grid <- param_grid[sample(1:nrow(param_grid), 5), ]
}

results <-
  by(param_grid, 1:nrow(param_grid),
     function(x) {
       with(x,
            compute_method(
              method = method,
              index_i = index_i,
              n_simulations = N_SIMULATIONS,
              prob_sim_data = prob_sim_data,
              prob_type = prob_type,
              stat_knockoff = stat_knockoff,
              with_ranger = with_ranger,
              with_impurity = with_impurity,
              with_python = with_python,
              refit = refit,
              scaled_statistics = scaled_statistics,
              type_forest = type_forest,
              n_samples = n_samples,
              n_features = n_features,
              n_signal = n_signal,
              rho = corr_coeff))
       }
)

results <- rbindlist(results)

out_fname <- "simulation_results.csv"

if (DEBUG)
  out_fname <- gsub(".csv", "-debug.csv", out_fname)

fwrite(results, out_fname)

if (!DEBUG)
    sfStop()
