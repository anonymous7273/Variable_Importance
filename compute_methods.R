alib <- import("alibi.explainers", convert = FALSE)
np <- import("numpy", convert = FALSE)
sklearn <- import("sklearn", convert = FALSE)
shap <- import("shap", convert = FALSE)
sandbox <- import_from_path("sandbox",
                            path = "../tuan_binh_nguyen/dev")

compute_janitza <- function(sim_data,
                            cv,
                            ntree = 5000L,
                            ncores = 4L,
                            with_ranger = FALSE,
                            with_impurity = FALSE,
                            with_python = FALSE,
                            replace = FALSE,
                            prob_type = "regression",
                            ...) {

    print("Applying HoldOut Method")

    mtry <- ceiling(sqrt(ncol(sim_data[, -1])))

    if (prob_type == "classification")
        sim_data$y <- as.factor(sim_data$y)

    if (!with_ranger) {
        res <- CVPVI(sim_data[, -1],
                     sim_data$y,
                     k = cv,
                     mtry = mtry,
                     ntree = ntree,
                     ncores = ncores)
        tryCatch({
            return(data.frame(
                method = "HoldOut_nr",
                importance = as.numeric(res$cv_varim),
                p_value = as.numeric(NTA(res$cv_varim)$pvalue)))},
            finally = {
                return(data.frame(
                    method = "HoldOut_nr",
                    importance = as.numeric(res$cv_varim)))})
    }

    else {
        if (!with_impurity) {
            rf_sim <- holdoutRF(y ~ .,
                                data = sim_data,
                                mtry = mtry,
                                num.trees = ntree)
            suffix <- "r"
        }
        else {
            rf_sim <- ranger(y ~ .,
                             data = sim_data,
                             importance = "impurity_corrected",
                             mtry = mtry,
                             replace = replace,
                             num.trees = ntree)
            suffix <- "ri"
        }

        res <- importance_pvalues(rf_sim, method = "janitza")
        return(data.frame(method = paste0("HoldOut_", suffix),
                          importance = as.numeric(res[, 1]),
                          p_value = as.numeric(res[, 2])))
    }
}


compute_altmann <- function(sim_data,
                            nper = 100L,
                            ntree = 500L,
                            replace = FALSE,
                            prob_type = "regression",
                            ...) {
    print("Applying Altmann Method")

    if (prob_type == "classification")
        sim_data$y <- as.factor(sim_data$y)

    rf_altmann <- ranger(y ~ .,
                         data = sim_data,
                         importance = "permutation",
                         mtry = ceiling(sqrt(ncol(sim_data[, -1]))),
                         num.trees = ntree,
                         replace = replace)

    res <- data.frame(importance_pvalues(rf_altmann,
                                         method = "altmann",
                                         num.permutations = nper,
                                         formula = y ~ .,
                                         data = sim_data))
    return(data.frame(method = "Altmann",
                      importance = res[, 1],
                      p_value = res[, 2]))
}


compute_d0crt <- function(sim_data,
                          loss = "least_square",
                          statistic = "residual",
                          ntree = 100L,
                          prob_type = "regression",
                          verbose = FALSE,
                          scaled_statistics = FALSE,
                          refit = FALSE,
                          ...) {
    print("Applying d0CRT Method")

    d0crt_results <- sandbox$dcrt_zero(
        sim_data[, -1],
        as.numeric(sim_data$y),
        loss = loss,
        screening = FALSE,
        statistic = statistic,
        ntree = ntree,
        type_prob = prob_type,
        refit = refit,
        scaled_statistics = scaled_statistics,
        verbose = TRUE)

    return(data.frame(method = ifelse(scaled_statistics,
                                      "d0CRT_scaled",
                                      "d0CRT"),
                      importance = d0crt_results[[3]],
                      p_value = d0crt_results[[2]]))
}


compute_strobl <- function(sim_data,
                           ntree = 150L,
                           mtry = 8L,
                           conditional = TRUE,
                           parallel = TRUE,
                           ncores = 3L,
                           prob_type = "regression",
                           ...) {
    print("Applying Strobl Method")

    if (prob_type == "classification") {
        measure <- "ACC"
        sim_data$y <- as.factor(sim_data$y)
    }
    else
        measure <- "RMSE"

    registerDoParallel(cores = ncores)

    f1 <- fastcforest(y ~ .,
                      data = sim_data,
                      control = cforest_unbiased(ntree = ntree,
                                                 mtry = mtry),
                      parallel = parallel)
    print(measure)
    result <- fastvarImp(f1,
                         conditional = conditional,
                         parallel = parallel,
                         measure = measure)

    stopImplicitCluster()

    return(data.frame(method = "Strobl",
                      importance = as.numeric(result)))
}


compute_shap <- function(sim_data,
                         seed = 2021L,
                         ntree = 100L,
                         prob_type = "regression",
                         ...) {
    print("Applying SHAP Method")

    if (prob_type == "classification")
        clf_rf <- sklearn$ensemble$
            RandomForestClassifier(n_estimators = ntree)

    if (prob_type == "regression")
        clf_rf <- sklearn$ensemble$
            RandomForestRegressor(n_estimators = ntree)

    data_tt <- sklearn$model_selection$train_test_split(
        sim_data[, -1],
        sim_data$y,
        test_size = 0.3,
        random_state = seed)

    x_train <- data_tt[[0]]
    x_test <- data_tt[[1]]
    y_train <- data_tt[[2]]

    clf_rf$fit(x_train, y_train)
    explainer <- shap$TreeExplainer(clf_rf)

    if (prob_type == "classification")
        shap_values <- as.matrix(explainer$shap_values(x_test)[[1]])
    if (prob_type == "regression")
        shap_values <- as.matrix(explainer$shap_values(x_test))

    return(data.frame(method = "Shap",
                      importance = colMeans(shap_values)))
}


compute_mdi <- function(sim_data,
                        ntree = 100L,
                        prob_type = "regression",
                        ...) {
    print("Applying MDI Method")

    if (prob_type == "classification")
        clf_rf <- sklearn$ensemble$
            RandomForestClassifier(n_estimators = ntree)

    if (prob_type == "regression")
        clf_rf <- sklearn$ensemble$
            RandomForestRegressor(n_estimators = ntree)
    clf_rf$fit(sim_data[, -1], sim_data$y)

    return(data.frame(
        method = "Mdi",
        importance = as.numeric(clf_rf$feature_importances_)))
}


compute_marginal <- function(sim_data,
                             prob_type = "regression",
                             ...) {
    print("Applying Marginal Method")

    marginal_imp <- numeric()
    marginal_pval <- numeric()
    sim_data[, 1] <- as.numeric(sim_data[, 1])

    if (prob_type == "classification")
        for (i in 1:ncol(sim_data[, -1])) {
            fit <- glm(formula(paste0("y ~ X", i)),
                        data = sim_data,
                        family = binomial())
            sum_fit <- summary(fit)
            marginal_imp[i] <- coef(sum_fit)[, 1][[2]]
            marginal_pval[i] <- coef(sum_fit)[, 4][[2]]
        }

    if (prob_type == "regression")
        for (i in 1:ncol(sim_data[, -1])) {
            fit <- lm(formula(paste0("y ~ X", i)),
                        data = sim_data)
            sum_fit <- summary(fit)
            marginal_imp[i] <- coef(sum_fit)[, 1][[2]]
            marginal_pval[i] <- coef(sum_fit)[, 4][[2]]
        }

    return(data.frame(method = "Marg",
                    importance = marginal_imp,
                    p_value = marginal_pval))
}


compute_bart <- function(sim_data,
                         ntree = 100L,
                         num_cores = 4,
                         ...) {
    print("Applying BART Method")
    options(java.parameters = "-Xmx2500m")
    library(bartMachine)
    set_bart_machine_num_cores(num_cores)
    bart_machine <- bartMachine(X = sim_data[, -1],
                                y = as.numeric(sim_data$y),
                                num_trees = ntree)

    imp <- investigate_var_importance(bart_machine,
                                      plot = FALSE)$avg_var_props
    imp <- imp[mixedsort(names(imp))]

    # p_val <- c()
    # for (name in colnames(sim_data[, -1])) {
    #     p_val <- c(p_val, cov_importance_test(bart_machine,
    #                                           covariates = c(name),
    #                                           plot = FALSE)$pval)
    # }
    return(data.frame(method = "Bart",
                      importance = as.numeric(imp)))
                    #   p_value = p_val))
}


compute_knockoff <- function(sim_data,
                             stat_knockoff = NULL,
                             with_bart = TRUE,
                             verbose = TRUE,
                             ...) {
    print("Applying Knockoff Method")

    res <- sandbox$model_x_knockoff(sim_data[, -1],
                                    as.numeric(sim_data$y),
                                    statistics = stat_knockoff,
                                    verbose = verbose)

    if (stat_knockoff == "l1_regu_path")
        return(data.frame(
            method = "Knockoff_path",
            importance = res[[2]][1:as.integer(length(res[[2]]) / 2)]))
    else if (stat_knockoff == "bart") {
        res_imp <- compute_bart(data.frame(y = sim_data$y, res[[1]]))
        test_score <- res_imp$importance[1:ncol(sim_data[, -1])]
         - res_imp$importance[ncol(sim_data[, -1]): (2 * ncol(sim_data[, -1]))]
        return(data.frame(
            method = "Knockoff_bart",
            importance = test_score))
    }
    return(data.frame(method = "Knockoff_lasso",
                      importance = res[[2]]))
}


compute_ale <- function(sim_data,
                        ntree = 100L,
                        prob_type = "regression",
                        ...) {
    print("Applying ALE Method")

    if (prob_type == "classification") {
        clf_rf <- sklearn$ensemble$
            RandomForestClassifier(n_estimators = ntree)
        clf_rf$fit(sim_data[, -1], sim_data$y)
        rf_ale <- alib$ALE(clf_rf$predict_proba)
    }
    if (prob_type == "regression") {
        clf_rf <- sklearn$ensemble$
            RandomForestRegressor(n_estimators = ntree)
        clf_rf$fit(sim_data[, -1], sim_data$y)
        rf_ale <- alib$ALE(clf_rf$predict)
    }
    rf_explain <- rf_ale$explain(as.matrix(sim_data[, -1]))
    imp <- c()
    for (i in 1:dim(sim_data[, -1])[[2]])
        imp <- c(imp,
                 mean(as.vector(rf_explain$ale_values[[i - 1]])))
    return(data.frame(method = "Ale",
                      importance = imp))
}


compute_dnn <- function(sim_data,
                        n = 1000L,
                        ...) {
    print("Applying DNN Method")

    ## 1.0 Hyper-parameters
    esCtrl <- list(n.hidden = c(50L, 40L, 30L, 20L),
                activate = "relu",
                l1.reg = 10**-4,
                early.stop.det = 1000L,
                n.batch = 50L,
                n.epoch = 200L,
                learning.rate.adaptive = "adam",
                plot = FALSE)
    n_ensemble <- 100L
    n_perm <- 100L

    dnn_obj <- importDnnet(x = sim_data[, -1],
                           y = as.numeric(sim_data$y))

    # PermFIT-DNN
    shuffle <- sample(n)
    dat_spl <- splitDnnet(dnn_obj, 0.8)
    permfit_dnn <- permfit(train = dat_spl$train,
                           validate = dat_spl$valid,
                           k_fold = 0,
                           pathway_list = list(),
                           n_perm = n_perm,
                           method = "ensemble_dnnet",
                           shuffle = shuffle,
                           n.ensemble = n_ensemble,
                           esCtrl = esCtrl)

    return(data.frame(method = "Dnn",
                      importance = permfit_dnn@importance$importance,
                    p_value = permfit_dnn@importance$importance_pval))
}


compute_grf <- function(sim_data,
                        type_forest = "regression",
                        ...) {
    print("Applying GRF Method")

    if (type_forest == "regression")
        forest <- regression_forest(sim_data[, -1],
                                    as.numeric(sim_data$y),
                                    tune.parameters = "all")
    if (type_forest == "quantile")
        forest <- quantile_forest(sim_data[, -1],
                                as.numeric(sim_data$y),
                               quantiles = c(0.1, 0.3, 0.5, 0.7, 0.9))
    return(data.frame(method = paste0("GRF_", type_forest),
                      importance = variable_importance(forest)[, 1]))
}


compute_bart_py <- function(sim_data,
                            ntree = 100L,
                            prob_type = "regression",
                            ...) {
    print("Applying BART Python Method")
    bartpy <- import_from_path("utils_py",
                                path = "utils")
    imp <- bartpy$compute_bart_py(sim_data[, -1],
                                  np$array(as.numeric(sim_data$y)))

    return(data.frame(method = "Bart_py",
                      importance = as.numeric(imp)))
}