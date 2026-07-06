# gmdh_combinatorial.R
# -----------------------------------------------------------------------
# Inductive combinatorial Group Method of Data Handling (GMDH), as used
# for the demand model (Fig. 2) and cost model (Fig. 3) in:
#   Osypenko et al., "Hybrid GMDH-Based Energy Pricing for Territorial
#   Communities with Renewable and Biofuel Sources".
#
# Implements the partial-quadratic combinatorial GMDH of eq. (23):
#   yhat = a0 + a1*xi + a2*xj + a3*xi^2 + a4*xj^2 + a5*xi*xj
# selected layer-by-layer using the external regularity criterion
# CR(f,B) of eq. (24), computed on a held-out validation subset B.
# Layers stop once the best validation error increases (overfitting).
#
# Environment: R 4.3.2, packages: Metrics, caret, ggplot2
# Seed = 42
# -----------------------------------------------------------------------

suppressPackageStartupMessages({
  library(Metrics)   # rmse(), mae()
  library(caret)     # createDataPartition() for stratified train/val split
})

set.seed(42)

#' Fit one partial quadratic model on (xi, xj) -> y via OLS
#' @return list(coef = named vector a0..a5, pred = function(xi,xj))
fit_partial_model <- function(xi, xj, y) {
  X <- cbind(1, xi, xj, xi^2, xj^2, xi * xj)
  colnames(X) <- c("a0", "a1", "a2", "a3", "a4", "a5")
  coefs <- tryCatch(
    qr.solve(X, y),
    error = function(e) rep(NA_real_, 6)
  )
  list(
    coef = coefs,
    predict = function(xi_new, xj_new) {
      Xn <- cbind(1, xi_new, xj_new, xi_new^2, xj_new^2, xi_new * xj_new)
      as.numeric(Xn %*% coefs)
    }
  )
}

#' External regularity criterion, eq. (24): normalized RMSE on validation set B
regularity_criterion <- function(y_true_B, y_pred_B) {
  if (any(!is.finite(y_pred_B))) return(Inf)
  sqrt(mean((y_true_B - y_pred_B)^2)) / (sd(y_true_B) + 1e-12)
}

#' Run one GMDH layer over the current candidate feature matrix Z
#' Returns the top `keep` models (by CR) as a list, plus their fitted
#' outputs on train (A) and validation (B), to be used as inputs to the
#' next layer.
run_layer <- function(Z_A, Z_B, y_A, y_B, keep = 6) {
  p <- ncol(Z_A)
  combos <- combn(p, 2, simplify = FALSE)

  results <- lapply(combos, function(idx) {
    i <- idx[1]; j <- idx[2]
    m <- fit_partial_model(Z_A[, i], Z_A[, j], y_A)
    pred_B <- m$predict(Z_B[, i], Z_B[, j])
    cr <- regularity_criterion(y_B, pred_B)
    list(model = m, i = i, j = j, cr = cr)
  })

  crs <- vapply(results, function(r) r$cr, numeric(1))
  ord <- order(crs)[seq_len(min(keep, length(crs)))]
  results[ord]
}

#' Full combinatorial GMDH training loop
#'
#' @param X data.frame/matrix of input features (columns = candidate inputs)
#' @param y numeric target vector
#' @param max_layers maximum number of layers to grow before forced stop
#' @param val_fraction fraction of rows used as the external validation set B
#' @return list(best_model_chain, best_cr_per_layer, predict = function(Xnew))
gmdh_fit <- function(X, y, max_layers = 8, val_fraction = 0.3, keep = 6) {
  X <- as.matrix(X)
  n <- nrow(X)

  val_idx <- createDataPartition(y, p = val_fraction, list = FALSE)
  A_idx <- setdiff(seq_len(n), val_idx)
  B_idx <- val_idx

  Z_A <- X[A_idx, , drop = FALSE]
  Z_B <- X[B_idx, , drop = FALSE]
  y_A <- y[A_idx]
  y_B <- y[B_idx]

  layer_chain <- list()
  best_cr_per_layer <- numeric(0)
  prev_best_cr <- Inf

  for (layer in seq_len(max_layers)) {
    layer_models <- run_layer(Z_A, Z_B, y_A, y_B, keep = keep)
    best_cr <- layer_models[[1]]$cr

    cat(sprintf("Layer %d: best CR(f,B) = %.5f (from %d candidates)\n",
                layer, best_cr, length(layer_models)))

    if (best_cr >= prev_best_cr) {
      cat(sprintf("Validation error increased -> stopping at layer %d (selecting layer %d model)\n",
                   layer, layer - 1))
      break
    }

    layer_chain[[layer]] <- layer_models
    best_cr_per_layer <- c(best_cr_per_layer, best_cr)
    prev_best_cr <- best_cr

    # outputs of retained models become inputs to the next layer
    Z_A <- sapply(layer_models, function(r) r$model$predict(Z_A[, r$i], Z_A[, r$j]))
    Z_B <- sapply(layer_models, function(r) r$model$predict(Z_B[, r$i], Z_B[, r$j]))
  }

  final_layer <- length(layer_chain)
  cat(sprintf("Selected model: layer %d, CR(f,B) = %.5f\n",
              final_layer, best_cr_per_layer[final_layer]))

  list(
    layer_chain = layer_chain,
    best_cr_per_layer = best_cr_per_layer,
    n_layers = final_layer
  )
}

# -------------------------------------------------------------------
# Example run on the reproducibility dataset
# -------------------------------------------------------------------
if (sys.nframe() == 0) {
  df <- read.csv("data/energy_balance_2019_2023.csv")

  inputs <- df[, c("E_SPP", "E_WPP", "E_HPP", "E_BCP", "E_CPSS")]

  cat("=== GMDH Demand Model ===\n")
  fit_D <- gmdh_fit(inputs, df$D, max_layers = 8, val_fraction = 0.3)

  cat("\n=== GMDH Cost Model ===\n")
  fit_C <- gmdh_fit(inputs, df$C, max_layers = 8, val_fraction = 0.3)

  saveRDS(fit_D, "results/gmdh_demand_model.rds")
  saveRDS(fit_C, "results/gmdh_cost_model.rds")
}
