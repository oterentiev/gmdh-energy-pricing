# cv_pipeline.R
# -----------------------------------------------------------------------
# Stratified 5-fold cross-validation over the 5-year (60-month) energy
# balance dataset, used to produce the SOTA comparison in Fig. 5 / Table
# (RMSE, MAE, R^2 for GMDH vs. RF / XGBoost / NN / MILP / SP baselines).
#
# This script only cross-validates the R-side GMDH model; baseline
# metrics produced by Python/baselines.py are merged in at the end via
# results/baseline_metrics.csv (written by that script).
#
# Environment: R 4.3.2, packages: Metrics, caret, ggplot2
# Seed = 42
# -----------------------------------------------------------------------

suppressPackageStartupMessages({
  library(Metrics)
  library(caret)
  library(ggplot2)
})

set.seed(42)
source("R/gmdh_combinatorial.R")

df <- read.csv("data/energy_balance_2019_2023.csv")
inputs <- df[, c("E_SPP", "E_WPP", "E_HPP", "E_BCP", "E_CPSS")]

k <- 5
folds <- createFolds(df$C, k = k, list = TRUE, returnTrain = FALSE)

cv_metrics <- data.frame(fold = integer(), rmse = double(), mae = double(), r2 = double())

for (f in seq_len(k)) {
  test_idx <- folds[[f]]
  train_idx <- setdiff(seq_len(nrow(df)), test_idx)

  fit <- gmdh_fit(inputs[train_idx, ], df$C[train_idx],
                   max_layers = 8, val_fraction = 0.3)

  # Propagate the held-out fold through the selected model chain
  Z <- as.matrix(inputs[test_idx, ])
  for (layer_models in fit$layer_chain) {
    Z <- sapply(layer_models, function(r) r$model$predict(Z[, r$i], Z[, r$j]))
  }
  y_pred <- Z[, 1]
  y_true <- df$C[test_idx]

  cv_metrics <- rbind(cv_metrics, data.frame(
    fold = f,
    rmse = rmse(y_true, y_pred),
    mae  = mae(y_true, y_pred),
    r2   = 1 - sum((y_true - y_pred)^2) / sum((y_true - mean(y_true))^2)
  ))
}

cat("=== 5-fold CV results: GMDH cost model ===\n")
print(cv_metrics)
cat(sprintf("\nMean RMSE = %.4f +/- %.4f\n", mean(cv_metrics$rmse), sd(cv_metrics$rmse)))
cat(sprintf("Mean MAE  = %.4f +/- %.4f\n", mean(cv_metrics$mae), sd(cv_metrics$mae)))
cat(sprintf("Mean R^2  = %.4f +/- %.4f\n", mean(cv_metrics$r2), sd(cv_metrics$r2)))

dir.create("results", showWarnings = FALSE)
write.csv(cv_metrics, "results/gmdh_cv_metrics.csv", row.names = FALSE)

p <- ggplot(cv_metrics, aes(x = factor(fold), y = rmse)) +
  geom_col(fill = "#2c7fb8") +
  labs(title = "GMDH Cost Model: RMSE by CV Fold", x = "Fold", y = "RMSE") +
  theme_minimal()
ggsave("results/cv_rmse_by_fold.png", p, width = 6, height = 4, dpi = 150)
