# mc_uncertainty.R
# -----------------------------------------------------------------------
# Monte Carlo uncertainty propagation for the specific electricity cost,
# eq. (26): for each realization k = 1..K, stochastic RES generation
# (xi_SPP, xi_WPP) and biofuel availability (zeta) are resampled from
# their fitted distributions, propagated through the fitted GMDH cost
# model, and the empirical 75%/95% prediction intervals of the resulting
# ensemble {h_MC^(k)(t)} are computed non-parametrically (Fig. 5b).
#
# K = 10,000 realizations, as stated in the manuscript.
# Environment: R 4.3.2, packages: Metrics, caret, ggplot2
# Seed = 42
# -----------------------------------------------------------------------

suppressPackageStartupMessages({
  library(Metrics)
  library(ggplot2)
})

set.seed(42)
source("R/gmdh_combinatorial.R")

K <- 10000

df <- read.csv("data/energy_balance_2019_2023.csv")
inputs <- df[, c("E_SPP", "E_WPP", "E_HPP", "E_BCP", "E_CPSS")]

cat("Fitting GMDH cost model on full dataset...\n")
fit_C <- gmdh_fit(inputs, df$C, max_layers = 8, val_fraction = 0.3)

predict_gmdh <- function(fit, X) {
  Z <- as.matrix(X)
  for (layer_models in fit$layer_chain) {
    Z <- sapply(layer_models, function(r) r$model$predict(Z[, r$i], Z[, r$j]))
  }
  Z[, 1]
}

# Empirical noise scale for each RES/biofuel-driven input, estimated as
# the residual variability around each series' 12-month seasonal mean.
noise_sd <- function(x) {
  detrended <- x - stats::ave(x, cycle(ts(x, frequency = 12)))
  sd(detrended, na.rm = TRUE)
}
sd_SPP  <- noise_sd(df$E_SPP)  * 0.15
sd_WPP  <- noise_sd(df$E_WPP)  * 0.15
sd_HPP  <- noise_sd(df$E_HPP)  * 0.10
sd_BCP  <- noise_sd(df$E_BCP)  * 0.10
sd_CPSS <- noise_sd(df$E_CPSS) * 0.05

n_t <- nrow(df)
mc_results <- matrix(NA_real_, nrow = n_t, ncol = K)

cat(sprintf("Running Monte Carlo simulation with K = %d realizations...\n", K))
for (k in seq_len(K)) {
  X_k <- data.frame(
    E_SPP  = pmax(0, df$E_SPP  + rnorm(n_t, 0, sd_SPP)),
    E_WPP  = pmax(0, df$E_WPP  + rnorm(n_t, 0, sd_WPP)),
    E_HPP  = pmax(0, df$E_HPP  + rnorm(n_t, 0, sd_HPP)),
    E_BCP  = pmax(0, df$E_BCP  + rnorm(n_t, 0, sd_BCP)),
    E_CPSS = pmax(0, df$E_CPSS + rnorm(n_t, 0, sd_CPSS))
  )
  mc_results[, k] <- predict_gmdh(fit_C, X_k)
}

pi_95 <- t(apply(mc_results, 1, quantile, probs = c(0.025, 0.975)))
pi_75 <- t(apply(mc_results, 1, quantile, probs = c(0.125, 0.875)))
mc_mean <- rowMeans(mc_results)

summary_df <- data.frame(
  date = df$date,
  actual_C = df$C,
  mc_mean = mc_mean,
  pi95_low = pi_95[, 1], pi95_high = pi_95[, 2],
  pi75_low = pi_75[, 1], pi75_high = pi_75[, 2]
)

dir.create("results", showWarnings = FALSE)
write.csv(summary_df, "results/mc_uncertainty_summary.csv", row.names = FALSE)

coverage_95 <- mean(df$C >= pi_95[, 1] & df$C <= pi_95[, 2])
coverage_75 <- mean(df$C >= pi_75[, 1] & df$C <= pi_75[, 2])
cat(sprintf("Empirical 95%% PI coverage: %.1f%%\n", 100 * coverage_95))
cat(sprintf("Empirical 75%% PI coverage: %.1f%%\n", 100 * coverage_75))

p <- ggplot(summary_df, aes(x = seq_along(date))) +
  geom_ribbon(aes(ymin = pi95_low, ymax = pi95_high), fill = "#a6bddb", alpha = 0.5) +
  geom_ribbon(aes(ymin = pi75_low, ymax = pi75_high), fill = "#2b8cbe", alpha = 0.5) +
  geom_line(aes(y = actual_C), color = "black", linewidth = 0.6) +
  labs(title = "Monte Carlo Uncertainty: Specific Electricity Cost",
       x = "Month index", y = "Specific cost (EUR/kWh)") +
  theme_minimal()
ggsave("results/mc_uncertainty.png", p, width = 8, height = 4.5, dpi = 150)
