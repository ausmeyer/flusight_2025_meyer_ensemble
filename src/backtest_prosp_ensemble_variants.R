#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(dplyr)
  library(readr)
  library(lubridate)
  library(tidyr)
})

# CLI args
args <- commandArgs(trailingOnly = TRUE)
LOOKBACK_WEEKS <- 3
HISTORY_WEEKS <- 6
SHRINK_FACTOR <- 0.3
LOC_LOOKBACK_WEEKS <- 12
LOC_HISTORY_WEEKS <- 52
LOC_MIN_WEEKS <- 20
LOC_BASE_SHRINK <- 0.2
KNN_K <- 10
ROBUST_K <- 3
META_HISTORY_WEEKS <- 26
META_MIN_ROWS <- 50
GATE_MIN_WEEKS <- 12
GATE_MAX_WEEKS <- 40
TIER_MIN_WEEKS <- 20
SEASON_START <- NA_character_
SEASON_END <- NA_character_
OUTPUT_DIR_MAIN <- "forecasts/retrospective/ensemble"
OUTPUT_DIR_SHRINK <- "forecasts/retrospective/ensemble_shrink"
OUTPUT_DIR_LOC <- "forecasts/retrospective/ensemble_loc"
OUTPUT_DIR_KNN <- "forecasts/retrospective/ensemble_knn"
OUTPUT_DIR_META <- "forecasts/retrospective/ensemble_meta"
OUTPUT_DIR_GATE <- "forecasts/retrospective/ensemble_gate"
OUTPUT_DIR_ROBUST <- "forecasts/retrospective/ensemble_robust"
OUTPUT_DIR_TIER <- "forecasts/retrospective/ensemble_tier"
INCLUDE_ARIMA <- TRUE
INCLUDE_SVM <- TRUE
INCLUDE_LGBM_BLENDED <- TRUE
INCLUDE_LGBM_BOUNDED <- TRUE
INCLUDE_LGBM_BOUNDED_WIDE_1 <- TRUE
INCLUDE_LGBM_BOUNDED_WIDE_2 <- TRUE
INCLUDE_LGBM_BOUNDED_WIDE_3 <- TRUE
INCLUDE_LGBM_BOUNDED_WIDE_4 <- TRUE
INCLUDE_LGBM_BOUNDED_WIDE_4_NE <- TRUE
INCLUDE_LGBM_BOUNDED_WIDE_5 <- TRUE

i <- 1
while (i <= length(args)) {
  key <- args[i]
  val <- if (i + 1 <= length(args)) args[i + 1] else NA_character_
  if (key == "--lookback-weeks") { LOOKBACK_WEEKS <- as.integer(val); i <- i + 2; next }
  if (key == "--history-weeks") { HISTORY_WEEKS <- as.integer(val); i <- i + 2; next }
  if (key == "--shrink") { SHRINK_FACTOR <- as.numeric(val); i <- i + 2; next }
  if (key == "--loc-lookback-weeks") { LOC_LOOKBACK_WEEKS <- as.integer(val); i <- i + 2; next }
  if (key == "--loc-history-weeks") { LOC_HISTORY_WEEKS <- as.integer(val); i <- i + 2; next }
  if (key == "--loc-min-weeks") { LOC_MIN_WEEKS <- as.integer(val); i <- i + 2; next }
  if (key == "--loc-base-shrink") { LOC_BASE_SHRINK <- as.numeric(val); i <- i + 2; next }
  if (key == "--knn-k") { KNN_K <- as.integer(val); i <- i + 2; next }
  if (key == "--robust-k") { ROBUST_K <- as.numeric(val); i <- i + 2; next }
  if (key == "--meta-history-weeks") { META_HISTORY_WEEKS <- as.integer(val); i <- i + 2; next }
  if (key == "--meta-min-rows") { META_MIN_ROWS <- as.integer(val); i <- i + 2; next }
  if (key == "--gate-min-weeks") { GATE_MIN_WEEKS <- as.integer(val); i <- i + 2; next }
  if (key == "--gate-max-weeks") { GATE_MAX_WEEKS <- as.integer(val); i <- i + 2; next }
  if (key == "--tier-min-weeks") { TIER_MIN_WEEKS <- as.integer(val); i <- i + 2; next }
  if (key == "--season-start") { SEASON_START <- val; i <- i + 2; next }
  if (key == "--season-end") { SEASON_END <- val; i <- i + 2; next }
  if (key == "--include-arima") { INCLUDE_ARIMA <- tolower(val) %in% c("1","true","t","yes","y"); i <- i + 2; next }
  if (key == "--include-svm") { INCLUDE_SVM <- tolower(val) %in% c("1","true","t","yes","y"); i <- i + 2; next }
  if (key == "--include-lgbm-blended") { INCLUDE_LGBM_BLENDED <- tolower(val) %in% c("1","true","t","yes","y"); i <- i + 2; next }
  if (key == "--include-lgbm-bounded") { INCLUDE_LGBM_BOUNDED <- tolower(val) %in% c("1","true","t","yes","y"); i <- i + 2; next }
  if (key == "--include-lgbm-bounded-wide-1") { INCLUDE_LGBM_BOUNDED_WIDE_1 <- tolower(val) %in% c("1","true","t","yes","y"); i <- i + 2; next }
  if (key == "--include-lgbm-bounded-wide-2") { INCLUDE_LGBM_BOUNDED_WIDE_2 <- tolower(val) %in% c("1","true","t","yes","y"); i <- i + 2; next }
  if (key == "--include-lgbm-bounded-wide-3") { INCLUDE_LGBM_BOUNDED_WIDE_3 <- tolower(val) %in% c("1","true","t","yes","y"); i <- i + 2; next }
  if (key == "--include-lgbm-bounded-wide-4") { INCLUDE_LGBM_BOUNDED_WIDE_4 <- tolower(val) %in% c("1","true","t","yes","y"); i <- i + 2; next }
  if (key == "--include-lgbm-bounded-wide-4-ne") { INCLUDE_LGBM_BOUNDED_WIDE_4_NE <- tolower(val) %in% c("1","true","t","yes","y"); i <- i + 2; next }
  if (key == "--include-lgbm-bounded-wide-5") { INCLUDE_LGBM_BOUNDED_WIDE_5 <- tolower(val) %in% c("1","true","t","yes","y"); i <- i + 2; next }
  i <- i + 1
}

dir.create(OUTPUT_DIR_MAIN, recursive = TRUE, showWarnings = FALSE)
dir.create(OUTPUT_DIR_SHRINK, recursive = TRUE, showWarnings = FALSE)
dir.create(OUTPUT_DIR_LOC, recursive = TRUE, showWarnings = FALSE)
dir.create(OUTPUT_DIR_KNN, recursive = TRUE, showWarnings = FALSE)
dir.create(OUTPUT_DIR_META, recursive = TRUE, showWarnings = FALSE)
dir.create(OUTPUT_DIR_GATE, recursive = TRUE, showWarnings = FALSE)
dir.create(OUTPUT_DIR_ROBUST, recursive = TRUE, showWarnings = FALSE)
dir.create(OUTPUT_DIR_TIER, recursive = TRUE, showWarnings = FALSE)

# Helper: latest stitched file
latest_stitched <- function() {
  files <- list.files("data/imputed_sets",
                      pattern = "imputed_and_stitched_hosp_\\d{4}-\\d{2}-\\d{2}\\.csv",
                      full.names = TRUE)
  if (length(files) == 0) stop("No stitched files found")
  files[order(files)][length(files)]
}

actual_path <- latest_stitched()
actual_raw <- read_csv(actual_path, show_col_types = FALSE)
actual_data <- actual_raw %>% select(location_name, date, total_hosp) %>%
  rename(state_name = location_name, actual_value = total_hosp) %>%
  mutate(date = as.Date(date)) %>%
  filter(!is.na(actual_value))

location_to_fips <- c(
  "Alabama" = "01", "Alaska" = "02", "Arizona" = "04", "Arkansas" = "05",
  "California" = "06", "Colorado" = "08", "Connecticut" = "09", "Delaware" = "10",
  "District of Columbia" = "11", "Florida" = "12", "Georgia" = "13", "Hawaii" = "15",
  "Idaho" = "16", "Illinois" = "17", "Indiana" = "18", "Iowa" = "19",
  "Kansas" = "20", "Kentucky" = "21", "Louisiana" = "22", "Maine" = "23",
  "Maryland" = "24", "Massachusetts" = "25", "Michigan" = "26", "Minnesota" = "27",
  "Mississippi" = "28", "Missouri" = "29", "Montana" = "30", "Nebraska" = "31",
  "Nevada" = "32", "New Hampshire" = "33", "New Jersey" = "34", "New Mexico" = "35",
  "New York" = "36", "North Carolina" = "37", "North Dakota" = "38", "Ohio" = "39",
  "Oklahoma" = "40", "Oregon" = "41", "Pennsylvania" = "42", "Puerto Rico" = "72",
  "Rhode Island" = "44", "South Carolina" = "45", "South Dakota" = "46", "Tennessee" = "47",
  "Texas" = "48", "Utah" = "49", "Vermont" = "50", "Virginia" = "51",
  "Washington" = "53", "West Virginia" = "54", "Wisconsin" = "55", "Wyoming" = "56",
  "US" = "US"
)
actual_data$location <- location_to_fips[actual_data$state_name]

CDC_QUANTILES <- c(0.01, 0.025, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5,
                   0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.975, 0.99)

calculate_wis_single <- function(quantile_values, quantile_levels, actual_value) {
  if (length(actual_value) == 0 || all(is.na(actual_value))) return(NA_real_)
  qv <- as.numeric(quantile_values)
  ql <- as.numeric(quantile_levels)
  aval <- as.numeric(actual_value)[1]
  alog <- log(aval + 1)
  if (!is.finite(alog)) return(NA_real_)
  keep <- is.finite(qv) & is.finite(ql)
  qv <- qv[keep]; ql <- ql[keep]
  if (length(qv) == 0 || length(ql) == 0) return(NA_real_)
  alphas <- unique(c(ql[ql <= 0.5], 1 - ql[ql > 0.5]))
  alphas <- alphas[is.finite(alphas) & alphas > 0]
  if (length(alphas) == 0) return(NA_real_)
  qlog <- log(pmax(qv, 0) + 1)
  scores <- c()
  for (a in alphas) {
    li <- suppressWarnings(which.min(abs(ql - a)))
    ui <- suppressWarnings(which.min(abs(ql - (1 - a))))
    if (length(li) == 0 || length(ui) == 0 || is.na(li) || is.na(ui)) { scores <- c(scores, NA_real_); next }
    L <- qlog[li]; U <- qlog[ui]
    if (!is.finite(L) || !is.finite(U)) { scores <- c(scores, NA_real_); next }
    width <- U - L
    pen <- if (is.finite(alog) && alog < L) (2/a) * (L - alog) else if (is.finite(alog) && alog > U) (2/a) * (alog - U) else 0
    score <- if (is.finite(width) && is.finite(pen)) width + pen else NA_real_
    scores <- c(scores, score)
  }
  if (all(is.na(scores))) NA_real_ else mean(scores, na.rm = TRUE)
}

calculate_linear_pool <- function(df_subset, weights, target_taus) {
  all_vals <- df_subset$value
  if (length(all_vals) == 0) return(rep(NA, length(target_taus)))
  min_v <- min(all_vals, na.rm = TRUE)
  max_v <- max(all_vals, na.rm = TRUE)
  if (!is.finite(min_v) || !is.finite(max_v)) return(rep(NA, length(target_taus)))
  if (min_v == max_v) return(rep(min_v, length(target_taus)))
  range_width <- max_v - min_v
  grid_min <- max(0, min_v - (range_width * 0.1))
  grid_max <- max_v + (range_width * 0.1)
  y_grid <- seq(grid_min, grid_max, length.out = 1000)
  ensemble_cdf <- rep(0, length(y_grid))
  total_weight <- 0
  models <- unique(df_subset$source_model)
  for (m in models) {
    w <- weights[m]
    if (is.na(w) || w <= 0) next
    m_data <- df_subset %>%
      filter(source_model == m, !is.na(value)) %>%
      arrange(output_type_id)
    if (nrow(m_data) < 2 || length(unique(m_data$value)) < 2) next
    model_cdf_probs <- suppressWarnings(approx(
      x = m_data$value,
      y = m_data$output_type_id,
      xout = y_grid,
      yleft = 0,
      yright = 1,
      ties = mean
    )$y)
    ensemble_cdf <- ensemble_cdf + (model_cdf_probs * w)
    total_weight <- total_weight + w
  }
  if (total_weight == 0) return(rep(NA, length(target_taus)))
  ensemble_cdf <- ensemble_cdf / total_weight
  final_values <- suppressWarnings(approx(
    x = ensemble_cdf,
    y = y_grid,
    xout = target_taus,
    rule = 2
  )$y)
  pmax(0, final_values)
}

load_prospective_files <- function(pattern, source_model) {
  files <- list.files("forecasts/prospective", pattern = pattern, full.names = TRUE)
  if (length(files) == 0) return(NULL)
  df <- bind_rows(lapply(files, read_csv, show_col_types = FALSE))
  df$source_model <- source_model
  df
}

normalize_columns <- function(df, horizon_index) {
  if ("type" %in% names(df)) df <- df %>% rename(output_type = type)
  if ("quantile" %in% names(df)) df <- df %>% rename(output_type_id = quantile)
  if (!("output_type" %in% names(df))) df$output_type <- "quantile"
  df$output_type_id <- as.numeric(df$output_type_id)
  df %>%
    filter(output_type == "quantile") %>%
    mutate(
      reference_date = as.Date(reference_date),
      target_end_date = as.Date(target_end_date),
      location = as.character(location),
      horizon = horizon_index
    )
}

get_models_for_h <- function(h) {
  model_dfs <- list()
  if (INCLUDE_ARIMA) {
    model_dfs[[length(model_dfs) + 1]] <- load_prospective_files(sprintf("^ARIMA_h%d_prospective_\\d{8}\\.csv$", h), "ARIMA")
  }
  if (INCLUDE_SVM) {
    model_dfs[[length(model_dfs) + 1]] <- load_prospective_files(sprintf("^SVM_h%d_prospective_\\d{8}\\.csv$", h), "SVM")
  }
  if (INCLUDE_LGBM_BLENDED) {
    model_dfs[[length(model_dfs) + 1]] <- load_prospective_files(sprintf("^LGBM-blended_h%d_prospective_\\d{8}\\.csv$", h), "LGBM_blended")
  }
  if (INCLUDE_LGBM_BOUNDED) {
    model_dfs[[length(model_dfs) + 1]] <- load_prospective_files(sprintf("^TwoStage-FrozenMu-bounded_h%d_prospective_\\d{8}\\.csv$", h), "LGBM_bounded")
  }
  for (v in 1:5) {
    include_flag <- get(sprintf("INCLUDE_LGBM_BOUNDED_WIDE_%d", v))
    if (!include_flag) next
    model_dfs[[length(model_dfs) + 1]] <- load_prospective_files(
      sprintf("^TwoStage-FrozenMu-bounded-wide-%d_h%d_prospective_\\d{8}\\.csv$", v, h),
      sprintf("LGBM_bounded_wide_%d", v)
    )
  }
  # LGBM bounded wide 4 non-enhanced (uses default state lag features)
  if (INCLUDE_LGBM_BOUNDED_WIDE_4_NE) {
    model_dfs[[length(model_dfs) + 1]] <- load_prospective_files(
      sprintf("^TwoStage-FrozenMu-bounded-wide-4-ne_h%d_prospective_\\d{8}\\.csv$", h),
      "LGBM_bounded_wide_4_ne"
    )
  }
  model_dfs <- model_dfs[!vapply(model_dfs, is.null, logical(1))]
  if (length(model_dfs) == 0) return(NULL)
  bind_rows(lapply(model_dfs, normalize_columns, horizon_index = h - 1))
}

weights_from_scores <- function(scores, models) {
  scores_vec <- scores[models]
  if (all(is.na(scores_vec))) {
    return(setNames(rep(1 / length(models), length(models)), models))
  }
  scores_vec[!is.finite(scores_vec)] <- NA
  if (all(is.na(scores_vec))) {
    return(setNames(rep(1 / length(models), length(models)), models))
  }
  inv <- 1 / (scores_vec + 1e-8)
  inv[!is.finite(inv)] <- NA
  if (all(is.na(inv))) {
    return(setNames(rep(1 / length(models), length(models)), models))
  }
  inv[is.na(inv)] <- 0
  if (sum(inv) <= 0) {
    return(setNames(rep(1 / length(models), length(models)), models))
  }
  inv / sum(inv)
}

compute_wis_by_ref <- function(df) {
  df %>%
    group_by(source_model, reference_date, target_end_date, location) %>%
    summarise(quantile_values = list(value), quantile_levels = list(output_type_id), .groups = "drop") %>%
    left_join(actual_data %>% select(date, location, actual_value),
              by = c("target_end_date" = "date", "location")) %>%
    filter(!is.na(actual_value)) %>%
    rowwise() %>%
    mutate(wis = calculate_wis_single(unlist(quantile_values), unlist(quantile_levels), actual_value)) %>%
    ungroup() %>%
    group_by(source_model, reference_date) %>%
    summarise(daily_avg_wis = mean(wis, na.rm = TRUE), .groups = "drop")
}

compute_wis_by_ref_loc <- function(df) {
  df %>%
    group_by(source_model, reference_date, target_end_date, location) %>%
    summarise(quantile_values = list(value), quantile_levels = list(output_type_id), .groups = "drop") %>%
    left_join(actual_data %>% select(date, location, actual_value),
              by = c("target_end_date" = "date", "location")) %>%
    filter(!is.na(actual_value)) %>%
    rowwise() %>%
    mutate(wis = calculate_wis_single(unlist(quantile_values), unlist(quantile_levels), actual_value)) %>%
    ungroup() %>%
    group_by(source_model, reference_date, location) %>%
    summarise(daily_avg_wis = mean(wis, na.rm = TRUE), .groups = "drop")
}

compute_scores_for_refs <- function(wis_df, eval_refs, lookback_weeks, models) {
  if (length(eval_refs) == 0) {
    return(setNames(rep(NA_real_, length(models)), models))
  }
  alpha <- 2 / (lookback_weeks + 1)
  lags <- seq(length(eval_refs) - 1, 0)
  date_decay_weights <- (1 - alpha) ^ lags
  date_weight_map <- setNames(date_decay_weights, as.character(eval_refs))
  scores <- setNames(rep(NA_real_, length(models)), models)
  for (mn in models) {
    dfm <- wis_df %>% filter(source_model == mn, reference_date %in% eval_refs)
    if (nrow(dfm) == 0) next
    w_subset <- date_weight_map[as.character(dfm$reference_date)]
    if (any(is.na(w_subset))) {
      w_subset <- rep(1, length(w_subset))
    }
    w_subset_norm <- w_subset / sum(w_subset)
    scores[mn] <- sum(dfm$daily_avg_wis * w_subset_norm)
  }
  scores
}

safe_slope <- function(y) {
  y <- as.numeric(y)
  y <- y[is.finite(y)]
  if (length(y) < 2) return(0)
  coef(lm(y ~ seq_along(y)))[2]
}

get_covariates <- function(ref_date, window_weeks = 52, recent_weeks = 6) {
  window_start <- ref_date - weeks(window_weeks)
  data_slice <- actual_data %>%
    filter(date <= ref_date, date > window_start) %>%
    arrange(location, date) %>%
    group_by(location) %>%
    summarise(
      values = list(log1p(actual_value)),
      .groups = "drop"
    )
  if (nrow(data_slice) == 0) return(tibble::tibble())
  data_slice %>%
    mutate(
      mean_log = vapply(values, function(v) mean(v, na.rm = TRUE), numeric(1)),
      sd_log = vapply(values, function(v) sd(v, na.rm = TRUE), numeric(1)),
      amp_log = vapply(values, function(v) max(v, na.rm = TRUE) - min(v, na.rm = TRUE), numeric(1)),
      trend_log = vapply(values, safe_slope, numeric(1)),
      recent_slope = vapply(values, function(v) safe_slope(tail(v, min(length(v), recent_weeks))), numeric(1)),
      last_log = vapply(values, function(v) tail(v, 1), numeric(1))
    ) %>%
    mutate(
      sd_log = ifelse(is.finite(sd_log), sd_log, 0),
      amp_log = ifelse(is.finite(amp_log), amp_log, 0),
      trend_log = ifelse(is.finite(trend_log), trend_log, 0),
      recent_slope = ifelse(is.finite(recent_slope), recent_slope, 0)
    ) %>%
    select(location, mean_log, sd_log, amp_log, trend_log, recent_slope, last_log)
}

covariate_cache <- new.env(parent = emptyenv())

get_covariates_cached <- function(ref_date) {
  if (!inherits(ref_date, "Date")) {
    ref_date <- as.Date(ref_date, origin = "1970-01-01")
  }
  key <- format(ref_date, "%Y-%m-%d")
  if (exists(key, envir = covariate_cache, inherits = FALSE)) {
    return(get(key, envir = covariate_cache, inherits = FALSE))
  }
  covs <- get_covariates(ref_date)
  assign(key, covs, envir = covariate_cache)
  covs
}

build_neighbor_map <- function(covariates, k = 10) {
  if (nrow(covariates) == 0) return(list())
  features <- covariates %>% select(mean_log, sd_log, amp_log, trend_log, recent_slope, last_log)
  features[!is.finite(as.matrix(features))] <- 0
  scaled <- scale(features)
  scaled[is.na(scaled)] <- 0
  locs <- covariates$location
  dist_mat <- as.matrix(dist(scaled))
  neighbors <- list()
  for (i in seq_along(locs)) {
    order_idx <- order(dist_mat[i, ], decreasing = FALSE)
    order_idx <- order_idx[order_idx != i]
    pick <- head(order_idx, k)
    neighbors[[locs[i]]] <- c(locs[i], locs[pick])
  }
  neighbors
}

robustify_wis_by_ref <- function(wis_by_ref, k = 3) {
  wis_by_ref %>%
    group_by(reference_date) %>%
    mutate(
      med = median(daily_avg_wis, na.rm = TRUE),
      mad = median(abs(daily_avg_wis - med), na.rm = TRUE),
      lower = pmax(med - k * mad, 0),
      upper = med + k * mad,
      daily_avg_wis = pmin(pmax(daily_avg_wis, lower), upper)
    ) %>%
    ungroup() %>%
    select(source_model, reference_date, daily_avg_wis)
}

build_ensemble <- function(df_slice, weights, horizon_index) {
  target_taus <- sort(CDC_QUANTILES)
  df_slice %>%
    group_by(reference_date, target_end_date, location, output_type) %>%
    reframe(
      value = calculate_linear_pool(pick(everything()), weights, target_taus),
      output_type_id = target_taus
    ) %>%
    mutate(
      horizon = horizon_index,
      target = "wk inc flu hosp"
    ) %>%
    select(reference_date, target, horizon, target_end_date, location,
           output_type, output_type_id, value)
}

fit_meta_models <- function(wis_by_ref_loc, eval_refs, models, covariate_fn) {
  eval_refs <- sort(as.Date(eval_refs, origin = "1970-01-01"))
  if (length(eval_refs) == 0) return(list())
  training_rows <- list()
  for (ref in eval_refs) {
    cov_ref <- covariate_fn(ref)
    if (nrow(cov_ref) == 0) next
    wis_ref <- wis_by_ref_loc %>% filter(reference_date == ref)
    if (nrow(wis_ref) == 0) next
    training_rows[[length(training_rows) + 1]] <- wis_ref %>%
      left_join(cov_ref, by = "location")
  }
  training_data <- bind_rows(training_rows)
  if (nrow(training_data) == 0) return(list())
  fits <- list()
  for (mn in models) {
    dfm <- training_data %>% filter(source_model == mn)
    if (nrow(dfm) < META_MIN_ROWS) next
    fit <- tryCatch(
      lm(log1p(daily_avg_wis) ~ mean_log + sd_log + amp_log + trend_log + recent_slope,
         data = dfm),
      error = function(e) NULL
    )
    if (!is.null(fit)) fits[[mn]] <- fit
  }
  fits
}

predict_meta_scores <- function(cov_ref, fits, models) {
  scores <- setNames(rep(NA_real_, length(models)), models)
  if (nrow(cov_ref) == 0) return(scores)
  for (mn in names(fits)) {
    fit <- fits[[mn]]
    preds <- tryCatch(predict(fit, newdata = cov_ref), error = function(e) rep(NA_real_, nrow(cov_ref)))
    scores[mn] <- preds
  }
  scores
}

write_ensembles <- function(ensembles_by_date, out_dir, prefix) {
  if (length(ensembles_by_date) == 0) return()
  for (key in names(ensembles_by_date)) {
    out_df <- ensembles_by_date[[key]] %>%
      arrange(reference_date, horizon, location, output_type_id)
    out_path <- file.path(out_dir, sprintf("%s_%s.csv", prefix, gsub("-", "", key)))
    write_csv(out_df, out_path)
  }
}

ensembles_main <- list()
ensembles_shrink <- list()
ensembles_loc <- list()
ensembles_knn <- list()
ensembles_meta <- list()
ensembles_gate <- list()
ensembles_robust <- list()
ensembles_tier <- list()

for (h in 1:4) {
  combined <- get_models_for_h(h)
  if (is.null(combined) || nrow(combined) == 0) {
    message(sprintf("H%d: No prospective model files found; skipping.", h))
    next
  }

  combined <- combined %>%
    mutate(output_type_id = as.numeric(output_type_id))

  ref_dates <- sort(unique(combined$reference_date))
  if (!is.na(SEASON_START)) {
    ref_dates <- ref_dates[ref_dates >= as.Date(SEASON_START)]
  }
  if (!is.na(SEASON_END)) {
    ref_dates <- ref_dates[ref_dates <= as.Date(SEASON_END)]
  }
  if (length(ref_dates) == 0) next

  models_all <- sort(unique(combined$source_model))

  wis_by_ref <- compute_wis_by_ref(combined)
  wis_by_ref_loc <- compute_wis_by_ref_loc(combined)
  robust_wis_by_ref <- robustify_wis_by_ref(wis_by_ref, ROBUST_K)

  for (i in seq_along(ref_dates)) {
    ref_date <- ref_dates[i]
    if (!inherits(ref_date, "Date")) {
      ref_date <- as.Date(ref_date, origin = "1970-01-01")
    }
    df_date <- combined %>% filter(reference_date == ref_date)
    available_models <- sort(unique(df_date$source_model))
    if (length(available_models) == 0) next

    # Current prospective strategy (EWMA inverse WIS)
    eval_refs <- sort(unique(ref_dates[ref_dates < ref_date]))
    eval_refs <- tail(eval_refs, HISTORY_WEEKS)
    scores <- compute_scores_for_refs(wis_by_ref, eval_refs, LOOKBACK_WEEKS, models_all)
    weights_main <- weights_from_scores(scores, available_models)
    ensemble_main <- build_ensemble(df_date, weights_main, h - 1)

    key <- format(ref_date, "%Y-%m-%d")
    if (!is.null(ensembles_main[[key]])) {
      ensembles_main[[key]] <- bind_rows(ensembles_main[[key]], ensemble_main)
    } else {
      ensembles_main[[key]] <- ensemble_main
    }

    # Shrink-to-equal variant
    equal_weights <- setNames(rep(1 / length(available_models), length(available_models)), available_models)
    weights_shrink <- (1 - SHRINK_FACTOR) * weights_main + SHRINK_FACTOR * equal_weights
    weights_shrink <- weights_shrink / sum(weights_shrink)
    ensemble_shrink <- build_ensemble(df_date, weights_shrink, h - 1)

    if (!is.null(ensembles_shrink[[key]])) {
      ensembles_shrink[[key]] <- bind_rows(ensembles_shrink[[key]], ensemble_shrink)
    } else {
      ensembles_shrink[[key]] <- ensemble_shrink
    }

    # Robust stacking variant (Huberized WIS)
    scores_robust <- compute_scores_for_refs(robust_wis_by_ref, eval_refs, LOOKBACK_WEEKS, models_all)
    weights_robust <- weights_from_scores(scores_robust, available_models)
    ensemble_robust <- build_ensemble(df_date, weights_robust, h - 1)
    if (!is.null(ensembles_robust[[key]])) {
      ensembles_robust[[key]] <- bind_rows(ensembles_robust[[key]], ensemble_robust)
    } else {
      ensembles_robust[[key]] <- ensemble_robust
    }

    # Location + horizon weights with EWMA + shrink to global
    global_weights <- weights_main
    locs <- sort(unique(df_date$location))
    loc_ensembles <- list()
    gate_ensembles <- list()
    tier_ensembles <- list()
    knn_ensembles <- list()
    meta_ensembles <- list()
    eval_refs_loc <- sort(unique(ref_dates[ref_dates < ref_date]))
    eval_refs_loc <- tail(eval_refs_loc, LOC_HISTORY_WEEKS)
    eval_refs_meta <- sort(unique(ref_dates[ref_dates < ref_date]))
    eval_refs_meta <- tail(eval_refs_meta, META_HISTORY_WEEKS)
    cov_ref <- get_covariates_cached(ref_date)
    neighbor_map <- build_neighbor_map(cov_ref, KNN_K)
    meta_fits <- fit_meta_models(wis_by_ref_loc, eval_refs_meta, models_all, get_covariates_cached)

    for (loc in locs) {
      df_loc <- df_date %>% filter(location == loc)
      available_loc_models <- sort(unique(df_loc$source_model))
      if (length(available_loc_models) == 0) next

      wis_loc <- wis_by_ref_loc %>% filter(location == loc, reference_date %in% eval_refs_loc)
      n_weeks <- length(unique(wis_loc$reference_date))
      scores_loc <- compute_scores_for_refs(wis_loc, eval_refs_loc, LOC_LOOKBACK_WEEKS, models_all)
      weights_loc_base <- weights_from_scores(scores_loc, available_loc_models)
      if (all(is.na(weights_loc_base))) {
        weights_loc_base <- global_weights[available_loc_models]
      }

      shrink_scale <- if (LOC_MIN_WEEKS <= 0) 1 else max(0, (LOC_MIN_WEEKS - n_weeks) / LOC_MIN_WEEKS)
      shrink <- min(1, LOC_BASE_SHRINK + (1 - LOC_BASE_SHRINK) * shrink_scale)
      weights_global_loc <- global_weights[available_loc_models]
      if (any(is.na(weights_global_loc))) {
        weights_global_loc <- setNames(rep(1 / length(available_loc_models), length(available_loc_models)), available_loc_models)
      }
      weights_loc <- (1 - shrink) * weights_loc_base + shrink * weights_global_loc
      if (sum(weights_loc) > 0) {
        weights_loc <- weights_loc / sum(weights_loc)
      } else {
        weights_loc <- setNames(rep(1 / length(available_loc_models), length(available_loc_models)), available_loc_models)
      }

      loc_ensemble <- build_ensemble(df_loc, weights_loc, h - 1)
      loc_ensembles[[length(loc_ensembles) + 1]] <- loc_ensemble

      # Mixture-of-experts gating
      scores_loc_vec <- scores_loc[available_loc_models]
      mean_score <- mean(scores_loc_vec, na.rm = TRUE)
      cv <- if (!is.finite(mean_score) || mean_score <= 0) 1 else sd(scores_loc_vec, na.rm = TRUE) / mean_score
      if (!is.finite(cv)) cv <- 1
      gate_scale <- if (GATE_MAX_WEEKS <= GATE_MIN_WEEKS) 1 else
        max(0, min(1, (n_weeks - GATE_MIN_WEEKS) / (GATE_MAX_WEEKS - GATE_MIN_WEEKS)))
      gate <- gate_scale * (1 / (1 + cv))
      weights_gate <- (1 - gate) * weights_global_loc + gate * weights_loc_base
      if (sum(weights_gate) > 0) {
        weights_gate <- weights_gate / sum(weights_gate)
      } else {
        weights_gate <- weights_global_loc
      }
      gate_ensemble <- build_ensemble(df_loc, weights_gate, h - 1)
      gate_ensembles[[length(gate_ensembles) + 1]] <- gate_ensemble

      # Two-tier model: only use loc weights if enough history
      if (n_weeks >= TIER_MIN_WEEKS) {
        weights_tier <- weights_loc_base
      } else {
        weights_tier <- weights_global_loc
      }
      if (sum(weights_tier) > 0) {
        weights_tier <- weights_tier / sum(weights_tier)
      }
      tier_ensemble <- build_ensemble(df_loc, weights_tier, h - 1)
      tier_ensembles[[length(tier_ensembles) + 1]] <- tier_ensemble

      # Borrowed-strength kNN pooling
      neighbors <- neighbor_map[[loc]]
      if (is.null(neighbors)) neighbors <- loc
      wis_knn <- wis_by_ref_loc %>%
        filter(location %in% neighbors, reference_date %in% eval_refs_loc)
      if (nrow(wis_knn) > 0) {
        wis_knn <- wis_knn %>%
          group_by(source_model, reference_date) %>%
          summarise(daily_avg_wis = mean(daily_avg_wis, na.rm = TRUE), .groups = "drop")
        scores_knn <- compute_scores_for_refs(wis_knn, eval_refs_loc, LOC_LOOKBACK_WEEKS, models_all)
        weights_knn <- weights_from_scores(scores_knn, available_loc_models)
      } else {
        weights_knn <- weights_global_loc
      }
      if (all(is.na(weights_knn))) {
        weights_knn <- weights_global_loc
      }
      if (sum(weights_knn) > 0) {
        weights_knn <- weights_knn / sum(weights_knn)
      } else {
        weights_knn <- weights_global_loc
      }
      knn_ensemble <- build_ensemble(df_loc, weights_knn, h - 1)
      knn_ensembles[[length(knn_ensembles) + 1]] <- knn_ensemble

      # Meta-learner regression weights
      cov_loc <- cov_ref %>% filter(location == loc)
      weights_meta <- weights_global_loc
      if (length(meta_fits) > 0 && nrow(cov_loc) > 0) {
        pred_scores <- setNames(rep(NA_real_, length(models_all)), models_all)
        for (mn in names(meta_fits)) {
          pred <- tryCatch(predict(meta_fits[[mn]], newdata = cov_loc), error = function(e) NA_real_)
          pred_scores[mn] <- pred
        }
        pred_wis <- exp(pred_scores) - 1
        weights_meta <- weights_from_scores(pred_wis, available_loc_models)
      }
      if (all(is.na(weights_meta)) || sum(weights_meta) <= 0) {
        weights_meta <- weights_global_loc
      } else {
        weights_meta <- weights_meta / sum(weights_meta)
      }
      meta_ensemble <- build_ensemble(df_loc, weights_meta, h - 1)
      meta_ensembles[[length(meta_ensembles) + 1]] <- meta_ensemble
    }

    if (length(loc_ensembles) > 0) {
      ensemble_loc <- bind_rows(loc_ensembles)
      if (!is.null(ensembles_loc[[key]])) {
        ensembles_loc[[key]] <- bind_rows(ensembles_loc[[key]], ensemble_loc)
      } else {
        ensembles_loc[[key]] <- ensemble_loc
      }
    }

    if (length(gate_ensembles) > 0) {
      ensemble_gate <- bind_rows(gate_ensembles)
      if (!is.null(ensembles_gate[[key]])) {
        ensembles_gate[[key]] <- bind_rows(ensembles_gate[[key]], ensemble_gate)
      } else {
        ensembles_gate[[key]] <- ensemble_gate
      }
    }

    if (length(tier_ensembles) > 0) {
      ensemble_tier <- bind_rows(tier_ensembles)
      if (!is.null(ensembles_tier[[key]])) {
        ensembles_tier[[key]] <- bind_rows(ensembles_tier[[key]], ensemble_tier)
      } else {
        ensembles_tier[[key]] <- ensemble_tier
      }
    }

    if (length(knn_ensembles) > 0) {
      ensemble_knn <- bind_rows(knn_ensembles)
      if (!is.null(ensembles_knn[[key]])) {
        ensembles_knn[[key]] <- bind_rows(ensembles_knn[[key]], ensemble_knn)
      } else {
        ensembles_knn[[key]] <- ensemble_knn
      }
    }

    if (length(meta_ensembles) > 0) {
      ensemble_meta <- bind_rows(meta_ensembles)
      if (!is.null(ensembles_meta[[key]])) {
        ensembles_meta[[key]] <- bind_rows(ensembles_meta[[key]], ensemble_meta)
      } else {
        ensembles_meta[[key]] <- ensemble_meta
      }
    }
  }
}

write_ensembles(ensembles_main, OUTPUT_DIR_MAIN, "AdaptiveEnsemble_retrospective")
write_ensembles(ensembles_shrink, OUTPUT_DIR_SHRINK, "AdaptiveEnsemble_shrink_retrospective")
write_ensembles(ensembles_loc, OUTPUT_DIR_LOC, "AdaptiveEnsemble_loc_retrospective")
write_ensembles(ensembles_knn, OUTPUT_DIR_KNN, "AdaptiveEnsemble_knn_retrospective")
write_ensembles(ensembles_meta, OUTPUT_DIR_META, "AdaptiveEnsemble_meta_retrospective")
write_ensembles(ensembles_gate, OUTPUT_DIR_GATE, "AdaptiveEnsemble_gate_retrospective")
write_ensembles(ensembles_robust, OUTPUT_DIR_ROBUST, "AdaptiveEnsemble_robust_retrospective")
write_ensembles(ensembles_tier, OUTPUT_DIR_TIER, "AdaptiveEnsemble_tier_retrospective")

message(sprintf("Wrote current strategy ensembles to %s", OUTPUT_DIR_MAIN))
message(sprintf("Wrote shrink ensembles to %s", OUTPUT_DIR_SHRINK))
message(sprintf("Wrote location ensembles to %s", OUTPUT_DIR_LOC))
message(sprintf("Wrote kNN ensembles to %s", OUTPUT_DIR_KNN))
message(sprintf("Wrote meta ensembles to %s", OUTPUT_DIR_META))
message(sprintf("Wrote gate ensembles to %s", OUTPUT_DIR_GATE))
message(sprintf("Wrote robust ensembles to %s", OUTPUT_DIR_ROBUST))
message(sprintf("Wrote tier ensembles to %s", OUTPUT_DIR_TIER))
