#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(dplyr)
  library(readr)
  library(lubridate)
  library(tidyr)
})

# CLI args (match generate_prosp_adaptive_ensemble.R)
args <- commandArgs(trailingOnly = TRUE)
LOOKBACK_WEEKS <- 3
HISTORY_WEEKS  <- 6
INCLUDE_ARIMA <- TRUE
INCLUDE_SVM   <- FALSE
INCLUDE_LGBM_BLENDED <- TRUE
INCLUDE_LGBM_BOUNDED <- TRUE
INCLUDE_LGBM_BOUNDED_WIDE_1 <- TRUE
INCLUDE_LGBM_BOUNDED_WIDE_2 <- TRUE
INCLUDE_LGBM_BOUNDED_WIDE_3 <- TRUE
INCLUDE_LGBM_BOUNDED_WIDE_4 <- TRUE
INCLUDE_LGBM_BOUNDED_WIDE_4_NE <- TRUE
INCLUDE_LGBM_BOUNDED_WIDE_5 <- TRUE
INCLUDE_JOINT_TWOSTAGE <- TRUE
JOINT_RETRO_FILE <- NA_character_
JOINT_PROSP_FILE <- NA_character_
JOINT_REFERENCE_SHIFT_DAYS <- -7
OUTPUT_DIR <- "joint_knn3_pool_test/outputs/ensemble_hedge"
AS_OF_OVERRIDE <- NA_character_

i <- 1
while (i <= length(args)) {
  key <- args[i]
  val <- if (i + 1 <= length(args)) args[i + 1] else NA_character_
  if (key == '--lookback-weeks') { LOOKBACK_WEEKS <- as.integer(val); i <- i + 2; next }
  if (key == '--history-weeks')  { HISTORY_WEEKS  <- as.integer(val); i <- i + 2; next }
  if (key == '--include-arima')  { INCLUDE_ARIMA  <- tolower(val) %in% c('1','true','t','yes','y'); i <- i + 2; next }
  if (key == '--include-svm')    { INCLUDE_SVM    <- tolower(val) %in% c('1','true','t','yes','y'); i <- i + 2; next }
  if (key == '--include-lgbm-blended') { INCLUDE_LGBM_BLENDED <- tolower(val) %in% c('1','true','t','yes','y'); i <- i + 2; next }
  if (key == '--include-lgbm-bounded') { INCLUDE_LGBM_BOUNDED <- tolower(val) %in% c('1','true','t','yes','y'); i <- i + 2; next }
  if (key == '--include-lgbm-bounded-wide-1') { INCLUDE_LGBM_BOUNDED_WIDE_1 <- tolower(val) %in% c('1','true','t','yes','y'); i <- i + 2; next }
  if (key == '--include-lgbm-bounded-wide-2') { INCLUDE_LGBM_BOUNDED_WIDE_2 <- tolower(val) %in% c('1','true','t','yes','y'); i <- i + 2; next }
  if (key == '--include-lgbm-bounded-wide-3') { INCLUDE_LGBM_BOUNDED_WIDE_3 <- tolower(val) %in% c('1','true','t','yes','y'); i <- i + 2; next }
  if (key == '--include-lgbm-bounded-wide-4') { INCLUDE_LGBM_BOUNDED_WIDE_4 <- tolower(val) %in% c('1','true','t','yes','y'); i <- i + 2; next }
  if (key == '--include-lgbm-bounded-wide-4-ne') { INCLUDE_LGBM_BOUNDED_WIDE_4_NE <- tolower(val) %in% c('1','true','t','yes','y'); i <- i + 2; next }
  if (key == '--include-lgbm-bounded-wide-5') { INCLUDE_LGBM_BOUNDED_WIDE_5 <- tolower(val) %in% c('1','true','t','yes','y'); i <- i + 2; next }
  if (key == '--include-joint-twostage') { INCLUDE_JOINT_TWOSTAGE <- tolower(val) %in% c('1','true','t','yes','y'); i <- i + 2; next }
  if (key == '--joint-retro-file') { JOINT_RETRO_FILE <- val; i <- i + 2; next }
  if (key == '--joint-prosp-file') { JOINT_PROSP_FILE <- val; i <- i + 2; next }
  if (key == '--joint-reference-shift-days') { JOINT_REFERENCE_SHIFT_DAYS <- as.integer(val); i <- i + 2; next }
  if (key == '--output-dir') { OUTPUT_DIR <- val; i <- i + 2; next }
  if (key == '--asof-date')      { AS_OF_OVERRIDE <- val; i <- i + 2; next }
  i <- i + 1
}

latest_file <- function(dir_path, pattern) {
  if (!dir.exists(dir_path)) return(NA_character_)
  files <- list.files(dir_path, pattern = pattern, full.names = TRUE)
  if (length(files) == 0) return(NA_character_)
  info <- file.info(files)
  files[order(info$mtime, decreasing = TRUE)][1]
}

if (is.na(JOINT_RETRO_FILE)) {
  JOINT_RETRO_FILE <- latest_file("joint_knn3_pool_test/outputs", "^backtest_cov.*\\.csv$")
}
if (is.na(JOINT_PROSP_FILE)) {
  JOINT_PROSP_FILE <- latest_file("joint_knn3_pool_test/outputs", "^prospective.*\\.csv$")
}

latest_stitched <- function() {
  files <- list.files("data/imputed_sets", pattern = "imputed_and_stitched_hosp_\\d{4}-\\d{2}-\\d{2}\\.csv", full.names = TRUE)
  if (length(files) == 0) stop("No stitched files found")
  files[order(files)][length(files)]
}

actual_path <- latest_stitched()
actual_raw <- read_csv(actual_path, show_col_types = FALSE)
actual_data <- actual_raw %>% select(location_name, date, total_hosp) %>%
  rename(state_name = location_name, actual_value = total_hosp) %>%
  mutate(date = as.Date(date)) %>%
  filter(!is.na(actual_value))

if (!is.na(AS_OF_OVERRIDE)) {
  as_of_date <- as.Date(AS_OF_OVERRIDE)
} else {
  as_of_date <- max(actual_data$date)
}
as_of_str  <- format(as_of_date, "%Y-%m-%d")
as_of_ts   <- format(as_of_date, "%Y%m%d")
submission_ref_date <- as_of_date + 7
submission_ref_ts   <- format(submission_ref_date, "%Y%m%d")

message(sprintf("Prospective hedge ensemble for as_of=%s (Submission Ref Date=%s)", as_of_str, format(submission_ref_date, "%Y-%m-%d")))
if (INCLUDE_JOINT_TWOSTAGE) {
  message(sprintf("Joint model enabled. retro=%s | prosp=%s | ref_shift_days=%d",
                  ifelse(is.na(JOINT_RETRO_FILE), "NA", JOINT_RETRO_FILE),
                  ifelse(is.na(JOINT_PROSP_FILE), "NA", JOINT_PROSP_FILE),
                  JOINT_REFERENCE_SHIFT_DAYS))
  if (is.na(JOINT_PROSP_FILE) || !file.exists(JOINT_PROSP_FILE)) {
    message("Joint prospective file not found; JointKNN3Pool will affect weight estimation but not current blended forecast values.")
  }
}

location_to_fips <- c(
  'Alabama' = '01', 'Alaska' = '02', 'Arizona' = '04', 'Arkansas' = '05',
  'California' = '06', 'Colorado' = '08', 'Connecticut' = '09', 'Delaware' = '10',
  'District of Columbia' = '11', 'Florida' = '12', 'Georgia' = '13', 'Hawaii' = '15',
  'Idaho' = '16', 'Illinois' = '17', 'Indiana' = '18', 'Iowa' = '19',
  'Kansas' = '20', 'Kentucky' = '21', 'Louisiana' = '22', 'Maine' = '23',
  'Maryland' = '24', 'Massachusetts' = '25', 'Michigan' = '26', 'Minnesota' = '27',
  'Mississippi' = '28', 'Missouri' = '29', 'Montana' = '30', 'Nebraska' = '31',
  'Nevada' = '32', 'New Hampshire' = '33', 'New Jersey' = '34', 'New Mexico' = '35',
  'New York' = '36', 'North Carolina' = '37', 'North Dakota' = '38', 'Ohio' = '39',
  'Oklahoma' = '40', 'Oregon' = '41', 'Pennsylvania' = '42', 'Puerto Rico' = '72',
  'Rhode Island' = '44', 'South Carolina' = '45', 'South Dakota' = '46', 'Tennessee' = '47',
  'Texas' = '48', 'Utah' = '49', 'Vermont' = '50', 'Virginia' = '51',
  'Washington' = '53', 'West Virginia' = '54', 'Wisconsin' = '55', 'Wyoming' = '56',
  'US' = 'US'
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

compute_hedge_weights <- function(model_dfs, as_of_date, lookback_weeks, history_weeks) {
  all_refs <- unique(do.call(c, lapply(model_dfs, function(x) unique(as.Date(x$reference_date)))))
  all_refs <- sort(all_refs)
  all_refs <- all_refs[all_refs < as_of_date]
  eval_refs <- tail(all_refs, history_weeks)
  if (length(eval_refs) == 0) {
    return(setNames(rep(1/length(model_dfs), length(model_dfs)), names(model_dfs)))
  }
  wis_df <- bind_rows(lapply(names(model_dfs), function(mn) {
    dfm <- model_dfs[[mn]] %>%
      filter(as.Date(reference_date) %in% eval_refs, output_type == 'quantile') %>%
      mutate(output_type_id = as.numeric(output_type_id), value = as.numeric(value)) %>%
      mutate(source_model = mn)
    if (nrow(dfm) == 0) return(NULL)
    dfm
  }))
  if (nrow(wis_df) == 0) {
    return(setNames(rep(1/length(model_dfs), length(model_dfs)), names(model_dfs)))
  }
  wis_by_ref <- wis_df %>%
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

  models <- sort(unique(wis_by_ref$source_model))
  scores <- compute_scores_for_refs(wis_by_ref, eval_refs, lookback_weeks, models)
  scores_vec <- scores[models]
  scores_vec <- scores_vec[is.finite(scores_vec)]
  if (length(scores_vec) == 0) {
    return(setNames(rep(1/length(model_dfs), length(model_dfs)), names(model_dfs)))
  }
  eta <- 1
  med <- median(scores_vec, na.rm = TRUE)
  if (is.finite(med) && med > 0) {
    eta <- 1 / med
  }
  weights <- exp(-eta * scores[models])
  weights[!is.finite(weights)] <- 0
  if (sum(weights) <= 0) {
    weights <- rep(1, length(models))
  }
  weights <- weights / sum(weights)
  out <- rep(0, length(model_dfs)); names(out) <- names(model_dfs)
  out[models] <- weights
  out
}

load_retro_for_h <- function(h) {
  lst <- list()
  arima_path <- file.path('forecasts/retrospective/arima', sprintf('ARIMA_h%d_forecasts.csv', h))
  lgbm_blended_path <- file.path('forecasts/retrospective/lgbm_blended', sprintf('LGBM-blended_h%d_forecasts.csv', h))
  lgbm_bounded_path <- file.path('forecasts/retrospective/lgbm_enhanced_t10_bounded', sprintf('TwoStage-FrozenMu_h%d_forecasts.csv', h))
  svm_glob   <- list.files('forecasts/retrospective/svm_t100', pattern = sprintf('^svm.*_h%d.*\\.csv$', h), full.names = TRUE, ignore.case = TRUE)

  if (INCLUDE_ARIMA && file.exists(arima_path)) {
    lst$ARIMA <- read_csv(arima_path, show_col_types = FALSE)
  }
  if (INCLUDE_LGBM_BLENDED && file.exists(lgbm_blended_path)) {
    lst$LGBM_blended <- read_csv(lgbm_blended_path, show_col_types = FALSE)
  }
  if (INCLUDE_LGBM_BOUNDED && file.exists(lgbm_bounded_path)) {
    lst$LGBM_bounded <- read_csv(lgbm_bounded_path, show_col_types = FALSE)
  }
  for (v in 1:5) {
    include_flag <- get(sprintf('INCLUDE_LGBM_BOUNDED_WIDE_%d', v))
    lgbm_bounded_wide_path <- file.path('forecasts/retrospective', sprintf('lgbm_enhanced_t10_bounded_wide_%d', v), sprintf('TwoStage-FrozenMu_h%d_forecasts.csv', h))
    if (include_flag && file.exists(lgbm_bounded_wide_path)) {
      lst[[sprintf('LGBM_bounded_wide_%d', v)]] <- read_csv(lgbm_bounded_wide_path, show_col_types = FALSE)
    }
  }
  # Bounded wide 4 non-enhanced (uses default state lag features, not enhanced features)
  lgbm_bounded_wide_4_ne_path <- file.path('forecasts/retrospective', 'lgbm_t10_bounded_wide_4', sprintf('TwoStage-FrozenMu_h%d_forecasts.csv', h))
  if (INCLUDE_LGBM_BOUNDED_WIDE_4_NE && file.exists(lgbm_bounded_wide_4_ne_path)) {
    lst$LGBM_bounded_wide_4_ne <- read_csv(lgbm_bounded_wide_4_ne_path, show_col_types = FALSE)
  }
  if (INCLUDE_SVM && length(svm_glob) > 0) {
    svm_df <- bind_rows(lapply(svm_glob, read_csv, show_col_types = FALSE))
    if ('type' %in% names(svm_df)) svm_df <- svm_df %>% rename(output_type = type)
    if ('quantile' %in% names(svm_df)) svm_df <- svm_df %>% rename(output_type_id = quantile)
    lst$SVM <- svm_df
  }
  if (INCLUDE_JOINT_TWOSTAGE && !is.na(JOINT_RETRO_FILE) && file.exists(JOINT_RETRO_FILE)) {
    joint_df <- read_csv(JOINT_RETRO_FILE, show_col_types = FALSE)
    if ('output_type' %in% names(joint_df) && 'horizon' %in% names(joint_df)) {
      joint_df <- joint_df %>%
        mutate(
          horizon = as.integer(horizon),
          reference_date = as.Date(reference_date) + JOINT_REFERENCE_SHIFT_DAYS
        ) %>%
        filter(output_type == 'quantile', horizon == (h - 1))
      if (nrow(joint_df) > 0) {
        lst$JointKNN3Pool <- joint_df
      }
    }
  }
  lst
}

load_prosp_for_h <- function(h, ts) {
  lst <- list()
  pdir <- 'forecasts/prospective'
  arima_fp <- file.path(pdir, sprintf('ARIMA_h%d_prospective_%s.csv', h, ts))
  svm_fp   <- file.path(pdir, sprintf('SVM_h%d_prospective_%s.csv', h, ts))
  if (file.exists(arima_fp)) lst$ARIMA <- read_csv(arima_fp, show_col_types = FALSE)
  if (file.exists(svm_fp))   lst$SVM   <- read_csv(svm_fp, show_col_types = FALSE)

  lgbm_blended_fp <- file.path(pdir, sprintf('LGBM-blended_h%d_prospective_%s.csv', h, ts))
  if (INCLUDE_LGBM_BLENDED && file.exists(lgbm_blended_fp)) {
    lst$LGBM_blended <- read_csv(lgbm_blended_fp, show_col_types = FALSE)
  }

  lgbm_bounded_fp <- file.path(pdir, sprintf('TwoStage-FrozenMu-bounded_h%d_prospective_%s.csv', h, ts))
  if (INCLUDE_LGBM_BOUNDED && file.exists(lgbm_bounded_fp)) {
    lst$LGBM_bounded <- read_csv(lgbm_bounded_fp, show_col_types = FALSE)
  }

  for (v in 1:5) {
    include_flag <- get(sprintf('INCLUDE_LGBM_BOUNDED_WIDE_%d', v))
    lgbm_bounded_wide_fp <- file.path(pdir, sprintf('TwoStage-FrozenMu-bounded-wide-%d_h%d_prospective_%s.csv', v, h, ts))
    if (include_flag && file.exists(lgbm_bounded_wide_fp)) {
      lst[[sprintf('LGBM_bounded_wide_%d', v)]] <- read_csv(lgbm_bounded_wide_fp, show_col_types = FALSE)
    }
  }
  # LGBM Bounded Wide 4 Non-Enhanced (default state lag features)
  lgbm_bounded_wide_4_ne_fp <- file.path(pdir, sprintf('TwoStage-FrozenMu-bounded-wide-4-ne_h%d_prospective_%s.csv', h, ts))
  if (INCLUDE_LGBM_BOUNDED_WIDE_4_NE && file.exists(lgbm_bounded_wide_4_ne_fp)) {
    lst$LGBM_bounded_wide_4_ne <- read_csv(lgbm_bounded_wide_4_ne_fp, show_col_types = FALSE)
  }
  if (INCLUDE_JOINT_TWOSTAGE && !is.na(JOINT_PROSP_FILE) && file.exists(JOINT_PROSP_FILE)) {
    joint_df <- read_csv(JOINT_PROSP_FILE, show_col_types = FALSE)
    if ('output_type' %in% names(joint_df) && 'horizon' %in% names(joint_df)) {
      joint_df <- joint_df %>%
        mutate(
          horizon = as.integer(horizon),
          reference_date = as.Date(reference_date) + JOINT_REFERENCE_SHIFT_DAYS
        ) %>%
        filter(output_type == 'quantile', horizon == (h - 1))
      if ('reference_date' %in% names(joint_df) && nrow(joint_df) > 0) {
        if (any(joint_df$reference_date == submission_ref_date, na.rm = TRUE)) {
          joint_df <- joint_df %>% filter(reference_date == submission_ref_date)
        } else {
          latest_ref <- max(joint_df$reference_date, na.rm = TRUE)
          joint_df <- joint_df %>% filter(reference_date == latest_ref)
        }
      }
      if (nrow(joint_df) > 0) {
        lst$JointKNN3Pool <- joint_df
      }
    }
  }
  lst
}

dir.create(OUTPUT_DIR, showWarnings = FALSE, recursive = TRUE)

all_ensembles <- list()

for (h in 1:4) {
  retro_models <- load_retro_for_h(h)
  retro_models <- lapply(retro_models, function(df) {
    if (!('output_type' %in% names(df))) df$output_type <- 'quantile'
    if ('output_type_id' %in% names(df)) df$output_type_id <- as.numeric(df$output_type_id)
    df
  })

  weights <- NULL
  if (length(retro_models) > 0) {
    if ('JointKNN3Pool' %in% names(retro_models)) {
      n_joint_refs <- retro_models[['JointKNN3Pool']] %>%
        mutate(reference_date = as.Date(reference_date)) %>%
        filter(reference_date < as_of_date) %>%
        distinct(reference_date) %>%
        nrow()
      if (n_joint_refs < HISTORY_WEEKS) {
        message(sprintf(
          "H%d: JointKNN3Pool has %d historical refs before as_of (target history=%d); weights may be noisy.",
          h, n_joint_refs, HISTORY_WEEKS
        ))
      }
    }
    weights <- compute_hedge_weights(retro_models, as_of_date, lookback_weeks = LOOKBACK_WEEKS, history_weeks = HISTORY_WEEKS)
    message(sprintf("H%d hedge weights: %s", h, paste(sprintf('%s=%.3f', names(weights), weights), collapse=', ')))
  } else {
    message(sprintf("H%d: No retrospective files found; will fall back to equal weights across available prospective models.", h))
  }

  prosp_models <- load_prosp_for_h(h, as_of_ts)
  if (length(prosp_models) == 0) { message(sprintf("H%d: No prospective model files found; skipping.", h)); next }
  prosp_models <- lapply(prosp_models, function(df) {
    if (!('output_type' %in% names(df))) df$output_type <- 'quantile'
    if ('output_type_id' %in% names(df)) df$output_type_id <- as.numeric(df$output_type_id)
    df
  })

  combined <- bind_rows(
    lapply(names(prosp_models), function(mn) mutate(prosp_models[[mn]], source_model = mn))
  ) %>% filter(output_type == 'quantile')

  if (is.null(weights) || length(weights) == 0) {
    unique_models <- unique(combined$source_model)
    weights <- setNames(rep(1/length(unique_models), length(unique_models)), unique_models)
  }

  target_taus <- sort(CDC_QUANTILES)

  ensemble <- combined %>%
    group_by(reference_date, target_end_date, location, output_type) %>%
    reframe(
      value = calculate_linear_pool(pick(everything()), weights, target_taus),
      output_type_id = target_taus
    ) %>%
    mutate(
      horizon = h - 1,
      target = 'wk inc flu hosp',
      reference_date = submission_ref_date
    ) %>%
    select(reference_date, horizon, target, target_end_date, location, output_type, output_type_id, value)

  all_ensembles[[paste0('h', h)]] <- ensemble
}

if (length(all_ensembles) > 0) {
  final_df <- bind_rows(all_ensembles)
  final_df <- final_df %>% select(reference_date, target, horizon, target_end_date, location, output_type, output_type_id, value)
  out_path <- file.path(OUTPUT_DIR, sprintf('AdaptiveEnsemble-hedge_prospective_%s.csv', submission_ref_ts))
  write_csv(final_df, out_path)
  message(sprintf('Saved hedge ensemble: %s (%d rows across %d horizons)', out_path, nrow(final_df), length(all_ensembles)))
} else {
  message('No hedge ensemble output generated (no prospective files found for any horizon).')
}
