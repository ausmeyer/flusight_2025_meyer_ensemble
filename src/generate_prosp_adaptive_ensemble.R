#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(dplyr)
  library(readr)
  library(lubridate)
  library(purrr)
  library(tidyr)
})

# CLI args
args <- commandArgs(trailingOnly = TRUE)
LOOKBACK_WEEKS <- 4
HISTORY_WEEKS  <- 8
INCLUDE_ARIMA <- TRUE
INCLUDE_SVM   <- TRUE
INCLUDE_LGBM  <- TRUE
AS_OF_OVERRIDE <- NA_character_

i <- 1
while (i <= length(args)) {
  key <- args[i]
  val <- if (i + 1 <= length(args)) args[i + 1] else NA_character_
  if (key == '--lookback-weeks') { LOOKBACK_WEEKS <- as.integer(val); i <- i + 2; next }
  if (key == '--history-weeks')  { HISTORY_WEEKS  <- as.integer(val); i <- i + 2; next }
  if (key == '--include-arima')  { INCLUDE_ARIMA  <- tolower(val) %in% c('1','true','t','yes','y'); i <- i + 2; next }
  if (key == '--include-svm')    { INCLUDE_SVM    <- tolower(val) %in% c('1','true','t','yes','y'); i <- i + 2; next }
  if (key == '--include-lgbm')   { INCLUDE_LGBM   <- tolower(val) %in% c('1','true','t','yes','y'); i <- i + 2; next }
  if (key == '--asof-date')      { AS_OF_OVERRIDE <- val; i <- i + 2; next }
  i <- i + 1
}

# Compute last Saturday as-of-date (or override)
if (!is.na(AS_OF_OVERRIDE)) {
  as_of_date <- as.Date(AS_OF_OVERRIDE)
} else {
  as_of_date <- floor_date(Sys.Date(), unit = "week", week_start = 7)
}
as_of_str  <- format(as_of_date, "%Y-%m-%d")
as_of_ts   <- format(as_of_date, "%Y%m%d")

message(sprintf("Prospective adaptive ensemble for as_of=%s", as_of_str))

# Helper: latest stitched file
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

# Map to FIPS
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

# Quantiles used
CDC_QUANTILES <- c(0.01, 0.025, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 
                   0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.975, 0.99)

# WIS function (log scale for CDC methodology)
calculate_wis_single <- function(quantile_values, quantile_levels, actual_value) {
  qlog <- log(as.numeric(quantile_values) + 1)
  alog <- log(as.numeric(actual_value) + 1)
  alphas <- unique(c(quantile_levels[quantile_levels <= 0.5], 1 - quantile_levels[quantile_levels > 0.5]))
  alphas <- alphas[alphas > 0]
  scores <- c()
  for (a in alphas) {
    li <- which.min(abs(quantile_levels - a))
    ui <- which.min(abs(quantile_levels - (1 - a)))
    L <- qlog[li]; U <- qlog[ui]
    width <- U - L
    pen <- if (alog < L) (2/a) * (L - alog) else if (alog > U) (2/a) * (alog - U) else 0
    scores <- c(scores, width + pen)
  }
  mean(scores)
}

# Compute weights for a horizon based on last 4 reference weeks (using up to 8 available)
compute_weights <- function(model_dfs, horizon, as_of_date, lookback_weeks = 4, history_weeks = 8) {
  # Find candidate reference_dates across models
  all_refs <- unique(do.call(c, lapply(model_dfs, function(x) unique(as.Date(x$reference_date)))))
  all_refs <- sort(all_refs)
  # restrict to references strictly before as_of_date
  all_refs <- all_refs[all_refs < as_of_date]
  if (length(all_refs) == 0) return(rep(1/length(model_dfs), length(model_dfs)))
  last_refs <- tail(all_refs, history_weeks)
  eval_refs <- tail(last_refs, lookback_weeks)

  scores <- list()
  for (mn in names(model_dfs)) {
    dfm <- model_dfs[[mn]] %>%
      filter(as.Date(reference_date) %in% eval_refs, output_type == 'quantile') %>%
      mutate(output_type_id = as.numeric(output_type_id), value = as.numeric(value))
    if (nrow(dfm) == 0) { scores[[mn]] <- NA; next }
    wis_df <- dfm %>% group_by(reference_date, target_end_date, location) %>%
      summarise(quantile_values = list(value), quantile_levels = list(output_type_id), .groups = 'drop') %>%
      left_join(actual_data %>% select(date, location, actual_value), by = c('target_end_date' = 'date', 'location')) %>%
      filter(!is.na(actual_value)) %>%
      rowwise() %>% mutate(wis = calculate_wis_single(unlist(quantile_values), unlist(quantile_levels), actual_value)) %>%
      ungroup()
    scores[[mn]] <- if (nrow(wis_df) > 0) mean(wis_df$wis, na.rm = TRUE) else NA
  }
  valid <- !is.na(unlist(scores))
  if (!any(valid)) {
    w <- rep(1/length(model_dfs), length(model_dfs)); names(w) <- names(model_dfs); return(w)
  }
  inv <- 1 / unlist(scores[valid])
  weights <- inv / sum(inv)
  out <- rep(0, length(model_dfs)); names(out) <- names(model_dfs)
  out[names(weights)] <- weights
  out
}

# Load retrospective model files (last 8 weeks window) for weighting
load_retro_for_h <- function(h) {
  lst <- list()
  arima_path <- file.path('forecasts/retrospective/saved_models/arima', sprintf('ARIMA_h%d_forecasts.csv', h))
  lgbm_path  <- file.path('forecasts/retrospective/saved_models/lgbm_enhanced_t10', sprintf('TwoStage-FrozenMu_h%d_forecasts.csv', h))
  svm_glob   <- list.files('forecasts/retrospective', pattern = sprintf('^svm.*_h%d.*\\.csv$', h), full.names = TRUE, ignore.case = TRUE)

  if (INCLUDE_ARIMA && file.exists(arima_path)) {
    lst$ARIMA <- read_csv(arima_path, show_col_types = FALSE)
  }
  if (INCLUDE_LGBM && file.exists(lgbm_path)) {
    lst$LGBM <- read_csv(lgbm_path, show_col_types = FALSE)
  }
  if (INCLUDE_SVM && length(svm_glob) > 0) {
    # prefer a single main SVM file; otherwise combine
    svm_df <- bind_rows(lapply(svm_glob, read_csv, show_col_types = FALSE))
    # Map legacy columns if present
    if ('type' %in% names(svm_df)) svm_df <- svm_df %>% rename(output_type = type)
    if ('quantile' %in% names(svm_df)) svm_df <- svm_df %>% rename(output_type_id = quantile)
    lst$SVM <- svm_df
  }
  lst
}

# Load prospective files for current week
load_prosp_for_h <- function(h, ts) {
  lst <- list()
  pdir <- 'forecasts/prospective'
  files <- c(
    file.path(pdir, sprintf('ARIMA_h%d_prospective_%s.csv', h, ts)),
    file.path(pdir, sprintf('SVM_h%d_prospective_%s.csv', h, ts)),
    file.path(pdir, sprintf('TwoStage-FrozenMu_h%d_prospective_%s.csv', h, ts))
  )
  for (fp in files) {
    if (file.exists(fp)) {
      key <- if (grepl('ARIMA', fp)) 'ARIMA' else if (grepl('SVM', fp)) 'SVM' else 'LGBM'
      lst[[key]] <- read_csv(fp, show_col_types = FALSE)
    }
  }
  lst
}

dir.create('forecasts/prospective', showWarnings = FALSE, recursive = TRUE)

for (h in 1:4) {
  retro_models <- load_retro_for_h(h)
  if (length(retro_models) == 0) next
  # Standardize minimal columns and ensure types
  retro_models <- lapply(retro_models, function(df) {
    if (!('output_type' %in% names(df))) df$output_type <- 'quantile'
    if ('output_type_id' %in% names(df)) df$output_type_id <- as.numeric(df$output_type_id)
    df
  })

  weights <- compute_weights(retro_models, h, as_of_date, lookback_weeks = LOOKBACK_WEEKS, history_weeks = HISTORY_WEEKS)
  message(sprintf("H%d weights: %s", h, paste(sprintf('%s=%.3f', names(weights), weights), collapse=', ')))

  prosp_models <- load_prosp_for_h(h, as_of_ts)
  if (length(prosp_models) == 0) next
  prosp_models <- lapply(prosp_models, function(df) {
    if (!('output_type' %in% names(df))) df$output_type <- 'quantile'
    if ('output_type_id' %in% names(df)) df$output_type_id <- as.numeric(df$output_type_id)
    df
  })

  # Combine using weights
  combined <- bind_rows(
    lapply(names(prosp_models), function(mn) mutate(prosp_models[[mn]], source_model = mn))
  ) %>% filter(output_type == 'quantile')

  ensemble <- combined %>%
    group_by(reference_date, target_end_date, location, output_type, output_type_id) %>%
    summarise(
      value = {
        vs <- value; ms <- source_model
        wsum <- 0; wtot <- 0
        for (i in seq_along(vs)) {
          w <- weights[ms[i]]; if (is.na(w) || w <= 0) next
          wsum <- wsum + vs[i] * w; wtot <- wtot + w
        }
        if (wtot > 0) wsum / wtot else mean(vs, na.rm = TRUE)
      }, .groups = 'drop') %>%
    mutate(horizon = 0, target = 'wk inc flu hosp') %>%
    select(reference_date, horizon, target, target_end_date, location, output_type, output_type_id, value)

  out_path <- file.path('forecasts/prospective', sprintf('AdaptiveEnsemble_h%d_prospective_%s.csv', h, as_of_ts))
  write_csv(ensemble, out_path)
  message(sprintf("Saved: %s (%d rows)", out_path, nrow(ensemble)))
}
