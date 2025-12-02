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
LOOKBACK_WEEKS <- 6
HISTORY_WEEKS  <- 8
INCLUDE_ARIMA <- TRUE
INCLUDE_SVM   <- TRUE
INCLUDE_LGBM  <- TRUE
INCLUDE_LGBM_BOUNDED <- TRUE
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
  if (key == '--include-lgbm-bounded') { INCLUDE_LGBM_BOUNDED <- tolower(val) %in% c('1','true','t','yes','y'); i <- i + 2; next }
  if (key == '--asof-date')      { AS_OF_OVERRIDE <- val; i <- i + 2; next }
  i <- i + 1
}

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

# Compute as-of-date: prefer override; else align with latest data date (default)

if (!is.na(AS_OF_OVERRIDE)) {

  as_of_date <- as.Date(AS_OF_OVERRIDE)

} else {

  as_of_date <- max(actual_data$date)

}

as_of_str  <- format(as_of_date, "%Y-%m-%d")

as_of_ts   <- format(as_of_date, "%Y%m%d")



# FluSight 2025-26: Reference Date is Saturday *after* the forecast due date.

# Our as_of_date is the Saturday *before* the due date (last available data).

submission_ref_date <- as_of_date + 7

submission_ref_ts   <- format(submission_ref_date, "%Y%m%d")



message(sprintf("Prospective adaptive ensemble for as_of=%s (Submission Ref Date=%s)", as_of_str, format(submission_ref_date, "%Y-%m-%d")))



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
  # Validate actual_value
  if (length(actual_value) == 0 || all(is.na(actual_value))) return(NA_real_)
  qv <- as.numeric(quantile_values)
  ql <- as.numeric(quantile_levels)
  aval <- as.numeric(actual_value)[1]
  alog <- log(aval + 1)
  if (!is.finite(alog)) return(NA_real_)
  # Keep finite pairs only
  keep <- is.finite(qv) & is.finite(ql)
  qv <- qv[keep]; ql <- ql[keep]
  if (length(qv) == 0 || length(ql) == 0) return(NA_real_)
  # Define alpha pairs from available quantiles
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

# HELPER: Linear Pooling (Mixture Distribution)
# Inverts the mixture CDF to return values for specific target quantiles
calculate_linear_pool <- function(df_subset, weights, target_taus) {
  # df_subset: must contain columns 'value', 'output_type_id', 'source_model'
  
  # 1. Define the support (grid) for the mixture distribution
  # We take the full range of all model predictions and add a small buffer
  all_vals <- df_subset$value
  if(length(all_vals) == 0) return(rep(NA, length(target_taus)))
  
  min_v <- min(all_vals, na.rm=TRUE)
  max_v <- max(all_vals, na.rm=TRUE)
  
  # Check for non-finite bounds (e.g. if all NAs)
  if(!is.finite(min_v) || !is.finite(max_v)) return(rep(NA, length(target_taus)))
  
  # If single point, return it
  if(min_v == max_v) return(rep(min_v, length(target_taus)))
  
  # Buffer to avoid boundary issues
  range_width <- max_v - min_v
  grid_min <- max(0, min_v - (range_width * 0.1)) # Bound at 0
  grid_max <- max_v + (range_width * 0.1)
  
  # Create a fine grid of hospitalizations (y-axis of CDF)
  # 1000 points is usually sufficient for smooth resolution
  y_grid <- seq(grid_min, grid_max, length.out = 1000)
  
  # 2. Build the Ensemble CDF
  ensemble_cdf <- rep(0, length(y_grid))
  total_weight <- 0
  
  models <- unique(df_subset$source_model)
  
  for(m in models) {
    w <- weights[m]
    if(is.na(w) || w <= 0) next
    
    # Extract this model's quantiles
    m_data <- df_subset %>% 
      filter(source_model == m, !is.na(value)) %>%
      arrange(output_type_id)
    
    # Need at least 2 distinct non-NA values to interpolate
    # If all values are identical (point mass), approx fails (nx=1)
    if(nrow(m_data) < 2 || length(unique(m_data$value)) < 2) next 
    
    # Create CDF approximation for this model: P(Y <= y)
    # We map Values (x) -> Quantiles (y)
    # rule=2 means: for values < min, return 0; for values > max, return 1
    # suppressWarnings() is used because approx warns when collapsing ties, which is expected here.
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
  
  if(total_weight == 0) return(rep(NA, length(target_taus)))
  
  # Normalize (in case weights didn't sum to exactly 1)
  ensemble_cdf <- ensemble_cdf / total_weight
  
  # 3. Invert the Ensemble CDF to get Quantiles
  # We map Probabilities (x) -> Values (y)
  # suppressWarnings() because ensemble_cdf may have flat regions (ties), causing benign warnings.
  final_values <- suppressWarnings(approx(
    x = ensemble_cdf, 
    y = y_grid, 
    xout = target_taus, 
    rule = 2 # Extrapolate (clamp) if target tau is outside grid range
  )$y)
  
  # Enforce zero bound on final values
  final_values <- pmax(0, final_values)
  
  return(final_values)
}

# REPLACEMENT FUNCTION: compute_weights (EWMA Version)
compute_weights <- function(model_dfs, horizon, as_of_date, lookback_weeks = 6, history_weeks = 8) {
  
  # 1. Identify all available reference dates from the history files
  all_refs <- unique(do.call(c, lapply(model_dfs, function(x) unique(as.Date(x$reference_date)))))
  all_refs <- sort(all_refs)
  
  # Restrict to references strictly before as_of_date
  all_refs <- all_refs[all_refs < as_of_date]
  
  # We use the 'history_weeks' to define the maximum window of data we consider
  eval_refs <- tail(all_refs, history_weeks)
  
  if (length(eval_refs) == 0) {
    return(setNames(rep(1/length(model_dfs), length(model_dfs)), names(model_dfs)))
  }

  # 2. Calculate EWMA Decay Weights for these dates
  # We interpret 'lookback_weeks' as the "Span" for the EWMA.
  # Formula: alpha = 2 / (span + 1)
  alpha <- 2 / (lookback_weeks + 1)
  
  # Assign weights to dates based on recency.
  # eval_refs is sorted Oldest -> Newest.
  # We want Newest to have weight ~ 1 (lag=0) and Oldest to have weight ~ (1-alpha)^lag
  n_dates <- length(eval_refs)
  lags <- seq(n_dates - 1, 0) # e.g. 7, 6, 5, ..., 0
  
  # Raw geometric decay weights
  date_decay_weights <- (1 - alpha) ^ lags
  
  # Map dates to their weights for easy lookup
  # Structure: date_weight_map['2025-01-01'] = 0.45
  date_weight_map <- setNames(date_decay_weights, as.character(eval_refs))

  scores <- list()
  
  for (mn in names(model_dfs)) {
    # Filter model data to the evaluation window
    dfm <- model_dfs[[mn]] %>%
      filter(as.Date(reference_date) %in% eval_refs, output_type == 'quantile') %>%
      mutate(output_type_id = as.numeric(output_type_id), value = as.numeric(value))
    
    if (nrow(dfm) == 0) { scores[[mn]] <- NA; next }
    
    # Calculate WIS for each specific date (grouping by reference_date)
    wis_by_date <- dfm %>% 
      group_by(reference_date, target_end_date, location) %>%
      summarise(quantile_values = list(value), quantile_levels = list(output_type_id), .groups = 'drop') %>%
      left_join(actual_data %>% select(date, location, actual_value), by = c('target_end_date' = 'date', 'location')) %>%
      filter(!is.na(actual_value)) %>%
      rowwise() %>% 
      mutate(wis = calculate_wis_single(unlist(quantile_values), unlist(quantile_levels), actual_value)) %>%
      group_by(reference_date) %>%
      summarise(daily_avg_wis = mean(wis, na.rm = TRUE), .groups = 'drop')
    
    if (nrow(wis_by_date) == 0) { scores[[mn]] <- NA; next }
    
    # 3. Apply the Time-Decay Weights
    # We must match the model's available dates to our master weight map.
    # If a model missed a week, we re-normalize the weights for the weeks it DID submit.
    
    model_dates_chr <- as.character(wis_by_date$reference_date)
    w_subset <- date_weight_map[model_dates_chr]
    
    # Safety check for missing lookups (shouldn't happen if logic is correct)
    if (any(is.na(w_subset))) {
      w_subset <- rep(1, length(w_subset)) 
    }
    
    # Re-normalize weights to sum to 1 for this specific model
    w_subset_norm <- w_subset / sum(w_subset)
    
    # The final Score is the Weighted Average of the weekly WIS scores
    scores[[mn]] <- sum(wis_by_date$daily_avg_wis * w_subset_norm)
  }
  
  valid <- !is.na(unlist(scores))
  
  # 4. Standard Inverse Weighting (Higher Score/WIS = Lower Ensemble Weight)
  if (!any(valid)) {
    return(setNames(rep(1/length(model_dfs), length(model_dfs)), names(model_dfs)))
  }
  
  # Invert scores (minimize WIS)
  # Add small epsilon to avoid division by zero if a model has perfect score (0)
  scores_vec <- unlist(scores[valid])
  inv <- 1 / (scores_vec + 1e-8)
  
  # Normalize to sum to 1
  weights <- inv / sum(inv)
  
  out <- rep(0, length(model_dfs)); names(out) <- names(model_dfs)
  out[names(weights)] <- weights
  out
}

# Load retrospective model files (last 8 weeks window) for weighting
load_retro_for_h <- function(h) {
  lst <- list()
  arima_path <- file.path('forecasts/retrospective/arima', sprintf('ARIMA_h%d_forecasts.csv', h))
  lgbm_t100_path <- file.path('forecasts/retrospective/lgbm_enhanced_t100', sprintf('TwoStage-FrozenMu_h%d_forecasts.csv', h))
  lgbm_t10_path  <- file.path('forecasts/retrospective/lgbm_enhanced_t10',  sprintf('TwoStage-FrozenMu_h%d_forecasts.csv', h))
  lgbm_bounded_path <- file.path('forecasts/retrospective/lgbm_enhanced_t10_bounded', sprintf('TwoStage-FrozenMu_h%d_forecasts.csv', h))
  svm_glob   <- list.files('forecasts/retrospective/svm_t100', pattern = sprintf('^svm.*_h%d.*\\.csv$', h), full.names = TRUE, ignore.case = TRUE)

  if (INCLUDE_ARIMA && file.exists(arima_path)) {
    lst$ARIMA <- read_csv(arima_path, show_col_types = FALSE)
  }
  if (INCLUDE_LGBM && file.exists(lgbm_t100_path)) {
    lst$LGBM_t100 <- read_csv(lgbm_t100_path, show_col_types = FALSE)
  }
  if (INCLUDE_LGBM && file.exists(lgbm_t10_path)) {
    lst$LGBM_t10 <- read_csv(lgbm_t10_path, show_col_types = FALSE)
  }
  if (INCLUDE_LGBM_BOUNDED && file.exists(lgbm_bounded_path)) {
    lst$LGBM_bounded <- read_csv(lgbm_bounded_path, show_col_types = FALSE)
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
  # ARIMA and SVM (single files)
  arima_fp <- file.path(pdir, sprintf('ARIMA_h%d_prospective_%s.csv', h, ts))
  svm_fp   <- file.path(pdir, sprintf('SVM_h%d_prospective_%s.csv', h, ts))
  if (file.exists(arima_fp)) lst$ARIMA <- read_csv(arima_fp, show_col_types = FALSE)
  if (file.exists(svm_fp))   lst$SVM   <- read_csv(svm_fp, show_col_types = FALSE)

  # LGBM variants: TwoStage-FrozenMu*, distinguish bounded vs t100 vs t10 by filename
  lgbm_files <- list.files(pdir, pattern = sprintf('^TwoStage-FrozenMu.*_h%d_prospective_%s\\.csv$', h, ts), full.names = TRUE)
  for (fp in lgbm_files) {
    key <- if (grepl('bounded', basename(fp), ignore.case = TRUE)) 'LGBM_bounded'
           else if (grepl('t100', basename(fp), ignore.case = TRUE)) 'LGBM_t100'
           else if (grepl('t10', basename(fp), ignore.case = TRUE)) 'LGBM_t10'
           else 'LGBM'
    lst[[key]] <- read_csv(fp, show_col_types = FALSE)
  }
  lst
}

dir.create('forecasts/prospective', showWarnings = FALSE, recursive = TRUE)

all_ensembles <- list()

for (h in 1:4) {
  retro_models <- load_retro_for_h(h)
  # Standardize minimal columns and ensure types
  retro_models <- lapply(retro_models, function(df) {
    if (!('output_type' %in% names(df))) df$output_type <- 'quantile'
    if ('output_type_id' %in% names(df)) df$output_type_id <- as.numeric(df$output_type_id)
    df
  })

  # We will compute weights if retrospective data exists; otherwise fall back to equal weights over prospective models
  weights <- NULL
  if (length(retro_models) > 0) {
    weights <- compute_weights(retro_models, h, as_of_date, lookback_weeks = LOOKBACK_WEEKS, history_weeks = HISTORY_WEEKS)
    message(sprintf("H%d weights: %s", h, paste(sprintf('%s=%.3f', names(weights), weights), collapse=', ')))
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

  # Combine using weights
  combined <- bind_rows(
    lapply(names(prosp_models), function(mn) mutate(prosp_models[[mn]], source_model = mn))
  ) %>% filter(output_type == 'quantile')

  # If weights are NULL (no retros), use equal weights across prospective models
  if (is.null(weights) || length(weights) == 0) {
    unique_models <- unique(combined$source_model)
    weights <- setNames(rep(1/length(unique_models), length(unique_models)), unique_models)
  }

  # We need the list of quantiles we are targeting (CDC_QUANTILES)
  # Ensure they are sorted
  target_taus <- sort(CDC_QUANTILES)

  ensemble <- combined %>%
    # Group by the specific forecasting instance (Location + Date)
    # CRITICAL: Do NOT group by output_type_id here. 
    group_by(reference_date, target_end_date, location, output_type) %>%
    reframe( # reframe allows returning multiple rows per group
      # Call the helper function
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

# Write a single final file combining all horizons
if (length(all_ensembles) > 0) {
  final_df <- bind_rows(all_ensembles)
  # Ensure proper column order
  final_df <- final_df %>% select(reference_date, target, horizon, target_end_date, location, output_type, output_type_id, value)
  out_path <- file.path('forecasts/prospective', sprintf('AdaptiveEnsemble_prospective_%s.csv', submission_ref_ts))
  write_csv(final_df, out_path)
  message(sprintf('Saved final ensemble: %s (%d rows across %d horizons)', out_path, nrow(final_df), length(all_ensembles)))
} else {
  message('No ensemble output generated (no prospective files found for any horizon).')
}
