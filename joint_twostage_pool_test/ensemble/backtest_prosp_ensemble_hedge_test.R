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
ETA <- NA_real_
SEASON_START <- NA_character_
WARMUP_START <- NA_character_
SEASON_END <- NA_character_
OUTPUT_DIR <- "joint_twostage_pool_test/outputs/ensemble_hedge_retrospective"
INCLUDE_ARIMA <- FALSE
INCLUDE_SVM <- TRUE
INCLUDE_LGBM_BLENDED <- FALSE
INCLUDE_LGBM_BOUNDED <- TRUE
INCLUDE_LGBM_BOUNDED_WIDE_1 <- TRUE
INCLUDE_LGBM_BOUNDED_WIDE_2 <- TRUE
INCLUDE_LGBM_BOUNDED_WIDE_3 <- TRUE
INCLUDE_LGBM_BOUNDED_WIDE_4 <- TRUE
INCLUDE_LGBM_BOUNDED_WIDE_4_NE <- TRUE
INCLUDE_LGBM_BOUNDED_WIDE_5 <- TRUE
INCLUDE_JOINT_TWOSTAGE <- TRUE
JOINT_RETRO_FILE <- NA_character_
JOINT_REFERENCE_SHIFT_DAYS <- -7
INCLUDE_JOINT_KNN3 <- TRUE
KNN3_RETRO_FILE <- NA_character_
KNN3_REFERENCE_SHIFT_DAYS <- -7
INCLUDE_ARGO <- TRUE
ARGO_BASE <- "../flusight_2025_mighte_joint/flu-forecast-2024/point_forecasts/live/2025-26"

i <- 1
while (i <= length(args)) {
  key <- args[i]
  val <- if (i + 1 <= length(args)) args[i + 1] else NA_character_
  if (key == "--lookback-weeks") { LOOKBACK_WEEKS <- as.integer(val); i <- i + 2; next }
  if (key == "--eta") { ETA <- as.numeric(val); i <- i + 2; next }
  if (key == "--season-start") { SEASON_START <- val; i <- i + 2; next }
  if (key == "--warmup-start") { WARMUP_START <- val; i <- i + 2; next }
  if (key == "--season-end") { SEASON_END <- val; i <- i + 2; next }
  if (key == "--output-dir") { OUTPUT_DIR <- val; i <- i + 2; next }
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
  if (key == "--include-joint-twostage") { INCLUDE_JOINT_TWOSTAGE <- tolower(val) %in% c("1","true","t","yes","y"); i <- i + 2; next }
  if (key == "--joint-retro-file") { JOINT_RETRO_FILE <- val; i <- i + 2; next }
  if (key == "--joint-reference-shift-days") { JOINT_REFERENCE_SHIFT_DAYS <- as.integer(val); i <- i + 2; next }
  if (key == "--include-joint-knn3") { INCLUDE_JOINT_KNN3 <- tolower(val) %in% c("1","true","t","yes","y"); i <- i + 2; next }
  if (key == "--knn3-retro-file") { KNN3_RETRO_FILE <- val; i <- i + 2; next }
  if (key == "--knn3-reference-shift-days") { KNN3_REFERENCE_SHIFT_DAYS <- as.integer(val); i <- i + 2; next }
  if (key == "--include-argo") { INCLUDE_ARGO <- tolower(val) %in% c("1","true","t","yes","y"); i <- i + 2; next }
  if (key == "--argo-base") { ARGO_BASE <- val; i <- i + 2; next }
  i <- i + 1
}

# Hard-disable ARIMA and LGBM-blended for this two-stage test hedge ensemble.
if (INCLUDE_ARIMA || INCLUDE_LGBM_BLENDED) {
  message("Forcing INCLUDE_ARIMA=FALSE and INCLUDE_LGBM_BLENDED=FALSE for two-stage test hedge ensemble.")
}
INCLUDE_ARIMA <- FALSE
INCLUDE_LGBM_BLENDED <- FALSE

dir.create(OUTPUT_DIR, recursive = TRUE, showWarnings = FALSE)

latest_file <- function(dir_path, pattern) {
  if (!dir.exists(dir_path)) return(NA_character_)
  files <- list.files(dir_path, pattern = pattern, full.names = TRUE)
  if (length(files) == 0) return(NA_character_)
  info <- file.info(files)
  files[order(info$mtime, decreasing = TRUE)][1]
}

if (is.na(JOINT_RETRO_FILE)) {
  JOINT_RETRO_FILE <- latest_file("joint_twostage_pool_test/outputs", "^backtest_cov.*\\.csv$")
}
if (is.na(KNN3_RETRO_FILE)) {
  KNN3_RETRO_FILE <- latest_file("joint_knn3_pool_test/outputs", "^backtest_cov.*\\.csv$")
}

if (INCLUDE_JOINT_TWOSTAGE) {
  message(sprintf("Backtest hedge: JointTwoStagePool enabled. retro=%s | ref_shift_days=%d",
                  ifelse(is.na(JOINT_RETRO_FILE), "NA", JOINT_RETRO_FILE),
                  JOINT_REFERENCE_SHIFT_DAYS))
  if (is.na(JOINT_RETRO_FILE) || !file.exists(JOINT_RETRO_FILE)) {
    message("Joint retrospective file not found; running hedge backtest without JointTwoStagePool.")
  }
}
if (INCLUDE_JOINT_KNN3) {
  message(sprintf("Backtest hedge: JointKNN3Pool enabled. retro=%s | ref_shift_days=%d",
                  ifelse(is.na(KNN3_RETRO_FILE), "NA", KNN3_RETRO_FILE),
                  KNN3_REFERENCE_SHIFT_DAYS))
  if (is.na(KNN3_RETRO_FILE) || !file.exists(KNN3_RETRO_FILE)) {
    message("KNN3 retrospective file not found; running hedge backtest without JointKNN3Pool.")
  }
}
if (INCLUDE_ARGO) {
  message(sprintf("Backtest hedge: ARGO enabled. base=%s", ARGO_BASE))
  if (!dir.exists(ARGO_BASE)) {
    message("ARGO base directory not found; running hedge backtest without ARGO components.")
  }
}

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

# Map to FIPS
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

# Quantiles used
CDC_QUANTILES <- c(0.01, 0.025, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5,
                   0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.975, 0.99)

# WIS function (log scale for CDC methodology)
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

# Linear Pooling (Mixture Distribution)
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

normalize_location_codes <- function(x) {
  out <- as.character(x)
  out <- gsub("\\.0+$", "", out)
  state_mask <- out %in% names(location_to_fips)
  if (any(state_mask)) {
    out[state_mask] <- unname(location_to_fips[out[state_mask]])
  }
  num_mask <- grepl("^[0-9]+$", out)
  if (any(num_mask)) {
    out[num_mask] <- sprintf("%02d", as.integer(out[num_mask]))
  }
  out
}

load_argo_backtest_component <- function(model_name, source_model, h) {
  if (!INCLUDE_ARGO) return(NULL)
  if (!dir.exists(ARGO_BASE)) return(NULL)

  base_file <- file.path(ARGO_BASE, paste0(model_name, ".csv"))
  dated_pattern <- paste0("^", model_name, "_\\d{4}-\\d{2}-\\d{2}\\.csv$")
  dated_files <- sort(list.files(ARGO_BASE, pattern = dated_pattern, full.names = TRUE))
  files <- c(if (file.exists(base_file)) base_file else character(0), dated_files)
  if (length(files) == 0) return(NULL)

  df <- bind_rows(lapply(files, read_csv, show_col_types = FALSE))
  needed <- c("reference_date", "horizon", "target_end_date", "location", "output_type_id", "value")
  if (!all(needed %in% names(df))) return(NULL)
  if (!("output_type" %in% names(df))) df$output_type <- "quantile"

  df <- df %>%
    mutate(
      reference_date = as.Date(reference_date),
      target_end_date = as.Date(target_end_date),
      horizon = as.integer(horizon),
      output_type_id = as.numeric(output_type_id),
      location = normalize_location_codes(location)
    ) %>%
    filter(output_type == "quantile", horizon == (h - 1))

  if (nrow(df) == 0) return(NULL)
  df <- df %>%
    group_by(reference_date, target_end_date, location, output_type, output_type_id) %>%
    slice_tail(n = 1) %>%
    ungroup()
  df$source_model <- source_model
  df
}

load_joint_backtest_component <- function(h) {
  if (!INCLUDE_JOINT_TWOSTAGE) return(NULL)
  if (is.na(JOINT_RETRO_FILE) || !file.exists(JOINT_RETRO_FILE)) return(NULL)
  df <- read_csv(JOINT_RETRO_FILE, show_col_types = FALSE)
  needed <- c("reference_date", "horizon", "target_end_date", "location", "output_type", "output_type_id", "value")
  if (!all(needed %in% names(df))) return(NULL)
  df <- df %>%
    mutate(
      reference_date = as.Date(reference_date) + JOINT_REFERENCE_SHIFT_DAYS,
      target_end_date = as.Date(target_end_date),
      horizon = as.integer(horizon)
    ) %>%
    filter(output_type == "quantile", horizon == (h - 1))
  if (nrow(df) == 0) return(NULL)
  df$source_model <- "JointTwoStagePool"
  df
}

load_joint_knn3_backtest_component <- function(h) {
  if (!INCLUDE_JOINT_KNN3) return(NULL)
  if (is.na(KNN3_RETRO_FILE) || !file.exists(KNN3_RETRO_FILE)) return(NULL)
  df <- read_csv(KNN3_RETRO_FILE, show_col_types = FALSE)
  needed <- c("reference_date", "horizon", "target_end_date", "location", "output_type", "output_type_id", "value")
  if (!all(needed %in% names(df))) return(NULL)
  df <- df %>%
    mutate(
      reference_date = as.Date(reference_date) + KNN3_REFERENCE_SHIFT_DAYS,
      target_end_date = as.Date(target_end_date),
      horizon = as.integer(horizon)
    ) %>%
    filter(output_type == "quantile", horizon == (h - 1))
  if (nrow(df) == 0) return(NULL)
  df$source_model <- "JointKNN3Pool"
  df
}

normalize_columns <- function(df) {
  if ("type" %in% names(df)) df <- df %>% rename(output_type = type)
  if ("quantile" %in% names(df)) df <- df %>% rename(output_type_id = quantile)
  if (!("output_type" %in% names(df))) df$output_type <- "quantile"
  if ("output_type_id" %in% names(df)) df$output_type_id <- as.numeric(df$output_type_id)
  df
}

infer_season_start <- function(max_ref) {
  yr <- year(max_ref)
  start_year <- if (month(max_ref) >= 10) yr else yr - 1
  as.Date(sprintf("%d-10-01", start_year))
}

compute_model_losses <- function(df_date) {
  df_date %>%
    filter(output_type == "quantile") %>%
    mutate(output_type_id = as.numeric(output_type_id), value = as.numeric(value)) %>%
    group_by(source_model, target_end_date, location) %>%
    summarise(quantile_values = list(value), quantile_levels = list(output_type_id), .groups = "drop") %>%
    left_join(actual_data %>% select(date, location, actual_value),
              by = c("target_end_date" = "date", "location")) %>%
    filter(!is.na(actual_value)) %>%
    rowwise() %>%
    mutate(wis = calculate_wis_single(unlist(quantile_values), unlist(quantile_levels), actual_value)) %>%
    ungroup() %>%
    group_by(source_model) %>%
    summarise(loss = mean(wis, na.rm = TRUE), .groups = "drop")
}

compute_softmin_weights <- function(loss_state, available_models, eta_use) {
  avail_state <- loss_state[available_models]
  finite_mask <- is.finite(avail_state)
  fill_val <- if (any(finite_mask)) mean(avail_state[finite_mask]) else 0
  avail_state[!finite_mask] <- fill_val
  if (is.na(eta_use) || !is.finite(eta_use) || eta_use <= 0) eta_use <- 1
  w <- exp(-eta_use * avail_state)
  if (!all(is.finite(w)) || sum(w) <= 0) w <- rep(1, length(available_models))
  w <- w / sum(w)
  setNames(w, available_models)
}

ensembles_by_date <- list()
weights_log <- list()

for (h in 1:4) {
  model_dfs <- list()
  if (INCLUDE_SVM) {
    model_dfs <- append(model_dfs, list(load_prospective_files(sprintf("^SVM_h%d_prospective_\\d{8}\\.csv$", h), "SVM")))
  }
  if (INCLUDE_LGBM_BOUNDED) {
    model_dfs <- append(model_dfs, list(load_prospective_files(sprintf("^TwoStage-FrozenMu-bounded_h%d_prospective_\\d{8}\\.csv$", h), "LGBM_bounded")))
  }
  for (v in 1:5) {
    include_flag <- get(sprintf("INCLUDE_LGBM_BOUNDED_WIDE_%d", v))
    if (!include_flag) next
    model_dfs <- append(model_dfs, list(load_prospective_files(
      sprintf("^TwoStage-FrozenMu-bounded-wide-%d_h%d_prospective_\\d{8}\\.csv$", v, h),
      sprintf("LGBM_bounded_wide_%d", v)
    )))
  }
  # LGBM bounded wide 4 non-enhanced (uses default state lag features)
  if (INCLUDE_LGBM_BOUNDED_WIDE_4_NE) {
    model_dfs <- append(model_dfs, list(load_prospective_files(
      sprintf("^TwoStage-FrozenMu-bounded-wide-4-ne_h%d_prospective_\\d{8}\\.csv$", h),
      "LGBM_bounded_wide_4_ne"
    )))
  }
  model_dfs <- append(model_dfs, list(load_joint_backtest_component(h)))
  model_dfs <- append(model_dfs, list(load_joint_knn3_backtest_component(h)))
  model_dfs <- append(model_dfs, list(load_argo_backtest_component("argo_smooth", "ARGO_smooth", h)))
  model_dfs <- append(model_dfs, list(load_argo_backtest_component("argo_smooth_log", "ARGO_smooth_log", h)))
  model_dfs <- append(model_dfs, list(load_argo_backtest_component("argo2_smooth", "ARGO2_smooth", h)))
  model_dfs <- append(model_dfs, list(load_argo_backtest_component("argo2_smooth_log", "ARGO2_smooth_log", h)))

  model_dfs <- compact(model_dfs)
  if (length(model_dfs) == 0) {
    message(sprintf("H%d: No prospective model files found; skipping.", h))
    next
  }

  combined <- bind_rows(lapply(model_dfs, normalize_columns)) %>%
    filter(output_type == "quantile") %>%
    mutate(reference_date = as.Date(reference_date),
           target_end_date = as.Date(target_end_date))

  ref_dates <- sort(unique(combined$reference_date))
  if (length(ref_dates) == 0) {
    message(sprintf("H%d: No reference dates found; skipping.", h))
    next
  }

  season_start <- if (!is.na(SEASON_START)) as.Date(SEASON_START) else infer_season_start(max(ref_dates, na.rm = TRUE))
  warmup_start <- if (!is.na(WARMUP_START)) as.Date(WARMUP_START) else season_start
  season_end <- if (!is.na(SEASON_END)) as.Date(SEASON_END) else max(ref_dates, na.rm = TRUE)
  if (warmup_start > season_start) {
    stop("warmup_start cannot be after season_start")
  }
  ref_dates <- ref_dates[ref_dates >= warmup_start & ref_dates <= season_end]
  if (length(ref_dates) == 0) {
    message(sprintf("H%d: No reference dates after warmup/season filter; skipping.", h))
    next
  }

  alpha <- 2 / (LOOKBACK_WEEKS + 1)
  model_names <- sort(unique(combined$source_model))
  loss_state <- setNames(rep(NA_real_, length(model_names)), model_names)
  eta_use <- ETA

  for (i in seq_along(ref_dates)) {
    ref_date <- ref_dates[i]
    if (!inherits(ref_date, "Date")) {
      ref_date <- as.Date(ref_date, origin = "1970-01-01")
    }
    df_date <- combined %>% filter(reference_date == ref_date)
    available_models <- sort(unique(df_date$source_model))
    if (length(available_models) == 0) next

    weights <- if (all(is.na(loss_state[available_models]))) {
      setNames(rep(1 / length(available_models), length(available_models)), available_models)
    } else {
      compute_softmin_weights(loss_state, available_models, eta_use)
    }

    target_taus <- sort(CDC_QUANTILES)
    ensemble <- df_date %>%
      group_by(reference_date, target_end_date, location, output_type) %>%
      reframe(
        value = calculate_linear_pool(pick(everything()), weights, target_taus),
        output_type_id = target_taus
      ) %>%
      mutate(
        horizon = h - 1,
        target = "wk inc flu hosp"
      ) %>%
      select(reference_date, target, horizon, target_end_date, location, output_type, output_type_id, value)

    if (ref_date >= season_start) {
      key <- format(ref_date, "%Y-%m-%d")
      if (!is.null(ensembles_by_date[[key]])) {
        ensembles_by_date[[key]] <- bind_rows(ensembles_by_date[[key]], ensemble)
      } else {
        ensembles_by_date[[key]] <- ensemble
      }
    }

    weights_log[[length(weights_log) + 1]] <- tibble(
      reference_date = ref_date,
      horizon = h - 1,
      model = names(weights),
      weight = as.numeric(weights)
    )

    losses <- compute_model_losses(df_date)
    if (nrow(losses) > 0 && is.na(eta_use)) {
      median_loss <- median(losses$loss, na.rm = TRUE)
      if (is.finite(median_loss) && median_loss > 0) eta_use <- 1 / median_loss
    }
    if (nrow(losses) > 0) {
      for (row_i in seq_len(nrow(losses))) {
        mn <- losses$source_model[row_i]
        loss_val <- losses$loss[row_i]
        if (!is.finite(loss_val)) next
        if (is.na(loss_state[mn])) {
          loss_state[mn] <- loss_val
        } else {
          loss_state[mn] <- (1 - alpha) * loss_state[mn] + alpha * loss_val
        }
      }
    }
  }
}

if (length(ensembles_by_date) > 0) {
  for (key in names(ensembles_by_date)) {
    out_df <- ensembles_by_date[[key]] %>%
      arrange(reference_date, horizon, location, output_type_id)
    out_path <- file.path(OUTPUT_DIR, sprintf("AdaptiveEnsemble_hedge_retrospective_%s.csv",
                                              gsub("-", "", key)))
    write_csv(out_df, out_path)
  }
  message(sprintf("Saved %d ensemble files in %s", length(ensembles_by_date), OUTPUT_DIR))
} else {
  message("No ensemble outputs generated.")
}

if (length(weights_log) > 0) {
  weights_df <- bind_rows(weights_log)
  write_csv(weights_df, file.path(OUTPUT_DIR, "AdaptiveEnsemble_hedge_weights.csv"))
}
