#!/usr/bin/env bash
set -euo pipefail

# Backtest using existing prospective base model forecasts
# Writes multiple ensemble variants to forecasts/retrospective/*

ROOT_DIR=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT_DIR"

SEASON_START=""
SEASON_END=""
LOOKBACK_WEEKS=""
HISTORY_WEEKS=""
ETA=""
SHRINK_FACTOR=""
LOC_LOOKBACK_WEEKS=""
LOC_HISTORY_WEEKS=""
LOC_MIN_WEEKS=""
LOC_BASE_SHRINK=""
INCLUDE_ARIMA=""
INCLUDE_SVM=""
INCLUDE_LGBM_BLENDED=""
INCLUDE_LGBM_BOUNDED=""
INCLUDE_LGBM_BOUNDED_WIDE_1=""
INCLUDE_LGBM_BOUNDED_WIDE_2=""
INCLUDE_LGBM_BOUNDED_WIDE_3=""
INCLUDE_LGBM_BOUNDED_WIDE_4=""
INCLUDE_LGBM_BOUNDED_WIDE_5=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --season-start)
      SEASON_START="$2"; shift 2;;
    --season-end)
      SEASON_END="$2"; shift 2;;
    --lookback)
      LOOKBACK_WEEKS="$2"; shift 2;;
    --history)
      HISTORY_WEEKS="$2"; shift 2;;
    --eta)
      ETA="$2"; shift 2;;
    --shrink)
      SHRINK_FACTOR="$2"; shift 2;;
    --loc-lookback)
      LOC_LOOKBACK_WEEKS="$2"; shift 2;;
    --loc-history)
      LOC_HISTORY_WEEKS="$2"; shift 2;;
    --loc-min-weeks)
      LOC_MIN_WEEKS="$2"; shift 2;;
    --loc-base-shrink)
      LOC_BASE_SHRINK="$2"; shift 2;;
    --include-arima)
      INCLUDE_ARIMA="$2"; shift 2;;
    --include-svm)
      INCLUDE_SVM="$2"; shift 2;;
    --include-lgbm-blended)
      INCLUDE_LGBM_BLENDED="$2"; shift 2;;
    --include-lgbm-bounded)
      INCLUDE_LGBM_BOUNDED="$2"; shift 2;;
    --include-lgbm-bounded-wide-1)
      INCLUDE_LGBM_BOUNDED_WIDE_1="$2"; shift 2;;
    --include-lgbm-bounded-wide-2)
      INCLUDE_LGBM_BOUNDED_WIDE_2="$2"; shift 2;;
    --include-lgbm-bounded-wide-3)
      INCLUDE_LGBM_BOUNDED_WIDE_3="$2"; shift 2;;
    --include-lgbm-bounded-wide-4)
      INCLUDE_LGBM_BOUNDED_WIDE_4="$2"; shift 2;;
    --include-lgbm-bounded-wide-5)
      INCLUDE_LGBM_BOUNDED_WIDE_5="$2"; shift 2;;
    *)
      shift;;
  esac
done

ARGS=()
if [[ -n "$SEASON_START" ]]; then ARGS+=(--season-start "$SEASON_START"); fi
if [[ -n "$SEASON_END" ]]; then ARGS+=(--season-end "$SEASON_END"); fi
if [[ -n "$INCLUDE_ARIMA" ]]; then ARGS+=(--include-arima "$INCLUDE_ARIMA"); fi
if [[ -n "$INCLUDE_SVM" ]]; then ARGS+=(--include-svm "$INCLUDE_SVM"); fi
if [[ -n "$INCLUDE_LGBM_BLENDED" ]]; then ARGS+=(--include-lgbm-blended "$INCLUDE_LGBM_BLENDED"); fi
if [[ -n "$INCLUDE_LGBM_BOUNDED" ]]; then ARGS+=(--include-lgbm-bounded "$INCLUDE_LGBM_BOUNDED"); fi
if [[ -n "$INCLUDE_LGBM_BOUNDED_WIDE_1" ]]; then ARGS+=(--include-lgbm-bounded-wide-1 "$INCLUDE_LGBM_BOUNDED_WIDE_1"); fi
if [[ -n "$INCLUDE_LGBM_BOUNDED_WIDE_2" ]]; then ARGS+=(--include-lgbm-bounded-wide-2 "$INCLUDE_LGBM_BOUNDED_WIDE_2"); fi
if [[ -n "$INCLUDE_LGBM_BOUNDED_WIDE_3" ]]; then ARGS+=(--include-lgbm-bounded-wide-3 "$INCLUDE_LGBM_BOUNDED_WIDE_3"); fi
if [[ -n "$INCLUDE_LGBM_BOUNDED_WIDE_4" ]]; then ARGS+=(--include-lgbm-bounded-wide-4 "$INCLUDE_LGBM_BOUNDED_WIDE_4"); fi
if [[ -n "$INCLUDE_LGBM_BOUNDED_WIDE_5" ]]; then ARGS+=(--include-lgbm-bounded-wide-5 "$INCLUDE_LGBM_BOUNDED_WIDE_5"); fi

VARIANT_ARGS=("${ARGS[@]}")
if [[ -n "$LOOKBACK_WEEKS" ]]; then VARIANT_ARGS+=(--lookback-weeks "$LOOKBACK_WEEKS"); fi
if [[ -n "$HISTORY_WEEKS" ]]; then VARIANT_ARGS+=(--history-weeks "$HISTORY_WEEKS"); fi
if [[ -n "$SHRINK_FACTOR" ]]; then VARIANT_ARGS+=(--shrink "$SHRINK_FACTOR"); fi
if [[ -n "$LOC_LOOKBACK_WEEKS" ]]; then VARIANT_ARGS+=(--loc-lookback-weeks "$LOC_LOOKBACK_WEEKS"); fi
if [[ -n "$LOC_HISTORY_WEEKS" ]]; then VARIANT_ARGS+=(--loc-history-weeks "$LOC_HISTORY_WEEKS"); fi
if [[ -n "$LOC_MIN_WEEKS" ]]; then VARIANT_ARGS+=(--loc-min-weeks "$LOC_MIN_WEEKS"); fi
if [[ -n "$LOC_BASE_SHRINK" ]]; then VARIANT_ARGS+=(--loc-base-shrink "$LOC_BASE_SHRINK"); fi

Rscript src/backtest_prosp_ensemble_variants.R "${VARIANT_ARGS[@]}"

HEDGE_ARGS=("${ARGS[@]}")
if [[ -n "$LOOKBACK_WEEKS" ]]; then HEDGE_ARGS+=(--lookback-weeks "$LOOKBACK_WEEKS"); fi
if [[ -n "$ETA" ]]; then HEDGE_ARGS+=(--eta "$ETA"); fi

Rscript src/backtest_prosp_ensemble_hedge.R "${HEDGE_ARGS[@]}"
