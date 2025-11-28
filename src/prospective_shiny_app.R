library(shiny)
library(dplyr)
library(readr)
library(lubridate)
library(ggplot2)
library(tidyr)

# ==============================================================================
# DATA LOADING & PREPARATION
# ==============================================================================

# Helper: latest file by pattern
latest_file <- function(dir, pattern) {
  files <- list.files(dir, pattern = pattern, full.names = TRUE)
  if (length(files) == 0) return(NA_character_)
  files[order(files)][length(files)]
}

# Helper: Process CDC quantiles into wide format (q01, q05, ..., median, ...)
process_quantiles <- function(df) {
  df %>%
    filter(output_type == 'quantile') %>%
    mutate(output_type_id = as.numeric(output_type_id),
           target_end_date = as.Date(target_end_date)) %>%
    group_by(location, target_end_date) %>%
    summarise(
      q01 = value[which.min(abs(output_type_id - 0.01))],
      q05 = value[which.min(abs(output_type_id - 0.05))],
      q25 = value[which.min(abs(output_type_id - 0.25))],
      median = value[which.min(abs(output_type_id - 0.5))],
      q75 = value[which.min(abs(output_type_id - 0.75))],
      q95 = value[which.min(abs(output_type_id - 0.95))],
      q99 = value[which.min(abs(output_type_id - 0.99))],
      .groups = 'drop'
    ) %>%
    mutate(
      q01 = pmin(q01, q05, q25, median, na.rm = TRUE),
      q05 = pmin(q05, q25, median, na.rm = TRUE),
      q25 = pmin(q25, median, na.rm = TRUE),
      q75 = pmax(q75, median, na.rm = TRUE),
      q95 = pmax(q95, q75, median, na.rm = TRUE),
      q99 = pmax(q99, q95, q75, median, na.rm = TRUE)
    )
}

# Determine paths based on working directory
if (dir.exists("data/imputed_sets")) {
  data_dir <- "data/imputed_sets"
  forecast_dir <- "forecasts/prospective"
} else if (dir.exists("../data/imputed_sets")) {
  data_dir <- "../data/imputed_sets"
  forecast_dir <- "../forecasts/prospective"
} else {
  stop("Could not locate data directory. Checked 'data/imputed_sets' and '../data/imputed_sets'.")
}

# Locate latest actuals and prospective ensemble
actual_path <- latest_file(data_dir, '^imputed_and_stitched_hosp_[0-9]{4}-[0-9]{2}-[0-9]{2}\\.csv$')
prospective_path <- latest_file(forecast_dir, '^AdaptiveEnsemble_prospective_[0-9]{8}\\.csv$')

if (is.na(actual_path)) stop(paste0('No stitched actuals found in ', data_dir))
if (is.na(prospective_path)) stop(paste0('No prospective adaptive ensemble found in ', forecast_dir))

message(paste("Loading Actuals:", actual_path))
message(paste("Loading Ensemble:", prospective_path))

actual_raw <- read_csv(actual_path, show_col_types = FALSE)
prospective <- read_csv(prospective_path, show_col_types = FALSE)

# Extract timestamp from ensemble filename
ts_match <- regexpr("[0-9]{8}", basename(prospective_path))
ts_str <- regmatches(basename(prospective_path), ts_match)

# Prepare Ground Truth (History since 2024-07-01)
actual <- actual_raw %>%
  select(location_name, date, total_hosp) %>%
  mutate(target_end_date = as.Date(date)) %>%
  filter(target_end_date >= as.Date('2024-07-01')) %>%
  arrange(location_name, target_end_date)

# FIPS Mapping
fips_to_name <- c(
  '01'='Alabama', '02'='Alaska', '04'='Arizona', '05'='Arkansas', '06'='California',
  '08'='Colorado', '09'='Connecticut', '10'='Delaware', '11'='District of Columbia',
  '12'='Florida', '13'='Georgia', '15'='Hawaii', '16'='Idaho', '17'='Illinois',
  '18'='Indiana', '19'='Iowa', '20'='Kansas', '21'='Kentucky', '22'='Louisiana',
  '23'='Maine', '24'='Maryland', '25'='Massachusetts', '26'='Michigan', '27'='Minnesota',
  '28'='Mississippi', '29'='Missouri', '30'='Montana', '31'='Nebraska', '32'='Nevada',
  '33'='New Hampshire', '34'='New Jersey', '35'='New Mexico', '36'='New York',
  '37'='North Carolina', '38'='North Dakota', '39'='Ohio', '40'='Oklahoma', '41'='Oregon',
  '42'='Pennsylvania', '72'='Puerto Rico', '44'='Rhode Island', '45'='South Carolina',
  '46'='South Dakota', '47'='Tennessee', '48'='Texas', '49'='Utah', '50'='Vermont',
  '51'='Virginia', '53'='Washington', '54'='West Virginia', '55'='Wisconsin',
  '56'='Wyoming', 'US'='US'
)

# Prepare Ensemble Quantiles
prospective_q <- process_quantiles(prospective) %>%
  mutate(location_name = fips_to_name[location], model = "Ensemble")

# Determine last actual date
last_actual <- actual %>% group_by(location_name) %>% summarise(last_date = max(target_end_date), .groups = 'drop')
prospective_q <- prospective_q %>% left_join(last_actual, by = 'location_name') %>%
  filter(target_end_date > last_date)

# Load Component Models
component_data <- list()
message(paste("Searching for components in:", forecast_dir))

for (h in 1:4) {
  find_comp <- function(p) {
    f <- latest_file(forecast_dir, p)
    if (!is.na(f)) {
      message(paste("  [FOUND]   Pattern:", p, "->", basename(f)))
    }
    return(f)
  }
  
  files <- list(
    ARIMA = find_comp(sprintf("^ARIMA_h%d_prospective_[0-9]{8}\\.csv$", h)),
    SVM   = find_comp(sprintf("^SVM_h%d_prospective_[0-9]{8}\\.csv$", h))
  )
  
  lgbm100 <- find_comp(sprintf("^TwoStage-FrozenMu-t100_h%d_prospective_[0-9]{8}\\.csv$", h))
  if(!is.na(lgbm100)) files$LGBM_t100 <- lgbm100
  
  lgbm10 <- find_comp(sprintf("^TwoStage-FrozenMu-t10_h%d_prospective_[0-9]{8}\\.csv$", h))
  if(!is.na(lgbm10)) files$LGBM_t10 <- lgbm10
  
  for (m_name in names(files)) {
    if (!is.na(files[[m_name]])) {
      df <- read_csv(files[[m_name]], show_col_types = FALSE)
      if(nrow(df) > 0) {
        # Process full quantiles for the component
        df_proc <- process_quantiles(df) %>%
          mutate(location_name = fips_to_name[as.character(location)],
                 model = m_name)
        component_data[[length(component_data)+1]] <- df_proc
      }
    }
  }
}

comp_df <- data.frame()
if(length(component_data) > 0) comp_df <- bind_rows(component_data)

if(nrow(comp_df) > 0) {
  comp_df <- comp_df %>%
    left_join(last_actual, by = 'location_name') %>%
    filter(target_end_date > last_date)
}

# Combine all models into one main dataframe for easier plotting
all_models_df <- bind_rows(prospective_q, comp_df)

# Available Models
avail_models <- unique(all_models_df$model)

# ==============================================================================
# SHINY UI
# ==============================================================================

ui <- fluidPage(
  titlePanel(paste("Prospective Forecast Viewer - ", ts_str)),
  
  sidebarLayout(
    sidebarPanel(
      selectInput("location", "Select Location:", 
                  choices = sort(unique(actual$location_name)), 
                  selected = "US"),
      
      checkboxGroupInput("show_models", "Models to Show:",
                         choices = avail_models,
                         selected = avail_models),
      
      checkboxGroupInput("intervals", "Intervals to Show:",
                         choices = c("50%" = "50", "90%" = "90", "99%" = "99"),
                         selected = c("50", "90", "99")),
      
      hr(),
      helpText("History shown since 2024-07-01.")
    ),
    
    mainPanel(
      plotOutput("forecastPlot", height = "600px")
    )
  )
)

# ==============================================================================
# SHINY SERVER
# ==============================================================================

server <- function(input, output) {
  
  output$forecastPlot <- renderPlot({
    loc <- input$location
    
    # Filter Data
    dat_actual <- actual %>% filter(location_name == loc)
    dat_models <- all_models_df %>%
      filter(location_name == loc, model %in% input$show_models)
    
    # Colors
    cols <- c("Ensemble" = "#cc5200", "ARIMA" = "#1f77b4", "SVM" = "#2ca02c", 
              "LGBM_t100" = "#9467bd", "LGBM_t10" = "#8c564b")
    
    # Base Plot
    p <- ggplot() +
      # Ground Truth
      geom_line(data = dat_actual, aes(x = target_end_date, y = total_hosp), color = "black", linewidth = 0.5) +
      geom_point(data = dat_actual, aes(x = target_end_date, y = total_hosp), size = 1) +
      theme_bw(base_size = 16) +
      labs(title = paste(loc, "- Weekly Hospitalizations"),
           x = "Date", y = "Hospitalizations")
    
    # Loop through models to add layers in consistent order
    # We use alpha transparency to show overlapping intervals
    
    # 99% Intervals
    if ("99" %in% input$intervals) {
      p <- p + geom_ribbon(data = dat_models, 
                           aes(x = target_end_date, ymin = q01, ymax = q99, fill = model, group = model), 
                           alpha = 0.1)
    }
    
    # 90% Intervals
    if ("90" %in% input$intervals) {
      p <- p + geom_ribbon(data = dat_models, 
                           aes(x = target_end_date, ymin = q05, ymax = q95, fill = model, group = model), 
                           alpha = 0.15)
    }
    
    # 50% Intervals
    if ("50" %in% input$intervals) {
      p <- p + geom_ribbon(data = dat_models, 
                           aes(x = target_end_date, ymin = q25, ymax = q75, fill = model, group = model), 
                           alpha = 0.2)
    }
    
    # Medians
    p <- p + geom_line(data = dat_models, 
                       aes(x = target_end_date, y = median, color = model, group = model), 
                       linewidth = 1)
    
    # Scales
    p <- p + 
      scale_color_manual(name = "Model", values = cols) +
      scale_fill_manual(name = "Model", values = cols)
    
    print(p)
  })
}

# Run the application 
shinyApp(ui = ui, server = server)
