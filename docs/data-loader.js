/**
 * Data Loader Module
 * Handles loading and parsing of forecast CSV files from GitHub Pages
 */

const DataLoader = {
    // FIPS code mappings
    locationToFips: {
        'Alabama': '01', 'Alaska': '02', 'Arizona': '04', 'Arkansas': '05',
        'California': '06', 'Colorado': '08', 'Connecticut': '09', 'Delaware': '10',
        'District of Columbia': '11', 'Florida': '12', 'Georgia': '13', 'Hawaii': '15',
        'Idaho': '16', 'Illinois': '17', 'Indiana': '18', 'Iowa': '19',
        'Kansas': '20', 'Kentucky': '21', 'Louisiana': '22', 'Maine': '23',
        'Maryland': '24', 'Massachusetts': '25', 'Michigan': '26', 'Minnesota': '27',
        'Mississippi': '28', 'Missouri': '29', 'Montana': '30', 'Nebraska': '31',
        'Nevada': '32', 'New Hampshire': '33', 'New Jersey': '34', 'New Mexico': '35',
        'New York': '36', 'North Carolina': '37', 'North Dakota': '38', 'Ohio': '39',
        'Oklahoma': '40', 'Oregon': '41', 'Pennsylvania': '42', 'Puerto Rico': '72',
        'Rhode Island': '44', 'South Carolina': '45', 'South Dakota': '46',
        'Tennessee': '47', 'Texas': '48', 'Utah': '49', 'Vermont': '50',
        'Virginia': '51', 'Washington': '53', 'West Virginia': '54',
        'Wisconsin': '55', 'Wyoming': '56', 'US': 'US'
    },

    fipsToName: {},

    // Local model configurations (loaded from this repo)
    modelPatterns: {
        'ARIMA': 'ARIMA_h{H}_prospective_{DATE}.csv',
        'SVM': 'SVM_h{H}_prospective_{DATE}.csv',
        'LGBM-blended': 'LGBM-blended_h{H}_prospective_{DATE}.csv',
        'LGBM-bounded': 'TwoStage-FrozenMu-bounded_h{H}_prospective_{DATE}.csv',
        'LGBM-bounded-wide-1': 'TwoStage-FrozenMu-bounded-wide-1_h{H}_prospective_{DATE}.csv',
        'LGBM-bounded-wide-2': 'TwoStage-FrozenMu-bounded-wide-2_h{H}_prospective_{DATE}.csv',
        'LGBM-bounded-wide-3': 'TwoStage-FrozenMu-bounded-wide-3_h{H}_prospective_{DATE}.csv'
    },

    // List of local models (for ordering - these appear first in the list)
    localModels: [
        'ARIMA', 'LGBM-blended', 'LGBM-bounded',
        'LGBM-bounded-wide-1', 'LGBM-bounded-wide-2', 'LGBM-bounded-wide-3', 'SVM'
    ],

    // GitHub CDC FluSight models (loaded from cdcepi/FluSight-forecast-hub)
    githubModels: [
        'CADPH-FluCAT_Ensemble', 'CEPH-Rtrend_fluH', 'CFA_Pyrenew-Pyrenew_E_Flu',
        'CFA_Pyrenew-Pyrenew_HE_Flu', 'CFA_Pyrenew-Pyrenew_H_Flu', 'CMU-TimeSeries',
        'CMU-climate_baseline', 'CU-ARNB_Net', 'CU-ensemble', 'Cornell_JHU-hierarchSIR',
        'FluSight-HJudge_ensemble', 'FluSight-base_seasonal', 'FluSight-baseline',
        'FluSight-baseline_cat', 'FluSight-ens_q_cat', 'FluSight-ensemble',
        'FluSight-equal_cat', 'FluSight-lop_norm', 'FluSight-national_cat',
        'FluSight-trained_mean', 'FluSight-trained_med', 'GH-model', 'GT-FluFNP',
        'Gatech-ensemble_point', 'Gatech-ensemble_prob', 'Gatech-ensemble_stat',
        'Google_SAI-FluBoostQR', 'Google_SAI-FluEns', 'ISU_NiemiLab-ENS',
        'ISU_NiemiLab-GPE', 'ISU_NiemiLab-NLH', 'ISU_NiemiLab-SIR', 'JHUAPL-DMD',
        'JHUAPL-Morris', 'JHU_CSSE-CSSE_Ensemble', 'LUcompUncertLab-chimera',
        'LosAlamos-DoSiDo', 'LosAlamos-ThinMint', 'LosAlamos_NAU-CModel_Flu',
        'MDPredict-SIRS', 'MIGHTE-Joint', 'MIGHTE-Nsemble', 'MOBS-EpyStrain_Flu',
        'MOBS-GLEAM_FLUH', 'MOBS-GLEAM_RL_FLUH', 'Metaculus-cp', 'NAU-FourCAT',
        'NAU-epymorph', 'NAU-vulPES', 'NEU_ISI-AdaptiveEnsemble', 'NEU_ISI-FluBcast',
        'NIH-Flu_ARIMA', 'NU-PGF_FLUH', 'NU_UCSD-GLEAM_AI_FLUH', 'OHT_JHU-nbxd',
        'PSI-PROF', 'PSI-PROF_MOA', 'PSI-PROF_beta', 'SGroup-RandomForest',
        'SigSci-BECAM', 'SigSci-CREG', 'SigSci-TSENS', 'Stevens-GBR',
        'Stevens-ILIForecast', 'UGA_CEID-Walk', 'UGA_CEID-auto_AVG_LB',
        'UGA_flucast-Copycat', 'UGA_flucast-INFLAenza', 'UGA_flucast-OKeeffe',
        'UGA_flucast-Scenariocast', 'UGuelph-CompositeCurve', 'UGuelphensemble-GRYPHON',
        'UI_CompEpi-EpiGen', 'UM-DeepOutbreak', 'UMass-AR2', 'UMass-flusion',
        'UMass-trends_ensemble', 'UNC_IDD-InfluPaint', 'UVAFluX-CESGCN',
        'UVAFluX-Ensemble', 'UVAFluX-FS_OptimWISE', 'UVAFluX-OptimWISE',
        'VTSanghani-Ensemble', 'VTSanghani-PRIME', 'cfa-flumech',
        'cfarenewal-cfaepimlight', 'fjordhest-ensemble'
    ],

    // GitHub base URL for CDC FluSight models
    cdcGithubBaseUrl: 'https://raw.githubusercontent.com/cdcepi/FluSight-forecast-hub/main/model-output',

    // Base colors for local models
    modelColors: {
        'ARIMA': '#1f77b4',
        'SVM': '#2ca02c',
        'LGBM-blended': '#9467bd',
        'LGBM-bounded': '#e377c2',
        'LGBM-bounded-wide-1': '#8c564b',
        'LGBM-bounded-wide-2': '#17becf',
        'LGBM-bounded-wide-3': '#bcbd22',
        'MIGHTE-Nsemble': '#cc5200'
    },

    modelFills: {
        'ARIMA': 'rgba(31, 119, 180, 0.2)',
        'SVM': 'rgba(44, 160, 44, 0.2)',
        'LGBM-blended': 'rgba(148, 103, 189, 0.2)',
        'LGBM-bounded': 'rgba(227, 119, 194, 0.2)',
        'LGBM-bounded-wide-1': 'rgba(140, 86, 75, 0.2)',
        'LGBM-bounded-wide-2': 'rgba(23, 190, 207, 0.2)',
        'LGBM-bounded-wide-3': 'rgba(188, 189, 34, 0.2)',
        'MIGHTE-Nsemble': 'rgba(204, 82, 0, 0.2)'
    },

    // Generate a color based on model name (deterministic hash)
    getModelColor(modelName) {
        if (this.modelColors[modelName]) {
            return this.modelColors[modelName];
        }
        // Generate a deterministic color based on model name hash
        let hash = 0;
        for (let i = 0; i < modelName.length; i++) {
            hash = modelName.charCodeAt(i) + ((hash << 5) - hash);
        }
        const h = Math.abs(hash) % 360;
        const s = 50 + (Math.abs(hash >> 8) % 30);  // 50-80% saturation
        const l = 40 + (Math.abs(hash >> 16) % 20); // 40-60% lightness
        return `hsl(${h}, ${s}%, ${l}%)`;
    },

    getModelFill(modelName) {
        if (this.modelFills[modelName]) {
            return this.modelFills[modelName];
        }
        // Generate fill color based on the line color
        const color = this.getModelColor(modelName);
        if (color.startsWith('hsl')) {
            // Extract HSL values and add alpha
            const match = color.match(/hsl\((\d+),\s*(\d+)%,\s*(\d+)%\)/);
            if (match) {
                return `hsla(${match[1]}, ${match[2]}%, ${match[3]}%, 0.2)`;
            }
        }
        return 'rgba(128, 128, 128, 0.2)';
    },

    // Quantile levels we care about
    targetQuantiles: [0.025, 0.25, 0.5, 0.75, 0.975],

    // Data storage
    forecasts: {},
    actualData: [],
    latestDate: null,

    init() {
        // Build reverse mapping
        for (const [name, fips] of Object.entries(this.locationToFips)) {
            this.fipsToName[fips] = name;
        }
    },

    // GitHub repository info for raw file access
    githubOwner: 'ausmeyer',
    githubRepo: 'flusight_2025_meyer_ensemble',
    githubBranch: 'main',

    /**
     * Get the base URL for data files
     * Works both locally and on GitHub Pages / custom domains
     */
    getBaseUrl() {
        const hostname = window.location.hostname;

        // Local development (localhost or 127.0.0.1)
        if (hostname === 'localhost' || hostname === '127.0.0.1') {
            const pathname = window.location.pathname;
            if (pathname.includes('/docs/')) {
                return pathname.replace(/\/docs\/.*$/, '');
            }
            return '';
        }

        // Any hosted environment (GitHub Pages, custom domain, etc.) - use raw GitHub
        return `https://raw.githubusercontent.com/${this.githubOwner}/${this.githubRepo}/${this.githubBranch}`;
    },

    /**
     * Find the latest date from available forecast files
     */
    async findLatestDate() {
        // Try fetching the file listing from GitHub API or use known dates
        // For static GitHub Pages, we'll check a few recent dates
        const today = new Date();
        const dates = [];

        // Generate last 8 Saturdays (forecast dates are typically Saturdays)
        for (let i = 0; i < 12; i++) {
            const d = new Date(today);
            d.setDate(d.getDate() - (d.getDay() + 1 + i * 7) % 7 - i * 7);
            // Adjust to get Saturday
            const dayOfWeek = d.getDay();
            const diff = (dayOfWeek === 0) ? -1 : (6 - dayOfWeek);
            d.setDate(d.getDate() + diff);
            dates.push(this.formatDateForFile(d));
        }

        // Also try recent specific dates we know exist
        dates.push('20251227', '20251220', '20251213', '20251206', '20251129', '20251122', '20251115', '20251108');

        // Remove duplicates and sort descending
        const uniqueDates = [...new Set(dates)].sort().reverse();

        // Try to find a date with actual data (use ARIMA as it's a local model)
        for (const dateStr of uniqueDates) {
            const testUrl = `${this.getBaseUrl()}/forecasts/prospective/ARIMA_h1_prospective_${dateStr}.csv`;
            try {
                const response = await fetch(testUrl, { method: 'HEAD' });
                if (response.ok) {
                    this.latestDate = dateStr;
                    return dateStr;
                }
            } catch (e) {
                // Continue
            }
        }

        throw new Error('Could not find any forecast files');
    },

    formatDateForFile(date) {
        const y = date.getFullYear();
        const m = String(date.getMonth() + 1).padStart(2, '0');
        const d = String(date.getDate()).padStart(2, '0');
        return `${y}${m}${d}`;
    },

    /**
     * Load actual/ground truth data
     */
    async loadActualData() {
        const baseUrl = this.getBaseUrl();

        // Try to find the latest imputed data file
        const dates = ['2025-12-27', '2025-12-20', '2025-12-13', '2025-12-06', '2025-11-29', '2025-11-22', '2025-11-15', '2025-11-01'];

        for (const dateStr of dates) {
            const url = `${baseUrl}/data/imputed_sets/imputed_and_stitched_hosp_${dateStr}.csv`;
            try {
                const response = await fetch(url);
                if (response.ok) {
                    const text = await response.text();
                    const parsed = Papa.parse(text, { header: true, skipEmptyLines: true });

                    this.actualData = parsed.data
                        .filter(row => row.location_name && row.date && row.total_hosp !== 'NA')
                        .map(row => ({
                            location: this.locationToFips[row.location_name] || row.location_name,
                            locationName: row.location_name,
                            date: new Date(row.date),
                            value: parseFloat(row.total_hosp)
                        }))
                        .filter(row => !isNaN(row.value) && row.date >= new Date('2024-07-01'));

                    console.log(`Loaded ${this.actualData.length} actual data points`);
                    return;
                }
            } catch (e) {
                console.warn(`Could not load ${url}:`, e);
            }
        }

        console.warn('Could not load any actual data files');
    },

    /**
     * Load forecast data for a specific model and horizon
     */
    async loadForecast(modelName, horizon, dateStr) {
        const baseUrl = this.getBaseUrl();
        let url;

        if (modelName === 'AdaptiveEnsemble') {
            url = `${baseUrl}/forecasts/prospective/AdaptiveEnsemble_prospective_${dateStr}.csv`;
        } else {
            const pattern = this.modelPatterns[modelName];
            if (!pattern) return null;
            const filename = pattern.replace('{H}', horizon).replace('{DATE}', dateStr);
            url = `${baseUrl}/forecasts/prospective/${filename}`;
        }

        try {
            const response = await fetch(url);
            if (!response.ok) return null;

            const text = await response.text();
            const parsed = Papa.parse(text, { header: true, skipEmptyLines: true });

            return parsed.data
                .filter(row => row.output_type === 'quantile')
                .map(row => {
                    // Handle location - US stays as US, numbers get padded
                    let location = String(row.location);
                    if (location !== 'US' && !isNaN(parseInt(location))) {
                        location = location.padStart(2, '0');
                    }
                    // Base models use data cutoff date as reference_date, but CDC convention
                    // uses the Saturday after (+7 days). AdaptiveEnsemble already uses CDC convention.
                    let refDate = new Date(row.reference_date);
                    if (modelName !== 'AdaptiveEnsemble') {
                        refDate.setDate(refDate.getDate() + 7);
                    }
                    return {
                        model: modelName,
                        referenceDate: refDate,
                        horizon: modelName === 'AdaptiveEnsemble' ? parseInt(row.horizon) + 1 : horizon,
                        targetEndDate: new Date(row.target_end_date),
                        location: location,
                        quantile: parseFloat(row.output_type_id),
                        value: parseFloat(row.value)
                    };
                })
                .filter(row => !isNaN(row.value));
        } catch (e) {
            console.warn(`Error loading ${url}:`, e);
            return null;
        }
    },

    /**
     * Try to load a forecast, falling back to older dates if needed
     */
    async loadForecastWithFallback(modelName, horizon) {
        // Try multiple dates in descending order
        const datesToTry = ['20251227', '20251220', '20251213', '20251206', '20251129', '20251122', '20251115', '20251108'];

        for (const dateStr of datesToTry) {
            const data = await this.loadForecast(modelName, horizon, dateStr);
            if (data && data.length > 0) {
                return { data, date: dateStr };
            }
        }
        return null;
    },

    /**
     * Load forecast from CDC GitHub for a specific model and date
     */
    async loadGithubForecast(modelName, dateStr) {
        // Convert YYYYMMDD to YYYY-MM-DD format for CDC URLs
        const cdcDate = `${dateStr.slice(0,4)}-${dateStr.slice(4,6)}-${dateStr.slice(6,8)}`;
        const url = `${this.cdcGithubBaseUrl}/${modelName}/${cdcDate}-${modelName}.csv`;

        try {
            const response = await fetch(url);
            if (!response.ok) return null;

            const text = await response.text();
            const parsed = Papa.parse(text, { header: true, skipEmptyLines: true });

            return parsed.data
                .filter(row => row.output_type === 'quantile' &&
                               row.target === 'wk inc flu hosp' &&
                               parseInt(row.horizon) >= 0)
                .map(row => {
                    let location = String(row.location);
                    if (location !== 'US' && !isNaN(parseInt(location))) {
                        location = location.padStart(2, '0');
                    }
                    return {
                        model: modelName,
                        referenceDate: new Date(row.reference_date),
                        horizon: parseInt(row.horizon) + 1,
                        targetEndDate: new Date(row.target_end_date),
                        location: location,
                        quantile: parseFloat(row.output_type_id),
                        value: parseFloat(row.value)
                    };
                })
                .filter(row => !isNaN(row.value));
        } catch (e) {
            // Silently fail for GitHub models - they may not have all dates
            return null;
        }
    },

    /**
     * Try to load a GitHub forecast, falling back to older dates if needed
     * Only returns data if forecast exists since 2025-11-01
     */
    async loadGithubForecastWithFallback(modelName) {
        // Only try dates since 11/1/2025 (2025-11-01)
        const datesToTry = ['20251227', '20251220', '20251213', '20251206', '20251129', '20251122', '20251115', '20251108', '20251101'];

        for (const dateStr of datesToTry) {
            const data = await this.loadGithubForecast(modelName, dateStr);
            if (data && data.length > 0) {
                return { data, date: dateStr };
            }
        }
        return null;
    },

    /**
     * Load all forecasts, trying multiple dates for each model
     */
    async loadAllForecasts() {
        const models = Object.keys(this.modelPatterns);
        const horizons = [1, 2, 3, 4];
        let latestDateFound = null;

        const loadPromises = [];

        // Load local models (all have separate files per horizon)
        for (const model of models) {
            for (const h of horizons) {
                loadPromises.push(
                    this.loadForecastWithFallback(model, h).then(result => {
                        if (result) {
                            const key = `${model}_h${h}`;
                            this.forecasts[key] = result.data;
                            if (!latestDateFound || result.date > latestDateFound) {
                                latestDateFound = result.date;
                            }
                        }
                    })
                );
            }
        }

        // Load GitHub models (all contain all horizons in one file)
        for (const model of this.githubModels) {
            loadPromises.push(
                this.loadGithubForecastWithFallback(model).then(result => {
                    if (result) {
                        this.forecasts[model] = result.data;
                    }
                })
            );
        }

        await Promise.all(loadPromises);

        if (latestDateFound) {
            this.latestDate = latestDateFound;
        }

        console.log('Loaded forecasts:', Object.keys(this.forecasts));
    },

    /**
     * Get summarized forecast data for a specific model and location
     * Combines all horizons and extracts key quantiles
     */
    getForecastSummary(modelName, location) {
        const results = [];

        // Check if model has a single combined file (AdaptiveEnsemble or GitHub models)
        const hasCombinedFile = this.forecasts[modelName] !== undefined;

        if (hasCombinedFile) {
            const data = this.forecasts[modelName]
                .filter(row => row.location === location);

            // Group by target date
            const byDate = {};
            for (const row of data) {
                const dateKey = row.targetEndDate.toISOString();
                if (!byDate[dateKey]) {
                    byDate[dateKey] = { date: row.targetEndDate, quantiles: {} };
                }
                byDate[dateKey].quantiles[row.quantile] = row.value;
            }

            for (const entry of Object.values(byDate)) {
                results.push({
                    date: entry.date,
                    q025: this.findClosestQuantile(entry.quantiles, 0.025),
                    q25: this.findClosestQuantile(entry.quantiles, 0.25),
                    median: this.findClosestQuantile(entry.quantiles, 0.5),
                    q75: this.findClosestQuantile(entry.quantiles, 0.75),
                    q975: this.findClosestQuantile(entry.quantiles, 0.975)
                });
            }
        } else {
            // Combine all horizon files for this model (local models with separate h files)
            for (let h = 1; h <= 4; h++) {
                const key = `${modelName}_h${h}`;
                if (!this.forecasts[key]) continue;

                const data = this.forecasts[key]
                    .filter(row => row.location === location);

                // Group by target date
                const byDate = {};
                for (const row of data) {
                    const dateKey = row.targetEndDate.toISOString();
                    if (!byDate[dateKey]) {
                        byDate[dateKey] = { date: row.targetEndDate, quantiles: {} };
                    }
                    byDate[dateKey].quantiles[row.quantile] = row.value;
                }

                for (const entry of Object.values(byDate)) {
                    results.push({
                        date: entry.date,
                        q025: this.findClosestQuantile(entry.quantiles, 0.025),
                        q25: this.findClosestQuantile(entry.quantiles, 0.25),
                        median: this.findClosestQuantile(entry.quantiles, 0.5),
                        q75: this.findClosestQuantile(entry.quantiles, 0.75),
                        q975: this.findClosestQuantile(entry.quantiles, 0.975)
                    });
                }
            }
        }

        // Sort by date and ensure quantile ordering
        results.sort((a, b) => a.date - b.date);

        // Fix quantile ordering (lower bounds should be <= median <= upper bounds)
        for (const r of results) {
            r.q025 = Math.min(r.q025, r.q25, r.median);
            r.q25 = Math.min(r.q25, r.median);
            r.q75 = Math.max(r.q75, r.median);
            r.q975 = Math.max(r.q975, r.q75, r.median);
        }

        return results;
    },

    findClosestQuantile(quantiles, target) {
        const keys = Object.keys(quantiles).map(Number);
        if (keys.length === 0) return null;

        let closest = keys[0];
        let minDiff = Math.abs(keys[0] - target);

        for (const k of keys) {
            const diff = Math.abs(k - target);
            if (diff < minDiff) {
                minDiff = diff;
                closest = k;
            }
        }

        return quantiles[closest];
    },

    /**
     * Get actual data for a specific location
     */
    getActualData(location) {
        return this.actualData
            .filter(row => row.location === location)
            .sort((a, b) => a.date - b.date);
    },

    /**
     * Get list of available models
     * Returns local models first (in defined order), then GitHub models alphabetically
     */
    getAvailableModels() {
        const loadedModels = new Set();

        for (const key of Object.keys(this.forecasts)) {
            // Check if it's a combined file (direct model name) or horizon-split file (model_h1)
            if (key.match(/_h\d$/)) {
                // Extract model name from key like "ARIMA_h1"
                const modelName = key.replace(/_h\d$/, '');
                loadedModels.add(modelName);
            } else {
                // Combined file - key is the model name
                loadedModels.add(key);
            }
        }

        // Separate into local and GitHub models
        const localLoaded = this.localModels.filter(m => loadedModels.has(m));
        const githubLoaded = this.githubModels.filter(m => loadedModels.has(m)).sort();

        // Return local models first (in defined order), then GitHub models alphabetically
        return [...localLoaded, ...githubLoaded];
    },

    /**
     * Get all state names (excluding US and PR)
     */
    getAllStates() {
        return Object.keys(this.locationToFips)
            .filter(name => name !== 'US' && name !== 'Puerto Rico')
            .sort();
    }
};

// Initialize on load
DataLoader.init();
