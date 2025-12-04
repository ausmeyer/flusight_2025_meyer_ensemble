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

    // Model configurations
    modelPatterns: {
        'ARIMA': 'ARIMA_h{H}_prospective_{DATE}.csv',
        'SVM': 'SVM_h{H}_prospective_{DATE}.csv',
        'LGBM-blended': 'LGBM-blended_h{H}_prospective_{DATE}.csv',
        'LGBM-bounded': 'TwoStage-FrozenMu-bounded_h{H}_prospective_{DATE}.csv',
        'AdaptiveEnsemble': 'AdaptiveEnsemble_prospective_{DATE}.csv'
    },

    modelColors: {
        'AdaptiveEnsemble': '#cc5200',
        'ARIMA': '#1f77b4',
        'SVM': '#2ca02c',
        'LGBM-blended': '#9467bd',
        'LGBM-bounded': '#e377c2'
    },

    modelFills: {
        'AdaptiveEnsemble': 'rgba(204, 82, 0, 0.2)',
        'ARIMA': 'rgba(31, 119, 180, 0.2)',
        'SVM': 'rgba(44, 160, 44, 0.2)',
        'LGBM-blended': 'rgba(148, 103, 189, 0.2)',
        'LGBM-bounded': 'rgba(227, 119, 194, 0.2)'
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
     * Works both locally and on GitHub Pages
     */
    getBaseUrl() {
        const hostname = window.location.hostname;

        // If running on GitHub Pages, use raw.githubusercontent.com
        if (hostname.includes('github.io')) {
            return `https://raw.githubusercontent.com/${this.githubOwner}/${this.githubRepo}/${this.githubBranch}`;
        }

        // Local development - go up from /docs/
        const pathname = window.location.pathname;
        if (pathname.includes('/docs/')) {
            return pathname.replace(/\/docs\/.*$/, '');
        }
        return '';
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
        dates.push('20251206', '20251129', '20251122', '20251115', '20251108', '20251101');

        // Remove duplicates and sort descending
        const uniqueDates = [...new Set(dates)].sort().reverse();

        // Try to find a date with actual data
        for (const dateStr of uniqueDates) {
            const testUrl = `${this.getBaseUrl()}/forecasts/prospective/AdaptiveEnsemble_prospective_${dateStr}.csv`;
            try {
                const response = await fetch(testUrl, { method: 'HEAD' });
                if (response.ok) {
                    this.latestDate = dateStr;
                    return dateStr;
                }
            } catch (e) {
                // Continue to next date
            }
        }

        // Fallback: try ARIMA files
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
        const dates = ['2025-11-29', '2025-11-22', '2025-11-15', '2025-11-01', '2025-09-27'];

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
                    return {
                        model: modelName,
                        referenceDate: new Date(row.reference_date),
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
        const datesToTry = ['20251206', '20251129', '20251122', '20251115', '20251108'];

        for (const dateStr of datesToTry) {
            const data = await this.loadForecast(modelName, horizon, dateStr);
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

        for (const model of models) {
            if (model === 'AdaptiveEnsemble') {
                // Ensemble file contains all horizons
                loadPromises.push(
                    this.loadForecastWithFallback(model, null).then(result => {
                        if (result) {
                            this.forecasts[model] = result.data;
                            if (!latestDateFound || result.date > latestDateFound) {
                                latestDateFound = result.date;
                            }
                        }
                    })
                );
            } else {
                // Other models have separate files per horizon
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

        // Handle AdaptiveEnsemble (single file with all horizons)
        if (modelName === 'AdaptiveEnsemble' && this.forecasts['AdaptiveEnsemble']) {
            const data = this.forecasts['AdaptiveEnsemble']
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
            // Combine all horizon files for this model
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
     */
    getAvailableModels() {
        const models = new Set();

        for (const key of Object.keys(this.forecasts)) {
            if (key === 'AdaptiveEnsemble') {
                models.add('AdaptiveEnsemble');
            } else {
                // Extract model name from key like "ARIMA_h1"
                const modelName = key.replace(/_h\d$/, '');
                models.add(modelName);
            }
        }

        // Sort with AdaptiveEnsemble first, then alphabetically
        const sorted = Array.from(models).sort();
        if (sorted.includes('AdaptiveEnsemble')) {
            return ['AdaptiveEnsemble', ...sorted.filter(m => m !== 'AdaptiveEnsemble')];
        }
        return sorted;
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
