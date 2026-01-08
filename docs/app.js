/**
 * Main Application Module
 * Handles UI interactions and orchestrates data loading/chart rendering
 */

const App = {
    selectedModels: [],
    selectedStates: [],
    show95CI: true,
    show50CI: true,
    showMedian: true,
    currentTab: 'states',

    async init() {
        console.log('Initializing app...');

        // Set generated date
        document.getElementById('generated-date').textContent =
            new Date().toISOString().split('T')[0];

        // Show loading state
        this.showLoading();

        try {
            // Load data
            await DataLoader.findLatestDate();
            document.getElementById('forecast-date').textContent = DataLoader.latestDate;

            await Promise.all([
                DataLoader.loadActualData(),
                DataLoader.loadAllForecasts()
            ]);

            // Initialize UI
            this.initModels();
            this.initStates();
            this.initTabs();
            this.initCheckboxes();

            // Initial render
            this.renderCharts();

        } catch (error) {
            console.error('Initialization error:', error);
            this.showError('Failed to load forecast data. Please check the console for details.');
        }
    },

    showLoading() {
        const container = document.getElementById('state-charts-container');
        container.innerHTML = '<div class="loading">Loading forecasts...</div>';
    },

    showError(message) {
        const container = document.getElementById('state-charts-container');
        container.innerHTML = `<div class="error-message">${message}</div>`;
    },

    initModels() {
        const models = DataLoader.getAvailableModels();
        const container = document.getElementById('model-checkboxes');
        container.innerHTML = '';

        // Select only MIGHTE-Nsemble by default
        this.selectedModels = models.includes('MIGHTE-Nsemble') ? ['MIGHTE-Nsemble'] : models.slice(0, 1);

        for (const model of models) {
            const label = document.createElement('label');
            const checkbox = document.createElement('input');
            checkbox.type = 'checkbox';
            checkbox.value = model;
            checkbox.checked = this.selectedModels.includes(model);
            checkbox.addEventListener('change', () => this.onModelChange());

            const colorIndicator = document.createElement('span');
            colorIndicator.className = 'model-color-indicator';
            colorIndicator.style.backgroundColor = DataLoader.getModelColor(model);

            label.appendChild(checkbox);
            label.appendChild(colorIndicator);
            label.appendChild(document.createTextNode(' ' + model));
            container.appendChild(label);
        }

        document.getElementById('model-count').textContent = models.length;
    },

    initStates() {
        const states = DataLoader.getAllStates();
        const select = document.getElementById('state-select');
        select.innerHTML = '';

        for (const state of states) {
            const option = document.createElement('option');
            option.value = state;
            option.textContent = state;
            select.appendChild(option);
        }

        select.addEventListener('change', () => {
            this.selectedStates = Array.from(select.selectedOptions).map(o => o.value);
            this.renderCharts();
        });

        document.getElementById('clear-states').addEventListener('click', () => {
            select.selectedIndex = -1;
            this.selectedStates = [];
            this.renderCharts();
        });
    },

    initTabs() {
        const tabBtns = document.querySelectorAll('.tab-btn');
        tabBtns.forEach(btn => {
            btn.addEventListener('click', () => {
                const tab = btn.dataset.tab;
                this.switchTab(tab);
            });
        });
    },

    initCheckboxes() {
        document.getElementById('show-95ci').addEventListener('change', (e) => {
            this.show95CI = e.target.checked;
            this.renderCharts();
        });

        document.getElementById('show-50ci').addEventListener('change', (e) => {
            this.show50CI = e.target.checked;
            this.renderCharts();
        });

        document.getElementById('show-median').addEventListener('change', (e) => {
            this.showMedian = e.target.checked;
            this.renderCharts();
        });
    },

    onModelChange() {
        const checkboxes = document.querySelectorAll('#model-checkboxes input[type="checkbox"]');
        this.selectedModels = Array.from(checkboxes)
            .filter(cb => cb.checked)
            .map(cb => cb.value);
        this.renderCharts();
    },

    switchTab(tab) {
        // Update tab buttons
        document.querySelectorAll('.tab-btn').forEach(btn => {
            btn.classList.toggle('active', btn.dataset.tab === tab);
        });

        // Update tab panels
        document.querySelectorAll('.tab-panel').forEach(panel => {
            panel.classList.toggle('active', panel.id === `${tab}-tab`);
        });

        this.currentTab = tab;

        // Render charts for the new tab
        if (tab === 'us') {
            this.renderUSChart();
        } else if (tab === 'pr') {
            this.renderPRChart();
        }
    },

    getChartOptions() {
        return {
            show95CI: this.show95CI,
            show50CI: this.show50CI,
            showMedian: this.showMedian
        };
    },

    renderCharts() {
        if (this.selectedModels.length === 0) {
            document.getElementById('state-charts-container').innerHTML =
                '<div class="error-message">Please select at least one model.</div>';
            return;
        }

        if (this.currentTab === 'states') {
            this.renderStateCharts();
        } else if (this.currentTab === 'us') {
            this.renderUSChart();
        } else if (this.currentTab === 'pr') {
            this.renderPRChart();
        }
    },

    renderStateCharts() {
        const container = document.getElementById('state-charts-container');
        container.innerHTML = '';

        // Determine which states to show
        let states = this.selectedStates.length > 0
            ? this.selectedStates
            : DataLoader.getAllStates();

        const options = this.getChartOptions();

        for (const stateName of states) {
            const fips = DataLoader.locationToFips[stateName];
            if (!fips) continue;

            const wrapper = document.createElement('div');
            wrapper.className = 'chart-wrapper';

            const title = document.createElement('h3');
            title.textContent = stateName;
            wrapper.appendChild(title);

            const canvas = document.createElement('canvas');
            canvas.id = `chart-${fips}`;
            wrapper.appendChild(canvas);

            container.appendChild(wrapper);

            // Create chart after element is in DOM
            requestAnimationFrame(() => {
                ChartUtils.createForecastChart(
                    canvas.id,
                    stateName,
                    fips,
                    this.selectedModels,
                    { ...options, showLegend: false, showAxisLabels: false }
                );
            });
        }

    },

    renderUSChart() {
        if (this.selectedModels.length === 0) return;

        const options = {
            ...this.getChartOptions(),
            title: `US National Prospective Forecasts (${DataLoader.latestDate})`,
            showLegend: false,
            showAxisLabels: true,
            location: 'US',
            locationName: 'US'
        };

        // Debug: log what we're trying to render
        console.log('Rendering US chart with models:', this.selectedModels);
        for (const model of this.selectedModels) {
            const data = DataLoader.getForecastSummary(model, 'US');
            console.log(`${model} US data points:`, data.length, data.slice(0, 2));
        }

        ChartUtils.createForecastChart('us-chart', 'US', 'US', this.selectedModels, options);
    },

    renderPRChart() {
        if (this.selectedModels.length === 0) return;

        const options = {
            ...this.getChartOptions(),
            title: `Puerto Rico Prospective Forecasts (${DataLoader.latestDate})`,
            showLegend: false,
            showAxisLabels: true,
            location: '72',
            locationName: 'Puerto Rico'
        };

        ChartUtils.createForecastChart('pr-chart', 'Puerto Rico', '72', this.selectedModels, options);
    }
};

// Initialize when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    App.init();
});
