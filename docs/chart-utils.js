/**
 * Chart Utilities Module
 * Provides helper functions for creating and managing Chart.js charts
 */

const ChartUtils = {
    // Store chart instances for cleanup
    charts: {},

    /**
     * Create a forecast chart for a specific location
     */
    createForecastChart(canvasId, locationName, location, models, options = {}) {
        const ctx = document.getElementById(canvasId);
        if (!ctx) return null;

        // Destroy existing chart if any
        if (this.charts[canvasId]) {
            this.charts[canvasId].destroy();
        }

        const datasets = [];

        // Add actual data (ground truth)
        const actualData = DataLoader.getActualData(location);
        if (actualData.length > 0) {
            datasets.push({
                label: 'Ground Truth',
                data: actualData.map(d => ({ x: d.date, y: d.value })),
                borderColor: '#000000',
                backgroundColor: 'transparent',
                borderWidth: 1.5,
                pointRadius: 0,
                tension: 0,
                order: 0 // Draw on top
            });
        }

        // Add forecast data for each selected model
        for (const model of models) {
            const forecastData = DataLoader.getForecastSummary(model, location);
            if (forecastData.length === 0) continue;

            const color = DataLoader.getModelColor(model);
            const fillColor = DataLoader.getModelFill(model);

            // 95% CI band (if enabled)
            if (options.show95CI !== false) {
                datasets.push({
                    label: `${model} 95% CI`,
                    data: forecastData.map(d => ({ x: d.date, y: d.q975 })),
                    borderColor: 'transparent',
                    backgroundColor: 'transparent',
                    pointRadius: 0,
                    fill: false,
                    order: 10
                });
                datasets.push({
                    label: `${model} 95% CI Lower`,
                    data: forecastData.map(d => ({ x: d.date, y: d.q025 })),
                    borderColor: 'transparent',
                    backgroundColor: fillColor,
                    pointRadius: 0,
                    fill: '-1', // Fill to previous dataset
                    order: 10
                });
            }

            // 50% CI band (if enabled)
            if (options.show50CI !== false) {
                // Handle both rgba and hsla formats
                let fillColor50 = fillColor;
                if (fillColor.includes('0.2)')) {
                    fillColor50 = fillColor.replace('0.2)', '0.35)');
                } else if (fillColor.includes('hsla')) {
                    fillColor50 = fillColor.replace(/,\s*0\.2\)$/, ', 0.35)');
                }
                datasets.push({
                    label: `${model} 50% CI`,
                    data: forecastData.map(d => ({ x: d.date, y: d.q75 })),
                    borderColor: 'transparent',
                    backgroundColor: 'transparent',
                    pointRadius: 0,
                    fill: false,
                    order: 5
                });
                datasets.push({
                    label: `${model} 50% CI Lower`,
                    data: forecastData.map(d => ({ x: d.date, y: d.q25 })),
                    borderColor: 'transparent',
                    backgroundColor: fillColor50,
                    pointRadius: 0,
                    fill: '-1',
                    order: 5
                });
            }

            // Median line (if enabled)
            if (options.showMedian !== false) {
                datasets.push({
                    label: model,
                    data: forecastData.map(d => ({ x: d.date, y: d.median })),
                    borderColor: color,
                    backgroundColor: color,
                    borderWidth: 2,
                    pointRadius: 3,
                    pointHoverRadius: 5,
                    tension: 0,
                    order: 1
                });
            }
        }

        // Determine date range
        let minDate = new Date('2024-07-01');
        let maxDate = new Date();
        maxDate.setDate(maxDate.getDate() + 35); // Add 5 weeks for forecasts

        const chart = new Chart(ctx, {
            type: 'line',
            data: { datasets },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                interaction: {
                    mode: 'nearest',
                    intersect: false
                },
                plugins: {
                    title: {
                        display: !!options.title,
                        text: options.title || '',
                        font: { size: 14 }
                    },
                    legend: {
                        display: false
                    },
                    tooltip: {
                        callbacks: {
                            label: (context) => {
                                const label = context.dataset.label || '';
                                const value = context.parsed.y;
                                if (label.includes('CI')) return null;
                                return `${label}: ${value.toFixed(1)}`;
                            }
                        },
                        filter: (item) => !item.dataset.label?.includes('CI')
                    }
                },
                scales: {
                    x: {
                        type: 'time',
                        time: {
                            unit: 'month',
                            displayFormats: {
                                month: 'MMM yyyy'
                            }
                        },
                        min: minDate,
                        max: maxDate,
                        title: {
                            display: options.showAxisLabels !== false,
                            text: 'Date'
                        },
                        ticks: {
                            maxRotation: 45,
                            minRotation: 45
                        }
                    },
                    y: {
                        beginAtZero: true,
                        title: {
                            display: options.showAxisLabels !== false,
                            text: 'Weekly Hospitalizations'
                        }
                    }
                }
            }
        });

        this.charts[canvasId] = chart;
        return chart;
    },

    /**
     * Update an existing chart with new options
     */
    updateChart(canvasId, models, options) {
        const chart = this.charts[canvasId];
        if (!chart) return;

        // Get location from chart's original data
        // For simplicity, we'll recreate the chart
        // This could be optimized to just update datasets
        const location = options.location;
        const locationName = options.locationName;

        if (location) {
            this.createForecastChart(canvasId, locationName, location, models, options);
        }
    },

    /**
     * Destroy all charts
     */
    destroyAll() {
        for (const [id, chart] of Object.entries(this.charts)) {
            chart.destroy();
        }
        this.charts = {};
    },

    /**
     * Destroy a specific chart
     */
    destroy(canvasId) {
        if (this.charts[canvasId]) {
            this.charts[canvasId].destroy();
            delete this.charts[canvasId];
        }
    }
};
