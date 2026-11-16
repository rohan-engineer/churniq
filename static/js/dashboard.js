/**
 * ChurnIQ - Dashboard Analytics Script
 * Handles real-time API calls, Chart.js rendering, and DOM updates for metrics and evaluations.
 */

document.addEventListener("DOMContentLoaded", function () {
    loadDashboardMetrics();
    loadModelPerformance();
});

// Color constants for clean, cohesive charts
const CHART_COLORS = {
    retained: "#2563eb",       // Blue
    churned: "#ef4444",        // Red / Coral
    accentPurple: "#8b5cf6",
    accentCyan: "#06b6d4",
    retainedLight: "rgba(37, 99, 235, 0.8)",
    churnedLight: "rgba(239, 68, 68, 0.8)",
    navy: "#0f172a",
    grid: "#e2e8f0"
};

/**
 * Loads dynamic dataset metrics and initializes charts.
 */
async function loadDashboardMetrics() {
    try {
        const response = await fetch("/api/dashboard");
        if (!response.ok) {
            throw new Error(`Dashboard API responded with status ${response.status}`);
        }
        const data = await response.json();
        if (data.status !== "success") return;

        // 1. Populate Overview Metric Cards
        const overview = data.overview;
        document.getElementById("totalCustomers").innerText = overview.total_customers.toLocaleString();
        document.getElementById("churnedCustomers").innerText = overview.churned_customers.toLocaleString();
        document.getElementById("retainedCustomers").innerText = overview.retained_customers.toLocaleString();
        document.getElementById("churnRate").innerText = `${overview.churn_rate_percent}%`;

        // 2. Render Overall Churn Doughnut Chart
        renderChurnDoughnut(overview.retained_customers, overview.churned_customers);

        // 3. Render Contract Churn Chart
        renderContractChart(data.contract_analysis);

        // 4. Render Payment Method Chart
        if (document.getElementById("paymentChart")) {
            renderPaymentChart(data.payment_analysis);
        }

        // 5. Render Tenure Cohort Chart
        if (document.getElementById("tenureChart")) {
            renderTenureChart(data.tenure_analysis);
        }

        // 6. Populate Service Adoption Table / Stats
        if (document.getElementById("serviceStatsTable") && data.service_adoption) {
            renderServiceStats(data.service_adoption, overview.total_customers);
        }

    } catch (error) {
        console.error("Failed to load dashboard metrics:", error);
    }
}

/**
 * Renders Overall Churn Doughnut Chart.
 */
function renderChurnDoughnut(retained, churned) {
    const ctx = document.getElementById("churnDoughnutChart");
    if (!ctx) return;

    new Chart(ctx, {
        type: "doughnut",
        data: {
            labels: ["Retained Customers", "Churned Customers"],
            datasets: [{
                data: [retained, churned],
                backgroundColor: [CHART_COLORS.retained, CHART_COLORS.churned],
                hoverOffset: 6,
                borderWidth: 2,
                borderColor: "#ffffff"
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: "bottom",
                    labels: {
                        color: CHART_COLORS.navy,
                        font: { size: 12, weight: 600 },
                        padding: 15
                    }
                },
                tooltip: {
                    callbacks: {
                        label: function (context) {
                            const total = retained + churned;
                            const val = context.raw;
                            const pct = ((val / total) * 100).toFixed(1);
                            return ` ${context.label}: ${val.toLocaleString()} (${pct}%)`;
                        }
                    }
                }
            },
            cutout: "65%"
        }
    });
}

/**
 * Renders Contract-wise Churn Stacked Bar Chart.
 */
function renderContractChart(contractData) {
    const ctx = document.getElementById("contractChart");
    if (!ctx) return;

    new Chart(ctx, {
        type: "bar",
        data: {
            labels: contractData.labels,
            datasets: [
                {
                    label: "Retained",
                    data: contractData.retained,
                    backgroundColor: CHART_COLORS.retained,
                    borderRadius: 6
                },
                {
                    label: "Churned",
                    data: contractData.churned,
                    backgroundColor: CHART_COLORS.churned,
                    borderRadius: 6
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                x: {
                    grid: { display: false },
                    ticks: { color: CHART_COLORS.navy, font: { weight: 600 } }
                },
                y: {
                    grid: { color: CHART_COLORS.grid },
                    ticks: { color: "#64748b" }
                }
            },
            plugins: {
                legend: { position: "top", labels: { font: { weight: 600 } } }
            }
        }
    });
}

/**
 * Renders Payment Method Analysis Horizontal Bar Chart.
 */
function renderPaymentChart(paymentData) {
    const ctx = document.getElementById("paymentChart");
    if (!ctx) return;

    new Chart(ctx, {
        type: "bar",
        data: {
            labels: paymentData.labels,
            datasets: [
                {
                    label: "Retained",
                    data: paymentData.retained,
                    backgroundColor: CHART_COLORS.retained,
                    borderRadius: 4
                },
                {
                    label: "Churned",
                    data: paymentData.churned,
                    backgroundColor: CHART_COLORS.churned,
                    borderRadius: 4
                }
            ]
        },
        options: {
            indexAxis: "y",
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                x: {
                    grid: { color: CHART_COLORS.grid },
                    ticks: { color: "#64748b" }
                },
                y: {
                    grid: { display: false },
                    ticks: { color: CHART_COLORS.navy, font: { weight: 600 } }
                }
            },
            plugins: {
                legend: { position: "top", labels: { font: { weight: 600 } } }
            }
        }
    });
}

/**
 * Renders Tenure Analysis Bar Chart.
 */
function renderTenureChart(tenureData) {
    const ctx = document.getElementById("tenureChart");
    if (!ctx) return;

    new Chart(ctx, {
        type: "bar",
        data: {
            labels: tenureData.labels,
            datasets: [
                {
                    label: "Retained",
                    data: tenureData.retained,
                    backgroundColor: CHART_COLORS.retained,
                    borderRadius: 5
                },
                {
                    label: "Churned",
                    data: tenureData.churned,
                    backgroundColor: CHART_COLORS.churned,
                    borderRadius: 5
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                x: {
                    grid: { display: false },
                    ticks: { color: CHART_COLORS.navy, font: { weight: 600 } }
                },
                y: {
                    grid: { color: CHART_COLORS.grid },
                    ticks: { color: "#64748b" }
                }
            },
            plugins: {
                legend: { position: "top", labels: { font: { weight: 600 } } }
            }
        }
    });
}

/**
 * Populates Service Adoption Table.
 */
function renderServiceStats(serviceData, totalCustomers) {
    const tbody = document.getElementById("serviceStatsTable");
    if (!tbody) return;

    tbody.innerHTML = "";
    for (const [service, count] of Object.entries(serviceData)) {
        const pct = ((count / totalCustomers) * 100).toFixed(1);
        const row = document.createElement("tr");
        row.innerHTML = `
            <td><strong>${formatServiceName(service)}</strong></td>
            <td>${count.toLocaleString()}</td>
            <td>
                <div class="d-flex align-items-center gap-2">
                    <div class="progress flex-grow-1" style="height: 8px;">
                        <div class="progress-bar bg-primary" role="progressbar" style="width: ${pct}%"></div>
                    </div>
                    <span class="small font-monospace">${pct}%</span>
                </div>
            </td>
        `;
        tbody.appendChild(row);
    }
}

function formatServiceName(name) {
    return name.replace(/([A-Z])/g, " $1").trim();
}

/**
 * Loads Model Information and Verified Evaluation Metrics from API.
 */
async function loadModelPerformance() {
    try {
        const response = await fetch("/api/model-info");
        if (!response.ok) return;
        const data = await response.json();
        if (data.status !== "success") return;

        const info = data.model_info;
        const metrics = info.evaluation_metrics;

        // Model Metrics display
        if (metrics && metrics.accuracy !== undefined) {
            const accEl = document.getElementById("metricAccuracy");
            const precEl = document.getElementById("metricPrecision");
            const recEl = document.getElementById("metricRecall");
            const f1El = document.getElementById("metricF1");

            if (accEl) accEl.innerText = `${(metrics.accuracy * 100).toFixed(1)}%`;
            if (precEl) precEl.innerText = `${(metrics.precision * 100).toFixed(1)}%`;
            if (recEl) recEl.innerText = `${(metrics.recall * 100).toFixed(1)}%`;
            if (f1El) f1El.innerText = `${(metrics.f1_score * 100).toFixed(1)}%`;

            // Confusion Matrix Counts
            if (metrics.confusion_matrix) {
                const cm = metrics.confusion_matrix;
                const tnEl = document.getElementById("cmTN");
                const fpEl = document.getElementById("cmFP");
                const fnEl = document.getElementById("cmFN");
                const tpEl = document.getElementById("cmTP");
                if (tnEl) tnEl.innerText = cm.true_negatives;
                if (fpEl) fpEl.innerText = cm.false_positives;
                if (fnEl) fnEl.innerText = cm.false_negatives;
                if (tpEl) tpEl.innerText = cm.true_positives;
            }

            // Sample Comparisons Table
            if (metrics.sample_comparisons && document.getElementById("sampleComparisonBody")) {
                const compBody = document.getElementById("sampleComparisonBody");
                compBody.innerHTML = "";
                metrics.sample_comparisons.forEach((item) => {
                    const tr = document.createElement("tr");
                    const badgeClass = item.match ? "badge bg-success" : "badge bg-danger";
                    const matchText = item.match ? "Correct" : "Mismatch";
                    tr.innerHTML = `
                        <td>#${item.sample_index + 1}</td>
                        <td><span class="badge ${item.actual_label === 1 ? 'bg-danger' : 'bg-secondary'}">${item.actual_text}</span></td>
                        <td>${(item.predicted_probability * 100).toFixed(1)}%</td>
                        <td><span class="badge ${item.predicted_label === 1 ? 'bg-danger' : 'bg-secondary'}">${item.predicted_text}</span></td>
                        <td><span class="${badgeClass}">${matchText}</span></td>
                    `;
                    compBody.appendChild(tr);
                });
            }
        }

    } catch (e) {
        console.error("Error loading model info:", e);
    }
}
