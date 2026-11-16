/**
 * ChurnIQ - Customer Prediction Script
 * Collects form data, communicates with /api/predict, and renders interactive risk results.
 */

let gaugeChartInstance = null;

document.addEventListener("DOMContentLoaded", function () {
    const form = document.getElementById("churnPredictionForm");
    const tenureInput = document.getElementById("tenure");
    const monthlyInput = document.getElementById("MonthlyCharges");
    const totalInput = document.getElementById("TotalCharges");
    const thresholdInput = document.getElementById("threshold");
    const thresholdValDisplay = document.getElementById("thresholdValueDisplay");

    // Auto calculate TotalCharges suggestion
    function autoEstimateTotal() {
        const tenure = parseFloat(tenureInput.value) || 0;
        const monthly = parseFloat(monthlyInput.value) || 0;
        if (!totalInput.value || totalInput.dataset.autoFilled === "true") {
            totalInput.value = (tenure * monthly).toFixed(2);
            totalInput.dataset.autoFilled = "true";
        }
    }

    if (tenureInput && monthlyInput && totalInput) {
        tenureInput.addEventListener("input", autoEstimateTotal);
        monthlyInput.addEventListener("input", autoEstimateTotal);
        totalInput.addEventListener("input", () => {
            totalInput.dataset.autoFilled = "false";
        });
    }

    // Threshold range slider update
    if (thresholdInput && thresholdValDisplay) {
        thresholdInput.addEventListener("input", function () {
            thresholdValDisplay.innerText = parseFloat(this.value).toFixed(2);
        });
    }

    // Preset profile buttons
    const btnHighRisk = document.getElementById("btnPresetHighRisk");
    const btnLowRisk = document.getElementById("btnPresetLowRisk");

    if (btnHighRisk) {
        btnHighRisk.addEventListener("click", () => loadPreset("high"));
    }
    if (btnLowRisk) {
        btnLowRisk.addEventListener("click", () => loadPreset("low"));
    }

    // Form submission
    if (form) {
        form.addEventListener("submit", async function (e) {
            e.preventDefault();
            await executePrediction();
        });
    }
});

/**
 * Loads predefined customer profiles for rapid testing.
 */
function loadPreset(type) {
    if (type === "high") {
        setFieldValue("gender", "Female");
        setFieldValue("SeniorCitizen", "0");
        setFieldValue("Partner", "No");
        setFieldValue("Dependents", "No");
        setFieldValue("tenure", "2");
        setFieldValue("PhoneService", "Yes");
        setFieldValue("MultipleLines", "No");
        setFieldValue("InternetService", "Fiber optic");
        setFieldValue("OnlineSecurity", "No");
        setFieldValue("OnlineBackup", "No");
        setFieldValue("DeviceProtection", "No");
        setFieldValue("TechSupport", "No");
        setFieldValue("StreamingTV", "Yes");
        setFieldValue("StreamingMovies", "Yes");
        setFieldValue("Contract", "Month-to-month");
        setFieldValue("PaperlessBilling", "Yes");
        setFieldValue("PaymentMethod", "Electronic check");
        setFieldValue("MonthlyCharges", "89.85");
        setFieldValue("TotalCharges", "179.70");
    } else {
        setFieldValue("gender", "Male");
        setFieldValue("SeniorCitizen", "0");
        setFieldValue("Partner", "Yes");
        setFieldValue("Dependents", "Yes");
        setFieldValue("tenure", "62");
        setFieldValue("PhoneService", "Yes");
        setFieldValue("MultipleLines", "Yes");
        setFieldValue("InternetService", "DSL");
        setFieldValue("OnlineSecurity", "Yes");
        setFieldValue("OnlineBackup", "Yes");
        setFieldValue("DeviceProtection", "Yes");
        setFieldValue("TechSupport", "Yes");
        setFieldValue("StreamingTV", "No");
        setFieldValue("StreamingMovies", "No");
        setFieldValue("Contract", "Two year");
        setFieldValue("PaperlessBilling", "No");
        setFieldValue("PaymentMethod", "Credit card (automatic)");
        setFieldValue("MonthlyCharges", "58.40");
        setFieldValue("TotalCharges", "3620.80");
    }
}

function setFieldValue(id, val) {
    const el = document.getElementById(id);
    if (el) el.value = val;
}

/**
 * Collects payload, posts to /api/predict, and renders outcome.
 */
async function executePrediction() {
    const submitBtn = document.getElementById("btnSubmitPrediction");
    const resultContainer = document.getElementById("predictionResultContainer");
    const spinner = document.getElementById("predictionSpinner");

    // Gather values
    const customer = {
        gender: document.getElementById("gender").value,
        SeniorCitizen: document.getElementById("SeniorCitizen").value,
        Partner: document.getElementById("Partner").value,
        Dependents: document.getElementById("Dependents").value,
        tenure: parseInt(document.getElementById("tenure").value, 10),
        PhoneService: document.getElementById("PhoneService").value,
        MultipleLines: document.getElementById("MultipleLines").value,
        InternetService: document.getElementById("InternetService").value,
        OnlineSecurity: document.getElementById("OnlineSecurity").value,
        OnlineBackup: document.getElementById("OnlineBackup").value,
        DeviceProtection: document.getElementById("DeviceProtection").value,
        TechSupport: document.getElementById("TechSupport").value,
        StreamingTV: document.getElementById("StreamingTV").value,
        StreamingMovies: document.getElementById("StreamingMovies").value,
        Contract: document.getElementById("Contract").value,
        PaperlessBilling: document.getElementById("PaperlessBilling").value,
        PaymentMethod: document.getElementById("PaymentMethod").value,
        MonthlyCharges: parseFloat(document.getElementById("MonthlyCharges").value),
        TotalCharges: parseFloat(document.getElementById("TotalCharges").value),
    };

    const thresholdInput = document.getElementById("threshold");
    const threshold = thresholdInput ? parseFloat(thresholdInput.value) : 0.50;

    // Show loading state
    if (submitBtn) submitBtn.disabled = true;
    if (spinner) spinner.classList.remove("d-none");

    try {
        const response = await fetch("/api/predict", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ customer, threshold }),
        });

        const resData = await response.json();
        if (!response.ok || resData.status !== "success") {
            alert(resData.message || "Prediction failed. Please verify inputs.");
            return;
        }

        const data = resData.data;

        // Render outcomes
        renderPredictionOutcome(data);
        if (resultContainer) {
            resultContainer.classList.remove("d-none");
            resultContainer.scrollIntoView({ behavior: "smooth" });
        }

    } catch (err) {
        console.error("Error during prediction:", err);
        alert("An error occurred while communicating with the prediction API.");
    } finally {
        if (submitBtn) submitBtn.disabled = false;
        if (spinner) spinner.classList.add("d-none");
    }
}

/**
 * Updates DOM elements with prediction output and renders doughnut gauge.
 */
function renderPredictionOutcome(data) {
    const isChurn = data.is_churn;
    const probPercent = data.churn_probability_percent;
    const resultBox = document.getElementById("predictionResultBox");
    const probDisplay = document.getElementById("probPercentDisplay");
    const resultText = document.getElementById("predictionResultText");
    const riskBadge = document.getElementById("riskLevelBadge");
    const explanationList = document.getElementById("explanationList");
    const thresholdApplied = document.getElementById("appliedThresholdDisplay");

    if (resultBox) {
        resultBox.className = "prediction-result-box " + (isChurn ? "high-risk" : "low-risk");
    }

    if (probDisplay) {
        probDisplay.innerText = `${probPercent}%`;
    }

    if (resultText) {
        resultText.innerText = data.prediction_result;
        resultText.style.color = isChurn ? "#ef4444" : "#10b981";
    }

    if (riskBadge) {
        riskBadge.innerText = `${data.risk_level} Risk`;
        riskBadge.className = `risk-badge ${isChurn ? 'high' : 'low'}`;
    }

    if (thresholdApplied) {
        thresholdApplied.innerText = data.threshold_used;
    }

    if (explanationList && data.explanation) {
        explanationList.innerHTML = "";
        data.explanation.forEach((exp) => {
            const li = document.createElement("li");
            li.innerText = exp;
            explanationList.appendChild(li);
        });
    }

    renderGauge(probPercent, isChurn);
}

/**
 * Visualizes probability with a clean gauge chart.
 */
function renderGauge(probPercent, isChurn) {
    const ctx = document.getElementById("gaugeChart");
    if (!ctx) return;

    if (gaugeChartInstance) {
        gaugeChartInstance.destroy();
    }

    const remaining = (100 - probPercent).toFixed(1);
    const activeColor = isChurn ? "#ef4444" : "#10b981";

    gaugeChartInstance = new Chart(ctx, {
        type: "doughnut",
        data: {
            datasets: [{
                data: [probPercent, remaining],
                backgroundColor: [activeColor, "#e2e8f0"],
                borderWidth: 0,
            }]
        },
        options: {
            circumference: 180,
            rotation: 270,
            cutout: "75%",
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                tooltip: { enabled: false }
            }
        }
    });
}
