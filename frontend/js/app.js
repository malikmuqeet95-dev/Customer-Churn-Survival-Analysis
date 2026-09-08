document.addEventListener("DOMContentLoaded", () => {
const form = document.getElementById("prediction-form");

if (!form) {
    console.error("Prediction form with id='prediction-form' was not found.");
    return;
}

form.addEventListener("submit", async function (e) {
    e.preventDefault();

    const payload = {
        tenure: parseInt(document.getElementById("tenure").value, 10),
        forecast_horizon: parseInt(document.getElementById("forecast_horizon").value, 10),
        MonthlyCharges: parseFloat(document.getElementById("MonthlyCharges").value),
        TotalCharges: parseFloat(document.getElementById("TotalCharges").value),
        SeniorCitizen: parseInt(document.getElementById("SeniorCitizen").value, 10),
        gender: document.getElementById("gender").value,
        Partner: document.getElementById("Partner").value,
        Dependents: document.getElementById("Dependents").value,
        PhoneService: document.getElementById("PhoneService").value,
        MultipleLines: document.getElementById("MultipleLines").value,
        Contract: document.getElementById("Contract").value,
        InternetService: document.getElementById("InternetService").value,
        TechSupport: document.getElementById("TechSupport").value,
        OnlineSecurity: document.getElementById("OnlineSecurity").value,
        OnlineBackup: document.getElementById("OnlineBackup").value,
        DeviceProtection: document.getElementById("DeviceProtection").value,
        StreamingTV: document.getElementById("StreamingTV").value,
        StreamingMovies: document.getElementById("StreamingMovies").value,
        PaymentMethod: document.getElementById("PaymentMethod").value,
        PaperlessBilling: document.getElementById("PaperlessBilling").value,
    };

    if (
        Number.isNaN(payload.tenure) ||
        Number.isNaN(payload.forecast_horizon) ||
        Number.isNaN(payload.MonthlyCharges) ||
        Number.isNaN(payload.TotalCharges)
    ) {
        showError("Please enter valid numeric values.");
        return;
    }

    setLoadingState(true);

    try {
        const response = await fetch("/api/predict", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify(payload)
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(
                data.detail || "Prediction request failed."
            );
        }

        if (data.status !== "success") {
            throw new Error(
                data.detail || "Prediction failed."
            );
        }

        const hazardElement = document.getElementById("res-hazard");
        const tierElement = document.getElementById("res-tier");
        const retentionElement = document.getElementById("res-retention");
        const actionElement = document.getElementById("res-action");

        if (hazardElement) {
            hazardElement.innerText = `${data.hazard_ratio}x`;
        }

        if (tierElement) {
            tierElement.innerText = data.risk_tier;
            updateRiskTierColor(tierElement, data.risk_tier);
        }

        if (retentionElement) {
            retentionElement.innerText = `${data.projected_retention}%`;
        }

        if (actionElement) {
            actionElement.innerText = data.recommended_action;
        }

        if (
            data.forecast_curve &&
            Array.isArray(data.forecast_curve.timeline) &&
            Array.isArray(data.forecast_curve.survival_prob)
        ) {
            renderSurvivalPlot(
                data.forecast_curve,
                data.current_tenure,
                data.target_month
            );
        } else {
            console.warn("No valid forecast curve returned by API.");
        }

        updateOptionalElement(
            "res-churn",
            `${data.projected_churn}%`
        );

        updateOptionalElement(
            "res-tenure",
            `${data.current_tenure} months`
        );

        updateOptionalElement(
            "res-target",
            `${data.target_month} months`
        );
    } catch (err) {
        console.error("Prediction error:", err);
        showError(
            err.message || "Unable to generate prediction."
        );
    } finally {
        setLoadingState(false);
    }
});

// Recalculate immediately after any account attribute changes.
form.addEventListener("change", function () {
    form.requestSubmit();
});

form.dispatchEvent(new Event("submit"));

});

function renderSurvivalPlot(curve, currentTenure, targetMonth) {
const chartElement = document.getElementById("plotly-chart");

if (!chartElement) {
    console.error(
        "Plotly chart element with id='plotly-chart' was not found."
    );
    return;
}

if (
    !curve ||
    !Array.isArray(curve.timeline) ||
    !Array.isArray(curve.survival_prob)
) {
    console.error("Invalid survival curve data:", curve);
    return;
}

if (
    curve.timeline.length === 0 ||
    curve.survival_prob.length === 0
) {
    console.warn("Survival curve contains no data.");
    return;
}

const trace = {
    x: curve.timeline,
    y: curve.survival_prob.map(value => value * 100),
    mode: "lines+markers",
    name: "Survival Probability",
    line: {
        color: "#38bdf8",
        width: 2.5
    },
    marker: {
        size: 4,
        color: "#0284c7"
    }
};

const layout = {
    title: {
        text: `Forecast Survival Curve (Active: ${currentTenure} Mo | Target: ${targetMonth} Mo)`,
        font: {
            color: "#f8fafc",
            size: 13
        }
    },
    paper_bgcolor: "#111827",
    plot_bgcolor: "#0b0f19",
    xaxis: {
        title: "Tenure Timeline (Months)",
        color: "#94a3b8",
        gridcolor: "#1f2937",
        zeroline: false
    },
    yaxis: {
        title: "Retention Probability (%)",
        color: "#94a3b8",
        gridcolor: "#1f2937",
        range: [0, 105],
        zeroline: false
    },
    margin: {
        l: 55,
        r: 20,
        t: 50,
        b: 50
    },
    legend: {
        font: {
            color: "#cbd5e1"
        }
    }
};

const config = {
    responsive: true,
    displayModeBar: false
};

Plotly.newPlot(
    chartElement,
    [trace],
    layout,
    config
);

}

function updateRiskTierColor(element, tier) {
if (!element) {
return;
}

if (tier === "CRITICAL") {
    element.style.color = "#f43f5e";
} else if (tier === "MODERATE") {
    element.style.color = "#fbbf24";
} else {
    element.style.color = "#34d399";
}

}

function updateOptionalElement(elementId, value) {
const element = document.getElementById(elementId);

if (element) {
    element.innerText = value;
}

}

function setLoadingState(loading) {
const form = document.getElementById("prediction-form");

if (!form) {
    return;
}

const submitButton = form.querySelector(
    'button[type="submit"]'
);

if (!submitButton) {
    return;
}

if (loading) {
    submitButton.disabled = true;
    submitButton.dataset.originalText = submitButton.innerText;
    submitButton.innerText = "Predicting...";
} else {
    submitButton.disabled = false;
    submitButton.innerText =
        submitButton.dataset.originalText || "Predict";
}

}

function showError(message) {
console.error("Prediction error:", message);

const errorElement = document.getElementById(
    "prediction-error"
);

if (errorElement) {
    errorElement.innerText = message;
    errorElement.style.display = "block";
    return;
}

alert(message);

}
