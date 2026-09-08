# 📊 Customer Churn Survival Analysis Engine

Live Demo: https://customer-churn-survival-analysis.vercel.app/
Live Demo: https://customer-churn-survival-analysis-nwz6qxxzq.streamlit.app/
An end-to-end MLOps platform that moves beyond traditional binary churn classification by modeling customer lifetime duration and real-time retention risk curves using **Cox Proportional Hazards Regression**.

---

## 📌 Overview

Standard machine learning models predict **if** a customer will churn ($y \in \{0, 1\}$). This project predicts **when** a customer is likely to churn and computes dynamic retention probabilities across future tenure horizons:

* **Survival Trajectory Modeling:** Generates personalized survival decay curves ($S(t)$) over a 72-month lifecycle.
* **Hazard Risk Scoring:** Computes relative risk ratios against baseline population cohorts.
* **Dynamic Time Horizons:** Forecasts retention probability for user-selected forecast windows (e.g., $+6, +12, +24$ months).
* **Dual Interface Architecture:** Provides a production-ready custom HTML/JS interface backed by FastAPI, alongside an exploratory Streamlit dashboard.

---

## 🛠️ Tech Stack

| Domain | Technology |
| :--- | :--- |
| **Statistical ML Engine** | Python 3.11+, `lifelines` (Cox Proportional Hazards), `scikit-learn`, `pandas`, `numpy`, `joblib` |
| **Backend API** | FastAPI, Uvicorn, Pydantic |
| **Frontend & UI** | Vanilla HTML5, CSS3, JavaScript (ES6+), Plotly.js, Streamlit |
| **Deployment & Hosting** | Vercel (Serverless Functions + Static Edge CDN), Streamlit Community Cloud |
| **Testing & CI/CD** | `pytest`, Git, GitHub Actions |

---