# 🚦 Smart City Traffic Patterns

> **A data science project for traffic analysis, forecasting, and infrastructure planning across four city junctions.**

---

## 🧭 Overview

This project is built for the government's initiative to transform the city into a **smart, digital, and intelligent city**. As a data scientist, the goal is to:

- Analyse historical traffic data across **4 junctions**
- Identify traffic patterns during **holidays vs. regular weekdays**
- Forecast **future traffic peaks** using machine learning
- Provide **infrastructure planning insights** such as peak-hour capacity, staffing recommendations, and lane thresholds

The project includes:
- A **script-based pipeline** (`traffic_forecasting.py`) using Random Forest for competition-style prediction.
- An **interactive Streamlit dashboard** (`app.py`) with EDA, forecasting, and infrastructure insights.

---

## 📁 Project Structure

```
smart-city-traffic-patterns/
├── smart-city-traffic-patterns/
│   ├── train_aWnotuB.csv                       # Training data (hourly vehicle counts)
│   ├── test_BdBKkAj.csv                        # Test data for submission
│   ├── datasets_8494_11879_test_BdBKkAj.csv    # Alternate test dataset
│   ├── submission.csv                          # Model predictions for submission
│   └── plots/                                  # Generated EDA plots
├── app.py                                      # 🖥️ Streamlit interactive dashboard
├── utils.py                                    # Helper functions (data, EDA, forecasting, insights)
├── traffic_forecasting.py                      # Batch pipeline (Random Forest prediction)
├── requirements.txt                            # Python dependencies
└── README.md                                   # Project documentation
```

---

## 🛠️ Getting Started

### Prerequisites

- Python 3.8 or higher
- pip (Python package installer)

### Installation

1. **Clone the repository:**
```bash
git clone <repository-url>
cd smart-city-traffic-patterns
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

---

## 🚀 Running the Dashboard

Launch the interactive Streamlit dashboard with:

```bash
streamlit run app.py
```

Then open your browser at **http://localhost:8501**.

### Dashboard Features

#### 📊 1. Exploratory Data Analysis (EDA)
| Tab | Description |
|-----|-------------|
| **Time Series** | Hourly vehicle counts over the full historical period per junction |
| **Distribution** | Histogram of vehicle counts showing traffic volume spread |
| **Heatmap** | Average vehicles per hour of day vs. day of week |

#### 🔮 2. Traffic Forecast
- Trained using **Facebook Prophet** with yearly, weekly, and daily seasonality
- Configurable **forecast horizon** (1–30 days) via sidebar slider
- Toggle between **full dataset training** or a **rolling 6-month window**
- Displays historical data alongside forecast + 95% confidence intervals
- **📥 Download forecast as CSV** directly from the dashboard

#### 🛠️ 3. Infrastructure Planning Insights
| Metric | Description |
|--------|-------------|
| **Peak Hour (Historical)** | The hour of day with highest average traffic |
| **Forecasted Peak Vehicles** | Maximum expected vehicle count within the forecast window |
| **Forecasted Peak Time** | Date and time of the forecasted traffic peak |
| **Recommended Capacity** | Suggested road/lane capacity with a 20% buffer above the peak |

### Sidebar Controls
| Control | Description |
|---------|-------------|
| Select Junction | Choose between Junctions 1–4 |
| Treat weekends as holidays | Flag Saturdays and Sundays as holiday days |
| Forecast horizon (days) | Number of future days to forecast |
| Rolling 6-month window | Train model only on the most recent 6 months of data |

---

## 🤖 Running the Batch Prediction Pipeline

To generate predictions using the Random Forest model:

```bash
python traffic_forecasting.py
```

This script:
1. **Loads** training and test datasets
2. **Engineers features** – year, month, day, hour, day-of-week, weekend/holiday flags
3. **Trains** a `RandomForestRegressor` model on the training data
4. **Predicts** vehicle counts for the test set
5. **Saves** results to `submission.csv`

---

## 🗂️ Dataset Description

| Column | Description |
|--------|-------------|
| `DateTime` | Timestamp of the observation (hourly) |
| `Junction` | Junction ID (1, 2, 3, or 4) |
| `Vehicles` | Number of vehicles counted at that junction and time |
| `ID` | Unique row identifier |

**Training data period:** November 2015 – June 2017  
**Test data:** Predictions required for the following months

---

## 🧰 Technologies Used

| Library | Purpose |
|---------|---------|
| `pandas` | Data manipulation and analysis |
| `numpy` | Numerical operations |
| `plotly` | Interactive visualisations in the dashboard |
| `streamlit` | Web-based interactive dashboard |
| `prophet` | Time-series forecasting with seasonality |
| `scikit-learn` | Random Forest model for batch predictions |
| `matplotlib` / `seaborn` | Static EDA plots (batch pipeline) |

---

## 📌 Key Findings

- **Junction 1 & 2** show the most consistent peak-hour patterns.
- **Traffic peaks between 8 AM–10 AM and 5 PM–7 PM**, matching typical commute hours.
- **Weekends and holidays** show significantly lower traffic volumes (~30–40% reduction).
- **Prophet forecasting** captures both daily and weekly seasonality effectively.
- The recommended infrastructure capacity includes a **20% safety buffer** above forecasted peaks.

---

## 📄 License

This project is for educational purposes as part of the **UpSkill Campus – University of Cape Town (UCT)** internship program.

