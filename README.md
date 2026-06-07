# 🌐 Supply Chain Risk Monitor
## 📸 Live Demo

🔗 **[Live App → supply-chain-risk-monitor-olfvjdzxxxjpbgqnzfdyap.streamlit.app](https://supply-chain-risk-monitor-olfvjdzxxxjpbgqnzfdyap.streamlit.app)**

> **952 nodes scored** across 80+ countries | **AUC 0.9923** | Real-time GDELT ingestion | Groq LLaMA-3.3-70b insights
A real-time supply chain disruption intelligence platform for India-linked trade nodes. Combines GDELT event data, XGBoost machine learning, SHAP explainability, Groq LLM insights, and an interactive Streamlit dashboard to monitor and alert on geopolitical and operational risks across 952 global supply chain nodes.

---

## 📸 Demo

![Global Risk Map](docs/screenshots/map.png)

> **952 nodes scored** across 80+ countries | **AUC 0.9923** | Real-time GDELT ingestion | Groq LLaMA-3.3-70b insights

---

## 🏗️ Architecture

```
GDELT Events (15-min)
       │
       ▼
┌─────────────────┐    ┌──────────────────────┐    ┌─────────────────┐
│  Event          │    │  Feature             │    │  XGBoost        │
│  Processor      │───▶│  Engineer            │───▶│  Classifier +   │
│  (CAMEO filter) │    │  (22 features,       │    │  Regressor      │
└─────────────────┘    │   354k rows)         │    │  AUC = 0.9923   │
                       └──────────────────────┘    └────────┬────────┘
                                                            │
┌─────────────────┐    ┌──────────────────────┐            ▼
│  ISB Trade Data │    │  Node Graph          │    ┌─────────────────┐
│  EMIS Firms     │───▶│  952 nodes           │    │  Risk Scores    │
│  FSI Country    │    │  13,757 edges        │    │  (disruption_   │
│  Risk           │    └──────────────────────┘    │   prob + tier)  │
└─────────────────┘                                └────────┬────────┘
                                                            │
                              ┌─────────────────┬──────────┴──────────┐
                              ▼                 ▼                     ▼
                    ┌──────────────┐  ┌──────────────────┐  ┌──────────────┐
                    │ Groq LLaMA   │  │  Streamlit       │  │  AWS SNS     │
                    │ Insight      │  │  Dashboard       │  │  Alerts      │
                    │ Generator    │  │  (3 pages)       │  │  (hourly)    │
                    └──────────────┘  └──────────────────┘  └──────────────┘
```

---

## ✨ Features

- **Real-time ingestion** — GDELT 2.0 event stream filtered for 20+ supply-chain CAMEO codes
- **952-node graph** — India import trade nodes (country × sector) + 10 global chokepoints (Malacca, Suez, Hormuz, etc.)
- **Dual XGBoost models** — binary disruption classifier (AUC 0.9923) + severity regressor
- **Weakly supervised labelling** — programmatic labels derived from event features (Ratner et al. 2017 / Snorkel framework)
- **PSI drift monitoring** — Population Stability Index on 10 dynamic features vs training baseline
- **Groq LLaMA-3.3-70b insights** — structured 3-part AI analysis for critical nodes (risk reason, downstream impact, confidence)
- **AWS SNS alerting** — spike detection (prob > 0.70) with hourly scheduled checks
- **Interactive dashboard** — Folium global risk map, Plotly trend explorer, live event feed

---

## 📊 Model Performance

| Metric | Value |
|---|---|
| Classifier AUC | **0.9923** |
| Precision | 0.9152 |
| Recall | 0.9848 |
| F1 Score | 0.9487 |
| Regressor MAE | 1.0228 |
| Regressor RMSE | 1.3153 |
| Nodes scored | 952 |
| Feature dimensions | 22 |
| Training snapshots | 372 days |

---

## 🗂️ Project Structure

```
supply-chain-risk-monitor/
├── config/
│   └── config.yaml
├── data/
│   ├── raw/gdelt/              ← GDELT event files (downloaded)
│   ├── processed/              ← all pipeline outputs
│   └── reference/
│       ├── emis/               ← 8 EMIS sector Excel files
│       ├── isb/                ← ISB India trade CSVs
│       ├── fsi/                ← Fund for Peace FSI 2023
│       ├── bse/
│       └── nse/
├── src/
│   ├── ingestion/
│   │   └── gdelt_collector.py  ← GDELT 2.0 downloader + filter
│   ├── processing/
│   │   ├── build_reference_data.py
│   │   ├── build_trade_data.py
│   │   ├── node_mapper.py
│   │   ├── event_processor.py
│   │   └── feature_engineer.py
│   ├── ml/
│   │   ├── train.py            ← XGBoost + SHAP + MLflow
│   │   ├── predict.py          ← inference pipeline
│   │   └── drift_monitor.py    ← PSI monitoring
│   ├── genai/
│   │   └── insight_generator.py ← Groq LLaMA-3.3-70b
│   ├── alerting/
│   │   └── alert_manager.py    ← AWS SNS spike alerts
│   └── dashboard/
│       └── app.py              ← Streamlit 3-page dashboard
├── models/
│   ├── classifier.json
│   ├── regressor.json
│   ├── feature_columns.json
│   ├── shap_summary.csv
│   ├── train_metrics.json
│   └── train_feature_stats.json
├── logs/
├── .env.example
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start

### 1. Clone and set up environment

```bash
git clone https://github.com/yourusername/supply-chain-risk-monitor.git
cd supply-chain-risk-monitor
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Mac/Linux
pip install -r requirements.txt
```

### 2. Configure environment variables

```bash
cp .env.example .env
```

Edit `.env`:
```
GROQ_API_KEY=your_groq_api_key_here
SNS_TOPIC_ARN=arn:aws:sns:region:account:topic    # optional
AWS_REGION=ap-south-1                              # optional
```

### 3. Add reference data

Place the following files in `data/reference/`:
- `emis/` — EMIS sector Excel files (8 files, one per sector)
- `isb/` — ISB India trade CSVs
- `fsi/fsi_2023.xlsx` — Fund for Peace Fragile States Index 2023

### 4. Run the full pipeline

```bash
# Collect GDELT events (last 7 days)
python src/ingestion/gdelt_collector.py --mode historical --days 7

# Build reference and trade data
python src/processing/build_reference_data.py
python src/processing/build_trade_data.py

# Build node graph and features
python src/processing/node_mapper.py
python src/processing/event_processor.py
python src/processing/feature_engineer.py

# Train models
python src/ml/train.py

# Score all nodes
python src/ml/predict.py

# Generate AI insights (requires GROQ_API_KEY)
python src/genai/insight_generator.py

# Launch dashboard
streamlit run src/dashboard/app.py
```

---

## 📋 Dashboard Pages

### Page 1 — Global Risk Map
Folium interactive map with 952 circle markers colour-coded by risk tier. Click any node for a popup with disruption probability, severity prediction, and AI-generated insight.

| Tier | Threshold | Colour |
|---|---|---|
| Critical | > 0.70 | 🔴 Red |
| High | 0.50 – 0.70 | 🟠 Orange |
| Medium | 0.30 – 0.50 | 🟡 Yellow |
| Low | < 0.30 | 🟢 Green |

### Page 2 — Risk Trend Explorer
Select any node to view its 372-day risk signal history (composite of static risk, severity, and event activity), overlaid with event counts and current disruption probability reference line.

### Page 3 — Live Event Feed
Filterable table of 1,296 GDELT events (Jun 1–7 2026), with country and event category filters, severity distribution histogram, and category breakdown.

---

## 🧠 Methodology

### Node Construction
- **Trade nodes** — 952 country × sector pairs derived from ISB India import data (6 CSV files)
- **Chokepoint nodes** — 10 strategic maritime chokepoints (Malacca, Suez, Hormuz, Panama, Taiwan Strait, etc.) with UNCTAD-sourced trade share weights

### Feature Engineering (22 features)
- **Event features** — rolling counts (7d/14d/30d), Goldstein scale mean/slope, mention acceleration, severity statistics
- **Network features** — upstream exposure via chokepoint dependency graph
- **Seasonal flags** — monsoon, typhoon, sanctions cycle indicators
- **Static node features** — trade vulnerability score, FSI country risk, HHI concentration, commodity importance

### Labelling Strategy
Weakly supervised binary labels derived programmatically from feature thresholds (Snorkel / Ratner et al. 2017):
- `severity_max_14d > 0.75`, OR
- `event_count_14d ≥ 5` AND `severity_mean_14d > 0.65`, OR
- `goldstein_slope_14d < -0.5` AND `event_count_14d ≥ 3`

### Drift Monitoring
Two separate checks to avoid PSI misuse on static features:
- **Dynamic features** (10 features) — PSI vs training decile bins; DRIFT > 0.20, MONITOR 0.10–0.20
- **Static features** (6 features) — mean deviation check; flag if > 5% deviation from training mean

---

## 📦 Key Dependencies

```
xgboost>=2.0          # gradient boosted trees
shap                  # model explainability
mlflow                # experiment tracking (SQLite backend)
streamlit             # dashboard
streamlit-folium      # Folium map in Streamlit
folium                # interactive maps
plotly                # trend charts
groq                  # LLaMA-3.3-70b API
boto3                 # AWS SNS alerts
schedule              # hourly alert checks
pandas, numpy         # data processing
scikit-learn          # preprocessing utilities
python-dotenv         # environment config
openpyxl              # Excel file reading
```

---

## 📚 Data Sources

| Source | Usage |
|---|---|
| [GDELT 2.0](https://www.gdeltproject.org/) | Real-time geopolitical event stream |
| [ISB India Data Portal](https://idbdata.isb.edu/trade) | India import trade data (2020–2024) |
| [EMIS](https://www.emis.com/) | Indian firm-level financial data by sector |
| [Fund for Peace FSI 2023](https://fragilestatesindex.org/) | Country fragility / political risk scores |
| [UNCTAD Maritime Transport 2024](https://unctad.org/rmt) | Chokepoint trade share weights |

---

## ⚠️ Known Limitations

- **Weakly supervised labels** — no ground truth disruption data exists; labels are rule-derived approximations
- **Single snapshot inference** — `disruption_prob` reflects the latest snapshot only; trend is approximated from feature history
- **India-centric** — nodes represent India's import dependency; export routes and domestic supply chains are out of scope
- **GDELT coverage** — event matching rate is ~40% (expected; US/UK/Australia have no India import trade nodes)

---

## 🪪 License

MIT License — see [LICENSE](LICENSE) for details.

---

*Built with Python 3.11 · XGBoost · Groq · Streamlit · GDELT *

🔗 **Live app:** https://supply-chain-risk-monitor-olfvjdzxxxjpbgqnzfdyap.streamlit.app  
📁 **GitHub:** https://github.com/Zirviy/supply-chain-risk-monitor

