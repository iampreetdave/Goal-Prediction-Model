# Goal Prediction Model

![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-EC6C35?style=flat)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=streamlit&logoColor=white)

> A football match prediction pipeline benchmarking six regression algorithms to predict goals, moneyline, and over/under outcomes.

## About

A highly engineered football prediction system that benchmarks Ridge, Bayesian Ridge, ElasticNet, Lasso, Poisson, and XGBoost regressors on historical match data. The pipeline fetches live match data from the FootyStats API, computes advanced features (Elo ratings, rolling xG, form, H2H), trains regression models with proper time-series splitting, and outputs goal predictions, moneyline picks, and over/under 2.5 recommendations. Includes a Streamlit dashboard for running the full pipeline and viewing results interactively.

## Tech Stack

- **Language:** Python 3
- **ML:** Ridge, Bayesian Ridge, ElasticNet, Lasso, Poisson, XGBoost
- **Data:** Pandas, NumPy, scikit-learn
- **Dashboard:** Streamlit
- **API:** FootyStats API
- **Serialization:** joblib (pickle)

## Features

- **Six regression models** benchmarked side-by-side with unified evaluation
- **Advanced feature engineering** — Elo ratings, momentum xG, rolling form, H2H stats, CTMCL from odds
- **Time-series split** — 60/20/20 train/validation/test to prevent data leakage
- **Multiple prediction targets** — total goals, home/away goals, moneyline, over/under 2.5
- **Streamlit dashboard** — interactive 3-step pipeline (fetch → extract → predict) with live status
- **Live data pipeline** — fetches today's matches from the FootyStats API
- **Confidence categories** — predictions graded as High, Medium, or Low confidence
- **Comprehensive metrics** — MAE, RMSE, R², accuracy, edge, and TypeA (error < 0.2) reliable predictions
- **CSV export** — downloadable results from the dashboard

## Getting Started

### Prerequisites

- Python 3.8+
- FootyStats API key

### Installation

```bash
git clone https://github.com/iampreetdave/Goal-Prediction-Model.git
cd Goal-Prediction-Model
pip install pandas numpy scikit-learn xgboost streamlit joblib requests
```

### Run

**Streamlit Dashboard:**

```bash
streamlit run app.py
```

**CLI — benchmark models:**

```bash
python vo10.py
```

**CLI — fetch today's matches:**

```bash
python today_matches.py
```

## How It Works

1. **Data Ingestion:** Historical match data is loaded from CSV (with xG, odds, goals, and team stats)
2. **Feature Engineering:** Computes rolling averages, Elo ratings, form points, H2H records, and CTMCL from betting odds
3. **Model Training:** Six regressors are trained on the training set with StandardScaler preprocessing
4. **Evaluation:** Models are compared on MAE, RMSE, R², moneyline accuracy, and O/U edge across validation and test sets
5. **Live Prediction:** The Streamlit app orchestrates: fetch matches → extract features → run trained models → display results

## Project Structure

```
Goal-Prediction-Model/
├── app.py                  # Streamlit dashboard
├── vo10.py                 # Main benchmarking script (6 models)
├── best_version.py         # Optimized model variant
├── model_genrator.py       # Model training pipeline
├── fetch.py                # API data fetcher
├── today_matches.py        # Live match fetcher
├── api-8.py                # API integration variant
├── 15f.py / 40f.py         # Feature engineering variants
├── best (1).py             # Best model configuration
├── scripts/                # Utility scripts
├── *.pkl                   # Pre-trained model files
├── *.csv                   # Data files
└── README.md
```

## License

This project is licensed under the [MIT License](LICENSE).
