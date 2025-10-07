# Goal-Prediction-Model
This repository contains a highly engineered football (soccer) match prediction pipeline that benchmarks six leading regression algorithms on football data. The model focuses on predicting match goals, match winner (moneyline), and over/under 2.5 goals outcomes using robust historical and statistical features.

This repository contains a highly engineered football (soccer) match prediction pipeline that benchmarks six leading regression algorithms on football data. The model focuses on predicting match goals, match winner (moneyline), and over/under 2.5 goals outcomes using robust historical and statistical features.

Features
Top Models: Ridge, Bayesian Ridge, ElasticNet, Lasso, Poisson, and XGBoost regressors, with optimized parameters and a unified API for comparison.

Data Handling: Automatic CSV importation, cleaning, proper missing value handling, and feature engineering, including dynamic calculation of match variables and odds-based probabilities like CTMCL.

Historical Features: Incorporates advanced historical and momentum stats, Elo ratings, recent team form, head-to-head records, and engineered features for each match.

Training & Validation: Uses a 60-20-20 time series split for train, validation, and test sets. Proper feature imputation and scaling are applied using only training data to avoid leakage.

Evaluation Metrics: Reports MAE, RMSE, R² for goal prediction, over/under accuracy and edge, moneyline accuracy and edge, median/mean error, and “TypeA” reliable predictions (absolute error under 0.2 goals).

Output: Generates a comparison table of all models, detailed breakdowns per metric, and highlights best-in-class models for each betting category.

Extensible: Easily add new features or models; supports custom CSV files and new league data with minimal changes.

Usage
Import historical football data (with expected goals, match stats, and betting odds).

Run the main script (vo10.py) to benchmark the six top models.

Review printed tables and logs for full performance and diagnostics.

Requirements
Python (>=3.7)

pandas, numpy, scikit-learn, xgboost

Summary
This repo enables robust, fair, and interpretable benchmarking of modern regression models for football outcome prediction, tailored for researchers and practitioners in sports analytics or betting.

