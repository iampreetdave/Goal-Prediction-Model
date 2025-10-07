"""
FOOTBALL PREDICTION MODEL - TOP 6 MODELS (FIXED VERSION)
Ridge, Bayesian Ridge, ElasticNet, Lasso, Poisson, XGBoost
With proper data handling and validation set
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge, Lasso, ElasticNet, PoissonRegressor, BayesianRidge
from sklearn.metrics import mean_absolute_error, accuracy_score, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.base import clone
from xgboost import XGBRegressor
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


class Top6FootballPredictor:
    """Optimized predictor using top 6 performing models including XGBoost"""

    def __init__(self):
        self.scaler = StandardScaler()
        self.feature_columns = []
        self.all_results = []
        self.train_medians = {}  # Store training medians for imputation

    @staticmethod
    def get_top6_models():
        """Return the top 6 models including XGBoost with improved parameters"""
        return {
            'ridge': Ridge(alpha=1.0, random_state=42),
            'bayesian_ridge': BayesianRidge(max_iter=300, alpha_1=1e-6, alpha_2=1e-6,
                                           lambda_1=1e-6, lambda_2=1e-6),
            'elasticnet': ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=5000, random_state=42),
            'lasso': Lasso(alpha=0.01, max_iter=5000, random_state=42),
            'poisson': PoissonRegressor(alpha=1.0, max_iter=1000),
            'xgboost': XGBRegressor(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                gamma=0,
                min_child_weight=1,
                random_state=42,
                objective='reg:squarederror',
                eval_metric='mae',
                early_stopping_rounds=10,
                n_jobs=-1
            ),
        }

    def load_and_clean_data(self, filepath):
        """Step 1: Load and clean data"""
        print("\n" + "="*80)
        print("STEP 1: LOADING AND CLEANING DATA")
        print("="*80)

        df = pd.read_csv(filepath)
        print(f"✓ Loaded {len(df)} matches from {filepath}")
        

        df['date'] = pd.to_datetime(df['date_GMT'], format='%b %d %Y - %I:%M%p', errors='coerce')
        df['home_team'] = df['home_team_name'].str.strip()
        df['away_team'] = df['away_team_name'].str.strip()

        if 'league_name' in df.columns:
            df['league'] = df['league_name']
        elif 'competition_name' in df.columns:
            df['league'] = df['competition_name']
        else:
            df['league'] = 'Unknown'

        df['home_goals'] = pd.to_numeric(df['home_team_goal_count'], errors='coerce')
        df['away_goals'] = pd.to_numeric(df['away_team_goal_count'], errors='coerce')
        df['total_goals'] = df['home_goals'] + df['away_goals']

        print(f"✓ Target variables created:")
        print(f"  - Mean home goals: {df['home_goals'].mean():.2f}")
        print(f"  - Mean away goals: {df['away_goals'].mean():.2f}")
        print(f"  - Mean total goals: {df['total_goals'].mean():.2f}")

        df['pre_home_xg'] = pd.to_numeric(df['Home Team Pre-Match xG'], errors='coerce')
        df['pre_away_xg'] = pd.to_numeric(df['Away Team Pre-Match xG'], errors='coerce')
        df['pre_total_xg'] = df['pre_home_xg'] + df['pre_away_xg']

        df['avg_goals_market'] = pd.to_numeric(df['average_goals_per_match_pre_match'], errors='coerce')
        df['home_ppg'] = pd.to_numeric(df['Pre-Match PPG (Home)'], errors='coerce')
        df['away_ppg'] = pd.to_numeric(df['Pre-Match PPG (Away)'], errors='coerce')
        df['pre_match_home_ppg'] = pd.to_numeric(df['Pre-Match PPG (Home)'], errors='coerce')
        df['pre_match_away_ppg'] = pd.to_numeric(df['Pre-Match PPG (Away)'], errors='coerce')
        df['btts_percentage_pre_match'] = pd.to_numeric(df['btts_percentage_pre_match'])
        if 'odds_ft_over25' in df.columns:
            df['odds_over25'] = pd.to_numeric(df['odds_ft_over25'], errors='coerce')

            print(f"  Rows before odds filtering: {len(df)}")
            df = df[df['odds_over25'].notna()]
            df = df[df['odds_over25'] > 1.01]
            print(f"  Rows after odds filtering: {len(df)}")

            df['IP_OVER'] = 1 / df['odds_over25']
            df['CTMCL'] = 2.5 + (df['IP_OVER'] - 0.5)

            df = df[np.isfinite(df['CTMCL'])]
            df = df[np.isfinite(df['IP_OVER'])]

            print(f"✓ CTMCL calculated:")
            print(f"  - Mean IP_OVER: {df['IP_OVER'].mean():.3f}")
            print(f"  - Mean CTMCL: {df['CTMCL'].mean():.3f}")
            print(f"  - CTMCL std dev: {df['CTMCL'].std():.3f}")
            print(f"  - CTMCL range: [{df['CTMCL'].min():.2f}, {df['CTMCL'].max():.2f}]")
        else:
            df['CTMCL'] = 2.5

        df['btts_pct'] = pd.to_numeric(df['btts_percentage_pre_match'], errors='coerce')
        df['over15_pct'] = pd.to_numeric(df['over_15_percentage_pre_match'], errors='coerce')
        df['over25_pct'] = pd.to_numeric(df['over_25_percentage_pre_match'], errors='coerce')
        df['over35_pct'] = pd.to_numeric(df['over_35_percentage_pre_match'], errors='coerce')

        for col in ['odds_ft_home_team_win', 'odds_ft_draw', 'odds_ft_away_team_win']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                df[f'{col}_prob'] = 1 / df[col].replace(0, np.nan)

        df['game_week'] = pd.to_numeric(df['Game Week'], errors='coerce')

        essential_cols = [
            'date', 'home_team', 'away_team', 'home_goals', 'away_goals',
            'pre_home_xg', 'pre_away_xg', 'CTMCL', 'home_ppg', 'away_ppg',
            'avg_goals_market','btts_pct', 'over35_pct', 'over25_pct', 'over15_pct',
        ]

        before_clean = len(df)
        for col in essential_cols:
            if col in ['home_goals', 'away_goals']:
                continue
            df = df[df[col].notna()]
            if col in ['pre_home_xg', 'pre_away_xg', 'home_ppg', 'away_ppg', 'avg_goals_market','btts_pct','over15_pct','over25_pct','over35_pct']:
                df = df[df[col] > 0]

        print(f"✓ Removed {before_clean - len(df)} rows with missing/zero essential data")
        print(f"✓ Dataset after cleaning: {len(df)} matches")

        return df.sort_values('date').reset_index(drop=True)

    def calculate_elo_rating(self, result, home_elo, away_elo, k_factor=20):
        """Calculate Elo rating change"""
        expected_home = 1 / (1 + 10 ** ((away_elo - home_elo) / 400))
        expected_away = 1 - expected_home
        new_home_elo = home_elo + k_factor * (result - expected_home)
        new_away_elo = away_elo + k_factor * ((1 - result) - expected_away)
        return new_home_elo, new_away_elo

    def create_historical_features(self, df):
        """Step 2: Create historical features"""
        print("\n" + "="*80)
        print("STEP 2: CREATING HISTORICAL FEATURES")
        print("="*80)

        df_sorted = df.copy()

        feature_cols = [
            'home_xg_avg', 'away_xg_avg',
            'home_xg_recent', 'away_xg_recent',
            'home_xg_momentum', 'away_xg_momentum',
            'home_goals_avg', 'away_goals_avg',
            'home_goals_conceded_avg', 'away_goals_conceded_avg',
            'home_recent_goals', 'away_recent_goals',
            'home_recent_conceded', 'away_recent_conceded',
            'home_elo', 'away_elo', 'elo_diff',
            'h2h_home_wins', 'h2h_away_wins', 'h2h_draws',
            'h2h_home_goals_avg', 'h2h_away_goals_avg', 'h2h_total_goals_avg',
            'home_win_streak', 'away_win_streak',
            'home_form_points', 'away_form_points',
        ]

        for col in feature_cols:
            df_sorted[col] = np.nan

        team_elo = {}

        print("Computing features row by row...")

        for i in range(len(df_sorted)):
            if i % 500 == 0 and i > 0:
                print(f"  Processed {i}/{len(df_sorted)} matches ({i/len(df_sorted)*100:.1f}% complete)...")

            home_team = df_sorted.iloc[i]['home_team']
            away_team = df_sorted.iloc[i]['away_team']
            past_data = df_sorted.iloc[:i]

            if home_team not in team_elo:
                team_elo[home_team] = 1500
            if away_team not in team_elo:
                team_elo[away_team] = 1500

            df_sorted.at[i, 'home_elo'] = team_elo[home_team]
            df_sorted.at[i, 'away_elo'] = team_elo[away_team]
            df_sorted.at[i, 'elo_diff'] = team_elo[home_team] - team_elo[away_team]

            if len(past_data) == 0:
                home_goals = df_sorted.iloc[i]['home_goals']
                away_goals = df_sorted.iloc[i]['away_goals']
                result = 1.0 if home_goals > away_goals else (0.0 if home_goals < away_goals else 0.5)
                new_home_elo, new_away_elo = self.calculate_elo_rating(
                    result, team_elo[home_team], team_elo[away_team]
                )
                team_elo[home_team] = new_home_elo
                team_elo[away_team] = new_away_elo
                continue

            # HEAD-TO-HEAD
            h2h_matches = past_data[
                ((past_data['home_team'] == home_team) & (past_data['away_team'] == away_team)) |
                ((past_data['home_team'] == away_team) & (past_data['away_team'] == home_team))
            ]

            if len(h2h_matches) > 0:
                home_wins = 0
                away_wins = 0
                draws = 0
                h2h_home_goals = []
                h2h_away_goals = []

                for _, match in h2h_matches.iterrows():
                    if match['home_team'] == home_team:
                        h2h_home_goals.append(match['home_goals'])
                        h2h_away_goals.append(match['away_goals'])
                        if match['home_goals'] > match['away_goals']:
                            home_wins += 1
                        elif match['home_goals'] < match['away_goals']:
                            away_wins += 1
                        else:
                            draws += 1
                    else:
                        h2h_home_goals.append(match['away_goals'])
                        h2h_away_goals.append(match['home_goals'])
                        if match['away_goals'] > match['home_goals']:
                            home_wins += 1
                        elif match['away_goals'] < match['home_goals']:
                            away_wins += 1
                        else:
                            draws += 1

                df_sorted.at[i, 'h2h_home_wins'] = home_wins
                df_sorted.at[i, 'h2h_away_wins'] = away_wins
                df_sorted.at[i, 'h2h_draws'] = draws
                df_sorted.at[i, 'h2h_home_goals_avg'] = np.mean(h2h_home_goals)
                df_sorted.at[i, 'h2h_away_goals_avg'] = np.mean(h2h_away_goals)
                df_sorted.at[i, 'h2h_total_goals_avg'] = np.mean(h2h_home_goals) + np.mean(h2h_away_goals)

            # HOME TEAM STATS
            home_past = past_data[past_data['home_team'] == home_team]
            if len(home_past) >= 3:
                df_sorted.at[i, 'home_xg_avg'] = home_past['pre_home_xg'].mean()
                df_sorted.at[i, 'home_goals_avg'] = home_past['home_goals'].mean()
                df_sorted.at[i, 'home_goals_conceded_avg'] = home_past['away_goals'].mean()

                recent = home_past.tail(5)
                df_sorted.at[i, 'home_xg_recent'] = recent['pre_home_xg'].mean()
                df_sorted.at[i, 'home_recent_goals'] = recent['home_goals'].mean()
                df_sorted.at[i, 'home_recent_conceded'] = recent['away_goals'].mean()

                recent_avg = recent['pre_home_xg'].mean()
                long_avg = home_past['pre_home_xg'].mean()
                df_sorted.at[i, 'home_xg_momentum'] = recent_avg - long_avg

                win_streak = 0
                for _, match in recent.iloc[::-1].iterrows():
                    if match['home_goals'] > match['away_goals']:
                        win_streak += 1
                    else:
                        break
                df_sorted.at[i, 'home_win_streak'] = win_streak

                form_points = 0
                for _, match in recent.iterrows():
                    if match['home_goals'] > match['away_goals']:
                        form_points += 3
                    elif match['home_goals'] == match['away_goals']:
                        form_points += 1
                df_sorted.at[i, 'home_form_points'] = form_points

            # AWAY TEAM STATS
            away_past = past_data[past_data['away_team'] == away_team]
            if len(away_past) >= 3:
                df_sorted.at[i, 'away_xg_avg'] = away_past['pre_away_xg'].mean()
                df_sorted.at[i, 'away_goals_avg'] = away_past['away_goals'].mean()
                df_sorted.at[i, 'away_goals_conceded_avg'] = away_past['home_goals'].mean()

                recent = away_past.tail(5)
                df_sorted.at[i, 'away_xg_recent'] = recent['pre_away_xg'].mean()
                df_sorted.at[i, 'away_recent_goals'] = recent['away_goals'].mean()
                df_sorted.at[i, 'away_recent_conceded'] = recent['home_goals'].mean()

                recent_avg = recent['pre_away_xg'].mean()
                long_avg = away_past['pre_away_xg'].mean()
                df_sorted.at[i, 'away_xg_momentum'] = recent_avg - long_avg

                win_streak = 0
                for _, match in recent.iloc[::-1].iterrows():
                    if match['away_goals'] > match['home_goals']:
                        win_streak += 1
                    else:
                        break
                df_sorted.at[i, 'away_win_streak'] = win_streak

                form_points = 0
                for _, match in recent.iterrows():
                    if match['away_goals'] > match['home_goals']:
                        form_points += 3
                    elif match['away_goals'] == match['home_goals']:
                        form_points += 1
                df_sorted.at[i, 'away_form_points'] = form_points

            # Update Elo
            home_goals = df_sorted.iloc[i]['home_goals']
            away_goals = df_sorted.iloc[i]['away_goals']
            result = 1.0 if home_goals > away_goals else (0.0 if home_goals < away_goals else 0.5)
            new_home_elo, new_away_elo = self.calculate_elo_rating(
                result, team_elo[home_team], team_elo[away_team]
            )
            team_elo[home_team] = new_home_elo
            team_elo[away_team] = new_away_elo

        print(f"  Processed {len(df_sorted)}/{len(df_sorted)} matches (100.0% complete)...")
        print("✓ Historical features computed")

        # Impute missing values
        defaults = {
            'home_xg_avg': 1.3, 'away_xg_avg': 1.1,
            'home_xg_recent': 1.3, 'away_xg_recent': 1.1,
            'home_xg_momentum': 0.0, 'away_xg_momentum': 0.0,
            'home_goals_avg': 1.4, 'away_goals_avg': 1.1,
            'home_goals_conceded_avg': 1.1, 'away_goals_conceded_avg': 1.4,
            'home_recent_goals': 1.4, 'away_recent_goals': 1.1,
            'home_recent_conceded': 1.1, 'away_recent_conceded': 1.4,
            'home_elo': 1500, 'away_elo': 1500, 'elo_diff': 0,
            'h2h_home_wins': 0, 'h2h_away_wins': 0, 'h2h_draws': 0,
            'h2h_home_goals_avg': 1.3, 'h2h_away_goals_avg': 1.3,
            'h2h_total_goals_avg': 2.6,
            'home_win_streak': 0, 'away_win_streak': 0,
            'home_form_points': 5, 'away_form_points': 5
        }

        for col, default_val in defaults.items():
            df_sorted[col].fillna(default_val, inplace=True)

        # Filter invalid rows
        feature_check_cols = ['home_xg_avg', 'away_xg_avg', 'home_elo', 'away_elo']
        before_filter = len(df_sorted)

        for col in feature_check_cols:
            df_sorted = df_sorted[df_sorted[col] > 0]

        if before_filter - len(df_sorted) > 0:
            print(f"✓ Removed {before_filter - len(df_sorted)} rows with invalid historical features")

        df_final = df_sorted.iloc[30:].reset_index(drop=True)
        print(f"✓ Skipped first 30 matches for stability")
        print(f"✓ Final dataset: {len(df_final)} matches")

        return df_final

    def prepare_features(self, df):
        """Step 3: Prepare features (WITHOUT imputation - that happens after split)"""
        print("\n" + "="*80)
        print("STEP 3: PREPARING FEATURES")
        print("="*80)

        self.feature_columns = [
            'CTMCL',
            'avg_goals_market',
            'pre_total_xg',
            'pre_match_home_ppg',
            'pre_match_away_ppg',
            'home_xg_avg',
            'away_xg_avg',
            'home_xg_momentum',
            'away_xg_momentum',
            'home_goals_conceded_avg',
            'away_goals_conceded_avg',
            'btts_pct',
            'over35_pct',
            'over25_pct',
            'over15_pct'
        ]

        for col in ['odds_ft_home_team_win_prob', 'odds_ft_away_team_win_prob']:
            if col in df.columns:
                self.feature_columns.append(col)

        valid_features = [f for f in self.feature_columns if f in df.columns]
        self.feature_columns = valid_features

        print(f"✓ Using {len(self.feature_columns)} features:")
        for i, feat in enumerate(self.feature_columns, 1):
            print(f"  {i:2d}. {feat}")

        X = df[self.feature_columns].copy()

        # Check for missing values but DON'T impute yet
        missing_count = X.isnull().sum().sum()
        if missing_count > 0:
            print(f"\n⚠️ Found {missing_count} missing values (will be imputed after train/val/test split)")
        else:
            print("\n✓ No missing values found in features")

        return X, df

    def train_single_model(self, model_name, model, X_train, y_home_train, y_away_train,
                          X_val, y_home_val, y_away_val,
                          X_test, y_home_test, y_away_test, y_total_test, ctmcl_test, df_test):
        """Train a single model with validation set"""

        print(f"\n{'='*80}")
        print(f"TRAINING MODEL: {model_name.upper()}")
        print(f"{'='*80}")

        # Scale data for linear models, but XGBoost doesn't need scaling
        if model_name == 'xgboost':
            X_train_scaled = X_train.values
            X_val_scaled = X_val.values
            X_test_scaled = X_test.values
            print("Note: XGBoost doesn't require feature scaling")
        else:
            X_train_scaled = self.scaler.transform(X_train)
            X_val_scaled = self.scaler.transform(X_val)
            X_test_scaled = self.scaler.transform(X_test)

        print(f"Training set shape: {X_train_scaled.shape}")
        print(f"Validation set shape: {X_val_scaled.shape}")
        print(f"Test set shape: {X_test_scaled.shape}")

        # Train home model with validation for XGBoost
        print(f"\n→ Training HOME goals predictor...")
        home_model = model
        
        if model_name == 'xgboost':
            home_model.fit(
                X_train_scaled, y_home_train,
                eval_set=[(X_val_scaled, y_home_val)],
                verbose=False
            )
        else:
            home_model.fit(X_train_scaled, y_home_train)
        
        pred_home = home_model.predict(X_test_scaled)
        pred_home = np.maximum(pred_home, 0)

        home_mae = mean_absolute_error(y_home_test, pred_home)
        home_rmse = np.sqrt(mean_squared_error(y_home_test, pred_home))
        home_r2 = r2_score(y_home_test, pred_home)

        print(f"  Home Goals - MAE: {home_mae:.4f} | RMSE: {home_rmse:.4f} | R²: {home_r2:.4f}")
        print(f"  Avg predicted home goals: {pred_home.mean():.3f} (actual: {y_home_test.mean():.3f})")

        # Train away model with validation for XGBoost
        print(f"\n→ Training AWAY goals predictor...")
        away_model = clone(model)
        
        if model_name == 'xgboost':
            away_model.fit(
                X_train_scaled, y_away_train,
                eval_set=[(X_val_scaled, y_away_val)],
                verbose=False
            )
        else:
            away_model.fit(X_train_scaled, y_away_train)
        
        pred_away = away_model.predict(X_test_scaled)
        pred_away = np.maximum(pred_away, 0)

        away_mae = mean_absolute_error(y_away_test, pred_away)
        away_rmse = np.sqrt(mean_squared_error(y_away_test, pred_away))
        away_r2 = r2_score(y_away_test, pred_away)

        print(f"  Away Goals - MAE: {away_mae:.4f} | RMSE: {away_rmse:.4f} | R²: {away_r2:.4f}")
        print(f"  Avg predicted away goals: {pred_away.mean():.3f} (actual: {y_away_test.mean():.3f})")

        # Calculate totals
        pred_total = pred_home + pred_away
        total_mae = mean_absolute_error(y_total_test, pred_total)
        total_rmse = np.sqrt(mean_squared_error(y_total_test, pred_total))
        total_r2 = r2_score(y_total_test, pred_total)

        print(f"\n→ TOTAL goals performance:")
        print(f"  MAE: {total_mae:.4f} | RMSE: {total_rmse:.4f} | R²: {total_r2:.4f}")
        print(f"  Avg predicted total: {pred_total.mean():.3f} (actual: {y_total_test.mean():.3f})")
        print(f"  Prediction bias: {pred_total.mean() - y_total_test.mean():+.3f} goals")

        # Over/Under analysis
        print(f"\n→ OVER/UNDER CTMCL Analysis:")
        y_ou_actual = (y_total_test.values > ctmcl_test.values).astype(int)
        y_ou_pred = (pred_total > ctmcl_test.values).astype(int)
        ou_accuracy = accuracy_score(y_ou_actual, y_ou_pred)

        actual_over_count = y_ou_actual.sum()
        pred_over_count = y_ou_pred.sum()

        print(f"  Accuracy: {ou_accuracy:.1%}")
        print(f"  Edge over random (50%): {(ou_accuracy - 0.5) * 100:+.1f}%")
        print(f"  Actual OVER matches: {actual_over_count}/{len(y_ou_actual)} ({actual_over_count/len(y_ou_actual)*100:.1f}%)")
        print(f"  Predicted OVER matches: {pred_over_count}/{len(y_ou_pred)} ({pred_over_count/len(y_ou_pred)*100:.1f}%)")

        # True positives, false positives, etc.
        tp = ((y_ou_pred == 1) & (y_ou_actual == 1)).sum()
        fp = ((y_ou_pred == 1) & (y_ou_actual == 0)).sum()
        tn = ((y_ou_pred == 0) & (y_ou_actual == 0)).sum()
        fn = ((y_ou_pred == 0) & (y_ou_actual == 1)).sum()

        print(f"  True Positives (predicted OVER, was OVER): {tp}")
        print(f"  False Positives (predicted OVER, was UNDER): {fp}")
        print(f"  True Negatives (predicted UNDER, was UNDER): {tn}")
        print(f"  False Negatives (predicted UNDER, was OVER): {fn}")

        if pred_over_count > 0:
            precision = tp / (tp + fp)
            print(f"  Precision (when you predict OVER, how often right): {precision:.1%}")

        # Moneyline analysis
        print(f"\n→ MONEYLINE (Match Winner) Analysis:")
        pred_home_win = (pred_home > pred_away).astype(int)
        pred_away_win = (pred_away > pred_home).astype(int)
        actual_home_win = (df_test['home_goals'] > df_test['away_goals']).astype(int)
        actual_away_win = (df_test['away_goals'] > df_test['home_goals']).astype(int)

        moneyline_correct = (
            (pred_home_win == actual_home_win.values) & (actual_home_win.values == 1) |
            (pred_away_win == actual_away_win.values) & (actual_away_win.values == 1)
        ).astype(int)
        ml_accuracy = moneyline_correct.mean()

        print(f"  Accuracy: {ml_accuracy:.1%}")
        print(f"  Edge over random (33.3%): {(ml_accuracy - 0.333) * 100:+.1f}%")
        print(f"  Predicted home wins: {pred_home_win.sum()} | Actual: {actual_home_win.sum()}")
        print(f"  Predicted away wins: {pred_away_win.sum()} | Actual: {actual_away_win.sum()}")

        # Prediction error (Delta) analysis
        print(f"\n→ PREDICTION ERROR (Delta) Analysis:")
        prediction_error = np.abs(pred_total - y_total_test.values)
        avg_error = prediction_error.mean()
        median_error = np.median(prediction_error)

        print(f"  Mean prediction error: {avg_error:.4f} goals")
        print(f"  Median prediction error: {median_error:.4f} goals")
        print(f"  Max prediction error: {prediction_error.max():.4f} goals")
        print(f"  Min prediction error: {prediction_error.min():.4f} goals")

        # Delta buckets
        print(f"\n→ DELTA BUCKET DISTRIBUTION:")
        buckets = [(0, 0.2, 'TypeA'), (0.2, 0.4, 'TypeB'), (0.4, 0.6, 'TypeC'),
                   (0.6, 0.8, 'TypeD'), (0.8, 1.0, 'TypeE'), (1.0, 100, 'TypeF')]

        for low, high, bucket_name in buckets:
            mask = (prediction_error >= low) & (prediction_error < high)
            count = mask.sum()
            pct = count / len(prediction_error) * 100

            if count > 0:
                bucket_ou_acc = (y_ou_pred[mask] == y_ou_actual[mask]).mean()
                bucket_ml_acc = moneyline_correct[mask].mean()
                avg_ctmcl = ctmcl_test.values[mask].mean()

                print(f"  {bucket_name} ({low:.1f}-{high:.1f}): {count:4d} matches ({pct:5.1f}%)")
                print(f"    O/U Acc: {bucket_ou_acc:.1%} | ML Acc: {bucket_ml_acc:.1%} | Avg CTMCL: {avg_ctmcl:.2f}")

        # TypeA detailed analysis
        type_a_mask = prediction_error <= 0.2
        type_a_count = type_a_mask.sum()

        if type_a_count > 0:
            type_a_accuracy = (y_ou_pred[type_a_mask] == y_ou_actual[type_a_mask]).mean()
            print(f"\n→ TypeA GOLD STANDARD Predictions (error ≤ 0.2 goals):")
            print(f"  Count: {type_a_count} matches ({type_a_count/len(prediction_error)*100:.1f}% of test set)")
            print(f"  O/U Accuracy: {type_a_accuracy:.1%}")
            print(f"  Average error in TypeA: {prediction_error[type_a_mask].mean():.4f} goals")
        else:
            type_a_accuracy = 0.0

        print(f"\n{'='*80}\n")

        return {
            'model_name': model_name,
            'home_mae': home_mae,
            'home_rmse': home_rmse,
            'home_r2': home_r2,
            'away_mae': away_mae,
            'away_rmse': away_rmse,
            'away_r2': away_r2,
            'total_mae': total_mae,
            'total_rmse': total_rmse,
            'total_r2': total_r2,
            'ou_accuracy': ou_accuracy,
            'ml_accuracy': ml_accuracy,
            'avg_prediction_error': avg_error,
            'median_prediction_error': median_error,
            'type_a_accuracy': type_a_accuracy,
            'type_a_count': type_a_count,
            'predictions': {
                'home': pred_home,
                'away': pred_away,
                'total': pred_total
            },
            'models': {
                'home': home_model,
                'away': away_model
            }
        }

    def compare_top6_models(self, X, df):
        """Compare the top 6 models with train/val/test split"""
        print("\n" + "="*80)
        print("STEP 4: COMPARING TOP 6 MODELS WITH VALIDATION SET")
        print("="*80)

        models_dict = self.get_top6_models()
        print(f"\nTesting 6 best-performing models:")
        for i, name in enumerate(models_dict.keys(), 1):
            print(f"  {i}. {name}")

        # Prepare data
        y_home = df['home_goals']
        y_away = df['away_goals']
        y_total = df['total_goals']
        ctmcl = df['CTMCL']

        # Split: 60% train, 20% validation, 20% test
        train_idx = int(len(X) * 0.6)
        val_idx = int(len(X) * 0.8)

        X_train = X.iloc[:train_idx].copy()
        X_val = X.iloc[train_idx:val_idx].copy()
        X_test = X.iloc[val_idx:].copy()

        y_home_train = y_home.iloc[:train_idx]
        y_home_val = y_home.iloc[train_idx:val_idx]
        y_home_test = y_home.iloc[val_idx:]

        y_away_train = y_away.iloc[:train_idx]
        y_away_val = y_away.iloc[train_idx:val_idx]
        y_away_test = y_away.iloc[val_idx:]

        y_total_test = y_total.iloc[val_idx:]
        ctmcl_test = ctmcl.iloc[val_idx:]

        df_test = df.iloc[val_idx:].copy()

        print(f"\n✓ Data split complete:")
        print(f"  Training set: {len(X_train)} matches ({len(X_train)/len(X)*100:.1f}%)")
        print(f"  Validation set: {len(X_val)} matches ({len(X_val)/len(X)*100:.1f}%)")
        print(f"  Test set: {len(X_test)} matches ({len(X_test)/len(X)*100:.1f}%)")
        print(f"  Date range (train): {df.iloc[:train_idx]['date'].min()} to {df.iloc[:train_idx]['date'].max()}")
        print(f"  Date range (val): {df.iloc[train_idx:val_idx]['date'].min()} to {df.iloc[train_idx:val_idx]['date'].max()}")
        print(f"  Date range (test): {df_test['date'].min()} to {df_test['date'].max()}")

        # ✅ FIX: Impute missing values using ONLY training data
        print(f"\n✓ Handling missing values...")
        for col in X_train.columns:
            if X_train[col].isnull().any():
                # Calculate median from training data only
                self.train_medians[col] = X_train[col].median()
                # Apply to all sets
                X_train[col].fillna(self.train_medians[col], inplace=True)
                X_val[col].fillna(self.train_medians[col], inplace=True)
                X_test[col].fillna(self.train_medians[col], inplace=True)
                print(f"  Imputed {col} with training median: {self.train_medians[col]:.4f}")

        # ✅ FIX: Fit scaler ONCE on training data
        print(f"\n✓ Fitting scaler on training data...")
        self.scaler.fit(X_train)
        print(f"  Scaler fitted and stored for all models")

        # Train all models
        results = []

        for i, (name, model) in enumerate(models_dict.items(), 1):
            print(f"\n[{i}/6] Processing {name}...")

            result = self.train_single_model(
                name, model, X_train, y_home_train, y_away_train,
                X_val, y_home_val, y_away_val,
                X_test, y_home_test, y_away_test, y_total_test, ctmcl_test, df_test
            )

            results.append(result)
            self.all_results.append(result)

        # Create comparison DataFrame
        comparison_df = pd.DataFrame([
            {
                'Model': r['model_name'],
                'Total MAE': r['total_mae'],
                'Total RMSE': r['total_rmse'],
                'R²': r['total_r2'],
                'Home MAE': r['home_mae'],
                'Away MAE': r['away_mae'],
                'O/U Acc': r['ou_accuracy'],
                'ML Acc': r['ml_accuracy'],
                'Avg Error': r['avg_prediction_error'],
                'Med Error': r['median_prediction_error'],
                'TypeA Acc': r['type_a_accuracy'],
                'TypeA Count': r['type_a_count']
            }
            for r in results
        ]).sort_values('Total MAE')

        # Display final comparison
        print("\n" + "="*80)
        print("FINAL MODEL COMPARISON - TOP 6 MODELS")
        print("="*80)
        print(comparison_df.to_string(index=False, float_format=lambda x: f'{x:.4f}'))

        print("\n" + "="*80)
        print("DETAILED RANKINGS")
        print("="*80)

        for rank, (idx, row) in enumerate(comparison_df.iterrows(), 1):
            print(f"\nRank #{rank}: {row['Model'].upper()}")
            print(f"  Goal Prediction: MAE {row['Total MAE']:.4f} | RMSE {row['Total RMSE']:.4f} | R² {row['R²']:.4f}")
            print(f"  Home/Away: H_MAE {row['Home MAE']:.4f} | A_MAE {row['Away MAE']:.4f}")
            print(f"  Over/Under: {row['O/U Acc']:.1%} (edge: {(row['O/U Acc']-0.5)*100:+.1f}%)")
            print(f"  Moneyline: {row['ML Acc']:.1%} (edge: {(row['ML Acc']-0.333)*100:+.1f}%)")
            print(f"  TypeA: {int(row['TypeA Count'])} matches at {row['TypeA Acc']:.1%} accuracy")

        print("\n" + "="*80)
        print("WINNER ANALYSIS")
        print("="*80)

        winner = comparison_df.iloc[0]
        print(f"\n🏆 BEST OVERALL: {winner['Model'].upper()}")
        print(f"\n  Why it won:")
        print(f"  • Lowest MAE: {winner['Total MAE']:.4f} goals")
        print(f"  • O/U Edge: +{(winner['O/U Acc']-0.5)*100:.1f}% over random")
        print(f"  • Moneyline Edge: +{(winner['ML Acc']-0.333)*100:.1f}% over random")
        print(f"  • TypeA reliability: {int(winner['TypeA Count'])} high-confidence predictions")

        # Category winners
        best_ou = comparison_df.loc[comparison_df['O/U Acc'].idxmax()]
        best_ml = comparison_df.loc[comparison_df['ML Acc'].idxmax()]
        most_typea = comparison_df.loc[comparison_df['TypeA Count'].idxmax()]

        print(f"\n  Category Leaders:")
        print(f"  • Best O/U: {best_ou['Model']} ({best_ou['O/U Acc']:.1%})")
        print(f"  • Best Moneyline: {best_ml['Model']} ({best_ml['ML Acc']:.1%})")
        print(f"  • Most TypeA: {most_typea['Model']} ({int(most_typea['TypeA Count'])} matches)")

        return comparison_df, results


def main_top6(filepath='ab_data.csv'):
    """Main function to run top 6 models comparison"""

    print("\n" + "="*80)
    print("FOOTBALL PREDICTION - TOP 6 MODELS ANALYSIS (FIXED VERSION)")
    print("Ridge | Bayesian Ridge | ElasticNet | Lasso | Poisson | XGBoost")
    print("="*80)

    predictor = Top6FootballPredictor()

    try:
        df = predictor.load_and_clean_data(filepath)
        df = predictor.create_historical_features(df)
        X, df = predictor.prepare_features(df)
        comparison_df, results = predictor.compare_top6_models(X, df)

        print("\n" + "="*80)
        print("✅ ANALYSIS COMPLETE")
        print("="*80)
        print(f"\nAll results stored in predictor.all_results")
        print(f"Comparison table available as comparison_df")

        return predictor, comparison_df, results

    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None


if __name__ == "__main__":
    predictor, comparison, results = main_top6(filepath='ab_data.csv')