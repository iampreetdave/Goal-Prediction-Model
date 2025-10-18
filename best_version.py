"""
FOOTBALL PREDICTION MODEL - ADAPTED FOR NEW DATASET
3 Models: Poisson, Ridge, Lasso
Enhanced features with pre-match focus + Historical balance
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge, Lasso, PoissonRegressor
from sklearn.metrics import mean_absolute_error, accuracy_score, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.base import clone
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ========== CONFIGURATION ==========
# Toggle this to limit rows processed (None = all rows)
MAX_ROWS_TO_PROCESS = None  # Change to 500, 1000, 5000, etc. to limit
# ===================================


class FeatureWeights:
    """Define feature importance weights - UPDATED for new features"""

    @staticmethod
    def get_weights():
        return {
            # Market & Odds (High importance)
            'CTMCL': 2.0,
            'avg_goals_market': 1.4,
            'odds_ft_1_prob': 1.3,
            'odds_ft_2_prob': 1.3,
            
            # Pre-match xG (High importance)
            'pre_total_xg': 1.3,
            'team_a_xg_prematch': 1.3,
            'team_b_xg_prematch': 1.3,
            
            # Historical xG
            'home_xg_avg': 1.2,
            'away_xg_avg': 1.2,
            'home_xg_momentum': 1.1,
            'away_xg_momentum': 1.1,
            
            # PPG & Form
            'pre_match_home_ppg': 1.2,
            'pre_match_away_ppg': 1.2,
            'home_form_points': 1.1,
            'away_form_points': 1.1,
            
            # Goals
            'home_goals_conceded_avg': 1.0,
            'away_goals_conceded_avg': 1.0,
            'home_goals_avg': 1.0,
            'away_goals_avg': 1.0,
            
            # Shots & Accuracy
            'home_shots_accuracy_avg': 1.1,
            'away_shots_accuracy_avg': 1.1,
            'home_shots_on_target_avg': 1.0,
            'away_shots_on_target_avg': 1.0,
            
            # NEW: Attacks
            'home_dangerous_attacks_avg': 1.1,
            'away_dangerous_attacks_avg': 1.1,
            'home_attacks_avg': 0.9,
            'away_attacks_avg': 0.9,
            
            # Corners
            'home_corners_avg': 0.9,
            'away_corners_avg': 0.9,
            
            # Cards
            'home_yellow_cards_avg': 0.8,
            'away_yellow_cards_avg': 0.8,
            'home_red_cards_avg': 0.7,
            'away_red_cards_avg': 0.7,
            
            # Potentials
            'btts_potential': 1.0,
            'o35_potential': 1.0,
            'o25_potential': 1.1,
            'o15_potential': 0.9,
            'o05_potential': 0.9,
            
            # NEW: Penalties
            'home_penalties_won_avg': 0.9,
            'away_penalties_won_avg': 0.9,
            
            # H2H
            'days_since_last_home': 0.8,
            'days_since_last_away': 0.8,
            'h2h_matches_played': 0.9,
            'h2h_home_wins': 1.0,
            'h2h_away_wins': 1.0,
            'h2h_total_goals_avg': 1.1,
            
            # Elo
            'elo_diff': 1.0,
            
            # NEW: League strength
            'league_avg_goals': 0.9,
        }


class DataLoader:
    """Handles data loading and strict cleaning - ADAPTED FOR NEW DATASET"""

    def __init__(self):
        self.rows_dropped = {}

    def load_and_clean(self, filepath):
        print("\n" + "="*80)
        print("STEP 1: LOADING AND CLEANING DATA (NEW DATASET FORMAT)")
        print("="*80)

        df = pd.read_csv(filepath)
        
        if MAX_ROWS_TO_PROCESS is not None:
            df = df.head(MAX_ROWS_TO_PROCESS)
            print(f"⚙️  Row limit applied: Processing {len(df)} rows (MAX_ROWS_TO_PROCESS={MAX_ROWS_TO_PROCESS})")
        
        initial_rows = len(df)
        print(f"✓ Initial dataset: {initial_rows} matches")

        df['date'] = pd.to_datetime(df['date_unix'], unit='s', errors='coerce')
        before = len(df)
        df = df.dropna(subset=['date'])
        self._log_drop('invalid_dates', before - len(df))

        df['home_team'] = df['home_name'].str.strip()
        df['away_team'] = df['away_name'].str.strip()
        df['league'] = df['fetched_league_name'].fillna('Unknown')

        df['home_goals'] = pd.to_numeric(df['homeGoalCount'], errors='coerce')
        df['away_goals'] = pd.to_numeric(df['awayGoalCount'], errors='coerce')

        before = len(df)
        df = df.dropna(subset=['home_goals', 'away_goals'])
        self._log_drop('missing_goals', before - len(df))

        df = df[(df['home_goals'] >= 0) & (df['away_goals'] >= 0)]
        df['total_goals'] = df['home_goals'] + df['away_goals']

        print(f"✓ Target variables:")
        print(f"  Home goals: μ={df['home_goals'].mean():.2f}, σ={df['home_goals'].std():.2f}")
        print(f"  Away goals: μ={df['away_goals'].mean():.2f}, σ={df['away_goals'].std():.2f}")
        print(f"  Total goals: μ={df['total_goals'].mean():.2f}, σ={df['total_goals'].std():.2f}")

        df['pre_home_xg'] = pd.to_numeric(df['team_a_xg_prematch'], errors='coerce')
        df['pre_away_xg'] = pd.to_numeric(df['team_b_xg_prematch'], errors='coerce')
        
        df['team_a_xg_prematch'] = df['pre_home_xg']
        df['team_b_xg_prematch'] = df['pre_away_xg']

        before = len(df)
        df = df.dropna(subset=['pre_home_xg', 'pre_away_xg'])
        df = df[(df['pre_home_xg'] > 0) & (df['pre_away_xg'] > 0)]
        self._log_drop('invalid_xg', before - len(df))

        df['pre_total_xg'] = df['pre_home_xg'] + df['pre_away_xg']

        df['home_ppg'] = pd.to_numeric(df['pre_match_home_ppg'], errors='coerce')
        df['away_ppg'] = pd.to_numeric(df['pre_match_away_ppg'], errors='coerce')
        df['pre_match_home_ppg'] = df['home_ppg']
        df['pre_match_away_ppg'] = df['away_ppg']

        before = len(df)
        df = df.dropna(subset=['home_ppg', 'away_ppg'])
        df = df[(df['home_ppg'] >= 0) & (df['away_ppg'] >= 0)]
        self._log_drop('invalid_ppg', before - len(df))

        if 'o25_potential' in df.columns and 'o15_potential' in df.columns:
            df['avg_goals_market'] = 2.5 + (df['o25_potential'] - df['o15_potential']) / 100
            df['avg_goals_market'] = df['avg_goals_market'].clip(0.5, 6.0)
        else:
            df['avg_goals_market'] = 2.5

        if 'odds_ft_over25' in df.columns:
            df['odds_over25'] = pd.to_numeric(df['odds_ft_over25'], errors='coerce')
            before = len(df)
            df = df.dropna(subset=['odds_over25'])
            df = df[df['odds_over25'] > 1.01]
            self._log_drop('invalid_odds', before - len(df))

            df['IP_OVER'] = 1 / df['odds_over25']
            df['CTMCL'] = 2.5 + (df['IP_OVER'] - 0.5)

            before = len(df)
            df = df[np.isfinite(df['CTMCL'])]
            df = df[(df['CTMCL'] > 0) & (df['CTMCL'] < 10)]
            self._log_drop('invalid_ctmcl', before - len(df))

            print(f"✓ CTMCL: μ={df['CTMCL'].mean():.3f}, σ={df['CTMCL'].std():.3f}, range=[{df['CTMCL'].min():.2f}, {df['CTMCL'].max():.2f}]")
        else:
            df['CTMCL'] = 2.5
            print("⚠️  No odds_ft_over25 found, using default CTMCL=2.5")

        potential_cols = ['btts_potential', 'o05_potential', 'o15_potential', 
                         'o25_potential', 'o35_potential', 'o45_potential']
        
        for col in potential_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(50)
            else:
                df[col] = 50

        df['home_shots'] = pd.to_numeric(df['team_a_shots'], errors='coerce')
        df['away_shots'] = pd.to_numeric(df['team_b_shots'], errors='coerce')
        df['home_shots_on_target'] = pd.to_numeric(df['team_a_shotsOnTarget'], errors='coerce')
        df['away_shots_on_target'] = pd.to_numeric(df['team_b_shotsOnTarget'], errors='coerce')

        before = len(df)
        df = df.dropna(subset=['home_shots', 'away_shots', 'home_shots_on_target', 'away_shots_on_target'])
        df = df[(df['home_shots'] >= 0) & (df['away_shots'] >= 0)]
        df = df[(df['home_shots_on_target'] >= 0) & (df['away_shots_on_target'] >= 0)]
        df = df[(df['home_shots_on_target'] <= df['home_shots']) & (df['away_shots_on_target'] <= df['away_shots'])]
        self._log_drop('invalid_shots', before - len(df))

        df['home_dangerous_attacks'] = pd.to_numeric(df['team_a_dangerous_attacks'], errors='coerce').fillna(0)
        df['away_dangerous_attacks'] = pd.to_numeric(df['team_b_dangerous_attacks'], errors='coerce').fillna(0)
        df['home_attacks'] = pd.to_numeric(df['team_a_attacks'], errors='coerce').fillna(0)
        df['away_attacks'] = pd.to_numeric(df['team_b_attacks'], errors='coerce').fillna(0)

        df['home_corners'] = pd.to_numeric(df['team_a_corners'], errors='coerce')
        df['away_corners'] = pd.to_numeric(df['team_b_corners'], errors='coerce')

        before = len(df)
        df = df.dropna(subset=['home_corners', 'away_corners'])
        df = df[(df['home_corners'] >= 0) & (df['away_corners'] >= 0)]
        self._log_drop('invalid_corners', before - len(df))

        df['home_yellow_cards'] = pd.to_numeric(df['team_a_yellow_cards'], errors='coerce')
        df['away_yellow_cards'] = pd.to_numeric(df['team_b_yellow_cards'], errors='coerce')
        df['home_red_cards'] = pd.to_numeric(df['team_a_red_cards'], errors='coerce')
        df['away_red_cards'] = pd.to_numeric(df['team_b_red_cards'], errors='coerce')

        before = len(df)
        df = df.dropna(subset=['home_yellow_cards', 'away_yellow_cards', 'home_red_cards', 'away_red_cards'])
        df = df[(df['home_yellow_cards'] >= 0) & (df['away_yellow_cards'] >= 0)]
        df = df[(df['home_red_cards'] >= 0) & (df['away_red_cards'] >= 0)]
        self._log_drop('invalid_cards', before - len(df))

        df['home_penalties_won'] = pd.to_numeric(df['team_a_penalties_won'], errors='coerce').fillna(0)
        df['away_penalties_won'] = pd.to_numeric(df['team_b_penalties_won'], errors='coerce').fillna(0)

        for old_col, new_col in [('odds_ft_1', 'odds_ft_home_team_win'), 
                                   ('odds_ft_x', 'odds_ft_draw'), 
                                   ('odds_ft_2', 'odds_ft_away_team_win')]:
            if old_col in df.columns:
                df[new_col] = pd.to_numeric(df[old_col], errors='coerce')
                df[new_col] = df[new_col].replace(0, np.nan)
                prob_col = f'{new_col}_prob'
                df[prob_col] = 1 / df[new_col]

                before = len(df)
                df = df[np.isfinite(df[prob_col])]
                df = df[(df[prob_col] > 0) & (df[prob_col] < 1)]
                if before > len(df):
                    self._log_drop(f'invalid_{prob_col}', before - len(df))

        if 'odds_ft_home_team_win_prob' in df.columns:
            df['odds_ft_1_prob'] = df['odds_ft_home_team_win_prob']
        else:
            df['odds_ft_1_prob'] = np.nan
            
        if 'odds_ft_away_team_win_prob' in df.columns:
            df['odds_ft_2_prob'] = df['odds_ft_away_team_win_prob']
        else:
            df['odds_ft_2_prob'] = np.nan

        df['game_week'] = pd.to_numeric(df['game_week'], errors='coerce')

        print(f"\n✓ Cleaning summary:")
        total_dropped = 0
        for reason, count in self.rows_dropped.items():
            if count > 0:
                print(f"  - {reason}: {count} rows")
                total_dropped += count
        print(f"\n✓ Total dropped: {total_dropped} rows")
        print(f"✓ Final clean dataset: {len(df)} matches ({len(df)/initial_rows*100:.1f}% retained)")

        df = df.sort_values('date').reset_index(drop=True)

        return df

    def _log_drop(self, reason, count):
        if count > 0:
            self.rows_dropped[reason] = self.rows_dropped.get(reason, 0) + count


class HistoricalFeatureEngine:
    """Creates historical features - ENHANCED with new dataset features"""

    def __init__(self):
        self.feature_list = []

    def create_features(self, df):
        print("\n" + "="*80)
        print("STEP 2: CREATING HISTORICAL FEATURES (NO DATA LEAKS)")
        print("="*80)

        df_sorted = df.copy()

        self.feature_list = [
            'home_xg_avg', 'away_xg_avg',
            'home_xg_recent', 'away_xg_recent',
            'home_xg_momentum', 'away_xg_momentum',
            'home_goals_avg', 'away_goals_avg',
            'home_goals_conceded_avg', 'away_goals_conceded_avg',
            'home_recent_goals', 'away_recent_goals',
            'home_recent_conceded', 'away_recent_conceded',
            'home_shots_accuracy_avg', 'away_shots_accuracy_avg',
            'home_shots_on_target_avg', 'away_shots_on_target_avg',
            'home_shots_avg', 'away_shots_avg',
            'home_dangerous_attacks_avg', 'away_dangerous_attacks_avg',
            'home_attacks_avg', 'away_attacks_avg',
            'home_dangerous_attacks_recent', 'away_dangerous_attacks_recent',
            'home_corners_avg', 'away_corners_avg',
            'home_corners_recent', 'away_corners_recent',
            'home_yellow_cards_avg', 'away_yellow_cards_avg',
            'home_red_cards_avg', 'away_red_cards_avg',
            'home_total_cards_avg', 'away_total_cards_avg',
            'home_penalties_won_avg', 'away_penalties_won_avg',
            'home_elo', 'away_elo', 'elo_diff',
            'h2h_home_wins', 'h2h_away_wins', 'h2h_draws',
            'h2h_home_goals_avg', 'h2h_away_goals_avg', 'h2h_total_goals_avg',
            'h2h_matches_played',
            'home_win_streak', 'away_win_streak',
            'home_form_points', 'away_form_points',
            'days_since_last_home', 'days_since_last_away',
            'league_avg_goals',
        ]

        for col in self.feature_list:
            df_sorted[col] = np.nan

        team_elo = {}
        team_last_match_date = {}
        league_goals_history = {}

        print("Computing historical features (leak-free)...")
        print("⚠️  CRITICAL: Features use only PAST data, Elo updated AFTER feature extraction")

        total_matches = len(df_sorted)
        update_frequency = max(1, total_matches // 20)

        for i in range(total_matches):
            if i % update_frequency == 0 or i == total_matches - 1:
                pct = (i / total_matches) * 100 if i > 0 else 0
                print(f"  Progress: {i:5d}/{total_matches} ({pct:5.1f}%)")

            home_team = df_sorted.iloc[i]['home_team']
            away_team = df_sorted.iloc[i]['away_team']
            current_date = df_sorted.iloc[i]['date']
            league = df_sorted.iloc[i]['league']

            past_data = df_sorted.iloc[:i]

            if home_team not in team_elo:
                team_elo[home_team] = 1500
            if away_team not in team_elo:
                team_elo[away_team] = 1500

            df_sorted.at[i, 'home_elo'] = team_elo[home_team]
            df_sorted.at[i, 'away_elo'] = team_elo[away_team]
            df_sorted.at[i, 'elo_diff'] = team_elo[home_team] - team_elo[away_team]

            if league in league_goals_history:
                df_sorted.at[i, 'league_avg_goals'] = league_goals_history[league]
            else:
                df_sorted.at[i, 'league_avg_goals'] = 2.5

            if home_team in team_last_match_date:
                days_rest = (current_date - team_last_match_date[home_team]).days
                df_sorted.at[i, 'days_since_last_home'] = days_rest

            if away_team in team_last_match_date:
                days_rest = (current_date - team_last_match_date[away_team]).days
                df_sorted.at[i, 'days_since_last_away'] = days_rest

            if len(past_data) == 0:
                self._update_elo_post_match(df_sorted, i, team_elo, home_team, away_team)
                team_last_match_date[home_team] = current_date
                team_last_match_date[away_team] = current_date
                
                league_past = past_data[past_data['league'] == league]
                if len(league_past) > 0:
                    league_goals_history[league] = league_past['total_goals'].mean()
                
                continue

            self._compute_h2h(df_sorted, i, past_data, home_team, away_team)

            home_past = past_data[past_data['home_team'] == home_team]
            if len(home_past) >= 3:
                self._compute_team_features(df_sorted, i, home_past, 'home', is_home=True)

            away_past = past_data[past_data['away_team'] == away_team]
            if len(away_past) >= 3:
                self._compute_team_features(df_sorted, i, away_past, 'away', is_home=False)

            self._update_elo_post_match(df_sorted, i, team_elo, home_team, away_team)
            
            team_last_match_date[home_team] = current_date
            team_last_match_date[away_team] = current_date
            
            league_past = past_data[past_data['league'] == league]
            if len(league_past) > 0:
                league_goals_history[league] = league_past['total_goals'].mean()

        print(f"  Progress: {total_matches:5d}/{total_matches} (100.0%)")
        print("✓ Historical features computed (no data leaks)")

        before = len(df_sorted)
        df_sorted = df_sorted.dropna(subset=self.feature_list)
        dropped = before - len(df_sorted)
        if dropped > 0:
            print(f"✓ Removed {dropped} matches with incomplete history")

        if len(df_sorted) > 30:
            df_final = df_sorted.iloc[30:].reset_index(drop=True)
            print(f"✓ Skipped first 30 matches for feature stability")
        else:
            df_final = df_sorted.reset_index(drop=True)
            print(f"⚠️  Dataset too small to skip 30 matches")

        print(f"✓ Final dataset: {len(df_final)} matches with complete features")

        return df_final

    def _compute_h2h(self, df, i, past_data, home_team, away_team):
        """Compute head-to-head features"""
        h2h = past_data[
            ((past_data['home_team'] == home_team) & (past_data['away_team'] == away_team)) |
            ((past_data['home_team'] == away_team) & (past_data['away_team'] == home_team))
        ]

        if len(h2h) == 0:
            return

        home_wins, away_wins, draws = 0, 0, 0
        h2h_home_goals, h2h_away_goals = [], []

        for _, match in h2h.iterrows():
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

        df.at[i, 'h2h_home_wins'] = home_wins
        df.at[i, 'h2h_away_wins'] = away_wins
        df.at[i, 'h2h_draws'] = draws
        df.at[i, 'h2h_matches_played'] = len(h2h)
        df.at[i, 'h2h_home_goals_avg'] = np.mean(h2h_home_goals) if h2h_home_goals else 0
        df.at[i, 'h2h_away_goals_avg'] = np.mean(h2h_away_goals) if h2h_away_goals else 0
        df.at[i, 'h2h_total_goals_avg'] = df.at[i, 'h2h_home_goals_avg'] + df.at[i, 'h2h_away_goals_avg']

    def _compute_team_features(self, df, i, team_past, prefix, is_home):
        """Compute team-specific features - ENHANCED"""
        if is_home:
            xg_col = 'pre_home_xg'
            goals_col, conceded_col = 'home_goals', 'away_goals'
            shots_col, sot_col = 'home_shots', 'home_shots_on_target'
            corners_col = 'home_corners'
            yellow_col, red_col = 'home_yellow_cards', 'home_red_cards'
            attacks_col = 'home_attacks'
            dangerous_attacks_col = 'home_dangerous_attacks'
            penalties_col = 'home_penalties_won'
        else:
            xg_col = 'pre_away_xg'
            goals_col, conceded_col = 'away_goals', 'home_goals'
            shots_col, sot_col = 'away_shots', 'away_shots_on_target'
            corners_col = 'away_corners'
            yellow_col, red_col = 'away_yellow_cards', 'away_red_cards'
            attacks_col = 'away_attacks'
            dangerous_attacks_col = 'away_dangerous_attacks'
            penalties_col = 'away_penalties_won'

        df.at[i, f'{prefix}_xg_avg'] = team_past[xg_col].mean()
        df.at[i, f'{prefix}_goals_avg'] = team_past[goals_col].mean()
        df.at[i, f'{prefix}_goals_conceded_avg'] = team_past[conceded_col].mean()
        df.at[i, f'{prefix}_shots_avg'] = team_past[shots_col].mean()
        df.at[i, f'{prefix}_shots_on_target_avg'] = team_past[sot_col].mean()

        total_shots = team_past[shots_col].sum()
        if total_shots > 0:
            accuracy = team_past[sot_col].sum() / total_shots
            df.at[i, f'{prefix}_shots_accuracy_avg'] = accuracy
        else:
            df.at[i, f'{prefix}_shots_accuracy_avg'] = 0.33

        df.at[i, f'{prefix}_corners_avg'] = team_past[corners_col].mean()
        df.at[i, f'{prefix}_yellow_cards_avg'] = team_past[yellow_col].mean()
        df.at[i, f'{prefix}_red_cards_avg'] = team_past[red_col].mean()
        df.at[i, f'{prefix}_total_cards_avg'] = (team_past[yellow_col] + team_past[red_col]).mean()

        if attacks_col in team_past.columns:
            df.at[i, f'{prefix}_attacks_avg'] = team_past[attacks_col].mean()
        else:
            df.at[i, f'{prefix}_attacks_avg'] = 0
            
        if dangerous_attacks_col in team_past.columns:
            df.at[i, f'{prefix}_dangerous_attacks_avg'] = team_past[dangerous_attacks_col].mean()
        else:
            df.at[i, f'{prefix}_dangerous_attacks_avg'] = 0

        if penalties_col in team_past.columns:
            df.at[i, f'{prefix}_penalties_won_avg'] = team_past[penalties_col].mean()
        else:
            df.at[i, f'{prefix}_penalties_won_avg'] = 0

        recent = team_past.tail(5)
        df.at[i, f'{prefix}_xg_recent'] = recent[xg_col].mean()
        df.at[i, f'{prefix}_recent_goals'] = recent[goals_col].mean()
        df.at[i, f'{prefix}_recent_conceded'] = recent[conceded_col].mean()
        df.at[i, f'{prefix}_corners_recent'] = recent[corners_col].mean()
        
        if dangerous_attacks_col in recent.columns:
            df.at[i, f'{prefix}_dangerous_attacks_recent'] = recent[dangerous_attacks_col].mean()
        else:
            df.at[i, f'{prefix}_dangerous_attacks_recent'] = 0

        df.at[i, f'{prefix}_xg_momentum'] = recent[xg_col].mean() - team_past[xg_col].mean()

        win_streak = 0
        for _, match in recent.iloc[::-1].iterrows():
            won = match[goals_col] > match[conceded_col]
            if won:
                win_streak += 1
            else:
                break
        df.at[i, f'{prefix}_win_streak'] = win_streak

        form_points = 0
        for _, match in recent.iterrows():
            if match[goals_col] > match[conceded_col]:
                form_points += 3
            elif match[goals_col] == match[conceded_col]:
                form_points += 1
        df.at[i, f'{prefix}_form_points'] = form_points

    def _update_elo_post_match(self, df, i, team_elo, home_team, away_team):
        """Update Elo ratings AFTER match (prevents data leak)"""
        home_goals = df.iloc[i]['home_goals']
        away_goals = df.iloc[i]['away_goals']

        result = 1.0 if home_goals > away_goals else (0.0 if home_goals < away_goals else 0.5)
        expected_home = 1 / (1 + 10 ** ((team_elo[away_team] - team_elo[home_team]) / 400))

        k_factor = 20
        team_elo[home_team] += k_factor * (result - expected_home)
        team_elo[away_team] += k_factor * ((1 - result) - (1 - expected_home))


class FeaturePreparator:
    """Prepares and weights features - ENHANCED"""

    def __init__(self):
        self.feature_columns = []
        self.weights = FeatureWeights.get_weights()
        self.scaler = StandardScaler()

    def prepare(self, df):
        print("\n" + "="*80)
        print("STEP 3: PREPARING AND WEIGHTING FEATURES (ENHANCED)")
        print("="*80)

        self.feature_columns = [
            'CTMCL', 'avg_goals_market',
            #'pre_total_xg', 
            'team_a_xg_prematch', 'team_b_xg_prematch',
            'pre_match_home_ppg', 'pre_match_away_ppg',
            'home_xg_avg', 'away_xg_avg',
            'home_xg_momentum', 'away_xg_momentum',
            'home_goals_conceded_avg', 'away_goals_conceded_avg',
            'o25_potential', 'o35_potential',
            'home_shots_accuracy_avg', 'away_shots_accuracy_avg',
            'home_dangerous_attacks_avg', 'away_dangerous_attacks_avg',
            'h2h_total_goals_avg',
            'home_form_points', 'away_form_points',
            'elo_diff',
            'league_avg_goals',
        ]

        for col in ['odds_ft_1_prob', 'odds_ft_2_prob']:
            if col in df.columns:
                self.feature_columns.append(col)

        valid_features = [f for f in self.feature_columns if f in df.columns]
        missing_features = [f for f in self.feature_columns if f not in df.columns]

        if missing_features:
            print(f"⚠️  Missing features (will be skipped): {missing_features}")

        self.feature_columns = valid_features
        print(f"✓ Using {len(self.feature_columns)} features")

        X = df[self.feature_columns].copy()

        nan_counts = X.isnull().sum()
        if nan_counts.sum() > 0:
            print(f"\n⚠️  Found NaN values in features:")
            for col in nan_counts[nan_counts > 0].index:
                print(f"  - {col}: {nan_counts[col]} NaN values")

            before = len(X)
            X = X.dropna()
            df = df.loc[X.index]
            print(f"✓ Removed {before - len(X)} rows with NaN features")

        print(f"✓ Final feature matrix: {X.shape}")

        print(f"\n✓ Feature weights applied:")
        weighted_features = []
        for feat in self.feature_columns:
            weight = self.weights.get(feat, 1.0)
            weighted_features.append(weight)
            if weight != 1.0:
                print(f"  {feat}: {weight:.2f}x")

        return X, df, weighted_features


class ModelTrainer:
    """Trains Poisson, Ridge, and Lasso models with home/away accuracy metrics"""

    def __init__(self, scaler, weights):
        self.scaler = scaler
        self.weights = np.array(weights)
        self.models = {
            'poisson': PoissonRegressor(alpha=1.0, max_iter=1000),
            'ridge': Ridge(alpha=1.0, random_state=42),
            'lasso': Lasso(alpha=0.01, max_iter=5000, random_state=42)
        }

    def train_and_evaluate(self, X_train, X_val, X_test,
                          y_home_train, y_home_val, y_home_test,
                          y_away_train, y_away_val, y_away_test,
                          y_total_test, ctmcl_test, df_test):
        """Train all models and evaluate"""

        print("\n" + "="*80)
        print("STEP 4: TRAINING MODELS (POISSON, RIDGE, LASSO)")
        print("="*80)

        X_train_weighted = X_train.values * self.weights
        X_val_weighted = X_val.values * self.weights
        X_test_weighted = X_test.values * self.weights

        X_train_scaled = self.scaler.fit_transform(X_train_weighted)
        X_val_scaled = self.scaler.transform(X_val_weighted)
        X_test_scaled = self.scaler.transform(X_test_weighted)

        print(f"\n✓ Data split:")
        print(f"  Train: {len(X_train)} matches ({len(X_train)/(len(X_train)+len(X_val)+len(X_test))*100:.1f}%)")
        print(f"  Val:   {len(X_val)} matches ({len(X_val)/(len(X_train)+len(X_val)+len(X_test))*100:.1f}%)")
        print(f"  Test:  {len(X_test)} matches ({len(X_test)/(len(X_train)+len(X_val)+len(X_test))*100:.1f}%)")

        results = []

        for model_name, model in self.models.items():
            print(f"\n{'='*80}")
            print(f"TRAINING: {model_name.upper()}")
            print(f"{'='*80}")

            result = self._train_single_model(
                model_name, model,
                X_train_scaled, X_val_scaled, X_test_scaled,
                y_home_train, y_home_val, y_home_test,
                y_away_train, y_away_val, y_away_test,
                y_total_test, ctmcl_test, df_test
            )
            results.append(result)

        return results

    def _train_single_model(self, name, model, X_train, X_val, X_test,
                           y_home_train, y_home_val, y_home_test,
                           y_away_train, y_away_val, y_away_test,
                           y_total_test, ctmcl_test, df_test):
        """Train and evaluate a single model with home/away accuracy"""

        print(f"\n→ Training HOME goals predictor...")
        home_model = clone(model)
        home_model.fit(X_train, y_home_train)

        pred_home_train = np.maximum(home_model.predict(X_train), 0)
        pred_home_val = np.maximum(home_model.predict(X_val), 0)
        pred_home_test = np.maximum(home_model.predict(X_test), 0)

        home_train_mae = mean_absolute_error(y_home_train, pred_home_train)
        home_val_mae = mean_absolute_error(y_home_val, pred_home_val)
        home_test_mae = mean_absolute_error(y_home_test, pred_home_test)
        home_test_rmse = np.sqrt(mean_squared_error(y_home_test, pred_home_test))
        home_test_r2 = r2_score(y_home_test, pred_home_test)
        home_within_05 = np.abs(pred_home_test - y_home_test.values) <= 0.5
        home_acc_within_05 = home_within_05.mean()

        print(f"  Train MAE: {home_train_mae:.4f} | Val MAE: {home_val_mae:.4f} | Test MAE: {home_test_mae:.4f}")
        print(f"  Test RMSE: {home_test_rmse:.4f} | R²: {home_test_r2:.4f}")
        print(f"  Predicted μ={pred_home_test.mean():.3f}, Actual μ={y_home_test.mean():.3f} (bias: {pred_home_test.mean() - y_home_test.mean():+.3f})")
        print(f"  🎯 ACCURACY (Δ ≤ 0.5): {home_acc_within_05:.1%}")

        print(f"\n→ Training AWAY goals predictor...")
        away_model = clone(model)
        away_model.fit(X_train, y_away_train)

        pred_away_train = np.maximum(away_model.predict(X_train), 0)
        pred_away_val = np.maximum(away_model.predict(X_val), 0)
        pred_away_test = np.maximum(away_model.predict(X_test), 0)

        away_train_mae = mean_absolute_error(y_away_train, pred_away_train)
        away_val_mae = mean_absolute_error(y_away_val, pred_away_val)
        away_test_mae = mean_absolute_error(y_away_test, pred_away_test)
        away_test_rmse = np.sqrt(mean_squared_error(y_away_test, pred_away_test))
        away_test_r2 = r2_score(y_away_test, pred_away_test)
        away_within_05 = np.abs(pred_away_test - y_away_test.values) <= 0.5
        away_acc_within_05 = away_within_05.mean()

        print(f"  Train MAE: {away_train_mae:.4f} | Val MAE: {away_val_mae:.4f} | Test MAE: {away_test_mae:.4f}")
        print(f"  Test RMSE: {away_test_rmse:.4f} | R²: {away_test_r2:.4f}")
        print(f"  Predicted μ={pred_away_test.mean():.3f}, Actual μ={y_away_test.mean():.3f} (bias: {pred_away_test.mean() - y_away_test.mean():+.3f})")
        print(f"  🎯 ACCURACY (Δ ≤ 0.5): {away_acc_within_05:.1%}")

        pred_total_test = pred_home_test + pred_away_test

        print(f"\n→ TOTAL GOALS Performance:")
        total_mae = mean_absolute_error(y_total_test, pred_total_test)
        total_rmse = np.sqrt(mean_squared_error(y_total_test, pred_total_test))
        total_r2 = r2_score(y_total_test, pred_total_test)

        print(f"  MAE:  {total_mae:.4f}")
        print(f"  RMSE: {total_rmse:.4f}")
        print(f"  R²:   {total_r2:.4f}")
        print(f"  Predicted μ={pred_total_test.mean():.3f}, Actual μ={y_total_test.mean():.3f}")
        print(f"  Bias: {pred_total_test.mean() - y_total_test.mean():+.3f} goals")

        print(f"\n→ OVER/UNDER CTMCL Analysis:")
        y_ou_actual = (y_total_test.values > ctmcl_test.values).astype(int)
        y_ou_pred = (pred_total_test > ctmcl_test.values).astype(int)
        ou_accuracy = accuracy_score(y_ou_actual, y_ou_pred)

        print(f"  Accuracy: {ou_accuracy:.1%}")
        print(f"  Edge over 50%: {(ou_accuracy - 0.5) * 100:+.1f}%")
        print(f"  Actual OVER: {y_ou_actual.sum()}/{len(y_ou_actual)} ({y_ou_actual.mean()*100:.1f}%)")
        print(f"  Predicted OVER: {y_ou_pred.sum()}/{len(y_ou_pred)} ({y_ou_pred.mean()*100:.1f}%)")

        print(f"\n→ MONEYLINE (Match Winner) Analysis:")
        actual_home_win = (df_test['home_goals'] > df_test['away_goals']).astype(int).values
        actual_away_win = (df_test['away_goals'] > df_test['home_goals']).astype(int).values
        actual_draw = (df_test['home_goals'] == df_test['away_goals']).astype(int).values

        pred_home_win = (pred_home_test > pred_away_test).astype(int)
        pred_away_win = (pred_away_test > pred_home_test).astype(int)
        pred_draw = ((np.abs(pred_home_test - pred_away_test) < 0.15)).astype(int)

        ml_correct = (
            ((pred_home_test > pred_away_test) & (actual_home_win == 1)) |
            ((pred_away_test > pred_home_test) & (actual_away_win == 1))
        )
        ml_accuracy = ml_correct.mean()

        print(f"  Accuracy: {ml_accuracy:.1%}")
        print(f"  Edge over 33.3%: {(ml_accuracy - 0.333) * 100:+.1f}%")
        print(f"  Predicted home wins: {pred_home_win.sum()} | Actual: {actual_home_win.sum()}")
        print(f"  Predicted away wins: {pred_away_win.sum()} | Actual: {actual_away_win.sum()}")
        print(f"  Actual draws: {actual_draw.sum()}")

        return {
            'model_name': name,
            'home_mae': home_test_mae,
            'home_rmse': home_test_rmse,
            'home_r2': home_test_r2,
            'home_acc_05': home_acc_within_05,
            'away_mae': away_test_mae,
            'away_rmse': away_test_rmse,
            'away_r2': away_test_r2,
            'away_acc_05': away_acc_within_05,
            'total_mae': total_mae,
            'total_rmse': total_rmse,
            'total_r2': total_r2,
            'ou_accuracy': ou_accuracy,
            'ml_accuracy': ml_accuracy,
            'predictions': {
                'home': pred_home_test,
                'away': pred_away_test,
                'total': pred_total_test,
                'ou': y_ou_pred,
                'ou_actual': y_ou_actual,
                'home_win': pred_home_win,
                'away_win': pred_away_win,
                'draw': pred_draw,
                'actual_home_win': actual_home_win,
                'actual_away_win': actual_away_win,
                'actual_draw': actual_draw,
                'ml_correct': ml_correct,
                'home_within_05': home_within_05,
                'away_within_05': away_within_05
            },
            'models': {
                'home': home_model,
                'away': away_model
            }
        }


class ConfidenceCalculator:
    """Calculate prediction confidence based on market alignment"""
    
    @staticmethod
    def calculate_ou_confidence(pred_total, ctmcl, odds_over_prob=None):
        distance = np.abs(pred_total - ctmcl)
        distance_confidence = np.minimum(distance / 1.5, 1.0)
        
        pred_over = (pred_total > ctmcl).astype(int)
        
        if odds_over_prob is not None:
            market_over = (odds_over_prob > 0.5).astype(int)
            alignment = (pred_over == market_over).astype(float)
            market_strength = np.abs(odds_over_prob - 0.5) * 2
            alignment_confidence = alignment * (0.5 + market_strength * 0.5)
        else:
            alignment_confidence = 0.5
        
        confidence = (distance_confidence * 0.6 + alignment_confidence * 0.4) * 100
        
        return np.clip(confidence, 0, 100)
    
    @staticmethod
    def calculate_ml_confidence(pred_home, pred_away, odds_home_prob=None, odds_away_prob=None):
        goal_diff = np.abs(pred_home - pred_away)
        diff_confidence = np.minimum(goal_diff / 2.0, 1.0)
        
        pred_home_win = (pred_home > pred_away).astype(int)
        pred_away_win = (pred_away > pred_home).astype(int)
        pred_draw = ((np.abs(pred_home - pred_away) < 0.15)).astype(int)
        
        if odds_home_prob is not None and odds_away_prob is not None:
            market_home_fav = (odds_home_prob > odds_away_prob).astype(int)
            market_away_fav = (odds_away_prob > odds_home_prob).astype(int)
            
            alignment = (
                (pred_home_win & market_home_fav) | 
                (pred_away_win & market_away_fav)
            ).astype(float)
            
            market_strength = np.abs(odds_home_prob - odds_away_prob)
            alignment_confidence = alignment * (0.5 + market_strength * 0.5)
        else:
            alignment_confidence = 0.5
        
        draw_penalty = pred_draw * 0.3
        
        confidence = ((diff_confidence * 0.6 + alignment_confidence * 0.4) - draw_penalty) * 100
        
        return np.clip(confidence, 0, 100)
    
    @staticmethod
    def calculate_overall_confidence(ou_confidence, ml_confidence, prediction_error=None):
        base_confidence = (ou_confidence + ml_confidence) / 2
        
        if prediction_error is not None:
            error_factor = np.maximum(0, 1 - (prediction_error / 2))
            base_confidence = base_confidence * error_factor
        
        return np.clip(base_confidence, 0, 100)
    
    @staticmethod
    def get_confidence_label(confidence):
        if confidence >= 75:
            return "VERY HIGH"
        elif confidence >= 60:
            return "HIGH"
        elif confidence >= 45:
            return "MEDIUM"
        elif confidence >= 30:
            return "LOW"
        else:
            return "VERY LOW"


class OutputGenerator:
    """Generates comprehensive output CSV file"""

    @staticmethod
    def save_predictions(df_test, result, feature_columns, model_name):
        print("\n" + "="*80)
        print(f"STEP 5: SAVING PREDICTIONS FOR {model_name.upper()}")
        print("="*80)

        pred_home = result['predictions']['home']
        pred_away = result['predictions']['away']
        pred_total = result['predictions']['total']
        pred_ou = result['predictions']['ou']
        ou_actual = result['predictions']['ou_actual']
        pred_home_win = result['predictions']['home_win']
        pred_away_win = result['predictions']['away_win']
        pred_draw = result['predictions']['draw']
        actual_home_win = result['predictions']['actual_home_win']
        actual_away_win = result['predictions']['away_win']
        actual_draw = result['predictions']['actual_draw']
        ml_correct = result['predictions']['ml_correct']
        home_within_05 = result['predictions']['home_within_05']
        away_within_05 = result['predictions']['away_within_05']

        output_cols = ['date', 'league', 'home_team', 'away_team',
                      'home_goals', 'away_goals', 'total_goals']
        
        output = df_test[output_cols].copy()

        prediction_error = np.abs(pred_total - df_test['total_goals'].values)
        
        print("\n→ Calculating prediction confidence...")
        
        odds_over_prob = None
        odds_home_prob = None
        odds_away_prob = None
        
        if 'IP_OVER' in df_test.columns:
            odds_over_prob = df_test['IP_OVER'].values
        
        if 'odds_ft_1_prob' in df_test.columns:
            odds_home_prob = df_test['odds_ft_1_prob'].values
        
        if 'odds_ft_2_prob' in df_test.columns:
            odds_away_prob = df_test['odds_ft_2_prob'].values
        
        ou_confidence = ConfidenceCalculator.calculate_ou_confidence(
            pred_total, 
            df_test['CTMCL'].values,
            odds_over_prob
        )
        
        ml_confidence = ConfidenceCalculator.calculate_ml_confidence(
            pred_home,
            pred_away,
            odds_home_prob,
            odds_away_prob
        )
        
        overall_confidence = ConfidenceCalculator.calculate_overall_confidence(
            ou_confidence,
            ml_confidence,
            prediction_error=None
        )
        
        ou_confidence_label = np.array([ConfidenceCalculator.get_confidence_label(c) for c in ou_confidence])
        ml_confidence_label = np.array([ConfidenceCalculator.get_confidence_label(c) for c in ml_confidence])
        overall_confidence_label = np.array([ConfidenceCalculator.get_confidence_label(c) for c in overall_confidence])
        
        for feat in feature_columns:
            if feat in df_test.columns:
                output[feat] = df_test[feat]

        if 'odds_over25' in df_test.columns:
            output['odds_over25'] = df_test['odds_over25']
        if 'IP_OVER' in df_test.columns:
            output['IP_OVER'] = df_test['IP_OVER']

        output['pred_home_goals'] = pred_home
        output['pred_away_goals'] = pred_away
        output['pred_total_goals'] = pred_total
        output['pred_over_ctmcl'] = pred_ou
        output['actual_over_ctmcl'] = ou_actual
        output['ou_correct'] = (pred_ou == ou_actual).astype(int)
        
        output['ou_confidence'] = ou_confidence
        output['ml_confidence'] = ml_confidence
        output['overall_confidence'] = overall_confidence
        output['ou_confidence_label'] = ou_confidence_label
        output['ml_confidence_label'] = ml_confidence_label
        output['overall_confidence_label'] = overall_confidence_label

        output['prediction_error'] = prediction_error
        output['diff_from_ctmcl'] = np.abs(pred_total - df_test['CTMCL'].values)

        output['pred_home_win'] = pred_home_win
        output['pred_away_win'] = pred_away_win
        output['pred_draw'] = pred_draw
        output['actual_home_win'] = actual_home_win
        output['actual_away_win'] = actual_away_win
        output['actual_draw'] = actual_draw
        output['moneyline_correct'] = ml_correct.astype(int)
        output['goal_diff_pred'] = np.abs(pred_home - pred_away)

        output['home_accurate'] = home_within_05.astype(int)
        output['away_accurate'] = away_within_05.astype(int)

        output['home_error'] = np.abs(output['home_goals'] - output['pred_home_goals'])
        output['away_error'] = np.abs(output['away_goals'] - output['pred_away_goals'])

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'football_predictions_{model_name}_{timestamp}.csv'
        output.to_csv(filename, index=False, float_format='%.3f')

        print(f"✓ Predictions saved to: {filename}")
        print(f"✓ Total predictions: {len(output)}")
        
        print(f"\n" + "="*80)
        print("CONFIDENCE-BASED PERFORMANCE ANALYSIS")
        print("="*80)
        
        high_conf_mask = output['overall_confidence'] >= 75
        high_conf_count = high_conf_mask.sum()
        
        if high_conf_count > 0:
            print(f"\n■ HIGH CONFIDENCE (75-100%): {high_conf_count} predictions")
            print(f"  Home Acc (Δ ≤ 0.5): {output.loc[high_conf_mask, 'home_accurate'].mean():.1%}")
            print(f"  Away Acc (Δ ≤ 0.5): {output.loc[high_conf_mask, 'away_accurate'].mean():.1%}")
            print(f"  Over/Under Acc: {output.loc[high_conf_mask, 'ou_correct'].mean():.1%}")
            print(f"  Moneyline Acc: {output.loc[high_conf_mask, 'moneyline_correct'].mean():.1%}")
        
        medium_conf_mask = (output['overall_confidence'] >= 50) & (output['overall_confidence'] < 75)
        medium_conf_count = medium_conf_mask.sum()
        
        if medium_conf_count > 0:
            print(f"\n■ MEDIUM CONFIDENCE (50-75%): {medium_conf_count} predictions")
            print(f"  Home Acc (Δ ≤ 0.5): {output.loc[medium_conf_mask, 'home_accurate'].mean():.1%}")
            print(f"  Away Acc (Δ ≤ 0.5): {output.loc[medium_conf_mask, 'away_accurate'].mean():.1%}")
            print(f"  Over/Under Acc: {output.loc[medium_conf_mask, 'ou_correct'].mean():.1%}")
            print(f"  Moneyline Acc: {output.loc[medium_conf_mask, 'moneyline_correct'].mean():.1%}")
        
        low_conf_mask = output['overall_confidence'] < 50
        low_conf_count = low_conf_mask.sum()
        
        if low_conf_count > 0:
            print(f"\n■ LOW CONFIDENCE (0-50%): {low_conf_count} predictions")
            print(f"  Home Acc (Δ ≤ 0.5): {output.loc[low_conf_mask, 'home_accurate'].mean():.1%}")
            print(f"  Away Acc (Δ ≤ 0.5): {output.loc[low_conf_mask, 'away_accurate'].mean():.1%}")
            print(f"  Over/Under Acc: {output.loc[low_conf_mask, 'ou_correct'].mean():.1%}")
            print(f"  Moneyline Acc: {output.loc[low_conf_mask, 'moneyline_correct'].mean():.1%}")
        
        print(f"\n■ OVERALL PERFORMANCE: {len(output)} predictions")
        print(f"  Home Acc (Δ ≤ 0.5): {output['home_accurate'].mean():.1%}")
        print(f"  Away Acc (Δ ≤ 0.5): {output['away_accurate'].mean():.1%}")
        print(f"  Over/Under Acc: {output['ou_correct'].mean():.1%}")
        print(f"  Moneyline Acc: {output['moneyline_correct'].mean():.1%}")

        print("\n" + "="*80)

        return filename, output


class ResultsAnalyzer:
    """Analyzes and compares model results"""

    @staticmethod
    def compare_models(results):
        print("\n" + "="*80)
        print("FINAL MODEL COMPARISON")
        print("="*80)

        comparison = pd.DataFrame([{
            'Model': r['model_name'].upper(),
            'Total MAE': r['total_mae'],
            'RMSE': r['total_rmse'],
            'R²': r['total_r2'],
            'Home MAE': r['home_mae'],
            'Home Acc': r['home_acc_05'],
            'Away MAE': r['away_mae'],
            'Away Acc': r['away_acc_05'],
            'O/U Acc': r['ou_accuracy'],
            'O/U Edge': (r['ou_accuracy'] - 0.5) * 100,
            'ML Acc': r['ml_accuracy'],
            'ML Edge': (r['ml_accuracy'] - 0.333) * 100,
        } for r in results]).sort_values('Total MAE')

        print("\n" + comparison.to_string(index=False, float_format=lambda x: f'{x:.4f}'))

        print("\n" + "="*80)
        print("DETAILED MODEL RANKINGS")
        print("="*80)

        for rank, (_, row) in enumerate(comparison.iterrows(), 1):
            print(f"\n#{rank} - {row['Model']}")
            print(f"  Goal Prediction:")
            print(f"    Total: MAE {row['Total MAE']:.4f} | RMSE {row['RMSE']:.4f} | R² {row['R²']:.4f}")
            print(f"    Home:  MAE {row['Home MAE']:.4f} | Accuracy (Δ ≤ 0.5): {row['Home Acc']:.1%}")
            print(f"    Away:  MAE {row['Away MAE']:.4f} | Accuracy (Δ ≤ 0.5): {row['Away Acc']:.1%}")
            print(f"  Over/Under:")
            print(f"    Accuracy: {row['O/U Acc']:.1%} (edge: {row['O/U Edge']:+.1f}%)")
            print(f"  Moneyline:")
            print(f"    Accuracy: {row['ML Acc']:.1%} (edge: {row['ML Edge']:+.1f}%)")

        print("\n" + "="*80)
        print("🏆 CHAMPION MODEL")
        print("="*80)

        winner = comparison.iloc[0]
        print(f"\nWinner: {winner['Model']}")
        print(f"\n  Key Strengths:")
        print(f"  ✓ Best Total MAE: {winner['Total MAE']:.4f} goals")
        print(f"  ✓ Home goal accuracy (Δ ≤ 0.5): {winner['Home Acc']:.1%}")
        print(f"  ✓ Away goal accuracy (Δ ≤ 0.5): {winner['Away Acc']:.1%}")
        print(f"  ✓ O/U edge: {winner['O/U Edge']:+.1f}% over random (50%)")
        print(f"  ✓ ML edge: {winner['ML Edge']:+.1f}% over random (33.3%)")

        print(f"\n  Category Leaders:")
        best_ou = comparison.loc[comparison['O/U Acc'].idxmax()]
        best_ml = comparison.loc[comparison['ML Acc'].idxmax()]
        best_home = comparison.loc[comparison['Home Acc'].idxmax()]
        best_away = comparison.loc[comparison['Away Acc'].idxmax()]

        print(f"  • Best O/U: {best_ou['Model']} ({best_ou['O/U Acc']:.1%})")
        print(f"  • Best ML: {best_ml['Model']} ({best_ml['ML Acc']:.1%})")
        print(f"  • Best Home Goal Acc: {best_home['Model']} ({best_home['Home Acc']:.1%})")
        print(f"  • Best Away Goal Acc: {best_away['Model']} ({best_away['Away Acc']:.1%})")

        return comparison


class FootballPredictor:
    """Main orchestrator class - NEW DATASET VERSION"""

    def __init__(self):
        self.data_loader = DataLoader()
        self.feature_engine = HistoricalFeatureEngine()
        self.feature_prep = FeaturePreparator()
        self.results = None
        self.comparison = None
        self.output_files = []

    def run(self, filepath, train_ratio=0.6, val_ratio=0.2):
        """Execute full pipeline"""

        print("\n" + "="*80)
        print("FOOTBALL PREDICTION SYSTEM - NEW DATASET ADAPTED")
        print("Models: Poisson | Ridge | Lasso")
        if MAX_ROWS_TO_PROCESS:
            print(f"Row Limit: {MAX_ROWS_TO_PROCESS} matches")
        print("="*80)

        try:
            df = self.data_loader.load_and_clean(filepath)
            df = self.feature_engine.create_features(df)
            X, df, weights = self.feature_prep.prepare(df)

            y_home = df['home_goals']
            y_away = df['away_goals']
            y_total = df['total_goals']
            ctmcl = df['CTMCL']

            train_idx = int(len(X) * train_ratio)
            val_idx = int(len(X) * (train_ratio + val_ratio))

            X_train = X.iloc[:train_idx]
            X_val = X.iloc[train_idx:val_idx]
            X_test = X.iloc[val_idx:]

            y_home_train = y_home.iloc[:train_idx]
            y_home_val = y_home.iloc[train_idx:val_idx]
            y_home_test = y_home.iloc[val_idx:]

            y_away_train = y_away.iloc[:train_idx]
            y_away_val = y_away.iloc[train_idx:val_idx]
            y_away_test = y_away.iloc[val_idx:]

            y_total_test = y_total.iloc[val_idx:]
            ctmcl_test = ctmcl.iloc[val_idx:]
            df_test = df.iloc[val_idx:].copy()

            print(f"\n✓ Date ranges:")
            print(f"  Train: {df.iloc[:train_idx]['date'].min()} to {df.iloc[:train_idx]['date'].max()}")
            print(f"  Val:   {df.iloc[train_idx:val_idx]['date'].min()} to {df.iloc[train_idx:val_idx]['date'].max()}")
            print(f"  Test:  {df_test['date'].min()} to {df_test['date'].max()}")

            trainer = ModelTrainer(StandardScaler(), weights)
            self.results = trainer.train_and_evaluate(
                X_train, X_val, X_test,
                y_home_train, y_home_val, y_home_test,
                y_away_train, y_away_val, y_away_test,
                y_total_test, ctmcl_test, df_test
            )

            self.comparison = ResultsAnalyzer.compare_models(self.results)

            print("\n" + "="*80)
            print("GENERATING OUTPUT CSV FILES")
            print("="*80)

            for result in self.results:
                filename, output_df = OutputGenerator.save_predictions(
                    df_test, 
                    result, 
                    self.feature_prep.feature_columns,
                    result['model_name']
                )
                self.output_files.append({
                    'model': result['model_name'],
                    'filename': filename,
                    'dataframe': output_df
                })

            print("\n" + "="*80)
            print("✅ PIPELINE COMPLETE - NO DATA LEAKS CONFIRMED")
            print("="*80)
            print("\n✓ All features use only PAST data")
            print("✓ Elo ratings updated AFTER feature extraction")
            print("✓ No future information leaked into predictions")
            print(f"✓ Generated {len(self.output_files)} output CSV files:")
            for of in self.output_files:
                print(f"  • {of['filename']}")

            return self.results, self.comparison, self.output_files

        except Exception as e:
            print(f"\n❌ ERROR: {e}")
            import traceback
            traceback.print_exc()
            return None, None, None


# ========== MAIN EXECUTION ==========
if __name__ == "__main__":
    print("\n" + "="*80)
    print("CONFIGURATION")
    print("="*80)
    print(f"MAX_ROWS_TO_PROCESS: {MAX_ROWS_TO_PROCESS if MAX_ROWS_TO_PROCESS else 'ALL ROWS'}")
    print("=" * 80)
    
    predictor = FootballPredictor()
    results, comparison, output_files = predictor.run('bet.csv')

    if results:
        print("\n" + "="*80)
        print("✅ ALL DONE! Check your directory for CSV files")
        print("="*80)