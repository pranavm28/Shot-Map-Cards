import pandas as pd
import numpy as np
from pathlib import Path
import gc
from typing import Dict, List, Optional

class ShotDataPreprocessor:
    """
    Preprocesses shot data to reduce computational load for the Streamlit app.
    This script should be run offline to generate optimized datasets.
    """
    
    def __init__(self):
        self.leagues_to_file = {
            'ESP-La Liga': 'La_Liga_24_25.parquet',
            'ENG-Premier League': 'Premier_League_2425.parquet',
            'ITA-Serie A': 'Serie_A_2425.parquet',
            'GER-Bundesliga': 'Bundesliga_2425.parquet',
            'FRA-Ligue 1': 'Ligue_1_2425.parquet',
            #'POR-Liga Portugal': 'Primeira_Liga_2324.parquet'
            #'BEL-Jupiler Pro League': 'Jupiler_Pro_League_2324.parquet'
        }
        
        self.required_columns = [
            "league", "season", "gameId", "period", "minute", "second", 
            "type", "outcomeType", "teamId", "team", "playerId", "player",
            "x", "y", "endX", "endY", "is_shot", "is_goal"
        ]
        
        self.output_columns = [
            "league", "season", "gameId", "team", "player", 
            "x", "y", "endX", "endY", "is_shot", "is_goal", 
            "total_seconds", "TimeToShot", "in_penalty_box", "in_six_yard_box"
        ]
        
        # Define penalty box coordinates
        self.penalty_box = {
            'x_min': 102,
            'x_max': 120,
            'y_min': 18,
            'y_max': 62
        }
        
        # Define 6-yard box coordinates
        self.six_yard_box = {
            'x_min': 114,
            'x_max': 120,
            'y_min': 30,
            'y_max': 50
        }
    
    def load_and_filter_data(self, file_path: str, league: str, season: int) -> pd.DataFrame:
        """Load and filter data by league and season."""
        print(f"Loading data for {league}...")
        try:
            df = pd.read_parquet(file_path, columns=self.required_columns)
            df = df[(df["league"] == league) & (df["season"] == season)]
            print(f"Loaded {len(df)} records for {league}")
            return df
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return pd.DataFrame()
    
    def scale_coordinates(self, data: pd.DataFrame) -> pd.DataFrame:
        """Scale coordinates to match pitch dimensions."""
        data = data.copy()
        data['x'] = data['x'] * 1.218
        data['y'] = data['y'] * 0.8
        data['endX'] = data['endX'] * 1.2
        data['endY'] = data['endY'] * 0.8
        return data
    
    def calculate_total_seconds(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate total seconds for time calculations."""
        data = data.copy()
        data['total_seconds'] = data['minute'] * 60 + data['second']
        return data
    
    def label_shot_areas(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Label shots based on whether they were taken inside penalty box or 6-yard box.
        """
        print("Labeling shot areas...")
        data = data.copy()
        
        # Initialize columns
        data['in_penalty_box'] = False
        data['in_six_yard_box'] = False
        
        # Label penalty box shots
        penalty_box_mask = (
            (data['x'] >= self.penalty_box['x_min']) & 
            (data['x'] <= self.penalty_box['x_max']) &
            (data['y'] >= self.penalty_box['y_min']) & 
            (data['y'] <= self.penalty_box['y_max'])
        )
        data.loc[penalty_box_mask, 'in_penalty_box'] = True
        
        # Label 6-yard box shots
        six_yard_box_mask = (
            (data['x'] >= self.six_yard_box['x_min']) & 
            (data['x'] <= self.six_yard_box['x_max']) &
            (data['y'] >= self.six_yard_box['y_min']) & 
            (data['y'] <= self.six_yard_box['y_max'])
        )
        data.loc[six_yard_box_mask, 'in_six_yard_box'] = True
        
        # Print statistics
        total_shots = len(data[data['is_shot'] == True])
        penalty_box_shots = len(data[(data['is_shot'] == True) & (data['in_penalty_box'] == True)])
        six_yard_box_shots = len(data[(data['is_shot'] == True) & (data['in_six_yard_box'] == True)])
        
        print(f"Total shots: {total_shots}")
        print(f"Penalty box shots: {penalty_box_shots} ({penalty_box_shots/total_shots*100:.1f}%)")
        print(f"6-yard box shots: {six_yard_box_shots} ({six_yard_box_shots/total_shots*100:.1f}%)")
        
        return data
    
    def calculate_time_to_shoot(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate time between receiving the ball and taking a shot.
        Fixed version that matches the original logic exactly.
        """
        print("Calculating time to shoot...")
        
        # Make a copy to avoid modifying the original
        df = data.copy()
        
        # Create time to shot column
        df['TimeToShot'] = np.nan
        
        # Process each game - this is the key fix: we need to process sequentially like the original
        games_with_data = df['gameId'].unique()
        print(f"Processing {len(games_with_data)} games...")
        
        processed_games = 0
        total_shots_with_timing = 0
        
        for game_id in games_with_data:
            # Get game data and sort by time (crucial for sequential processing)
            game_df = df[df['gameId'] == game_id].copy()
            game_df = game_df.sort_values('total_seconds').reset_index()
            
            # Iterate through each action in chronological order
            for i in range(1, len(game_df)):
                current_row = game_df.iloc[i]
                prev_row = game_df.iloc[i-1]
                
                # Only process shot actions
                if current_row['is_shot'] == True:
                    current_player = current_row['player']
                    current_team = current_row['team']
                    
                    # Check if previous action was a pass TO this player FROM a teammate
                    if (prev_row['type'] == 'Pass' and 
                        prev_row['team'] == current_team and
                        prev_row['player'] != current_player):
                        
                        # Calculate time difference
                        reception_time = prev_row['total_seconds']
                        shot_time = current_row['total_seconds']
                        time_to_shoot = shot_time - reception_time
                        
                        # Update the original dataframe using the original index
                        original_idx = current_row['index']  # This is the original index
                        df.loc[original_idx, 'TimeToShot'] = time_to_shoot
                        total_shots_with_timing += 1
            
            processed_games += 1
            if processed_games % 100 == 0:
                print(f"Processed {processed_games}/{len(games_with_data)} games")
        
        print(f"Completed time to shoot calculations")
        print(f"Total shots with timing data: {total_shots_with_timing}")
        
        # Verify the results
        shots_with_timing = df[df['is_shot'] == True]['TimeToShot'].notna().sum()
        print(f"Verification: {shots_with_timing} shots have timing data")
        
        return df
    
    def create_shot_only_dataset(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create dataset containing only shot-related events."""
        print("Creating shot-only dataset...")
        
        # Keep only shots
        shot_events = data[data['is_shot'] == True].copy()
        
        print(f"Total shots: {len(shot_events)}")
        print(f"Shots with timing data: {shot_events['TimeToShot'].notna().sum()}")
        print(f"Penalty box shots: {len(shot_events[shot_events['in_penalty_box'] == True])}")
        print(f"6-yard box shots: {len(shot_events[shot_events['in_six_yard_box'] == True])}")
        
        return shot_events[self.output_columns]
    
    def create_player_aggregations(self, shot_data: pd.DataFrame, minutes_df: pd.DataFrame) -> pd.DataFrame:
        """Create pre-aggregated player statistics including area-specific stats."""
        print("Creating player aggregations...")
        
        player_stats = []
        
        for player in shot_data['player'].unique():
            player_data = shot_data[shot_data['player'] == player]
            team = player_data['team'].iloc[0]
            league = player_data['league'].iloc[0]
            
            # Get minutes played
            player_minutes_data = minutes_df[minutes_df['player'] == player]
            minutes_played = player_minutes_data['Mins'].iloc[0] if len(player_minutes_data) > 0 else 0
            
            # Calculate basic stats
            total_shots = len(player_data)
            total_goals = len(player_data[player_data['is_goal'] == True])
            conversion_rate = (total_goals / total_shots * 100) if total_shots > 0 else 0
            
            # Calculate per 90 stats
            shots_per_90 = (total_shots / minutes_played * 90) if minutes_played > 0 else 0
            goals_per_90 = (total_goals / minutes_played * 90) if minutes_played > 0 else 0
            
            # Calculate average time to shoot
            valid_timing = player_data[player_data['TimeToShot'].notna()]
            avg_time_to_shoot = valid_timing['TimeToShot'].mean() if len(valid_timing) > 0 else np.nan
            
            # Penalty box statistics
            penalty_box_data = player_data[player_data['in_penalty_box'] == True]
            penalty_box_shots = len(penalty_box_data)
            penalty_box_goals = len(penalty_box_data[penalty_box_data['is_goal'] == True])
            penalty_box_conversion = (penalty_box_goals / penalty_box_shots * 100) if penalty_box_shots > 0 else 0
            penalty_box_timing = penalty_box_data[penalty_box_data['TimeToShot'].notna()]
            penalty_box_avg_time = penalty_box_timing['TimeToShot'].mean() if len(penalty_box_timing) > 0 else np.nan
            
            # 6-yard box statistics
            six_yard_data = player_data[player_data['in_six_yard_box'] == True]
            six_yard_shots = len(six_yard_data)
            six_yard_goals = len(six_yard_data[six_yard_data['is_goal'] == True])
            six_yard_conversion = (six_yard_goals / six_yard_shots * 100) if six_yard_shots > 0 else 0
            six_yard_timing = six_yard_data[six_yard_data['TimeToShot'].notna()]
            six_yard_avg_time = six_yard_timing['TimeToShot'].mean() if len(six_yard_timing) > 0 else np.nan
            
            player_stats.append({
                'player': player,
                'team': team,
                'league': league,
                'minutes_played': minutes_played,
                'total_shots': total_shots,
                'total_goals': total_goals,
                'conversion_rate': conversion_rate,
                'shots_per_90': shots_per_90,
                'goals_per_90': goals_per_90,
                'avg_time_to_shoot': avg_time_to_shoot,
                'shots_with_timing': len(valid_timing),
                # Penalty box stats
                'penalty_box_shots': penalty_box_shots,
                'penalty_box_goals': penalty_box_goals,
                'penalty_box_conversion_rate': penalty_box_conversion,
                'penalty_box_avg_time_to_shoot': penalty_box_avg_time,
                'penalty_box_shots_with_timing': len(penalty_box_timing),
                # 6-yard box stats
                'six_yard_shots': six_yard_shots,
                'six_yard_goals': six_yard_goals,
                'six_yard_conversion_rate': six_yard_conversion,
                'six_yard_avg_time_to_shoot': six_yard_avg_time,
                'six_yard_shots_with_timing': len(six_yard_timing)
            })
        
        return pd.DataFrame(player_stats)
    
    def process_league(self, league: str, season: int = 2425) -> tuple:
        """Process a single league and return shot data and player stats."""
        file_path = self.leagues_to_file[league]
        
        # Load raw data
        raw_data = self.load_and_filter_data(file_path, league, season)
        if raw_data.empty:
            return pd.DataFrame(), pd.DataFrame()
        
        # Process data step by step
        scaled_data = self.scale_coordinates(raw_data)
        timed_data = self.calculate_total_seconds(scaled_data)
        area_labeled_data = self.label_shot_areas(timed_data)
        final_data = self.calculate_time_to_shoot(area_labeled_data)
        
        # Create shot-only dataset
        shot_data = self.create_shot_only_dataset(final_data)
        
        # Clean up memory
        del raw_data, scaled_data, timed_data, area_labeled_data, final_data
        gc.collect()
        
        return shot_data
    
    def process_all_leagues(self, minutes_file: str = 'T5_League_Mins_2025.csv'):
        """Process all leagues and create optimized datasets."""
        print("Starting preprocessing for all leagues...")
        
        # Load minutes data
        try:
            minutes_df = pd.read_csv(minutes_file)
            print(f"Loaded minutes data for {len(minutes_df)} players")
        except Exception as e:
            print(f"Error loading minutes data: {e}")
            return
        
        all_shot_data = []
        
        # Process each league
        for league in self.leagues_to_file.keys():
            print(f"\n{'='*50}")
            print(f"Processing {league}")
            print(f"{'='*50}")
            
            shot_data = self.process_league(league)
            
            if not shot_data.empty:
                all_shot_data.append(shot_data)
                
                # Save individual league file
                output_file = f"processed_{league.replace('-', '_').replace(' ', '_').lower()}_shots.parquet"
                shot_data.to_parquet(output_file, index=False)
                print(f"Saved {output_file} with {len(shot_data)} shot records")
                
                # Print timing and area stats for this league
                timing_count = shot_data['TimeToShot'].notna().sum()
                penalty_box_count = shot_data['in_penalty_box'].sum()
                six_yard_count = shot_data['in_six_yard_box'].sum()
                
                print(f"Shots with timing data in {league}: {timing_count}")
                print(f"Penalty box shots in {league}: {penalty_box_count}")
                print(f"6-yard box shots in {league}: {six_yard_count}")
            else:
                print(f"No data processed for {league}")
        
        # Combine all leagues
        if all_shot_data:
            combined_shots = pd.concat(all_shot_data, ignore_index=True)
            print(f"\nSaved combined dataset with {len(combined_shots)} total shot records")
            
            # Create player aggregations
            player_stats = self.create_player_aggregations(combined_shots, minutes_df)
            player_stats.to_parquet('processed_player_stats.parquet', index=False)
            print(f"Saved player stats with {len(player_stats)} players")
            
            # Print comprehensive summary
            print(f"\n{'='*50}")
            print("PREPROCESSING SUMMARY")
            print(f"{'='*50}")
            print(f"Total shots processed: {len(combined_shots):,}")
            print(f"Total players: {len(player_stats):,}")
            print(f"Shots with timing data: {combined_shots['TimeToShot'].notna().sum():,}")
            print(f"Penalty box shots: {combined_shots['in_penalty_box'].sum():,}")
            print(f"6-yard box shots: {combined_shots['in_six_yard_box'].sum():,}")
            
            # Show percentages
            total_shots = len(combined_shots)
            shots_with_timing = combined_shots['TimeToShot'].notna().sum()
            penalty_box_shots = combined_shots['in_penalty_box'].sum()
            six_yard_shots = combined_shots['in_six_yard_box'].sum()
            
            timing_percentage = (shots_with_timing / total_shots * 100) if total_shots > 0 else 0
            penalty_percentage = (penalty_box_shots / total_shots * 100) if total_shots > 0 else 0
            six_yard_percentage = (six_yard_shots / total_shots * 100) if total_shots > 0 else 0
            
            print(f"Percentage of shots with timing: {timing_percentage:.1f}%")
            print(f"Percentage of penalty box shots: {penalty_percentage:.1f}%")
            print(f"Percentage of 6-yard box shots: {six_yard_percentage:.1f}%")
            
            # Calculate conversion rates by area
            penalty_box_goals = combined_shots[(combined_shots['in_penalty_box'] == True) & 
                                             (combined_shots['is_goal'] == True)]
            six_yard_goals = combined_shots[(combined_shots['in_six_yard_box'] == True) & 
                                          (combined_shots['is_goal'] == True)]
            
            penalty_conversion = (len(penalty_box_goals) / penalty_box_shots * 100) if penalty_box_shots > 0 else 0
            six_yard_conversion = (len(six_yard_goals) / six_yard_shots * 100) if six_yard_shots > 0 else 0
            
            print(f"Penalty box conversion rate: {penalty_conversion:.1f}%")
            print(f"6-yard box conversion rate: {six_yard_conversion:.1f}%")
            
            # File sizes
            for league in self.leagues_to_file.keys():
                filename = f"processed_{league.replace('-', '_').replace(' ', '_').lower()}_shots.parquet"
                if Path(filename).exists():
                    size_mb = Path(filename).stat().st_size / (1024 * 1024)
                    print(f"{filename}: {size_mb:.1f} MB")
        
        print("\nPreprocessing completed!")

def main():
    """Main execution function."""
    preprocessor = ShotDataPreprocessor()
    preprocessor.process_all_leagues()

if __name__ == "__main__":
    main()