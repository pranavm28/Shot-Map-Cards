import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mplsoccer import VerticalPitch
import io
import base64
from highlight_text import fig_text
import matplotlib as mpl
import matplotlib.font_manager as fm
from pathlib import Path
import seaborn as sns

# Font setup
try:
    font_path = Path(__file__).parent / "fonts" / "Arial Rounded Bold.ttf"
    if font_path.exists():
        fm.fontManager.addfont(str(font_path))
        font_prop = fm.FontProperties(fname=str(font_path))
        mpl.rcParams['font.family'] = font_prop.get_name()
    else:
        print("Font file not found, using default font")
except Exception as e:
    print(f"Font loading error: {e}")

class OptimizedShotMapApp:
    """Optimized Streamlit app using preprocessed data."""
    
    def __init__(self):
        self.league_files = {
            '2024/25': {
                'ESP-La Liga': 'processed_esp_la_liga_shots.parquet',
                'ENG-Premier League': 'processed_eng_premier_league_shots.parquet',
                'ITA-Serie A': 'processed_ita_serie_a_shots.parquet',
                'GER-Bundesliga': 'processed_ger_bundesliga_shots.parquet',
                'FRA-Ligue 1': 'processed_fra_ligue_1_shots.parquet'
            },
            '2023/24': {
                'ENG-Premier League': 'processed_eng_premier_league_2324_shots.parquet',
                'POR-Liga Portugal': 'processed_por_liga_portugal_2324_shots.parquet'
            }
        }

        self.player_stats_files = {
            '2024/25': 'processed_player_stats.parquet',
            '2023/24': 'processed_player_stats_2324.parquet'
        }

            # --- NEW: area bounds (StatsBomb 120x80) ---
        # Keys here map to UI labels below.
        self.area_bounds = {
            'Zone-14':           {'x_min': 80, 'x_max': 102, 'y_min': 30, 'y_max': 50},
            'Left Half-Space':   {'x_min': 60, 'x_max': 102, 'y_min': 50, 'y_max': 62},
            'Right Half-Space':  {'x_min': 60, 'x_max': 102, 'y_min': 18, 'y_max': 30},
        }

    
    @st.cache_data
    def load_shot_data(_self, season:str, league: str) -> pd.DataFrame:
        """Load preprocessed shot data for a league and season."""
        try:
            file_path = _self.league_files[season][league]
            df = pd.read_parquet(file_path)
            return df
        except Exception as e:
            st.error(f"Failed to load shot data for {league} ({season}): {e}")
            return pd.DataFrame()
    
    @st.cache_data
    def load_player_stats(_self, season: str) -> pd.DataFrame:
        """Load preprocessed player statistics."""
        try:
            file_path = _self.player_stats_files[season]
            df = pd.read_parquet(file_path)
            return df
        except Exception as e:
            st.error(f"Failed to load player stats: {season} {e}")
            return pd.DataFrame()

    def _mask_rect(self, df: pd.DataFrame, bounds: dict) -> pd.Series:
        """Return boolean mask for points inside a rectangular area."""
        return (
            (df['x'] >= bounds['x_min']) & (df['x'] <= bounds['x_max']) &
            (df['y'] >= bounds['y_min']) & (df['y'] <= bounds['y_max'])
        )


    
    def filter_player_shots(self, shot_data: pd.DataFrame, player_name: str, max_time: float = None, area_filter: str = None) -> pd.DataFrame:
        """Filter shots for a specific player with optional time and area filters."""
        player_shots = shot_data[shot_data['player'] == player_name].copy()
        
        # Apply area filter
        if area_filter == "Penalty Box":
            player_shots = player_shots[player_shots['in_penalty_box'] == True]
        elif area_filter == "Six Yard Box":
            player_shots = player_shots[player_shots['in_six_yard_box'] == True]

        elif area_filter in ("Zone-14", "Left Half-Space", "Right Half-Space"):
            b = self.area_bounds[area_filter]
            mask = self._mask_rect(player_shots, b)
            player_shots = player_shots[mask]
        
        if max_time is not None:
            time_mask = (player_shots['TimeToShot'] <= max_time) & (player_shots['TimeToShot'].notna())
            player_shots = player_shots[time_mask]
        
        return player_shots

    
    def calculate_filtered_stats(self, shot_data: pd.DataFrame, player_name: str, max_time: float = None, area_filter: str = None) -> dict:
        """Calculate statistics for filtered data."""
        filtered_shots = self.filter_player_shots(shot_data, player_name, max_time, area_filter)
        
        total_shots = len(filtered_shots)
        total_goals = len(filtered_shots[filtered_shots['is_goal'] == True])
        conversion_rate = (total_goals / total_shots * 100) if total_shots > 0 else 0
        
        # Calculate average time to shoot for filtered shots
        valid_timing = filtered_shots[filtered_shots['TimeToShot'].notna()]
        avg_time_to_shoot = valid_timing['TimeToShot'].mean() if len(valid_timing) > 0 else 0
        
        return {
            'total_shots': total_shots,
            'total_goals': total_goals,
            'conversion_rate': conversion_rate,
            'avg_time_to_shoot': avg_time_to_shoot if not np.isnan(avg_time_to_shoot) else 0
        }
    
    def create_shot_map(self, shot_data: pd.DataFrame, player_stats: pd.DataFrame, 
                       player_name: str, max_time: float = None, area_filter: str = None) -> tuple:
        """Create shot map visualization."""
        # Get filtered shots
        filtered_shots = self.filter_player_shots(shot_data, player_name, max_time, area_filter)
        
        if filtered_shots.empty:
            return None, None
        
        # Get player info
        player_info = player_stats[player_stats['player'] == player_name].iloc[0]
        player_team = player_info['team']
        player_minutes = player_info['minutes_played']
        
        # Separate shots and goals
        shots = filtered_shots[(filtered_shots['is_shot'] == True) & (filtered_shots['is_goal'] != True)]
        goals = filtered_shots[filtered_shots['is_goal'] == True]
        
        # Calculate stats
        stats = self.calculate_filtered_stats(shot_data, player_name, max_time, area_filter)
        
        # Create the plot
        fig, ax = plt.subplots(figsize=(15.5, 12))
        fig.set_facecolor('#181818')
        ax.patch.set_facecolor('#181818')
        
        pitch = VerticalPitch(pitch_type='statsbomb',
                             pitch_color='#181818', line_color='#FFFFFF',
                             half=True, label=False)
        pitch.draw(ax=ax)
        
        # Plot shots (non-goals)
        if not shots.empty:
            pitch.scatter(shots.x, shots.y, s=100,
                         marker='o', edgecolors='none', c='#FF5959', zorder=2, ax=ax, alpha=0.45)
            pitch.scatter(shots.x, shots.y, s=100,
                         marker='o', edgecolors='#FF5959', c='none', zorder=3, ax=ax, alpha=1, lw=2.2)
        
        # Plot goals
        if not goals.empty:
            pitch.scatter(goals.x, goals.y, s=200, marker='o', edgecolors='#8ff00f',
                         c='none', ax=ax, alpha=1, zorder=4, lw=2.5)
            pitch.scatter(goals.x, goals.y, s=200, marker='o', edgecolors='none',
                         c='#8ff00f', ax=ax, alpha=0.45, zorder=4)
        
        # Stats circles and annotations
        # Circle 1: Total Shots
        pitch.scatter(64*1.2, 10*0.8, s=1800, marker='o', edgecolors='#FF5959', c='none', 
                     ax=ax, ls='-', lw=2.5, alpha=0.9, zorder=4)
        pitch.scatter(64*1.2, 10*0.8, s=1800, marker='o', edgecolors='none', c='#FF5959', 
                     ax=ax, ls='-', lw=2.5, alpha=0.3, zorder=4)
        pitch.annotate(f'{stats["total_shots"]:03d}', xy=(64*1.2, 10*0.8), xytext=(77.5, 6.15), 
                      ax=ax, font='Arial Rounded MT Bold', fontsize=22, color='#FFFFFF', 
                      fontweight='bold', zorder=5)
        pitch.annotate('Shots', xy=(64*1.2, 10*0.8), xytext=(77.5, 11.2), 
                      ax=ax, font='Arial Rounded MT Bold', fontsize=22, color='#FFFFFF', 
                      fontweight='bold', zorder=5)
        
        # Circle 2: Goals
        pitch.scatter(64*1.2, 30*0.8, s=1800, marker='o', edgecolors='#8ff00f', c='none', 
                     ax=ax, ls='-', lw=2.5, alpha=0.9, zorder=4)
        pitch.scatter(64*1.2, 30*0.8, s=1800, marker='o', edgecolors='none', c='#8ff00f', 
                     ax=ax, ls='-', lw=2.5, alpha=0.3, zorder=4)
        pitch.annotate(f'{stats["total_goals"]:02d}', xy=(64*1.2, 30*0.8), xytext=(77.5, 22.85), 
                      ax=ax, font='Arial Rounded MT Bold', fontsize=22, color='#FFFFFF', 
                      fontweight='bold', zorder=5)
        pitch.annotate('Goals', xy=(64*1.2, 30*0.8), xytext=(77.5, 27.2), 
                      ax=ax, font='Arial Rounded MT Bold', fontsize=22, color='#FFFFFF', 
                      fontweight='bold', zorder=5)
        
        # Circle 3: Conversion Rate
        pitch.scatter(64*1.2, 50*0.8, s=1800, marker='o', edgecolors='#FFFFFF', c='none', 
                     ax=ax, ls='-', lw=2.5, alpha=0.9, zorder=4)
        pitch.scatter(64*1.2, 50*0.8, s=1800, marker='o', edgecolors='none', c='#FFFFFF', 
                     ax=ax, ls='-', lw=2.5, alpha=0.2, zorder=4)
        pitch.annotate(f'{stats["conversion_rate"]:.1f}', xy=(64*1.2, 50*0.8), xytext=(77.5, 38.33), 
                      ax=ax, font='Arial Rounded MT Bold', fontsize=18, color='#FFFFFF', 
                      fontweight='bold', zorder=5)
        pitch.annotate('Conv. %', xy=(64*1.2, 50*0.8), xytext=(77.5, 43.2), 
                      ax=ax, font='Arial Rounded MT Bold', fontsize=22, color='#FFFFFF', 
                      fontweight='bold', zorder=5)
        
        # Circle 4: Average Time to Shoot
        pitch.scatter(64*1.2, 78*0.8, s=1800, marker='o', edgecolors='#FFFFFF', c='none', 
                     ax=ax, ls='-', lw=2.5, alpha=0.9, zorder=4)
        pitch.scatter(64*1.2, 78*0.8, s=1800, marker='o', edgecolors='none', c='#FFFFFF', 
                     ax=ax, ls='-', lw=2.5, alpha=0.2, zorder=4)
        pitch.annotate(f'{stats["avg_time_to_shoot"]:.1f}s', xy=(64*1.2, 78*0.8), xytext=(77.5, 60.8), 
                      ax=ax, font='Arial Rounded MT Bold', fontsize=18, color='#FFFFFF', 
                      fontweight='bold', zorder=5)
        pitch.annotate('Avg. Time', xy=(64*1.2, 78*0.8), xytext=(77.5, 65.4), 
                      ax=ax, font='Arial Rounded MT Bold', fontsize=22, color='#FFFFFF', 
                      fontweight='bold', zorder=5)
        
        # Divider line
        pitch.lines(80, 0, 80, 80, ls='-', lw=1.5, color='#FFFFFF', ax=ax, zorder=1, alpha=0.4)
        
        # Invert y-axis
        plt.gca().invert_yaxis()
        
        # Title - Updated to reflect that only shots after passes are shown when filtered
        area_filter_text = f" ({area_filter})" if area_filter else ""
        time_filter_text = f" (within {max_time}s)" if max_time is not None else ""
        fig_text(0.512, 0.980, f"<{player_name}>", font='Arial Rounded MT Bold', size=30,
                 ha="center", color="#FFFFFF", fontweight='bold', highlight_textprops=[{"color": '#FFFFFF'}])
        
        fig_text(0.512, 0.933,
         f"{player_team} | {int(player_minutes)} Mins | Shot Map Card | Made by @pranav_m28",
         font='Arial Rounded MT Bold', size=24,
         ha="center", color="#FFFFFF", fontweight='bold')

        fig_text(0.752, 0.786,
         f"shot-map-cards.streamlit.app",
         font='Arial Rounded MT Bold', size=14,
         ha="center", color="#FFFFFF", fontweight='bold', alpha=0.6)
        
        if area_filter or max_time is not None:
            filter_parts = []
            if area_filter:
                filter_parts.append(f"Area: {area_filter}")
            if max_time is not None:
                filter_parts.append(f"Time: within {max_time}s")
            
            filter_text = " | ".join(filter_parts)
            fig_text(0.512, 0.886,  # Position below the main info line
                    filter_text,
                    font='Arial Rounded MT Bold', size=20,  # Slightly smaller font
                    ha="center", color="#FFFFFF", fontweight='bold')
        
        # Save for display
        buffer = io.BytesIO()
        plt.tight_layout()
        plt.savefig(buffer, format='png', facecolor='#181818', edgecolor='none', dpi=300, bbox_inches='tight')
        buffer.seek(0)
        plot_data = base64.b64encode(buffer.read()).decode()
        
        # Save for download
        download_buffer = io.BytesIO()
        plt.savefig(download_buffer, format='png', facecolor='#181818', edgecolor='none', dpi=400, bbox_inches='tight')
        download_buffer.seek(0)
        download_data = download_buffer.getvalue()
        
        plt.close(fig)
        return plot_data, download_data
    
    def analyze_time_distribution(self, shot_data: pd.DataFrame, player_stats: pd.DataFrame, 
                                selected_teams: list, min_minutes: int) -> pd.DataFrame:
        """Analyze shot distribution by time periods."""
        # Filter players by teams and minutes
        eligible_players = player_stats[
            (player_stats['team'].isin(selected_teams)) & 
            (player_stats['minutes_played'] >= min_minutes)
        ]['player'].tolist()
        
        # Filter shot data for eligible players and shots with timing data
        filtered_shots = shot_data[
            (shot_data['player'].isin(eligible_players)) & 
            (shot_data['TimeToShot'].notna())
        ].copy()
        
        if filtered_shots.empty:
            return pd.DataFrame()
        
        # Define time bins
        bins = [0, 1, 2, 3, 4, 5, 7.5, 10, float('inf')]
        labels = ['0-1s', '1-2s', '2-3s', '3-4s', '4-5s', '5-7.5s', '7.5-10s', '10s+']
        
        # Create time categories
        filtered_shots['time_category'] = pd.cut(filtered_shots['TimeToShot'], 
                                       bins=bins, labels=labels, right=False, ordered=True)
        
        # Group by player and time category
        distribution_data = []
        
        for player in eligible_players:
            player_shots = filtered_shots[filtered_shots['player'] == player]
            player_info = player_stats[player_stats['player'] == player].iloc[0]
            
            if len(player_shots) == 0:
                continue
                
            for category in labels:
                category_shots = player_shots[player_shots['time_category'] == category]
                shots_count = len(category_shots)
                goals_count = len(category_shots[category_shots['is_goal'] == True])
                
                distribution_data.append({
                    'player': player,
                    'team': player_info['team'],
                    'minutes_played': player_info['minutes_played'],
                    'time_category': category,
                    'shots': shots_count,
                    'goals': goals_count,
                    'conversion_rate': (goals_count / shots_count * 100) if shots_count > 0 else 0
                })
        
        return pd.DataFrame(distribution_data)
    
    def create_time_distribution_chart(self, distribution_df: pd.DataFrame, 
                                     chart_type: str, selected_players: list = None) -> tuple:
        """Create time distribution visualization."""
        if distribution_df.empty:
            return None, None
        
        # Filter by selected players if specified
        if selected_players:
            distribution_df = distribution_df[distribution_df['player'].isin(selected_players)]
        
        if distribution_df.empty:
            return None, None
        # Determine title based on filters
        # Aggregate data by time category
        agg_data = distribution_df.groupby('time_category').agg({
            'shots': 'sum',
            'goals': 'sum'
        }).reset_index()
        
        # Calculate conversion rate
        agg_data['conversion_rate'] = (agg_data['goals'] / agg_data['shots'] * 100).fillna(0)
        
        # Create the plot based on chart type
        plt.style.use('dark_background')

        category_order = ['0-1s', '1-2s', '2-3s', '3-4s', '4-5s', '5-7.5s', '7.5-10s', '10s+']
        agg_data['time_category'] = pd.Categorical(agg_data['time_category'], categories=category_order, ordered=True)
        agg_data = agg_data.sort_values('time_category')
        if selected_players:
            if len(selected_players) == 1:
                title = f"{selected_players[0]} - Shot Time Distribution"
            else:
                title = f"Selected Players - Shot Time Distribution"
        else:
    # Check if all players are from the same team
            unique_teams = distribution_df['team'].unique()
            if len(unique_teams) == 1:
                title = f"{unique_teams[0]} - Shot Time Distribution"
            else:
                title = "Shot Time Distribution"
        
        if chart_type == "Bar Chart":
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
            fig.patch.set_facecolor('#181818')
            fig.suptitle(title, fontsize=20, fontweight='bold', color='white', y=0.98)
            
            # Shots bar chart
            bars1 = ax1.bar(agg_data['time_category'], agg_data['shots'], 
                           color='#FF5959', alpha=0.8, edgecolor='white', linewidth=1)
            ax1.set_title('Shot Distribution by Time Category | @pranav_m28', fontsize=12, fontweight='bold', color='white')
            ax1.set_ylabel('Number of Shots', fontsize=12, color='white')
            ax1.tick_params(colors='white')
            ax1.set_facecolor('#181818')
            
            # Add value labels on bars
            for bar in bars1:
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                        f'{int(height)}', ha='center', va='bottom', color='white', fontweight='bold')
            
            # Goals bar chart
            bars2 = ax2.bar(agg_data['time_category'], agg_data['goals'], 
                           color='#8ff00f', alpha=0.8, edgecolor='white', linewidth=1)
            ax2.set_title('Goal Distribution by Time Category | @pranav_m28', fontsize=12, fontweight='bold', color='white')
            ax2.set_ylabel('Number of Goals', fontsize=12, color='white')
            ax2.set_xlabel('Time to Shoot', fontsize=12, color='white')
            ax2.tick_params(colors='white')
            ax2.set_facecolor('#181818')
            
            # Add value labels on bars
            for bar in bars2:
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                        f'{int(height)}', ha='center', va='bottom', color='white', fontweight='bold')
            
            plt.xticks(rotation=45)
            plt.tight_layout()
        
        elif chart_type == "Stacked Bar":
            fig, ax = plt.subplots(figsize=(12, 8))
            fig.patch.set_facecolor('#181818')
            ax.set_facecolor('#181818')
            fig.suptitle(title, fontsize=20, fontweight='bold', color='white', y=0.98)
            
            # Create stacked bar chart
            width = 0.6
            x_pos = range(len(agg_data))
            
            # Goals (bottom)
            bars1 = ax.bar(x_pos, agg_data['goals'], width, 
                          color='#8ff00f', alpha=0.8, label='Goals', edgecolor='white', linewidth=1)
            
            # Shots minus goals (top)
            shots_minus_goals = agg_data['shots'] - agg_data['goals']
            bars2 = ax.bar(x_pos, shots_minus_goals, width, bottom=agg_data['goals'],
                          color='#FF5959', alpha=0.8, label='Shots (no goal)', edgecolor='white', linewidth=1)
            
            ax.set_title('Shot vs Goal Distribution by Time Category | @pranav_m28', fontsize=12, fontweight='bold', color='white')
            ax.set_ylabel('Count', fontsize=12, color='white')
            ax.set_xlabel('Time to Shoot', fontsize=12, color='white')
            ax.set_xticks(x_pos)
            ax.set_xticklabels(agg_data['time_category'], rotation=45)
            ax.tick_params(colors='white')
            ax.legend(loc='upper right')
            
            # Add conversion rate labels on top of bars
            for i, (shots, goals, conv_rate) in enumerate(zip(agg_data['shots'], agg_data['goals'], agg_data['conversion_rate'])):
                ax.text(i, shots + 1, f'{conv_rate:.1f}%', ha='center', va='bottom', 
                       color='white', fontweight='bold', fontsize=10)
            
            plt.tight_layout()
        
        elif chart_type == "Conversion Rate Line":
            fig, ax = plt.subplots(figsize=(12, 8))
            fig.patch.set_facecolor('#181818')
            ax.set_facecolor('#181818')
            fig.suptitle(title, fontsize=20, fontweight='bold', color='white', y=0.98)
            
            # Line plot for conversion rate
            ax.plot(agg_data['time_category'], agg_data['conversion_rate'], 
                   color='#00D4FF', marker='o', linewidth=3, markersize=8, alpha=0.9)
            
            # Fill area under the line
            ax.fill_between(agg_data['time_category'], agg_data['conversion_rate'], 
                           alpha=0.3, color='#00D4FF')
            
            ax.set_title('Conversion Rate by Time to Shoot | @pranav_m28', fontsize=12, fontweight='bold', color='white')
            ax.set_ylabel('Conversion Rate (%)', fontsize=12, color='white')
            ax.set_xlabel('Time to Shoot', fontsize=12, color='white')
            ax.tick_params(colors='white')
            
            # Add value labels on points
            for i, (cat, rate) in enumerate(zip(agg_data['time_category'], agg_data['conversion_rate'])):
                ax.text(i, rate + 0.5, f'{rate:.1f}%', ha='center', va='bottom', 
                       color='white', fontweight='bold')
            
            plt.xticks(rotation=45)
            plt.tight_layout()
        
        # Save plots
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', facecolor='#181818', edgecolor='none', dpi=300, bbox_inches='tight')
        buffer.seek(0)
        plot_data = base64.b64encode(buffer.read()).decode()
        
        download_buffer = io.BytesIO()
        plt.savefig(download_buffer, format='png', facecolor='#181818', edgecolor='none', dpi=400, bbox_inches='tight')
        download_buffer.seek(0)
        download_data = download_buffer.getvalue()
        
        plt.close(fig)
        return plot_data, download_data
    """
    def create_player_comparison_heatmap(self, distribution_df: pd.DataFrame, top_n: int = 15) -> tuple:
        
        if distribution_df.empty:
            return None, None
        
        # Get top players by total shots with timing data
        player_totals = distribution_df.groupby('player')['shots'].sum().sort_values(ascending=False)
        top_players = player_totals.head(top_n).index.tolist()
        
        # Filter for top players
        filtered_df = distribution_df[distribution_df['player'].isin(top_players)]
        
        # Create pivot table for heatmap
        heatmap_data = filtered_df.pivot_table(
            index='player', 
            columns='time_category', 
            values='shots', 
            fill_value=0
        )
        
        # Reorder columns properly
        desired_order = ['0-1s', '1-2s', '2-3s', '3-4s', '4-5s', '5-7.5s', '7.5-10s', '10s+']
        heatmap_data = heatmap_data.reindex(columns=[col for col in desired_order if col in heatmap_data.columns])
        
        # Create the heatmap
        plt.style.use('dark_background')
        fig, ax = plt.subplots(figsize=(12, 10))
        fig.patch.set_facecolor('#181818')
        
        # Create heatmap
        sns.heatmap(heatmap_data, annot=True, fmt='d', cmap='Reds', 
                   cbar_kws={'label': 'Number of Shots'}, ax=ax,
                   linewidths=0.5, linecolor='white')
        
        ax.set_title(f'Shot Distribution Heatmap - Top {top_n} Players', 
                    fontsize=16, fontweight='bold', color='white', pad=20)
        ax.set_xlabel('Time to Shoot', fontsize=12, color='white')
        ax.set_ylabel('Player', fontsize=12, color='white')
        ax.tick_params(colors='white')
        
        # Rotate x-axis labels
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        # Save plots
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', facecolor='#181818', edgecolor='none', dpi=300, bbox_inches='tight')
        buffer.seek(0)
        plot_data = base64.b64encode(buffer.read()).decode()
        
        download_buffer = io.BytesIO()
        plt.savefig(download_buffer, format='png', facecolor='#181818', edgecolor='none', dpi=400, bbox_inches='tight')
        download_buffer.seek(0)
        download_data = download_buffer.getvalue()
        
        plt.close(fig)
        return plot_data, download_data
    """
    def run_shot_map_tab(self, shot_data: pd.DataFrame, player_stats: pd.DataFrame, selected_league: str, selected_season: str):
        """Run the shot map analysis tab."""
        # Filter player stats by league
        league_player_stats = player_stats[player_stats['league'] == selected_league]
        
        # Team selection
        teams = sorted(league_player_stats['team'].unique())
        selected_teams = st.sidebar.multiselect("Select Teams", teams, default=teams, key="shotmap_teams")
        
        if not selected_teams:
            st.warning("Please select at least one team.")
            return
        
        # Filter by selected teams
        filtered_player_stats = league_player_stats[league_player_stats['team'].isin(selected_teams)]
        
        # Minimum minutes filter
        st.sidebar.markdown("---")
        st.sidebar.subheader("Player Filters")
        
        max_minutes = int(filtered_player_stats['minutes_played'].max()) if len(filtered_player_stats) > 0 else 1000
        min_minutes = st.sidebar.slider(
            "Minimum Minutes Played", 
            min_value=0, 
            max_value=max_minutes,
            value=0, 
            step=50,
            key="shotmap_min_minutes"
        )
        
        # Filter players by minimum minutes
        eligible_players = filtered_player_stats[filtered_player_stats['minutes_played'] >= min_minutes]
        
        if len(eligible_players) == 0:
            st.warning(f"No players found with at least {min_minutes} minutes played.")
            return
        
        # Player selection
        players = sorted(eligible_players['player'].unique())
        selected_player = st.sidebar.selectbox("Select Player", players, key="shotmap_player")
        
        # Time filter
        st.sidebar.markdown("---")
        st.sidebar.subheader("⏱️ Shot Timing Filter")
        st.sidebar.info("ℹ️ Time filter only shows shots taken after receiving a pass from a teammate (excludes penalties, free kicks, etc.)")
        use_time_filter = st.sidebar.checkbox("Filter by Time to Shoot", key="shotmap_time_filter")
        max_time = None
        if use_time_filter:
            max_time = st.sidebar.slider("Maximum Time to Shoot (seconds)", 
                                       min_value=0.5, max_value=10.0, value=5.0, step=0.5,
                                       key="shotmap_max_time")
            

        # Area filter
        st.sidebar.markdown("---")
        st.sidebar.subheader("🎯 Area Filter")
        area_filter = st.sidebar.selectbox(
            "Filter by shooting area",
            ["All Areas", "Penalty Box", "Six Yard Box", "Zone-14", "Left Half-Space", "Right Half-Space"],
            help="Filter shots by specific areas of the pitch"
        )

        area_filter = None if area_filter == "All Areas" else area_filter
        
        # Main content
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("📊 Shot Map Visualization")
            if selected_player:
                with st.spinner("Creating shot map..."):
                    result = self.create_shot_map(shot_data, league_player_stats, selected_player, max_time, area_filter)
                
                if result and result[0] is not None:
                    plot_data, download_data = result
                    st.image(f"data:image/png;base64,{plot_data}")
                    
                    # Download button
                    area_suffix = f"_{area_filter.lower().replace(' ', '_')}" if area_filter else ""
                    time_filter_suffix = f"_within_{max_time}s" if max_time is not None else ""
                    filename = f"{selected_player.replace(' ', '_')}_shot_map{area_suffix}{time_filter_suffix}.png"
                    
                    st.download_button(
                        label="📥 Download High Quality Image",
                        data=download_data,
                        file_name=filename,
                        mime="image/png",
                        help="Download the shot map in high quality (400 DPI)"
                    )
                else:
                    st.error("Could not generate shot map for the selected player.")
        
        with col2:
            st.subheader("📈 Player Statistics")
            if selected_player:
                # Get player info
                player_info = league_player_stats[league_player_stats['player'] == selected_player].iloc[0]
                
                # Calculate filtered stats
                filtered_stats = self.calculate_filtered_stats(shot_data, selected_player, max_time, area_filter)
                
                # Display metrics with context
                if max_time is not None:
                    st.info(f"📊 Filtered stats show only shots within {max_time}s after receiving a pass")
                
                st.metric("Total Shots", f"{filtered_stats['total_shots']}")
                st.metric("Goals", f"{filtered_stats['total_goals']}")
                st.metric("Conversion Rate", f"{filtered_stats['conversion_rate']:.1f}%")
                st.metric("Minutes Played", f"{int(player_info['minutes_played'])}")
                st.metric("Shots/90", f"{player_info['shots_per_90']:.2f}")
                st.metric("Goals/90", f"{player_info['goals_per_90']:.2f}")
                st.metric("Avg. Time to Shoot", f"{filtered_stats['avg_time_to_shoot']:.2f}s")
                
                # Show context about timing data
                if max_time is not None:
                    all_player_shots = shot_data[shot_data['player'] == selected_player]
                    total_shots = len(all_player_shots)
                    shots_with_timing = len(all_player_shots[all_player_shots['TimeToShot'].notna()])
                    st.markdown(f"**Context:** {shots_with_timing}/{total_shots} total shots have timing data")

                # Display area-specific stats from preprocessed data
                #if area_filter and selected_player:
                 #   player_info = league_player_stats[league_player_stats['player'] == selected_player].iloc[0]
                    
                  #  if area_filter == "Penalty Box":
                   #     st.markdown("**Penalty Box Stats (All Season):**")
                    #    st.metric("Penalty Box Shots", f"{int(player_info['penalty_box_shots'])}")
                     #   st.metric("Penalty Box Goals", f"{int(player_info['penalty_box_goals'])}")
                       # st.metric("Penalty Box Conv. Rate", f"{player_info['penalty_box_conversion_rate']:.1f}%")
                      #  st.metric("Penalty Box Avg. Time", f"{player_info['penalty_box_avg_time_to_shoot']:.2f}s")
                        
                    #elif area_filter == "Six Yard Box":
                     #   st.markdown("**Six Yard Box Stats (All Season):**")
                      #  st.metric("Six Yard Shots", f"{int(player_info['six_yard_shots'])}")
                       # st.metric("Six Yard Goals", f"{int(player_info['six_yard_goals'])}")
                        #st.metric("Six Yard Conv. Rate", f"{player_info['six_yard_conversion_rate']:.1f}%")
                        #st.metric("Six Yard Avg. Time", f"{player_info['six_yard_avg_time_to_shoot']:.2f}s")
        
        # Summary table
        st.subheader(f"📋 All Players Summary - {selected_league} ({selected_season}) (Min. {min_minutes} minutes)")
        
        # Create enhanced summary table
        summary_data = []
        for _, player_row in eligible_players.iterrows():
            player_name = player_row['player']
            
            # Calculate filtered stats if time filter is applied

            if area_filter is not None:
                area_filtered_stats = self.calculate_filtered_stats(shot_data, player_name, max_time, area_filter)
                area_filtered_shots = area_filtered_stats['total_shots']
                area_filtered_goals = area_filtered_stats['total_goals']
                area_filtered_conv = area_filtered_stats['conversion_rate']
            else:
                area_filtered_shots = 'N/A'
                area_filtered_goals = 'N/A'
                area_filtered_conv = 'N/A'

            if max_time is not None:
                filtered_stats = self.calculate_filtered_stats(shot_data, player_name, max_time)
                filtered_shots = filtered_stats['total_shots']
                filtered_goals = filtered_stats['total_goals']
                filtered_conv = filtered_stats['conversion_rate']
            else:
                filtered_shots = 'N/A'
                filtered_goals = 'N/A'
                filtered_conv = 'N/A'
            
            summary_data.append({
                'Player': player_name,
                'Team': player_row['team'],
                'Minutes': int(player_row['minutes_played']),
                'Total Shots': int(player_row['total_shots']),
                'Total Goals': int(player_row['total_goals']),
                'Overall Conv. %': player_row['conversion_rate'],
                'Shots/90': player_row['shots_per_90'],
                'Goals/90': player_row['goals_per_90'],
                'Avg. Time to Shoot': player_row['avg_time_to_shoot'],
                'Filtered Shots': filtered_shots,
                'Filtered Goals': filtered_goals,
                'Filtered Conv. %': filtered_conv, 
                f'Area Filtered Shots ({area_filter})': area_filtered_shots,
                f'Area Filtered Goals ({area_filter})': area_filtered_goals,
                f'Area Filtered Conv. % ({area_filter})': area_filtered_conv
            })

            # Calculate area-specific filtered stats if area filter is applied
        
        summary_df = pd.DataFrame(summary_data)
        summary_df = summary_df.sort_values('Total Shots', ascending=False)
        
        # Format the dataframe for display
        formatted_df = summary_df.copy()
        for col in ['Overall Conv. %', 'Shots/90', 'Goals/90', 'Avg. Time to Shoot']:
            if col in formatted_df.columns:
                formatted_df[col] = formatted_df[col].apply(lambda x: f"{x:.2f}" if isinstance(x, (int, float)) else x)
        
        if max_time is not None and 'Filtered Conv. %' in formatted_df.columns:
            formatted_df['Filtered Conv. %'] = formatted_df['Filtered Conv. %'].apply(
                lambda x: f"{x:.1f}%" if isinstance(x, (int, float)) else x
            )
        
        st.dataframe(formatted_df, use_container_width=True)
        
        # Show summary stats
        total_eligible = len(eligible_players)
        total_league = len(league_player_stats)
        st.info(f"📊 Showing {total_eligible} out of {total_league} players with at least {min_minutes} minutes played")
    
    def run_time_distribution_tab(self, shot_data: pd.DataFrame, player_stats: pd.DataFrame, selected_league: str, selected_season: str):
        """Run the time distribution analysis tab."""
        st.header(f"⏱️ Shot Time Distribution Analysis - {selected_league} ({selected_season})")
        st.markdown("Analyze how quickly players shoot after receiving the ball from teammates")
        
        # Filter player stats by league
        league_player_stats = player_stats[player_stats['league'] == selected_league]
        
        # Sidebar filters for time distribution
        st.sidebar.markdown("---")
        st.sidebar.subheader("🔍 Distribution Filters")
        
        # Team selection
        teams = sorted(league_player_stats['team'].unique())
        selected_teams_dist = st.sidebar.multiselect("Select Teams", teams, default=teams, key="dist_teams")
        
        if not selected_teams_dist:
            st.warning("Please select at least one team.")
            return
        
        # Minimum minutes filter
        filtered_player_stats = league_player_stats[league_player_stats['team'].isin(selected_teams_dist)]
        max_minutes = int(filtered_player_stats['minutes_played'].max()) if len(filtered_player_stats) > 0 else 1000
        min_minutes_dist = st.sidebar.slider(
            "Minimum Minutes Played", 
            min_value=0, 
            max_value=max_minutes,
            value=300, 
            step=50,
            key="dist_min_minutes"
        )
        
        # Analyze time distribution
        with st.spinner("Analyzing time distribution..."):
            distribution_df = self.analyze_time_distribution(
                shot_data, league_player_stats, selected_teams_dist, min_minutes_dist
            )
        
        if distribution_df.empty:
            st.error("No timing data available for the selected filters.")
            return
        
        # Show overview metrics
        total_shots_with_timing = distribution_df['shots'].sum()
        total_goals_with_timing = distribution_df['goals'].sum()
        overall_conversion = (total_goals_with_timing / total_shots_with_timing * 100) if total_shots_with_timing > 0 else 0
        unique_players = distribution_df['player'].nunique()
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Players Analyzed", unique_players)
        with col2:
            st.metric("Total Shots", total_shots_with_timing)
        with col3:
            st.metric("Total Goals", total_goals_with_timing)
        with col4:
            st.metric("Overall Conversion", f"{overall_conversion:.1f}%")
        
        st.markdown("---")
        
        # Chart options
        col1, col2 = st.columns([2, 1])
        
        with col2:
            st.subheader("📊 Visualization Options")
            
            chart_type = st.selectbox(
                "Select Chart Type",
                ["Bar Chart", "Stacked Bar", "Conversion Rate Line"],
                help="Choose how to visualize the time distribution data"
            )
            
            # Player filter for individual analysis
            st.markdown("**Player-Specific Analysis:**")
            all_players = sorted(distribution_df['player'].unique())
            selected_players_analysis = st.multiselect(
                "Filter by specific players (leave empty for all)",
                all_players,
                help="Select specific players to analyze their time distribution"
            )
            
            # Show heatmap option
            #show_heatmap = st.checkbox("Show Player Comparison Heatmap", value=True)
            #if show_heatmap:
             #   top_n_players = st.slider("Top N players for heatmap", 5, 25, 15)
        
        with col1:
            st.subheader("📈 Time Distribution Charts")
            
            # Create and display main chart
            with st.spinner("Creating visualization..."):
                chart_result = self.create_time_distribution_chart(
                    distribution_df, chart_type, selected_players_analysis
                )
            
            if chart_result and chart_result[0] is not None:
                plot_data, download_data = chart_result
                st.image(f"data:image/png;base64,{plot_data}")
                
                # Download button
                players_suffix = f"_{'_'.join(selected_players_analysis[:2])}" if selected_players_analysis else "_all_players"
                chart_filename = f"time_distribution_{chart_type.lower().replace(' ', '_')}{players_suffix}.png"
                
                st.download_button(
                    label="📥 Download Chart",
                    data=download_data,
                    file_name=chart_filename,
                    mime="image/png"
                )
            else:
                st.error("Could not generate chart with selected filters.")
        
        # Player comparison heatmap
        # """        
        # if show_heatmap:
        #     st.markdown("---")
        #     st.subheader("🔥 Player Comparison Heatmap")
            
        #     with st.spinner("Creating heatmap..."):
        #         heatmap_result = self.create_player_comparison_heatmap(distribution_df, top_n_players)
            
        #     if heatmap_result and heatmap_result[0] is not None:
        #         heatmap_plot, heatmap_download = heatmap_result
        #         st.image(f"data:image/png;base64,{heatmap_plot}")
                
        #         st.download_button(
        #             label="📥 Download Heatmap",
        #             data=heatmap_download,
        #             file_name=f"player_time_distribution_heatmap_top_{top_n_players}.png",
        #             mime="image/png"
        #         )
        #     else:
        #         st.error("Could not generate heatmap.")
        # """
        # Detailed statistics table
        st.markdown("---")
        st.subheader("📊 Detailed Time Distribution Statistics")
        
        # Create summary by player
        player_summary = distribution_df.groupby(['player', 'team']).agg({
            'shots': 'sum',
            'goals': 'sum',
            'minutes_played': 'first'
        }).reset_index()
        
        player_summary['conversion_rate'] = (player_summary['goals'] / player_summary['shots'] * 100).fillna(0)
        player_summary = player_summary.sort_values('shots', ascending=False)
        
        # Format for display
        display_summary = player_summary.copy()
        display_summary['conversion_rate'] = display_summary['conversion_rate'].apply(lambda x: f"{x:.1f}%")
        display_summary['minutes_played'] = display_summary['minutes_played'].apply(lambda x: int(x))
        
        display_summary.columns = ['Player', 'Team', 'Shots (w/ timing)', 'Goals (w/ timing)', 'Minutes', 'Conversion Rate']
        
        st.dataframe(display_summary, use_container_width=True)
        
        # Time category breakdown
        st.subheader("⏰ Time Category Breakdown")
        
        category_summary = distribution_df.groupby('time_category').agg({
            'shots': 'sum',
            'goals': 'sum'
        }).reset_index()
        
        category_summary['conversion_rate'] = (category_summary['goals'] / category_summary['shots'] * 100).fillna(0)
        category_summary['percentage_of_total'] = (category_summary['shots'] / category_summary['shots'].sum() * 100)
        
        # Format for display
        display_category = category_summary.copy()
        display_category['conversion_rate'] = display_category['conversion_rate'].apply(lambda x: f"{x:.1f}%")
        display_category['percentage_of_total'] = display_category['percentage_of_total'].apply(lambda x: f"{x:.1f}%")
        
        display_category.columns = ['Time Category', 'Total Shots', 'Total Goals', 'Conversion Rate', '% of All Shots']
        
        st.dataframe(display_category, use_container_width=True)
        
        # Key insights
        st.markdown("---")
        st.subheader("🎯 Key Insights")
        
        # Find best conversion rate category
        best_conversion_cat = category_summary.loc[category_summary['conversion_rate'].idxmax(), 'time_category']
        best_conversion_rate = category_summary['conversion_rate'].max()
        
        # Find most common shooting time
        most_common_cat = category_summary.loc[category_summary['shots'].idxmax(), 'time_category']
        most_common_shots = category_summary['shots'].max()
        
        # Quick vs slow shooters
        quick_shots = category_summary[category_summary['time_category'].isin(['0-1s', '1-2s', '2-3s'])]['shots'].sum()
        slow_shots = category_summary[category_summary['time_category'].isin(['5-7.5s', '7.5-10s', '10s+'])]['shots'].sum()
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.info(f"**Best Conversion Rate:** {best_conversion_cat} ({best_conversion_rate:.1f}%)")
        
        with col2:
            st.info(f"**Most Common Time:** {most_common_cat} ({most_common_shots} shots)")
        
        with col3:
            quick_percentage = (quick_shots / (quick_shots + slow_shots) * 100) if (quick_shots + slow_shots) > 0 else 0
            st.info(f"**Quick Shooters:** {quick_percentage:.1f}% shoot within 3s")
    
    def run(self):
        """Main app execution."""
        st.set_page_config(page_title="Shot Map Analysis", layout="wide")
        
        st.title("⚽ Shot Map Analysis Tool")
        
        # Create tabs
        tab1, tab2 = st.tabs(["🎯 Shot Maps", "⏱️ Time Distribution"])
        
        # Sidebar - Common filters
        st.sidebar.header("🔧 Settings")
        
        # Season selection
        seasons = list(self.league_files.keys())
        selected_season = st.sidebar.selectbox("Select Season", seasons)

        # League selection
        leagues = list(self.league_files[selected_season].keys())
        selected_league = st.sidebar.selectbox("Select League", leagues)
        
        # Load data
        with st.spinner(f"Loading {selected_league} ({selected_season}) data..."):
            shot_data = self.load_shot_data(selected_season, selected_league)
            player_stats = self.load_player_stats(selected_season)
        
        if shot_data.empty or player_stats.empty:
            st.error("Could not load data. Please ensure preprocessed files are available.")
            st.stop()
        
        # Tab 1: Shot Maps
        with tab1:
            self.run_shot_map_tab(shot_data, player_stats, selected_league, selected_season)
        
        # Tab 2: Time Distribution
        with tab2:
            self.run_time_distribution_tab(shot_data, player_stats, selected_league, selected_season)
        
        # Sidebar social links
        with st.sidebar:
            st.markdown("---")
            st.markdown("### Connect")
            st.markdown("- 🐦 [Twitter](https://twitter.com/pranav_m28)")
            st.markdown("- 🔗 [GitHub](https://github.com/pranavm28)")
            st.markdown("- ❤️ [BuyMeACoffee](https://buymeacoffee.com/pranav_m28)")

def main():
    """Main execution function."""
    app = OptimizedShotMapApp()
    app.run()

if __name__ == "__main__":
    main()
