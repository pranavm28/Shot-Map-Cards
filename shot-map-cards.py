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
import plotly.express as px
import plotly.graph_objects as go

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
            'ESP-La Liga': 'processed_esp_la_liga_shots.parquet',
            'ENG-Premier League': 'processed_eng_premier_league_shots.parquet',
            'ITA-Serie A': 'processed_ita_serie_a_shots.parquet',
            'GER-Bundesliga': 'processed_ger_bundesliga_shots.parquet',
            'FRA-Ligue 1': 'processed_fra_ligue_1_shots.parquet'
        }
    
    @st.cache_data
    def load_shot_data(_self, leagues: list) -> pd.DataFrame:
        """Load preprocessed shot data for selected leagues."""
        try:
            all_data = []
            for league in leagues:
                file_path = _self.league_files[league]
                df = pd.read_parquet(file_path)
                df['league'] = league  # Add league column if not present
                all_data.append(df)
            
            if all_data:
                combined_df = pd.concat(all_data, ignore_index=True)
                return combined_df
            else:
                return pd.DataFrame()
        except Exception as e:
            st.error(f"Failed to load shot data: {e}")
            return pd.DataFrame()
    
    @st.cache_data
    def load_player_stats(_self) -> pd.DataFrame:
        """Load preprocessed player statistics."""
        try:
            df = pd.read_parquet('processed_player_stats.parquet')
            return df
        except Exception as e:
            st.error(f"Failed to load player stats: {e}")
            return pd.DataFrame()
    
    def filter_player_shots(self, shot_data: pd.DataFrame, player_name: str, max_time: float = None) -> pd.DataFrame:
        """Filter shots for a specific player with optional time filter."""
        player_shots = shot_data[shot_data['player'] == player_name].copy()
        
        if max_time is not None:
            # FIXED: Only include shots that have valid timing data AND are within the time limit
            # This excludes penalties and other shots that weren't preceded by a teammate's pass
            time_mask = (player_shots['TimeToShot'] <= max_time) & (player_shots['TimeToShot'].notna())
            player_shots = player_shots[time_mask]
        
        return player_shots
    
    def calculate_filtered_stats(self, shot_data: pd.DataFrame, player_name: str, 
                                max_time: float = None, minutes_played: float = None) -> dict:
        """Calculate statistics for filtered data."""
        filtered_shots = self.filter_player_shots(shot_data, player_name, max_time)
        
        total_shots = len(filtered_shots)
        total_goals = len(filtered_shots[filtered_shots['is_goal'] == True])
        conversion_rate = (total_goals / total_shots * 100) if total_shots > 0 else 0
        
        # Calculate average time to shoot for all player shots (not just filtered)
        all_player_shots = shot_data[shot_data['player'] == player_name]
        valid_timing = all_player_shots[all_player_shots['TimeToShot'].notna()]
        avg_time_to_shoot = valid_timing['TimeToShot'].mean() if len(valid_timing) > 0 else 0
        
        # Calculate shots with timing data
        shots_with_timing = len(valid_timing)
        
        # Calculate per 90 stats if minutes_played is provided
        per_90_stats = {}
        if minutes_played and minutes_played > 0:
            per_90_stats = {
                'shots_per_90': (total_shots / minutes_played) * 90,
                'goals_per_90': (total_goals / minutes_played) * 90,
                'shots_with_timing_per_90': (shots_with_timing / minutes_played) * 90
            }
        
        return {
            'total_shots': total_shots,
            'total_goals': total_goals,
            'conversion_rate': conversion_rate,
            'avg_time_to_shoot': avg_time_to_shoot if not np.isnan(avg_time_to_shoot) else 0,
            'shots_with_timing': shots_with_timing,
            **per_90_stats
        }
    
    def calculate_filtered_player_stats(self, shot_data: pd.DataFrame, player_stats: pd.DataFrame,
                                      selected_leagues: list, max_time: float = None) -> pd.DataFrame:
        """Calculate filtered statistics for all players in selected leagues."""
        # Filter player stats by selected leagues
        filtered_player_stats = player_stats[player_stats['league'].isin(selected_leagues)].copy()
        
        # If no time filter, return original stats with shots_with_timing_per_90 added
        if max_time is None:
            # Add shots_with_timing_per_90 to existing stats
            shots_with_timing_per_90 = []
            for _, player_row in filtered_player_stats.iterrows():
                player_name = player_row['player']
                player_shots = shot_data[shot_data['player'] == player_name]
                valid_timing = player_shots[player_shots['TimeToShot'].notna()]
                shots_with_timing = len(valid_timing)
                minutes = player_row['minutes_played']
                shots_with_timing_p90 = (shots_with_timing / minutes * 90) if minutes > 0 else 0
                shots_with_timing_per_90.append(shots_with_timing_p90)
            
            filtered_player_stats['shots_with_timing_per_90'] = shots_with_timing_per_90
            return filtered_player_stats
        
        # Calculate filtered stats for each player
        filtered_stats_list = []
        for _, player_row in filtered_player_stats.iterrows():
            player_name = player_row['player']
            minutes = player_row['minutes_played']
            
            # Calculate filtered stats
            filtered_stats = self.calculate_filtered_stats(shot_data, player_name, max_time, minutes)
            
            # Create new row with filtered stats
            new_row = {
                'player': player_name,
                'team': player_row['team'],
                'league': player_row['league'],
                'minutes_played': minutes,
                'total_shots': filtered_stats['total_shots'],
                'total_goals': filtered_stats['total_goals'],
                'conversion_rate': filtered_stats['conversion_rate'],
                'shots_per_90': filtered_stats.get('shots_per_90', 0),
                'goals_per_90': filtered_stats.get('goals_per_90', 0),
                'avg_time_to_shoot': filtered_stats['avg_time_to_shoot'],
                'shots_with_timing': filtered_stats['shots_with_timing'],
                'shots_with_timing_per_90': filtered_stats.get('shots_with_timing_per_90', 0)
            }
            filtered_stats_list.append(new_row)
        
        return pd.DataFrame(filtered_stats_list)
    
    def create_shot_map(self, shot_data: pd.DataFrame, player_stats: pd.DataFrame, 
                       player_name: str, max_time: float = None) -> tuple:
        """Create shot map visualization."""
        # Get filtered shots
        filtered_shots = self.filter_player_shots(shot_data, player_name, max_time)
        
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
        stats = self.calculate_filtered_stats(shot_data, player_name, max_time, player_minutes)
        
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
        pitch.lines(85, 0, 85, 80, ls='-', lw=1.5, color='#FFFFFF', ax=ax, zorder=1, alpha=0.8)
        
        # Invert y-axis
        plt.gca().invert_yaxis()
        
        # Title - Updated to reflect that only shots after passes are shown when filtered
        time_filter_text = f" (within {max_time}s)" if max_time is not None else ""
        fig_text(0.512, 0.975, f"<{player_name}>", font='Arial Rounded MT Bold', size=30,
                 ha="center", color="#FFFFFF", fontweight='bold', highlight_textprops=[{"color": '#FFFFFF'}])
        
        fig_text(0.512, 0.928,
                 f"{player_team} | {int(player_minutes)} Minutes | Shot Map Card{time_filter_text} | Made by @pranav_m28",
                 font='Arial Rounded MT Bold', size=24,
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
    
    def get_scatter_plot_fields(self, player_stats: pd.DataFrame) -> list:
        """Get available numeric fields for scatter plot."""
        # Exclude non-numeric and identifier fields
        exclude_fields = ['player', 'team', 'league', 'minutes_played']
        numeric_fields = []
        
        for col in player_stats.columns:
            if col not in exclude_fields and pd.api.types.is_numeric_dtype(player_stats[col]):
                numeric_fields.append(col)
        
        return sorted(numeric_fields)
    
    def create_scatter_plot(self, player_stats: pd.DataFrame, x_field: str, y_field: str, 
                           min_minutes: int = 0, selected_teams: list = None) -> go.Figure:
        """Create interactive scatter plot using Plotly."""
        # Filter data
        filtered_data = player_stats[player_stats['minutes_played'] >= min_minutes].copy()
        
        if selected_teams:
            filtered_data = filtered_data[filtered_data['team'].isin(selected_teams)]
        
        if filtered_data.empty:
            return None
        
        # Create scatter plot
        fig = px.scatter(
            filtered_data,
            x=x_field,
            y=y_field,
            color='team',
            hover_name='player',
            hover_data={
                'team': True,
                'minutes_played': ':.0f',
                x_field: ':.2f',
                y_field: ':.2f'
            },
            title=f"{y_field.replace('_', ' ').title()} vs {x_field.replace('_', ' ').title()}",
            labels={
                x_field: x_field.replace('_', ' ').title(),
                y_field: y_field.replace('_', ' ').title()
            }
        )
        
        # Update layout
        fig.update_layout(
            template='plotly_dark',
            width=800,
            height=600,
            title_font_size=16,
            showlegend=True,
            legend=dict(
                orientation="v",
                yanchor="top",
                y=1,
                xanchor="left",
                x=1.02
            )
        )
        
        # Update traces
        fig.update_traces(
            marker=dict(size=8, opacity=0.7),
            hovertemplate='<b>%{hovertext}</b><br>' +
                         'Team: %{customdata[0]}<br>' +
                         'Minutes: %{customdata[1]:.0f}<br>' +
                         f'{x_field.replace("_", " ").title()}: %{{x:.2f}}<br>' +
                         f'{y_field.replace("_", " ").title()}: %{{y:.2f}}<br>' +
                         '<extra></extra>'
        )
        
        return fig
    
    def render_scatter_plot_tab(self, selected_leagues: list, max_time: float = None):
        """Render the scatter plot analysis tab."""
        st.header("📊 Interactive Scatter Plot Analysis")
        
        # Show time filter context
        if max_time is not None:
            st.markdown(f"*Showing filtered statistics for shots within {max_time} seconds after receiving a pass*")
        else:
            st.markdown("*Showing overall season statistics*")
        
        # Load data
        shot_data = self.load_shot_data(selected_leagues)
        player_stats = self.load_player_stats()
        
        if shot_data.empty or player_stats.empty:
            st.error("Could not load data.")
            return
        
        # Calculate filtered player stats based on time filter
        filtered_player_stats = self.calculate_filtered_player_stats(
            shot_data, player_stats, selected_leagues, max_time
        )
        
        if filtered_player_stats.empty:
            st.error("No data available for selected leagues.")
            return
        
        # Team selection for scatter plot
        teams = sorted(filtered_player_stats['team'].unique())
        scatter_teams = st.sidebar.multiselect(
            "Select Teams for Scatter Plot", 
            teams, 
            default=teams, 
            key="scatter_teams"
        )
        
        if not scatter_teams:
            st.warning("Please select at least one team.")
            return
        
        # Minimum minutes for scatter plot
        max_minutes = int(filtered_player_stats['minutes_played'].max()) if len(filtered_player_stats) > 0 else 1000
        scatter_min_minutes = st.sidebar.slider(
            "Minimum Minutes for Scatter Plot", 
            min_value=0, 
            max_value=max_minutes,
            value=0, 
            step=50,
            key="scatter_min_minutes"
        )
        
        # Get available fields
        available_fields = self.get_scatter_plot_fields(filtered_player_stats)
        
        if len(available_fields) < 2:
            st.error("Not enough numeric fields available for scatter plot.")
            return
        
        col1, col2 = st.columns(2)
        
        with col1:
            x_field = st.selectbox("X-Axis", available_fields, index=0)
        
        with col2:
            # Default Y-axis to second field if available
            default_y = 1 if len(available_fields) > 1 else 0
            y_field = st.selectbox("Y-Axis", available_fields, index=default_y)
        
        if x_field == y_field:
            st.warning("Please select different fields for X and Y axes.")
            return
        
        # Create and display scatter plot
        with st.spinner("Creating scatter plot..."):
            fig = self.create_scatter_plot(
                filtered_player_stats, 
                x_field, 
                y_field, 
                scatter_min_minutes, 
                scatter_teams
            )
        
        if fig is not None:
            st.plotly_chart(fig, use_container_width=True)
            
            # Show correlation
            final_filtered_stats = filtered_player_stats[
                (filtered_player_stats['minutes_played'] >= scatter_min_minutes) & 
                (filtered_player_stats['team'].isin(scatter_teams))
            ]
            
            if len(final_filtered_stats) > 1:
                correlation = final_filtered_stats[x_field].corr(final_filtered_stats[y_field])
                st.info(f"📊 Correlation coefficient: {correlation:.3f}")
            
            # Summary stats
            st.subheader("📋 Summary Statistics")
            col1, col2 = st.columns(2)
            
            with col1:
                st.write(f"**{x_field.replace('_', ' ').title()}**")
                st.write(f"Mean: {final_filtered_stats[x_field].mean():.3f}")
                st.write(f"Median: {final_filtered_stats[x_field].median():.3f}")
                st.write(f"Std Dev: {final_filtered_stats[x_field].std():.3f}")
            
            with col2:
                st.write(f"**{y_field.replace('_', ' ').title()}**")
                st.write(f"Mean: {final_filtered_stats[y_field].mean():.3f}")
                st.write(f"Median: {final_filtered_stats[y_field].median():.3f}")
                st.write(f"Std Dev: {final_filtered_stats[y_field].std():.3f}")
            
            st.write(f"**Total players shown:** {len(final_filtered_stats)}")
        else:
            st.error("Could not create scatter plot with the selected parameters.")
    
    def render_global_controls(self):
        """Render global controls in sidebar."""
        st.sidebar.header("🌍 Global Settings")
        
        # League selection (multiple)
        leagues = list(self.league_files.keys())
        selected_leagues = st.sidebar.multiselect(
            "Select Leagues", 
            leagues, 
            default=leagues[:1],
            key="global_leagues"
        )
        
        if not selected_leagues:
            st.sidebar.warning("Please select at least one league.")
            return None, None
        
        # Time filter
        st.sidebar.markdown("---")
        st.sidebar.subheader("⏱️ Global Time Filter")
        st.sidebar.info("ℹ️ Time filter applies to both shot maps and scatter plots")
        use_time_filter = st.sidebar.checkbox("Filter by Time to Shoot")
        max_time = None
        if use_time_filter:
            max_time = st.sidebar.slider("Maximum Time to Shoot (seconds)", 
                                       min_value=0.5, max_value=10.0, value=5.0, step=0.5)
        
        return selected_leagues, max_time
    
    def run(self):
        """Main app execution."""
        st.set_page_config(page_title="Shot Map Analysis", layout="wide")
        
        st.title("⚽ Shot Map Analysis Tool")
        
        # Global controls
        selected_leagues, max_time = self.render_global_controls()
        
        if not selected_leagues:
            st.error("Please select at least one league from the sidebar.")
            return
        
        # Create tabs
        tab1, tab2 = st.tabs(["🎯 Shot Map Analysis", "📊 Scatter Plot Analysis"])
        
        with tab1:
            self.render_shot_map_tab(selected_leagues, max_time)
        
        with tab2:
            self.render_scatter_plot_tab(selected_leagues, max_time)
    
    def render_shot_map_tab(self, selected_leagues: list, max_time: float = None):
        """Render the shot map analysis tab."""
        st.header("🎯 Shot Map Analysis")
        
        # Show selected leagues and time filter
        leagues_text = ", ".join(selected_leagues)
        time_text = f" (filtered to {max_time}s)" if max_time else ""
        st.markdown(f"*Analyzing: {leagues_text}{time_text}*")
        
        # Load data
        with st.spinner(f"Loading data for {len(selected_leagues)} league(s)..."):
            shot_data = self.load_shot_data(selected_leagues)
            player_stats = self.load_player_stats()
        
        if shot_data.empty or player_stats.empty:
            st.error("Could not load data. Please ensure preprocessed files are available.")
            st.stop()
        
        # Filter player stats by selected leagues
        league_player_stats = player_stats[player_stats['league'].isin(selected_leagues)]
        
        # Team selection
        teams = sorted(league_player_stats['team'].unique())
        selected_teams = st.sidebar.multiselect(
            "Select Teams", 
            teams, 
            default=teams,
            key="shot_map_teams"
        )
        
        if not selected_teams:
            st.warning("Please select at least one team.")
            st.stop()
        
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
            step=50
        )
        
        # Filter players by minimum minutes
        eligible_players = filtered_player_stats[filtered_player_stats['minutes_played'] >= min_minutes]
        
        if len(eligible_players) == 0:
            st.warning(f"No players found with at least {min_minutes} minutes played.")
            st.stop()
        
        # Player selection
        players = sorted(eligible_players['player'].unique())
        selected_player = st.sidebar.selectbox("Select Player", players)
        
        # Main content
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("📊 Shot Map Visualization")
            if selected_player:
                with st.spinner("Creating shot map..."):
                    result = self.create_shot_map(shot_data, league_player_stats, selected_player, max_time)
                
                if result and result[0] is not None:
                    plot_data, download_data = result
                    st.image(f"data:image/png;base64,{plot_data}")
                    
                    # Download button
                    time_filter_suffix = f"_within_{max_time}s" if max_time is not None else ""
                    filename = f"{selected_player.replace(' ', '_')}_shot_map{time_filter_
