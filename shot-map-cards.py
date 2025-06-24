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
        """Load preprocessed shot data for multiple leagues."""
        all_data = []
        for league in leagues:
            try:
                file_path = _self.league_files[league]
                df = pd.read_parquet(file_path)
                df['league'] = league  # Add league identifier
                all_data.append(df)
            except Exception as e:
                st.error(f"Failed to load shot data for {league}: {e}")
        
        if all_data:
            return pd.concat(all_data, ignore_index=True)
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
    
    def calculate_filtered_stats(self, shot_data: pd.DataFrame, player_name: str, max_time: float = None) -> dict:
        """Calculate statistics for filtered data."""
        filtered_shots = self.filter_player_shots(shot_data, player_name, max_time)
        
        total_shots = len(filtered_shots)
        total_goals = len(filtered_shots[filtered_shots['is_goal'] == True])
        conversion_rate = (total_goals / total_shots * 100) if total_shots > 0 else 0
        
        # Calculate average time to shoot for all player shots (not just filtered)
        all_player_shots = shot_data[shot_data['player'] == player_name]
        valid_timing = all_player_shots[all_player_shots['TimeToShot'].notna()]
        avg_time_to_shoot = valid_timing['TimeToShot'].mean() if len(valid_timing) > 0 else 0
        
        return {
            'total_shots': total_shots,
            'total_goals': total_goals,
            'conversion_rate': conversion_rate,
            'avg_time_to_shoot': avg_time_to_shoot if not np.isnan(avg_time_to_shoot) else 0
        }
    
    def calculate_filtered_stats_for_all_players(self, shot_data: pd.DataFrame, player_stats: pd.DataFrame, max_time: float = None) -> pd.DataFrame:
        """Calculate filtered statistics for all players and create enhanced player stats."""
        enhanced_stats = player_stats.copy()
        
        # Initialize new columns
        enhanced_stats['filtered_shots'] = 0
        enhanced_stats['filtered_goals'] = 0
        enhanced_stats['filtered_conversion_rate'] = 0.0
        enhanced_stats['filtered_shots_per_90'] = 0.0
        enhanced_stats['filtered_goals_per_90'] = 0.0
        enhanced_stats['shots_with_timing'] = 0
        enhanced_stats['shots_with_timing_p90'] = 0.0
        
        for idx, player_row in enhanced_stats.iterrows():
            player_name = player_row['player']
            minutes_played = player_row['minutes_played']
            
            # Calculate shots with timing data
            all_player_shots = shot_data[shot_data['player'] == player_name]
            shots_with_timing = len(all_player_shots[all_player_shots['TimeToShot'].notna()])
            shots_with_timing_p90 = (shots_with_timing / minutes_played * 90) if minutes_played > 0 else 0
            
            enhanced_stats.at[idx, 'shots_with_timing'] = shots_with_timing
            enhanced_stats.at[idx, 'shots_with_timing_p90'] = shots_with_timing_p90
            
            # Calculate filtered stats if time filter is applied
            if max_time is not None:
                filtered_stats = self.calculate_filtered_stats(shot_data, player_name, max_time)
                enhanced_stats.at[idx, 'filtered_shots'] = filtered_stats['total_shots']
                enhanced_stats.at[idx, 'filtered_goals'] = filtered_stats['total_goals']
                enhanced_stats.at[idx, 'filtered_conversion_rate'] = filtered_stats['conversion_rate']
                
                # Calculate per 90 stats for filtered data
                if minutes_played > 0:
                    enhanced_stats.at[idx, 'filtered_shots_per_90'] = filtered_stats['total_shots'] / minutes_played * 90
                    enhanced_stats.at[idx, 'filtered_goals_per_90'] = filtered_stats['total_goals'] / minutes_played * 90
            else:
                # If no time filter, use original stats
                enhanced_stats.at[idx, 'filtered_shots'] = player_row['total_shots']
                enhanced_stats.at[idx, 'filtered_goals'] = player_row['total_goals']
                enhanced_stats.at[idx, 'filtered_conversion_rate'] = player_row['conversion_rate']
                enhanced_stats.at[idx, 'filtered_shots_per_90'] = player_row['shots_per_90']
                enhanced_stats.at[idx, 'filtered_goals_per_90'] = player_row['goals_per_90']
        
        return enhanced_stats
    
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
        stats = self.calculate_filtered_stats(shot_data, player_name, max_time)
        
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
        leagues_text = " | ".join(shot_data[shot_data['player'] == player_name]['league'].unique())
        
        fig_text(0.512, 0.975, f"<{player_name}>", font='Arial Rounded MT Bold', size=30,
                 ha="center", color="#FFFFFF", fontweight='bold', highlight_textprops=[{"color": '#FFFFFF'}])
        
        fig_text(0.512, 0.928,
                 f"{player_team} | {leagues_text} | {int(player_minutes)} Minutes | Shot Map Card{time_filter_text} | Made by @pranav_m28",
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
        # Include both original and filtered fields
        base_fields = ['total_shots', 'total_goals', 'conversion_rate', 'shots_per_90', 'goals_per_90', 
                      'avg_time_to_shoot', 'shots_with_timing', 'shots_with_timing_p90']
        
        filtered_fields = ['filtered_shots', 'filtered_goals', 'filtered_conversion_rate', 
                          'filtered_shots_per_90', 'filtered_goals_per_90']
        
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
                'league': True,
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
                         'League: %{customdata[1]}<br>' +
                         'Minutes: %{customdata[2]:.0f}<br>' +
                         f'{x_field.replace("_", " ").title()}: %{{x:.2f}}<br>' +
                         f'{y_field.replace("_", " ").title()}: %{{y:.2f}}<br>' +
                         '<extra></extra>'
        )
        
        return fig
    
    def render_global_controls(self):
        """Render global controls in sidebar."""
        st.sidebar.header("🌍 Global Controls")
        
        # League selection (global)
        leagues = list(self.league_files.keys())
        selected_leagues = st.sidebar.multiselect(
            "Select Leagues", 
            leagues, 
            default=leagues[:1],
            key="global_leagues",
            help="Select leagues to analyze (applies to both shot maps and scatter plots)"
        )
        
        if not selected_leagues:
            st.sidebar.warning("Please select at least one league.")
            return None, None, None, None
        
        # Load data for selected leagues
        with st.spinner(f"Loading data for {len(selected_leagues)} league(s)..."):
            shot_data = self.load_shot_data(selected_leagues)
            player_stats = self.load_player_stats()
        
        if shot_data.empty or player_stats.empty:
            st.sidebar.error("Could not load data.")
            return None, None, None, None
        
        # Filter player stats by selected leagues
        filtered_player_stats = player_stats[player_stats['league'].isin(selected_leagues)]
        
        # Time filter (global)
        st.sidebar.markdown("---")
        st.sidebar.subheader("⏱️ Global Time Filter")
        st.sidebar.info("ℹ️ Applies to both shot maps and scatter plots. Shows only shots taken after receiving a pass from a teammate.")
        
        use_time_filter = st.sidebar.checkbox("Apply Time Filter", key="global_time_filter")
        max_time = None
        if use_time_filter:
            max_time = st.sidebar.slider(
                "Maximum Time to Shoot (seconds)", 
                min_value=0.5, 
                max_value=10.0, 
                value=5.0, 
                step=0.5,
                key="global_max_time"
            )
        
        # Calculate enhanced stats with time filter
        enhanced_player_stats = self.calculate_filtered_stats_for_all_players(
            shot_data, filtered_player_stats, max_time
        )
        
        return shot_data, enhanced_player_stats, selected_leagues, max_time
    
    def render_scatter_plot_tab(self, shot_data, enhanced_player_stats, selected_leagues, max_time):
        """Render the scatter plot analysis tab."""
        st.header("📊 Interactive Scatter Plot Analysis")
        
        if max_time is not None:
            st.markdown(f"*Analysis includes time filter: shots within {max_time}s after receiving a pass*")
        else:
            st.markdown("*Analysis includes all shots (no time filter applied)*")
        
        if enhanced_player_stats.empty:
            st.error("No data available for scatter plot analysis.")
            return
        
        # Team selection
        teams = sorted(enhanced_player_stats['team'].unique())
        scatter_teams = st.multiselect(
            "Select Teams", 
            teams, 
            default=teams,
            key="scatter_teams",
            help="Filter players by team"
        )
        
        if not scatter_teams:
            st.warning("Please select at least one team.")
            return
        
        # Minimum minutes filter
        max_minutes = int(enhanced_player_stats['minutes_played'].max()) if len(enhanced_player_stats) > 0 else 1000
        scatter_min_minutes = st.slider(
            "Minimum Minutes Played", 
            min_value=0, 
            max_value=max_minutes,
            value=0, 
            step=50,
            key="scatter_min_minutes"
        )
        
        # Get available fields
        available_fields = self.get_scatter_plot_fields(enhanced_player_stats)
        
        if len(available_fields) < 2:
            st.error("Not enough numeric fields available for scatter plot.")
            return
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Suggest filtered fields if time filter is applied
            default_x = 'filtered_shots_per_90' if max_time is not None and 'filtered_shots_per_90' in available_fields else 0
            if isinstance(default_x, str):
                default_x = available_fields.index(default_x) if default_x in available_fields else 0
            x_field = st.selectbox("X-Axis", available_fields, index=default_x)
        
        with col2:
            # Suggest filtered fields if time filter is applied
            default_y = 'filtered_goals_per_90' if max_time is not None and 'filtered_goals_per_90' in available_fields else 1
            if isinstance(default_y, str):
                default_y = available_fields.index(default_y) if default_y in available_fields else 1
            if isinstance(default_y, int) and default_y >= len(available_fields):
                default_y = 1
            y_field = st.selectbox("Y-Axis", available_fields, index=default_y)
        
        if x_field == y_field:
            st.warning("Please select different fields for X and Y axes.")
            return
        
        # Create and display scatter plot
        with st.spinner("Creating scatter plot..."):
            fig = self.create_scatter_plot(
                enhanced_player_stats, 
                x_field, 
                y_field, 
                scatter_min_minutes, 
                scatter_teams
            )
        
        if fig is not None:
            st.plotly_chart(fig, use_container_width=True)
            
            # Show correlation
            filtered_stats = enhanced_player_stats[
                (enhanced_player_stats['minutes_played'] >= scatter_min_minutes) & 
                (enhanced_player_stats['team'].isin(scatter_teams))
            ]
            
            if len(filtered_stats) > 1:
                correlation = filtered_stats[x_field].corr(filtered_stats[y_field])
                st.info(f"📊 Correlation coefficient: {correlation:.3f}")
            
            # Summary stats
            st.subheader("📋 Summary Statistics")
            col1, col2 = st.columns(2)
            
            with col1:
                st.write(f"**{x_field.replace('_', ' ').title()}**")
                st.write(f"Mean: {filtered_stats[x_field].mean():.3f}")
                st.write(f"Median: {filtered_stats[x_field].median():.3f}")
                st.write(f"Std Dev: {filtered_stats[x_field].std():.3f}")
            
            with col2:
                st.write(f"**{y_field.replace('_', ' ').title()}**")
                st.write(f"Mean: {filtered_stats[y_field].mean():.3f}")
                st.write(f"Median: {filtered_stats[y_field].median():.3f}")
                st.write(f"Std Dev: {filtered_stats[y_field].std():.3f}")
            
            leagues_shown = ", ".join(selected_leagues)
            st.write(f"**Leagues:** {leagues_shown}")
            st.write(f"**Total players shown:** {len(filtered_stats)}")
            
            if max_time is not None:
                st.info(f"📊 Stats shown are filtered for shots within {max_time}s after receiving a pass")
        else:
            st.error("Could not create scatter plot with the selected parameters.")
    
    def render_shot_map_tab(self, shot_data, enhanced_player_stats, selected_leagues, max_time):
        """Render the shot map analysis tab."""
        st.header("🎯 Shot Map Analysis")
        
        if max_time is not None:
            st.markdown(f"*Shot maps show shots within {max_time}s after receiving a pass*")
        else:
            st.markdown("*Shot maps show all shots (no time filter applied)*")
        
        # Team selection
        teams = sorted(enhanced_player_stats['team'].unique())
        selected_teams = st.multiselect(
            "Select Teams", 
            teams, 
            default=teams,
            key="shotmap_teams",
            help="Filter players by team"
        )
        
        if not selected_teams:
            st.warning("Please select at least one team.")
            return
        
        # Filter by selected teams
        team_filtered_stats = enhanced_player_stats[enhanced_player_stats['team'].isin(selected_teams)]
        
        # Minimum minutes filter
        max_minutes = int(team_filtered_stats['minutes_played'].max()) if len(team_filtered_stats) > 0 else 1000
        min_minutes = st.slider(
            "Minimum Minutes Played", 
            min_value=0, 
            max_value=max_minutes,
            value=0, 
            step=50,
            key="shotmap_min_minutes"
        )
        
        # Filter players by minimum minutes
        eligible_players = team_filtered_stats[team_filtered_stats['minutes_played'] >= min_minutes]
        
        if len(eligible_players) == 0:
            st.warning(f"No players found with at least {min_minutes} minutes played.")
            return
        
        # Player selection
        players = sorted(eligible_players['player'].unique())
        selected_player = st.selectbox("Select Player", players, key="shotmap_player")
        
        # Main content
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("📊 Shot Map Visualization")
            if selected_player:
                with st.spinner("Creating shot map..."):
                    result = self.create_shot_map(shot_data, enhanced_player_stats, selected_player, max_time)
        
            if result and result[0] is not None:
                plot_data, download_data = result
                st.image(f"data:image/png;base64,{plot_data}")
            
            # Download button
                time_filter_suffix = f"_within_{max_time}s" if max_time is not None else ""
                leagues_suffix = "_".join([league.split('-')[0] for league in selected_leagues])
                filename = f"{selected_player.replace(' ', '_')}_shot_map_{leagues_suffix}{time_filter_suffix}.png"
            
                st.download_button(
                    label="📥 Download High Quality Image",
                    data=download_data,
                    file_name=filename,
                    mime="image/png",
                    help="Download the shot map in high quality (400 DPI)"
             )
            else:
                    st.error("Could not generate shot map for the selected player.")
