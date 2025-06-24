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
    def load_shot_data(_self, league: str) -> pd.DataFrame:
        """Load preprocessed shot data for a league."""
        try:
            file_path = _self.league_files[league]
            df = pd.read_parquet(file_path)
            return df
        except Exception as e:
            st.error(f"Failed to load shot data for {league}: {e}")
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
        fig_text(0.512, 0.975, f"<{player_name}>", font='Arial Rounded MT Bold', size=30,
                 ha="center", color="#FFFFFF", fontweight='bold', highlight_textprops=[{"color": '#FFFFFF'}])
        
        fig_text(0.512, 0.928,
                 f"{player_team} | {int(player_minutes)} Mins Played | Shot Map Card{time_filter_text} | Made by @pranav_m28",
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
    
    def render_scatter_plot_tab(self):
        """Render the scatter plot analysis tab."""
        st.header("📊 Interactive Scatter Plot Analysis")
        st.markdown("*Explore relationships between different player statistics*")
        
        # Load player stats
        player_stats = self.load_player_stats()
        if player_stats.empty:
            st.error("Could not load player statistics.")
            return
        
        # Sidebar controls for scatter plot
        st.sidebar.markdown("---")
        st.sidebar.subheader("📈 Scatter Plot Controls")
        
        # League selection for scatter plot
        leagues = list(self.league_files.keys())
        scatter_league = st.sidebar.selectbox("Select League", leagues, key="scatter_league")
        
        # Filter by league
        league_stats = player_stats[player_stats['league'] == scatter_league]
        
        # Team selection for scatter plot
        teams = sorted(league_stats['team'].unique())
        scatter_teams = st.sidebar.multiselect("Select Teams", teams, default=teams, key="scatter_teams")
        
        if not scatter_teams:
            st.warning("Please select at least one team.")
            return
        
        # Minimum minutes for scatter plot
        max_minutes = int(league_stats['minutes_played'].max()) if len(league_stats) > 0 else 1000
        scatter_min_minutes = st.sidebar.slider(
            "Minimum Minutes", 
            min_value=0, 
            max_value=max_minutes,
            value=0, 
            step=50,
            key="scatter_min_minutes"
        )
        
        # Get available fields
        available_fields = self.get_scatter_plot_fields(league_stats)
        
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
                league_stats, 
                x_field, 
                y_field, 
                scatter_min_minutes, 
                scatter_teams
            )
        
        if fig is not None:
            st.plotly_chart(fig, use_container_width=True)
            
            # Show correlation
            filtered_stats = league_stats[
                (league_stats['minutes_played'] >= scatter_min_minutes) & 
                (league_stats['team'].isin(scatter_teams))
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
            
            st.write(f"**Total players shown:** {len(filtered_stats)}")
        else:
            st.error("Could not create scatter plot with the selected parameters.")
    
    def run(self):
        """Main app execution."""
        st.set_page_config(page_title="Shot Map Analysis", layout="wide")
        
        st.title("⚽ Shot Map Analysis Tool")
        
        # Create tabs
        tab1, tab2 = st.tabs(["🎯 Shot Map Analysis", "📊 Scatter Plot Analysis"])
        
        with tab1:
            self.render_shot_map_tab()
        
        with tab2:
            self.render_scatter_plot_tab()
    
    def render_shot_map_tab(self):
        """Render the original shot map analysis tab."""
        # Sidebar
        st.sidebar.header("🎯 Filters")
        
        # League selection
        leagues = list(self.league_files.keys())
        selected_league = st.sidebar.selectbox("Select League", leagues)
        
        # Load data
        with st.spinner(f"Loading {selected_league} data..."):
            shot_data = self.load_shot_data(selected_league)
            player_stats = self.load_player_stats()
        
        if shot_data.empty or player_stats.empty:
            st.error("Could not load data. Please ensure preprocessed files are available.")
            st.stop()
        
        # Filter player stats by league
        league_player_stats = player_stats[player_stats['league'] == selected_league]
        
        # Team selection
        teams = sorted(league_player_stats['team'].unique())
        selected_teams = st.sidebar.multiselect("Select Teams", teams, default=teams)
        
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
        
        # Time filter
        st.sidebar.markdown("---")
        st.sidebar.subheader("⏱️ Shot Timing Filter")
        st.sidebar.info("ℹ️ Time filter only shows shots taken after receiving a pass from a teammate (excludes penalties, free kicks, etc.)")
        use_time_filter = st.sidebar.checkbox("Filter by Time to Shoot")
        max_time = None
        if use_time_filter:
            max_time = st.sidebar.slider("Maximum Time to Shoot (seconds)", 
                                       min_value=0.5, max_value=10.0, value=5.0, step=0.5)
        
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
                    filename = f"{selected_player.replace(' ', '_')}_shot_map{time_filter_suffix}.png"
                    
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
                filtered_stats = self.calculate_filtered_stats(shot_data, selected_player, max_time)
                
                # Display metrics with context
                if max_time is not None:
                    st.info(f"📊 Filtered stats show only shots within {max_time}s after receiving a pass")
                
                st.metric("Total Shots", f"{filtered_stats['total_shots']}")
                st.metric("Goals", f"{filtered_stats['total_goals']}")
                st.metric("Conversion Rate", f"{filtered_stats['conversion_rate']:.1f}%")
                st.metric("Minutes Played", f"{int(player_info['minutes_played'])}")
                st.metric("Shots/90", f"{player_info['shots_per_90']:.2f}")
                st.metric("Goals/90", f"{player_info['goals_per_90']:.2f}")
                st.metric("Avg. Time to Shoot (after rec. a pass)", f"{filtered_stats['avg_time_to_shoot']:.2f}s")
                
                # Show context about timing data
                if max_time is not None:
                    all_player_shots = shot_data[shot_data['player'] == selected_player]
                    total_shots = len(all_player_shots)
                    shots_with_timing = len(all_player_shots[all_player_shots['TimeToShot'].notna()])
                    st.markdown(f"**Context:** {shots_with_timing}/{total_shots} total shots have timing data")
        
        # Summary table
        st.subheader(f"📋 All Players Summary (Min. {min_minutes} minutes)")
        
        # Create enhanced summary table
        summary_data = []
        for _, player_row in eligible_players.iterrows():
            player_name = player_row['player']
            
            # Calculate filtered stats if time filter is applied
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
                'Filtered Conv. %': filtered_conv
            })
        
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
