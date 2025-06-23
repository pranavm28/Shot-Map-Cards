import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mplsoccer import VerticalPitch
import io
import base64
import gc
from highlight_text import fig_text
import matplotlib as mpl
import matplotlib.font_manager as fm
from pathlib import Path

# ✅ Updated path to match your filename exactly
font_path = Path(__file__).parent / "fonts" / "Arial Rounded Bold.ttf"
fm.fontManager.addfont(str(font_path))

# Optional: check the font's internal name
font_prop = fm.FontProperties(fname=str(font_path))
print("Registered font name:", font_prop.get_name())

# ✅ Use the correct name returned above
mpl.rcParams['font.family'] = font_prop.get_name()

# --- Function Definitions ---

@st.cache_data(show_spinner="Loading data...", max_entries=5)
def load_data_filtered(data_path: str, league: str, season_internal: str, columns=None):
    """Loads data filtered by league and season directly from the source."""
    try:
        df = pd.read_parquet(data_path, columns=columns)
        df = df[(df["league"] == league) & (df["season"] == season_internal)]
        return df
    except Exception as e:
        st.error(f"Failed to load data: {e}")
        return pd.DataFrame()

@st.cache_data(show_spinner="Loading minutes data...", max_entries=5)
def load_minutes_data():
    """Loads minutes played data from CSV."""
    try:
        df = pd.read_csv('T5_League_Mins_2025.csv')
        return df
    except Exception as e:
        st.error(f"Failed to load minutes data: {e}")
        return pd.DataFrame()

@st.cache_data
def prepare_shot_data(data):
    """Scales coordinates and prepares shot data."""
    if data.empty:
        return pd.DataFrame()
    
    data = data.copy()
    
    # Scale coordinates
    data['x'] = data['x'] * 1.2
    data['y'] = data['y'] * 0.8
    data['endX'] = data['endX'] * 1.2
    data['endY'] = data['endY'] * 0.8
    
    # Create total_seconds column for time calculations
    data['total_seconds'] = data['minute'] * 60 + data['second']
    
    return data

@st.cache_data
def calculate_time_to_shoot(data):
    """
    Calculate time between receiving the ball and taking a shot.
    
    This approach carefully identifies when a player receives the ball
    and then tracks how long until they take a shot.
    """
    # Make a copy to avoid modifying the original
    df = data.copy()
    
    # Create time to shot column
    df['TimeToShot'] = np.nan
    
    # Define shot types
    shot_types = ['MissedShots', 'SavedShot', 'ShotOnPost', 'Goal']
    
    # Process each game
    for game_id, game_df in df.groupby('gameId'):
        # Sort actions by time
        game_df = game_df.sort_values('total_seconds')
        
        # Iterate through each action
        for i in range(1, len(game_df)):
            current_row = game_df.iloc[i]
            prev_row = game_df.iloc[i-1]
            
            # Only process shot actions
            if current_row['is_shot'] == True:
                current_player = current_row['player']
                current_team = current_row['team']
                current_idx = current_row.name
                
                # Check if previous action was a pass TO this player FROM a teammate
                if (prev_row['type'] == 'Pass' and 
                    prev_row['team'] == current_team and
                    prev_row['player'] != current_player):
                    
                    # Calculate time difference
                    reception_time = prev_row['total_seconds']
                    shot_time = current_row['total_seconds']
                    time_to_shoot = shot_time - reception_time
                    
                    # Update the original dataframe
                    df.loc[current_idx, 'TimeToShot'] = time_to_shoot
    
    return df

@st.cache_data
def get_player_shot_stats(df, player_name, max_time=None):
    """Calculate shot statistics for a player, optionally filtered by time."""
    player_data = df[df['player'] == player_name].copy()
    
    if player_data.empty:
        return {}
    
    # Filter by time if specified
    if max_time is not None:
        # Only include shots with TimeToShot <= max_time (exclude NaN values when filtering)
        time_mask = (player_data['TimeToShot'] <= max_time) & (player_data['TimeToShot'].notna())
        player_data = player_data[time_mask]
    
    # Calculate stats
    all_shots = player_data[player_data.get('is_shot', False) == True]
    goals = player_data[player_data.get('is_goal', False) == True]
    
    total_shots = len(all_shots)
    total_goals = len(goals)
    conversion_rate = (total_goals / total_shots * 100) if total_shots > 0 else 0
    
    # Calculate average time to shoot (ONLY for shots with valid TimeToShot)
    all_player_shots = df[(df['player'] == player_name) & (df.get('is_shot', False) == True)]
    valid_time_shots = all_player_shots[all_player_shots['TimeToShot'].notna()]
    avg_time_to_shoot = valid_time_shots['TimeToShot'].mean() if len(valid_time_shots) > 0 else 0
    
    return {
        'total_shots': total_shots,
        'total_goals': total_goals,
        'conversion_rate': conversion_rate,
        'avg_time_to_shoot': avg_time_to_shoot if not np.isnan(avg_time_to_shoot) else 0
    }

@st.cache_data
def get_player_minutes(minutes_df, player_name):
    """Get minutes played for a player."""
    if minutes_df.empty:
        return 0
    
    player_minutes = minutes_df[minutes_df['player'] == player_name]['Mins']
    return player_minutes.iloc[0] if len(player_minutes) > 0 else 0

@st.cache_data
def create_shot_map(df, player_name, minutes_df, max_time=None):
    """Create shot map visualization for a player."""
    player_data = df[df['player'] == player_name].copy()
    
    if player_data.empty:
        return None
    
    # Get player team and minutes
    player_team = player_data['team'].iloc[0]
    player_minutes = get_player_minutes(minutes_df, player_name)
    
    # Filter by time if specified
    if max_time is not None:
        # Only include shots with TimeToShot <= max_time (exclude NaN values when filtering)
        time_mask = (player_data['TimeToShot'] <= max_time) & (player_data['TimeToShot'].notna())
        player_data = player_data[time_mask]
    
    # Separate shots and goals
    shots = player_data[(player_data.get('is_shot', False) == True) & (player_data.get('is_goal', False) != True)]
    goals = player_data[player_data.get('is_goal', False) == True]
    
    # Get player stats
    stats = get_player_shot_stats(df, player_name, max_time)
    
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
    
    # Stats circles and annotations with exact positioning from reference code
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
    
    # Title with team and minutes
    time_filter_text = f" (within {max_time}s)" if max_time is not None else ""
    fig_text(0.512, 0.975, f"<{player_name}>", font='Arial Rounded MT Bold', size=30,
             ha="center", color="#FFFFFF", fontweight='bold', highlight_textprops=[{"color": '#FFFFFF'}])
    
    fig_text(0.512, 0.928,
             f"{player_team} | {player_minutes} Mins Played | Shot Map Card{time_filter_text} | Made by @pranav_m28",
             font='Arial Rounded MT Bold', size=24,
             ha="center", color="#FFFFFF", fontweight='bold')
    
    # Save plot to buffer for display
    buffer = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buffer, format='png', facecolor='#181818', edgecolor='none', dpi=300, bbox_inches='tight')
    buffer.seek(0)
    plot_data = base64.b64encode(buffer.read()).decode()
    
    # Save high-quality version for download
    download_buffer = io.BytesIO()
    plt.savefig(download_buffer, format='png', facecolor='#181818', edgecolor='none', dpi=400, bbox_inches='tight')
    download_buffer.seek(0)
    download_data = download_buffer.getvalue()
    
    plt.close(fig)
    
    return plot_data, download_data

def main():
    st.set_page_config(page_title="Shot Map Analysis", layout="wide")
    
    # Configuration
    required_event_columns = [
        "league", "season", "gameId", "period", "minute", "second", "expandedMinute",
        "type", "outcomeType", "teamId", "team", "playerId", "player",
        "x", "y", "endX", "endY", "is_shot", "is_goal"
    ]
    
    leagues = ['ESP-La Liga', 'ENG-Premier League', 'ITA-Serie A', 'GER-Bundesliga', 'FRA-Ligue 1']
    season_mapping = {"2024/2025": 2425}
    season_display_options = list(season_mapping.keys())
    
    st.title("Shot Map Analysis Tool")
    st.sidebar.header("Filters")
    
    # Sidebar widgets
    selected_season_display = st.sidebar.selectbox("Select Season", season_display_options)
    selected_season_internal = season_mapping[selected_season_display]
    selected_league = st.sidebar.selectbox("Select League", leagues)
    
    # Data loading
    leagues_to_file = {
        'ESP-La Liga': 'La_Liga_24_25.parquet',
        'ENG-Premier League': 'Premier_League_2425.parquet',
        'ITA-Serie A': 'Serie_A_2425.parquet',
        'GER-Bundesliga': 'Bundesliga_2425.parquet',
        'FRA-Ligue 1': 'Ligue_1_2425.parquet'
    }
    
    hf_url = leagues_to_file[selected_league]
    
    # Load minutes data
    minutes_df = load_minutes_data()
    
    with st.spinner("Loading data..."):
        data = load_data_filtered(hf_url, selected_league, selected_season_internal, columns=required_event_columns)
    
    if data.empty:
        st.error("No data found for the selected filters.")
        st.stop()
    
    # Prepare data
    prepared_data = prepare_shot_data(data)
    
    # Calculate time to shoot
    with st.spinner("Calculating shot timing data..."):
        data_with_timing = calculate_time_to_shoot(prepared_data)
    
    # Team selection
    league_teams = sorted(data_with_timing['team'].unique())
    selected_teams = st.sidebar.multiselect("Select Teams", league_teams, default=league_teams)
    
    if not selected_teams:
        st.warning("Please select at least one team.")
        st.stop()
    
    # Filter by selected teams
    filtered_data = data_with_timing[data_with_timing['team'].isin(selected_teams)].copy()
    
    # Add minimum minutes filter
    st.sidebar.markdown("---")
    st.sidebar.subheader("Player Filters")
    
    # Get all players with shots to determine max minutes
    all_players_with_shots = filtered_data[filtered_data.get('is_shot', False) == True]['player'].unique()
    max_minutes_available = 0
    if len(all_players_with_shots) > 0 and not minutes_df.empty:
        player_minutes = []
        for player in all_players_with_shots:
            mins = get_player_minutes(minutes_df, player)
            if mins > 0:
                player_minutes.append(mins)
        if player_minutes:
            max_minutes_available = max(player_minutes)
    
    # Minimum minutes slider
    min_minutes = st.sidebar.slider(
        "Minimum Minutes Played", 
        min_value=0, 
        max_value=max(max_minutes_available, 1000),  # Ensure we have a reasonable max
        value=0, 
        step=50,
        help="Filter players by minimum minutes played"
    )
    
    # Filter players by minimum minutes and shots
    players_with_shots = []
    for player in all_players_with_shots:
        player_minutes = get_player_minutes(minutes_df, player)
        if player_minutes >= min_minutes:
            players_with_shots.append(player)
    
    players_with_shots = sorted(players_with_shots)
    
    if len(players_with_shots) == 0:
        st.warning(f"No players found with shot data and at least {min_minutes} minutes played for the selected teams.")
        st.stop()
    
    # Player selection
    selected_player = st.sidebar.selectbox("Select Player", players_with_shots)
    
    # Time filter
    st.sidebar.markdown("---")
    st.sidebar.subheader("Shot Timing Filter")
    use_time_filter = st.sidebar.checkbox("Filter by Time to Shoot")
    max_time = None
    if use_time_filter:
        max_time = st.sidebar.slider("Maximum Time to Shoot (seconds)", 
                                   min_value=0.5, max_value=10.0, value=5.0, step=0.5)
    
    # Create two columns for layout
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Shot Map Visualization")
        if selected_player:
            with st.spinner("Creating shot map..."):
                result = create_shot_map(filtered_data, selected_player, minutes_df, max_time)
            
            if result:
                plot_data, download_data = result
                st.image(f"data:image/png;base64,{plot_data}")
                
                # Add download button
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
        st.subheader("Player Statistics")
        if selected_player:
            stats = get_player_shot_stats(filtered_data, selected_player, max_time)
            
            # Create a nice stats display
            st.metric("Total Shots", stats['total_shots'])
            st.metric("Goals", stats['total_goals'])
            st.metric("Conversion Rate", f"{stats['conversion_rate']:.1f}%")
            st.metric("Avg. Time to Shoot (after rec. a pass)", f"{stats['avg_time_to_shoot']:.2f}s")
    
    # Summary table
    st.subheader(f"All Players Summary (Min. {min_minutes} minutes)")
    
    # Create summary for all players that meet the minutes requirement
    summary_data = []
    for player in players_with_shots:
        player_data = filtered_data[filtered_data['player'] == player]
        team = player_data['team'].iloc[0]
        
        stats_all = get_player_shot_stats(filtered_data, player, None)
        stats_filtered = get_player_shot_stats(filtered_data, player, max_time) if max_time else stats_all
        
        # Get minutes played
        player_minutes = get_player_minutes(minutes_df, player)
        
        # Calculate per 90 stats
        shots_per_90 = (stats_all['total_shots'] / player_minutes * 90) if player_minutes > 0 else 0
        goals_per_90 = (stats_all['total_goals'] / player_minutes * 90) if player_minutes > 0 else 0
        
        summary_data.append({
            'Player': player,
            'Team': team,
            'Minutes': player_minutes,
            'Total Shots': stats_all['total_shots'],
            'Total Goals': stats_all['total_goals'],
            'Overall Conv. %': stats_all['conversion_rate'],
            'Shots/90': shots_per_90,
            'Goals/90': goals_per_90,
            'Avg. Time to Shoot': stats_all['avg_time_to_shoot'],
            'Filtered Shots': stats_filtered['total_shots'] if max_time else 'N/A',
            'Filtered Goals': stats_filtered['total_goals'] if max_time else 'N/A',
            'Filtered Conv. %': stats_filtered['conversion_rate'] if max_time else 'N/A'
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_df = summary_df.sort_values('Total Shots', ascending=False)
    
    st.dataframe(summary_df.round(2), use_container_width=True)
    
    # Show filtered count
    total_players = len(all_players_with_shots)
    filtered_players = len(players_with_shots)
    st.info(f"Showing {filtered_players} out of {total_players} players with at least {min_minutes} minutes played")
    
    # Sidebar social links
    with st.sidebar:
        st.markdown("---")
        st.markdown("### Connect with me")
        st.markdown("- 🐦 [Twitter](https://twitter.com/pranav_m28)")
        st.markdown("- 🔗 [GitHub](https://github.com/pranavm28)")
        st.markdown("- ❤️ [BuyMeACoffee](https://buymeacoffee.com/pranav_m28)")

if __name__ == "__main__":
    main()
