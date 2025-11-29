import streamlit as st
import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

def show():
    st.markdown('<div class="main-header"><h1>🏆 Gamification & Leaderboard</h1></div>', unsafe_allow_html=True)
    
    # Load gamification data
    try:
        with open('gamification.json', 'r') as f:
            all_users_data = json.load(f)
    except:
        st.warning("No gamification data yet. Start analyzing plants to earn points!")
        return
    
    if not all_users_data:
        st.info("Be the first to earn points! Analyze plants in the Dashboard.")
        return
    
    current_user = st.session_state.username
    user_data = all_users_data.get(current_user, {
        'total_points': 0,
        'total_scans': 0,
        'plants_saved': 0,
        'diseases_detected': 0,
        'perfect_detections': 0,
        'badges': [],
        'streak': 0
    })
    
    # User Profile Section
    st.markdown("## 👤 Your Profile")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h4>🏅 Total Points</h4>
            <h2>{user_data['total_points']}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <h4>🔬 Total Scans</h4>
            <h2>{user_data['total_scans']}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <h4>🦸 Plants Saved</h4>
            <h2>{user_data['plants_saved']}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <h4>🎯 Perfect Scans</h4>
            <h2>{user_data['perfect_detections']}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    # Badges Section
    st.markdown("---")
    st.markdown("## 🎖️ Your Badges")
    st.markdown('<div class="card">', unsafe_allow_html=True)
    
    if user_data['badges']:
        cols = st.columns(min(len(user_data['badges']), 5))
        for idx, badge in enumerate(user_data['badges']):
            with cols[idx % 5]:
                st.markdown(f"""
                <div style='text-align: center; padding: 1rem; background: linear-gradient(135deg, #ffd700 0%, #ffed4e 100%); 
                border-radius: 12px; margin: 0.5rem; box-shadow: 0 4px 10px rgba(255, 215, 0, 0.3);'>
                    <h2 style='margin: 0;'>{badge.split()[0]}</h2>
                    <p style='margin: 0; font-weight: bold; color: #333;'>{' '.join(badge.split()[1:])}</p>
                </div>
                """, unsafe_allow_html=True)
    else:
        st.info("No badges yet! Keep analyzing plants to unlock achievements.")
    
    # Badge Requirements
    with st.expander("🔓 How to Unlock Badges"):
        st.markdown("""
        - 🔬 **Lab Technician**: Complete 10 scans
        - 🌿 **Plant Doctor**: Complete 50 scans
        - 🏆 **Master Botanist**: Complete 100 scans
        - 🦸 **Plant Savior**: Save 5 plants by detecting critical diseases
        - 🎯 **Precision Expert**: Achieve 20 perfect detections (95%+ accuracy)
        - 🐛 **Disease Hunter**: Detect 25 different diseases
        """)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Leaderboard Section
    st.markdown("---")
    st.markdown("## 🏆 Global Leaderboard")
    
    # Create leaderboard dataframe
    leaderboard_data = []
    for username, data in all_users_data.items():
        leaderboard_data.append({
            'Username': username,
            'Points': data['total_points'],
            'Scans': data['total_scans'],
            'Plants Saved': data['plants_saved'],
            'Badges': len(data['badges']),
            'Perfect Scans': data['perfect_detections']
        })
    
    df_leaderboard = pd.DataFrame(leaderboard_data)
    df_leaderboard = df_leaderboard.sort_values('Points', ascending=False).reset_index(drop=True)
    df_leaderboard.index += 1  # Start ranking from 1
    
    # Highlight current user
    def highlight_user(row):
        if row['Username'] == current_user:
            return ['background-color: #4caf50; color: white'] * len(row)
        return [''] * len(row)
    
    st.markdown('<div class="card">', unsafe_allow_html=True)
    
    # Display top 10
    st.markdown("### 👑 Top 10 Plant Doctors")
    st.dataframe(
        df_leaderboard.head(10).style.apply(highlight_user, axis=1),
        use_container_width=True,
        hide_index=False
    )
    
    # User's rank
    user_rank = df_leaderboard[df_leaderboard['Username'] == current_user].index[0] if current_user in df_leaderboard['Username'].values else None
    if user_rank is not None:
        st.success(f"🎯 Your Rank: #{user_rank} out of {len(df_leaderboard)} users")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Visualizations
    st.markdown("---")
    st.markdown("## 📊 Community Statistics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("🏅 Top Points Leaders")
        
        fig_points = px.bar(
            df_leaderboard.head(10),
            x='Username',
            y='Points',
            title="Top 10 by Points",
            color='Points',
            color_continuous_scale='Greens'
        )
        fig_points.update_layout(showlegend=False, height=400)
        st.plotly_chart(fig_points, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("🦸 Plant Saviors")
        
        fig_saved = px.bar(
            df_leaderboard.head(10),
            x='Username',
            y='Plants Saved',
            title="Top 10 by Plants Saved",
            color='Plants Saved',
            color_continuous_scale='RdYlGn'
        )
        fig_saved.update_layout(showlegend=False, height=400)
        st.plotly_chart(fig_saved, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Distribution Charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("🎖️ Badge Distribution")
        
        badge_counts = df_leaderboard['Badges'].value_counts().sort_index()
        fig_badges = px.pie(
            values=badge_counts.values,
            names=[f"{count} badges" for count in badge_counts.index],
            title="Users by Badge Count",
            color_discrete_sequence=px.colors.sequential.YlOrRd
        )
        st.plotly_chart(fig_badges, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("🎯 Accuracy Champions")
        
        fig_perfect = px.bar(
            df_leaderboard.head(10),
            x='Username',
            y='Perfect Scans',
            title="Top 10 by Perfect Detections",
            color='Perfect Scans',
            color_continuous_scale='Blues'
        )
        fig_perfect.update_layout(showlegend=False, height=400)
        st.plotly_chart(fig_perfect, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Your Progress Chart
    st.markdown("---")
    st.markdown("## 📈 Your Progress")
    st.markdown('<div class="card">', unsafe_allow_html=True)
    
    # Create radar chart for user stats
    categories = ['Total Scans', 'Plants Saved', 'Perfect Scans', 'Diseases Detected', 'Badges']
    
    # Normalize values for radar chart
    max_scans = max([u['total_scans'] for u in all_users_data.values()]) or 1
    max_saved = max([u['plants_saved'] for u in all_users_data.values()]) or 1
    max_perfect = max([u['perfect_detections'] for u in all_users_data.values()]) or 1
    max_diseases = max([u['diseases_detected'] for u in all_users_data.values()]) or 1
    max_badges = max([len(u['badges']) for u in all_users_data.values()]) or 1
    
    user_values = [
        (user_data['total_scans'] / max_scans) * 100,
        (user_data['plants_saved'] / max_saved) * 100,
        (user_data['perfect_detections'] / max_perfect) * 100,
        (user_data['diseases_detected'] / max_diseases) * 100,
        (len(user_data['badges']) / max_badges) * 100
    ]
    
    fig_radar = go.Figure()
    fig_radar.add_trace(go.Scatterpolar(
        r=user_values,
        theta=categories,
        fill='toself',
        name=current_user,
        line_color='#4caf50'
    ))
    
    fig_radar.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 100])
        ),
        showlegend=True,
        title="Your Performance Profile (Normalized %)"
    )
    
    st.plotly_chart(fig_radar, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Achievements Progress
    st.markdown("---")
    st.markdown("## 🎯 Achievement Progress")
    st.markdown('<div class="card">', unsafe_allow_html=True)
    
    achievements = [
        {"name": "Lab Technician", "requirement": 10, "current": user_data['total_scans'], "emoji": "🔬"},
        {"name": "Plant Doctor", "requirement": 50, "current": user_data['total_scans'], "emoji": "🌿"},
        {"name": "Master Botanist", "requirement": 100, "current": user_data['total_scans'], "emoji": "🏆"},
        {"name": "Plant Savior", "requirement": 5, "current": user_data['plants_saved'], "emoji": "🦸"},
        {"name": "Precision Expert", "requirement": 20, "current": user_data['perfect_detections'], "emoji": "🎯"},
        {"name": "Disease Hunter", "requirement": 25, "current": user_data['diseases_detected'], "emoji": "🐛"}
    ]
    
    for achievement in achievements:
        progress = min(achievement['current'] / achievement['requirement'], 1.0)
        status = "✅ Unlocked!" if progress >= 1.0 else f"{achievement['current']}/{achievement['requirement']}"
        
        col1, col2 = st.columns([3, 1])
        with col1:
            st.progress(progress, text=f"{achievement['emoji']} {achievement['name']} - {status}")
        with col2:
            st.metric("", f"{int(progress * 100)}%")
    
    st.markdown('</div>', unsafe_allow_html=True)