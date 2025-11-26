import streamlit as st
import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import base64
from io import BytesIO
from PIL import Image
import numpy as np

def show():
    st.markdown('<div class="main-header"><h1>📜 Analysis History & Advanced Analytics</h1></div>', unsafe_allow_html=True)
    
    # Load history
    def load_history():
        with open('historique.json', 'r') as f:
            history = json.load(f)
        return [h for h in history if h['username'] == st.session_state.username]
    
    user_history = load_history()
    
    if not user_history:
        st.info("📭 No analysis history yet. Upload your first plant image in the Dashboard!")
        return
    
    # Statistics Overview
    st.markdown("## 📊 Overview Dashboard")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h4>Total Analyses</h4>
            <h2>{len(user_history)}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    predictions = [h['prediction'] for h in user_history]
    unique_plants = len(set(predictions))
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <h4>Plant Types</h4>
            <h2>{unique_plants}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    anomalies = sum(1 for p in predictions if 'healthy' not in p.lower())
    healthy = len(predictions) - anomalies
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <h4>Diseases Found</h4>
            <h2>{anomalies}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <h4>Healthy Plants</h4>
            <h2>{healthy}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    avg_accuracy = sum(h['accuracy'] for h in user_history) / len(user_history)
    
    with col5:
        st.markdown(f"""
        <div class="metric-card">
            <h4>Avg Accuracy</h4>
            <h2>{avg_accuracy:.1f}%</h2>
        </div>
        """, unsafe_allow_html=True)
    
    # Advanced Visualizations
    st.markdown("---")
    st.markdown("## 📈 Advanced Analytics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("🥧 Disease Distribution")
        
        pred_counts = pd.Series(predictions).value_counts()
        fig_pie = px.pie(
            values=pred_counts.values,
            names=pred_counts.index,
            title="Plant Condition Distribution",
            color_discrete_sequence=px.colors.sequential.Greens,
            hole=0.3
        )
        fig_pie.update_traces(textposition='inside', textinfo='percent+label', textfont_size=11)
        st.plotly_chart(fig_pie, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("🔍 Health vs Disease")
        
        health_data = pd.DataFrame({
            'Status': ['Healthy 🌿', 'Disease Detected 🦠'],
            'Count': [healthy, anomalies]
        })
        
        fig_health = px.bar(
            health_data,
            x='Status',
            y='Count',
            title="Plant Health Summary",
            color='Status',
            color_discrete_map={'Healthy 🌿': '#4caf50', 'Disease Detected 🦠': '#ff5722'}
        )
        fig_health.update_layout(showlegend=False)
        st.plotly_chart(fig_health, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Timeline and Heatmap
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("📅 Analysis Timeline & Accuracy Trends")
    
    df_history = pd.DataFrame(user_history)
    df_history['date'] = pd.to_datetime(df_history['date'])
    df_history_sorted = df_history.sort_values('date')
    
    # Create timeline with accuracy
    fig_timeline = go.Figure()
    
    fig_timeline.add_trace(go.Scatter(
        x=df_history_sorted['date'],
        y=df_history_sorted['accuracy'],
        mode='lines+markers',
        name='Accuracy',
        line=dict(color='#4caf50', width=3),
        marker=dict(size=10, color=df_history_sorted['accuracy'], 
                   colorscale='Greens', showscale=True,
                   colorbar=dict(title="Accuracy %"))
    ))
    
    fig_timeline.update_layout(
        title="Prediction Accuracy Over Time",
        xaxis_title="Date",
        yaxis_title="Accuracy (%)",
        hovermode='x unified',
        height=400
    )
    
    st.plotly_chart(fig_timeline, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Heatmap of predictions by date
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("🗓️ Activity Heatmap")
        
        df_history['date_only'] = df_history['date'].dt.date
        df_history['hour'] = df_history['date'].dt.hour
        
        activity_pivot = df_history.groupby(['date_only', 'hour']).size().reset_index(name='count')
        
        if len(activity_pivot) > 0:
            fig_heatmap = px.density_heatmap(
                activity_pivot,
                x='hour',
                y='date_only',
                z='count',
                title="Analysis Activity by Time",
                color_continuous_scale='Greens',
                labels={'hour': 'Hour of Day', 'date_only': 'Date', 'count': 'Analyses'}
            )
            st.plotly_chart(fig_heatmap, use_container_width=True)
        else:
            st.info("Not enough data for heatmap yet")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("🏆 Model Performance")
        
        model_counts = pd.Series([h['best_model'] for h in user_history]).value_counts()
        
        fig_models = go.Figure(data=[
            go.Bar(
                x=model_counts.index,
                y=model_counts.values,
                marker_color=['#66bb6a', '#81c784'],
                text=model_counts.values,
                textposition='auto',
            )
        ])
        
        fig_models.update_layout(
            title="Best Model Selection Frequency",
            xaxis_title="Model",
            yaxis_title="Times Selected",
            showlegend=False
        )
        
        st.plotly_chart(fig_models, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Score Gauges
    st.markdown("---")
    st.markdown("## 🎯 Performance Scores")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        
        fig_gauge1 = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=avg_accuracy,
            title={'text': "Average Accuracy"},
            delta={'reference': 90},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "#4caf50"},
                'steps': [
                    {'range': [0, 60], 'color': "#ffcdd2"},
                    {'range': [60, 80], 'color': "#fff9c4"},
                    {'range': [80, 100], 'color': "#c8e6c9"}
                ],
                'threshold': {
                    'line': {'color': "gold", 'width': 4},
                    'thickness': 0.75,
                    'value': 95
                }
            }
        ))
        fig_gauge1.update_layout(height=300)
        st.plotly_chart(fig_gauge1, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        
        detection_rate = (anomalies / len(user_history)) * 100
        
        fig_gauge2 = go.Figure(go.Indicator(
            mode="gauge+number",
            value=detection_rate,
            title={'text': "Disease Detection Rate"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "#ff9800"},
                'steps': [
                    {'range': [0, 30], 'color': "#e8f5e9"},
                    {'range': [30, 70], 'color': "#fff9c4"},
                    {'range': [70, 100], 'color': "#ffccbc"}
                ]
            }
        ))
        fig_gauge2.update_layout(height=300)
        st.plotly_chart(fig_gauge2, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        
        perfect_scans = sum(1 for h in user_history if h['accuracy'] >= 95)
        perfect_rate = (perfect_scans / len(user_history)) * 100
        
        fig_gauge3 = go.Figure(go.Indicator(
            mode="gauge+number",
            value=perfect_rate,
            title={'text': "Perfect Scan Rate"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "#2196f3"},
                'steps': [
                    {'range': [0, 40], 'color': "#e3f2fd"},
                    {'range': [40, 70], 'color': "#bbdefb"},
                    {'range': [70, 100], 'color': "#90caf9"}
                ]
            }
        ))
        fig_gauge3.update_layout(height=300)
        st.plotly_chart(fig_gauge3, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Eco Metrics
    st.markdown("---")
    st.markdown("## ♻️ Eco Impact Metrics")
    st.markdown('<div class="card">', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    # Calculate eco metrics
    plants_saved = sum(1 for h in user_history if 'CRITICAL' in h.get('advice', {}).get('urgency', '').upper() 
                      or 'URGENT' in h.get('advice', {}).get('urgency', '').upper())
    
    pesticide_saved = plants_saved * 0.5  # kg estimate
    water_saved = healthy * 20  # liters estimate
    co2_offset = len(user_history) * 0.1  # kg CO2 estimate
    
    with col1:
        st.metric("🌱 Plants Saved", plants_saved, help="Critical diseases detected early")
    
    with col2:
        st.metric("💧 Water Saved", f"{water_saved}L", help="Efficient care recommendations")
    
    with col3:
        st.metric("🧪 Pesticide Reduced", f"{pesticide_saved:.1f}kg", help="Targeted treatment only")
    
    with col4:
        st.metric("🌍 CO₂ Offset", f"{co2_offset:.1f}kg", help="Sustainable gardening impact")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Detailed History Table
    st.markdown("---")
    st.markdown("## 📋 Detailed Analysis History")
    st.markdown('<div class="card">', unsafe_allow_html=True)
    
    # Filter options
    col1, col2, col3 = st.columns(3)
    with col1:
        filter_prediction = st.selectbox("Filter by Prediction", ["All"] + list(set(predictions)))
    with col2:
        sort_by = st.selectbox("Sort by", ["Date (Newest)", "Date (Oldest)", "Accuracy (High)", "Accuracy (Low)"])
    with col3:
        show_images = st.checkbox("Show Images", value=True)
    
    # Apply filters and sorting
    filtered_history = user_history.copy()
    if filter_prediction != "All":
        filtered_history = [h for h in filtered_history if h['prediction'] == filter_prediction]
    
    if "Newest" in sort_by:
        filtered_history = sorted(filtered_history, key=lambda x: x['date'], reverse=True)
    elif "Oldest" in sort_by:
        filtered_history = sorted(filtered_history, key=lambda x: x['date'])
    elif "High" in sort_by:
        filtered_history = sorted(filtered_history, key=lambda x: x['accuracy'], reverse=True)
    else:
        filtered_history = sorted(filtered_history, key=lambda x: x['accuracy'])
    
    # Display entries
    for idx, entry in enumerate(filtered_history):
        with st.expander(f"📅 {entry['date']} - {entry['prediction']} ({entry['accuracy']:.2f}%) - +{entry.get('points_earned', 0)} pts"):
            
            if show_images:
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    # Decode and display image
                    img_data = base64.b64decode(entry['image'])
                    img = Image.open(BytesIO(img_data))
                    st.image(img, caption="Analyzed Image", use_container_width=True)
            else:
                col1, col2 = st.columns([0.01, 1])
            
            with col2:
                st.markdown(f"**🎯 Prediction:** {entry['prediction']}")
                st.markdown(f"**✅ Confidence:** {entry['accuracy']:.2f}%")
                st.markdown(f"**🏆 Best Model:** {entry['best_model']}")
                st.markdown(f"**📅 Date:** {entry['date']}")
                st.markdown(f"**🎮 Points Earned:** +{entry.get('points_earned', 0)}")
                
                if entry.get('context'):
                    st.markdown(f"**📝 User Context:** {entry['context']}")
                
                st.markdown("**📊 Model Comparison:**")
                col_a, col_b = st.columns(2)
                with col_a:
                    st.metric("Model 1", f"{entry['model1_accuracy']:.2f}%")
                with col_b:
                    st.metric("Model 2", f"{entry['model2_accuracy']:.2f}%")
                
                st.markdown("**🏆 Top Predictions:**")
                for pred in entry['top_predictions']:
                    st.progress(
                        pred['confidence'] / 100,
                        text=f"{pred['class']}: {pred['confidence']:.2f}%"
                    )
                
                # Show advice if available
                if 'advice' in entry:
                    with st.expander("💡 View Personalized Advice"):
                        advice = entry['advice']
                        
                        col_adv1, col_adv2 = st.columns(2)
                        with col_adv1:
                            st.markdown("**🔍 Symptoms:**")
                            for symptom in advice.get('symptoms', [])[:2]:
                                st.markdown(f"• {symptom}")
                        
                        with col_adv2:
                            st.markdown("**💊 Treatment:**")
                            for treatment in advice.get('treatment', [])[:2]:
                                st.markdown(f"• {treatment}")
                        
                        st.info(f"⏰ Urgency: {advice.get('urgency', 'Moderate')}")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Export option
    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("📥 Export History as JSON", use_container_width=True):
            json_str = json.dumps(user_history, indent=2)
            st.download_button(
                label="Download JSON File",
                data=json_str,
                file_name=f"plantdoctor_history_{st.session_state.username}.json",
                mime="application/json",
                use_container_width=True
            )
    
    with col2:
        if st.button("📊 Export as CSV", use_container_width=True):
            # Create simplified CSV
            csv_data = []
            for h in user_history:
                csv_data.append({
                    'Date': h['date'],
                    'Prediction': h['prediction'],
                    'Accuracy': h['accuracy'],
                    'Best Model': h['best_model'],
                    'Points': h.get('points_earned', 0)
                })
            
            df_csv = pd.DataFrame(csv_data)
            csv_str = df_csv.to_csv(index=False)
            
            st.download_button(
                label="Download CSV File",
                data=csv_str,
                file_name=f"plantdoctor_history_{st.session_state.username}.csv",
                mime="text/csv",
                use_container_width=True
            )