import streamlit as st
import json
import os
from datetime import datetime
from PIL import Image
import numpy as np
import tensorflow as tf
import base64
from io import BytesIO
from rag_system import rag_system

def calculate_points(accuracy, disease_severity):
    """Calculate points based on early detection and accuracy"""
    base_points = int(accuracy)
    
    # Bonus for high accuracy
    if accuracy >= 95:
        base_points += 50
    elif accuracy >= 90:
        base_points += 30
    elif accuracy >= 85:
        base_points += 20
    
    # Urgency bonus (detecting critical diseases early)
    if 'CRITICAL' in disease_severity.upper():
        base_points += 100
    elif 'URGENT' in disease_severity.upper():
        base_points += 75
    
    return base_points

def update_gamification(username, prediction, accuracy, advice):
    """Update user's gamification stats"""
    with open('gamification.json', 'r') as f:
        data = json.load(f)
    
    if username not in data:
        data[username] = {
            'total_points': 0,
            'total_scans': 0,
            'plants_saved': 0,
            'diseases_detected': 0,
            'perfect_detections': 0,
            'badges': [],
            'streak': 0,
            'last_scan': None
        }
    
    user_data = data[username]
    user_data['total_scans'] += 1
    
    # Calculate points
    points = calculate_points(accuracy, advice.get('urgency', 'Moderate'))
    user_data['total_points'] += points
    
    # Update stats
    if 'healthy' not in prediction.lower():
        user_data['diseases_detected'] += 1
        if 'CRITICAL' in advice.get('urgency', '').upper() or 'URGENT' in advice.get('urgency', '').upper():
            user_data['plants_saved'] += 1
    
    if accuracy >= 95:
        user_data['perfect_detections'] += 1
    
    # Award badges
    badges = []
    if user_data['total_scans'] >= 10:
        badges.append('🔬 Lab Technician')
    if user_data['total_scans'] >= 50:
        badges.append('🌿 Plant Doctor')
    if user_data['total_scans'] >= 100:
        badges.append('🏆 Master Botanist')
    if user_data['plants_saved'] >= 5:
        badges.append('🦸 Plant Savior')
    if user_data['perfect_detections'] >= 20:
        badges.append('🎯 Precision Expert')
    if user_data['diseases_detected'] >= 25:
        badges.append('🐛 Disease Hunter')
    
    user_data['badges'] = list(set(badges))
    user_data['last_scan'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    with open('gamification.json', 'w') as f:
        json.dump(data, f, indent=2)
    
    return points, badges

def show():
    st.markdown('<div class="main-header"><h1>🌿 Plant Disease Detection Dashboard</h1></div>', unsafe_allow_html=True)
    
    # Initialize gamification file
    if not os.path.exists('gamification.json'):
        with open('gamification.json', 'w') as f:
            json.dump({}, f)
    
    # Load models
    @st.cache_resource
    def load_models():
        try:
            model1 = tf.keras.models.load_model('model1.keras')
            model2 = tf.keras.models.load_model('model2.keras')
            return model1, model2, True
        except Exception as e:
            st.error(f"Error loading models: {e}")
            return None, None, False
    
    model1, model2, models_loaded = load_models()
    
    if not models_loaded:
        st.warning("⚠️ Models not found. Please ensure model1.keras and model2.keras are in the directory.")
        return
    
    # Initialize RAG system
    if not rag_system.initialized:
        with st.spinner("🧠 Initializing AI Knowledge Base..."):
            rag_system.initialize()
    
    # Class names
    class_names = ['Healthy', 'Bacterial Spot', 'Early Blight', 'Late Blight', 
                   'Leaf Mold', 'Septoria Leaf Spot', 'Spider Mites', 
                   'Target Spot', 'Yellow Leaf Curl Virus', 'Mosaic Virus']
    
    # Multi-modal input section
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("📤 Multi-Modal Plant Analysis")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        uploaded_file = st.file_uploader(
            "Upload Plant Image (PNG, JPG, JPEG)", 
            type=['png', 'jpg', 'jpeg'],
            help="Upload a clear image of the plant leaf"
        )
    
    with col2:
        text_description = st.text_area(
            "Additional Context (Optional)",
            placeholder="e.g., 'Yellow spots appeared after rain', 'Leaves wilting in afternoon'",
            height=100,
            help="Provide additional symptoms or context for better diagnosis"
        )
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    if uploaded_file is not None:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.subheader("📷 Uploaded Image")
            image = Image.open(uploaded_file)
            st.image(image, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.subheader("🔬 AI Analysis")
            
            # Preprocess image
            def preprocess_image(img):
                img = img.resize((224, 224))
                img_array = np.array(img) / 255.0
                img_array = np.expand_dims(img_array, axis=0)
                return img_array
            
            # Process button
            if st.button("🚀 Analyze Plant", use_container_width=True, type="primary"):
                with st.spinner("🔍 Analyzing plant health with AI..."):
                    preprocessed = preprocess_image(image)
                    
                    # Get predictions from both models
                    pred1 = model1.predict(preprocessed, verbose=0)
                    pred2 = model2.predict(preprocessed, verbose=0)
                    
                    # Calculate accuracies
                    acc1 = np.max(pred1) * 100
                    acc2 = np.max(pred2) * 100
                    
                    # Get predicted classes
                    class1_idx = np.argmax(pred1)
                    class2_idx = np.argmax(pred2)
                    
                    # Determine best model
                    if acc1 >= acc2:
                        best_accuracy = acc1
                        best_prediction = class_names[class1_idx] if class1_idx < len(class_names) else f"Class {class1_idx}"
                        best_model = "Model 1"
                        best_probs = pred1[0]
                    else:
                        best_accuracy = acc2
                        best_prediction = class_names[class2_idx] if class2_idx < len(class_names) else f"Class {class2_idx}"
                        best_model = "Model 2"
                        best_probs = pred2[0]
                    
                    # Get RAG-based advice
                    context = text_description if text_description else ""
                    advice = rag_system.get_personalized_advice(best_prediction, context)
                    quick_tips = rag_system.get_quick_tips(best_prediction)
                    
                    # Update gamification
                    points_earned, new_badges = update_gamification(
                        st.session_state.username, 
                        best_prediction, 
                        best_accuracy,
                        advice
                    )
                    
                    # Display results
                    st.success(f"✅ Analysis Complete! +{points_earned} points earned!")
                    
                    if new_badges:
                        st.balloons()
                        st.info(f"🎉 New badges: {' '.join(new_badges[:2])}")
                    
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3>🎯 Diagnosis</h3>
                        <h2>{best_prediction}</h2>
                        <p>Confidence: {best_accuracy:.2f}%</p>
                        <p>Best Model: {best_model}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Model comparison
                    st.markdown("### 📊 Model Comparison")
                    col_m1, col_m2 = st.columns(2)
                    with col_m1:
                        st.metric("Model 1 Confidence", f"{acc1:.2f}%")
                    with col_m2:
                        st.metric("Model 2 Confidence", f"{acc2:.2f}%")
                    
                    st.markdown('</div>', unsafe_allow_html=True)
        
        # Interactive Recommendations Section
        if 'best_prediction' in locals():
            st.markdown("---")
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.subheader("💡 Personalized Recommendations")
            
            # Quick Tips with Emojis
            st.markdown("### 🎯 Quick Action Tips")
            for tip in quick_tips:
                st.info(tip)
            
            # Detailed Advice Tabs
            tab1, tab2, tab3, tab4 = st.tabs(["🔍 Symptoms", "🦠 Causes", "🛡️ Prevention", "💊 Treatment"])
            
            with tab1:
                st.markdown("**Identified Symptoms:**")
                for symptom in advice['symptoms'][:3]:
                    st.markdown(f"• {symptom}")
            
            with tab2:
                st.markdown("**Root Causes:**")
                for cause in advice['causes'][:3]:
                    st.markdown(f"• {cause}")
            
            with tab3:
                st.markdown("**Prevention Strategies:**")
                for prev in advice['prevention'][:4]:
                    st.markdown(f"• {prev}")
            
            with tab4:
                st.markdown("**Treatment Options:**")
                for treat in advice['treatment'][:4]:
                    st.markdown(f"• {treat}")
                
                # Urgency indicator
                urgency = advice.get('urgency', 'Moderate')
                if 'CRITICAL' in urgency.upper():
                    st.error(f"⚠️ {urgency}")
                elif 'URGENT' in urgency.upper():
                    st.warning(f"⏰ {urgency}")
                else:
                    st.info(f"ℹ️ {urgency}")
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Top 3 predictions
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("### 🏆 Top Predictions")
            top_3_idx = np.argsort(best_probs)[-3:][::-1]
            for idx in top_3_idx:
                class_name = class_names[idx] if idx < len(class_names) else f"Class {idx}"
                conf = best_probs[idx] * 100
                st.progress(conf / 100, text=f"{class_name}: {conf:.2f}%")
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Convert image to base64 for storage
            buffered = BytesIO()
            image.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode()
            
            # Save to historique.json
            def save_to_history(username, img_base64, prediction, accuracy, metrics, advice_data):
                with open('historique.json', 'r') as f:
                    history = json.load(f)
                
                entry = {
                    'username': username,
                    'image': img_base64,
                    'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'prediction': prediction,
                    'accuracy': accuracy,
                    'best_model': metrics['best_model'],
                    'model1_accuracy': metrics['model1_accuracy'],
                    'model2_accuracy': metrics['model2_accuracy'],
                    'top_predictions': metrics['top_predictions'],
                    'advice': advice_data,
                    'context': text_description if text_description else "",
                    'points_earned': points_earned
                }
                
                history.append(entry)
                
                with open('historique.json', 'w') as f:
                    json.dump(history, f, indent=2)
            
            metrics = {
                'best_model': best_model,
                'model1_accuracy': float(acc1),
                'model2_accuracy': float(acc2),
                'top_predictions': [
                    {
                        'class': class_names[idx] if idx < len(class_names) else f"Class {idx}",
                        'confidence': float(best_probs[idx] * 100)
                    } for idx in top_3_idx
                ]
            }
            
            save_to_history(
                st.session_state.username,
                img_str,
                best_prediction,
                float(best_accuracy),
                metrics,
                advice
            )
            
            st.success("💾 Complete analysis saved to history!")
    
    # Information cards
    st.markdown("---")
    st.markdown("### 💡 How Multi-Modal Analysis Works")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="card">
            <h4>1️⃣ Upload & Describe</h4>
            <p>Upload image + add text context for enhanced accuracy</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="card">
            <h4>2️⃣ AI Analysis</h4>
            <p>Dual models + RAG knowledge base analyze your plant</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="card">
            <h4>3️⃣ Personalized Care</h4>
            <p>Get custom advice + earn points & badges!</p>
        </div>
        """, unsafe_allow_html=True)