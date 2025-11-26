import streamlit as st
import json
import os
from datetime import datetime
import hashlib

# Page configuration
st.set_page_config(
    page_title="PlantDoctor",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize JSON files
def init_json_files():
    if not os.path.exists('users.json'):
        with open('users.json', 'w') as f:
            json.dump([], f)
    
    if not os.path.exists('historique.json'):
        with open('historique.json', 'w') as f:
            json.dump([], f)

init_json_files()

# Session state initialization
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'username' not in st.session_state:
    st.session_state.username = None
if 'theme' not in st.session_state:
    st.session_state.theme = 'light'

# Theme toggle
def toggle_theme():
    st.session_state.theme = 'dark' if st.session_state.theme == 'light' else 'light'

# Apply custom CSS based on theme
def apply_theme():
    if st.session_state.theme == 'dark':
        st.markdown("""
        <style>
        .stApp {
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        }
        .main-header {
            background: linear-gradient(90deg, #0f3460 0%, #16213e 100%);
            padding: 2rem;
            border-radius: 15px;
            margin-bottom: 2rem;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        }
        .card {
            background: rgba(255, 255, 255, 0.05);
            backdrop-filter: blur(10px);
            padding: 2rem;
            border-radius: 15px;
            border: 1px solid rgba(255, 255, 255, 0.1);
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
            margin-bottom: 1rem;
        }
        .metric-card {
            background: linear-gradient(135deg, #1a5f3c 0%, #2d8659 100%);
            padding: 1.5rem;
            border-radius: 12px;
            text-align: center;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
        }
        h1, h2, h3 { color: #4ecca3; }
        .stButton>button {
            background: linear-gradient(90deg, #2d8659 0%, #4ecca3 100%);
            color: white;
            border: none;
            padding: 0.5rem 2rem;
            border-radius: 25px;
            font-weight: 600;
            transition: all 0.3s;
        }
        .stButton>button:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(78, 204, 163, 0.4);
        }
        </style>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <style>
        .stApp {
            background: linear-gradient(135deg, #e8f5e9 0%, #c8e6c9 100%);
        }
        .main-header {
            background: linear-gradient(90deg, #4caf50 0%, #81c784 100%);
            padding: 2rem;
            border-radius: 15px;
            margin-bottom: 2rem;
            box-shadow: 0 8px 32px rgba(76, 175, 80, 0.3);
            color: white;
        }
        .card {
            background: white;
            padding: 2rem;
            border-radius: 15px;
            border: 2px solid #a5d6a7;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
            margin-bottom: 1rem;
        }
        .metric-card {
            background: linear-gradient(135deg, #66bb6a 0%, #81c784 100%);
            padding: 1.5rem;
            border-radius: 12px;
            text-align: center;
            color: white;
            box-shadow: 0 4px 15px rgba(76, 175, 80, 0.3);
        }
        h1, h2, h3 { color: #2e7d32; }
        .stButton>button {
            background: linear-gradient(90deg, #4caf50 0%, #66bb6a 100%);
            color: white;
            border: none;
            padding: 0.5rem 2rem;
            border-radius: 25px;
            font-weight: 600;
            transition: all 0.3s;
        }
        .stButton>button:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(76, 175, 80, 0.4);
        }
        .logo { font-size: 3rem; text-align: center; margin-bottom: 1rem; }
        </style>
        """, unsafe_allow_html=True)

apply_theme()

# Hash password
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

# User authentication functions
def load_users():
    with open('users.json', 'r') as f:
        return json.load(f)

def save_user(username, password):
    users = load_users()
    users.append({'username': username, 'password': hash_password(password)})
    with open('users.json', 'w') as f:
        json.dump(users, f, indent=2)

def verify_user(username, password):
    users = load_users()
    hashed = hash_password(password)
    return any(u['username'] == username and u['password'] == hashed for u in users)

def user_exists(username):
    users = load_users()
    return any(u['username'] == username for u in users)

# Logout function
def logout():
    st.session_state.logged_in = False
    st.session_state.username = None
    st.rerun()

# Main app logic
if not st.session_state.logged_in:
    # Login/Register page
    st.markdown('<div class="logo">🌿</div>', unsafe_allow_html=True)
    st.markdown('<h1 style="text-align: center;">PlantDoctor</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 1.2rem;">Your AI-Powered Plant Health Assistant</p>', unsafe_allow_html=True)
    
    tab1, tab2 = st.tabs(["🔑 Login", "📝 Register"])
    
    with tab1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        with st.form("login_form"):
            st.subheader("Welcome Back!")
            username = st.text_input("Username", key="login_user")
            password = st.text_input("Password", type="password", key="login_pass")
            submit = st.form_submit_button("Login", use_container_width=True)
            
            if submit:
                if verify_user(username, password):
                    st.session_state.logged_in = True
                    st.session_state.username = username
                    st.success("✅ Login successful!")
                    st.rerun()
                else:
                    st.error("❌ Invalid username or password")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with tab2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        with st.form("register_form"):
            st.subheader("Create New Account")
            new_username = st.text_input("Username", key="reg_user")
            new_password = st.text_input("Password", type="password", key="reg_pass")
            confirm_password = st.text_input("Confirm Password", type="password", key="reg_confirm")
            submit = st.form_submit_button("Register", use_container_width=True)
            
            if submit:
                if len(new_username) < 3:
                    st.error("Username must be at least 3 characters")
                elif len(new_password) < 6:
                    st.error("Password must be at least 6 characters")
                elif new_password != confirm_password:
                    st.error("Passwords don't match")
                elif user_exists(new_username):
                    st.error("Username already exists")
                else:
                    save_user(new_username, new_password)
                    st.success("✅ Account created! Please login.")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Theme toggle at bottom
    st.divider()
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        if st.button("🌓 Toggle Theme", use_container_width=True):
            toggle_theme()
            st.rerun()

else:
    # Logged in - Show navigation
    st.sidebar.markdown(f"""
    <div style='text-align: center; padding: 1rem;'>
        <h1 style='color: #4caf50;'>🌿 PlantDoctor</h1>
        <p style='font-size: 1.1rem;'>Welcome, <strong>{st.session_state.username}</strong>!</p>
    </div>
    """, unsafe_allow_html=True)
    
    page = st.sidebar.radio("Navigation", ["📊 Dashboard", "📜 History", "🏆 Leaderboard"], label_visibility="collapsed")
    
    st.sidebar.divider()
    
    col1, col2 = st.sidebar.columns(2)
    with col1:
        if st.button("🚪 Logout", use_container_width=True):
            logout()
    with col2:
        if st.button("🌓 Theme", use_container_width=True):
            toggle_theme()
            st.rerun()
    
    if page == "📊 Dashboard":
        # Import and run dashboard page
        import dashboard
        dashboard.show()
    elif page == "🏆 Leaderboard":
        # Import and run leaderboard page
        import leaderboard
        leaderboard.show()
    else:
        # Import and run history page
        import history
        history.show()
