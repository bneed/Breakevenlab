"""
Authentication and subscription management for Break-even Lab
"""
import streamlit as st
import sqlite3
import os
from typing import Optional, Dict, Any

# Initialize session state
if 'user_tier' not in st.session_state:
    st.session_state.user_tier = 'free'
if 'user_id' not in st.session_state:
    st.session_state.user_id = None
if 'stripe_customer_id' not in st.session_state:
    st.session_state.stripe_customer_id = None

def init_database():
    """Initialize the SQLite database for user management"""
    db_path = "users.db"
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Create users table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT UNIQUE,
            tier TEXT DEFAULT 'free',
            stripe_customer_id TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Create alerts table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS alerts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            alert_type TEXT,
            parameters TEXT,
            is_active BOOLEAN DEFAULT 1,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    ''')
    
    conn.commit()
    conn.close()

def get_user_tier() -> str:
    """Get the current user's subscription tier"""
    if 'user_tier' not in st.session_state:
        st.session_state.user_tier = 'free'
    return st.session_state.user_tier

def check_user_subscription() -> bool:
    """Check if user has active subscription"""
    if 'user_tier' not in st.session_state:
        st.session_state.user_tier = 'free'
    return st.session_state.user_tier in ['pro', 'founder']

def is_pro_feature() -> bool:
    """Check if current user can access pro features"""
    return check_user_subscription()

def show_pro_upgrade_prompt(feature_name: str = "this feature"):
    """Show a prompt to upgrade to pro for a specific feature"""
    st.warning(f"🔒 {feature_name.title()} is a Pro feature. Upgrade to access unlimited usage!")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("⭐ Upgrade to Pro", key=f"upgrade_{feature_name}"):
            st.info("Pro upgrade coming soon! Contact us for early access.")
    with col2:
        if st.button("📧 Get Notified", key=f"notify_{feature_name}"):
            st.success("We'll notify you when Pro features are available!")

def create_user(email: str, tier: str = 'free', stripe_customer_id: Optional[str] = None) -> int:
    """Create a new user in the database"""
    init_database()
    conn = sqlite3.connect("users.db")
    cursor = conn.cursor()
    
    try:
        cursor.execute('''
            INSERT INTO users (email, tier, stripe_customer_id)
            VALUES (?, ?, ?)
        ''', (email, tier, stripe_customer_id))
        
        user_id = cursor.lastrowid
        conn.commit()
        return user_id
    except sqlite3.IntegrityError:
        # User already exists
        cursor.execute('SELECT id FROM users WHERE email = ?', (email,))
        result = cursor.fetchone()
        return result[0] if result else None
    finally:
        conn.close()

def update_user_tier(user_id: int, tier: str):
    """Update user's subscription tier"""
    init_database()
    conn = sqlite3.connect("users.db")
    cursor = conn.cursor()
    
    cursor.execute('''
        UPDATE users SET tier = ?, updated_at = CURRENT_TIMESTAMP
        WHERE id = ?
    ''', (tier, user_id))
    
    conn.commit()
    conn.close()
    
    # Update session state
    st.session_state.user_tier = tier

def get_user_by_email(email: str) -> Optional[Dict[str, Any]]:
    """Get user information by email"""
    init_database()
    conn = sqlite3.connect("users.db")
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT id, email, tier, stripe_customer_id, created_at
        FROM users WHERE email = ?
    ''', (email,))
    
    result = cursor.fetchone()
    conn.close()
    
    if result:
        return {
            'id': result[0],
            'email': result[1],
            'tier': result[2],
            'stripe_customer_id': result[3],
            'created_at': result[4]
        }
    return None

def add_alert(user_id: int, alert_type: str, parameters: Dict[str, Any]) -> int:
    """Add a new alert for a user"""
    init_database()
    conn = sqlite3.connect("users.db")
    cursor = conn.cursor()
    
    import json
    cursor.execute('''
        INSERT INTO alerts (user_id, alert_type, parameters)
        VALUES (?, ?, ?)
    ''', (user_id, alert_type, json.dumps(parameters)))
    
    alert_id = cursor.lastrowid
    conn.commit()
    conn.close()
    
    return alert_id

def get_user_alerts(user_id: int) -> list:
    """Get all alerts for a user"""
    init_database()
    conn = sqlite3.connect("users.db")
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT id, alert_type, parameters, is_active, created_at
        FROM alerts WHERE user_id = ? AND is_active = 1
    ''', (user_id,))
    
    results = cursor.fetchall()
    conn.close()
    
    alerts = []
    for result in results:
        import json
        alerts.append({
            'id': result[0],
            'alert_type': result[1],
            'parameters': json.loads(result[2]),
            'is_active': result[3],
            'created_at': result[4]
        })
    
    return alerts

def deactivate_alert(alert_id: int):
    """Deactivate an alert"""
    init_database()
    conn = sqlite3.connect("users.db")
    cursor = conn.cursor()
    
    cursor.execute('''
        UPDATE alerts SET is_active = 0 WHERE id = ?
    ''', (alert_id,))
    
    conn.commit()
    conn.close()

# Initialize database on import
init_database()
