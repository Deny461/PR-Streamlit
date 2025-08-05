from flask import Flask, render_template, request, redirect, url_for, flash, session, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash, check_password_hash
import pandas as pd
import plotly.graph_objects as go
import plotly.utils
import json
import os
from datetime import datetime

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-change-this-in-production'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///player_readiness.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# Database Models
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(120), nullable=False)
    player_name = db.Column(db.String(120), nullable=True)
    team = db.Column(db.String(120), nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# Helper functions
def load_data(path):
    df = pd.read_csv(path)
    df['Date'] = pd.to_datetime(df['Start Date'], format='%m/%d/%y', errors='coerce')
    if df['Date'].isna().all():
        df['Date'] = pd.to_datetime(df['Start Date'], errors='coerce')
    return df

def get_color(ratio):
    if ratio < 0.5: return "red"
    if ratio < 0.75: return "orange"
    if ratio < 1.0: return "yellow"
    if ratio <= 1.30: return "green"
    return "black"

def create_readiness_gauge(value, benchmark, label):
    ratio = 0 if pd.isna(benchmark) or benchmark == 0 else value / benchmark
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(ratio, 2),
        number={"font": {"size": 20}},
        gauge={
            "axis": {"range": [0, max(1.5, ratio)], "showticklabels": False},
            "bar": {"color": get_color(ratio)},
            "steps": [
                {"range": [0, 0.5], "color": "#ffcccc"},
                {"range": [0.5, 0.75], "color": "#ffe0b3"},
                {"range": [0.75, 1.0], "color": "#ffffcc"},
                {"range": [1.0, 1.3], "color": "#ccffcc"},
                {"range": [1.3, max(1.5, ratio)], "color": "#e6e6e6"}
            ]
        }
    ))
    fig.update_layout(margin=dict(t=10, b=10, l=10, r=10), height=180)
    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

# Routes
@app.route('/')
def index():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user = User.query.filter_by(username=username).first()
        
        if user and check_password_hash(user.password_hash, password):
            login_user(user)
            return redirect(url_for('dashboard'))
        else:
            flash('Invalid username or password')
    
    return render_template('login.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        player_name = request.form['player_name']
        team = request.form['team']
        
        if User.query.filter_by(username=username).first():
            flash('Username already exists')
            return render_template('signup.html')
        
        if User.query.filter_by(email=email).first():
            flash('Email already registered')
            return render_template('signup.html')
        
        user = User(
            username=username,
            email=email,
            password_hash=generate_password_hash(password),
            player_name=player_name,
            team=team
        )
        db.session.add(user)
        db.session.commit()
        
        flash('Account created successfully! Please log in.')
        return redirect(url_for('login'))
    
    return render_template('signup.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('index'))

@app.route('/dashboard')
@login_required
def dashboard():
    return render_template('dashboard.html', user=current_user)

@app.route('/api/player_data')
@login_required
def player_data():
    team = current_user.team
    if not team:
        return jsonify({'error': 'No team assigned'})
    
    path = f"Player Data/{team}_PD_Data.csv"
    if not os.path.exists(path):
        return jsonify({'error': 'Data file not found'})
    
    try:
        df = load_data(path)
        df = df.dropna(subset=["Date","Session Type","Athlete Name","Segment Name"])
        df = df[df["Segment Name"]=="Whole Session"].sort_values("Date")
        
        # Get player data
        player_df = df[df["Athlete Name"] == current_user.player_name]
        if player_df.empty:
            return jsonify({'error': 'Player data not found'})
        
        # Process metrics
        METRICS = ["Distance (m)", "High Intensity Running (m)", "Sprint Distance (m)", "No. of Sprints", "Top Speed (kph)"]
        
        # Get latest data
        latest_data = player_df.iloc[-1] if not player_df.empty else None
        
        # Create gauge data
        gauge_data = {}
        for metric in METRICS:
            if metric in player_df.columns:
                value = player_df[metric].iloc[-1] if not player_df.empty else 0
                benchmark = player_df[metric].max() if not player_df.empty else 1
                gauge_data[metric] = create_readiness_gauge(value, benchmark, metric)
        
        return jsonify({
            'player_name': current_user.player_name,
            'team': current_user.team,
            'latest_data': latest_data.to_dict() if latest_data is not None else {},
            'gauge_data': gauge_data
        })
        
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    port = int(os.environ.get('PORT', 3000))
    app.run(debug=False, host='0.0.0.0', port=port)