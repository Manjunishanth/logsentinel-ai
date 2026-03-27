"""
LogSentinel AI — Startup Script
Run: python3 run.py
Then open: http://localhost:5000
"""
import os, sys

# Initialize DB before starting
from app import app, init_db
with app.app_context():
    init_db()
    print("✅ Database initialized")

print("=" * 50)
print("  LogSentinel AI — Real-Time Log Analysis")
print("=" * 50)
print("  🌐  http://localhost:5000")
print("  📁  Upload .log / .txt / .csv files")
print("  🎲  Or click 'Generate Sample Logs'")
print("  ⚙️   Run Preprocessing + ML detection")
print("  📊  View real charts from your data")
print("=" * 50)

app.run(debug=False, host='0.0.0.0', port=5000)
