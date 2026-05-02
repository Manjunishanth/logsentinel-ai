# LogSentinel AI — Real-Time Log Analysis System

## Quick Start

### Requirements
```
pip install flask pandas scikit-learn
```

### Run
```bash
cd logsentinel
python3 run.py
```
Open **http://localhost:5000**

---

## What's Real (Not Demo)

| Feature | How it works |
|---------|-------------|
| File upload | Flask multipart POST → parsed into SQLite |
| Log parsing | 14 regex patterns extract fields from raw lines |
| Preprocessing | 8-step pipeline strips noise, tokenizes, removes stopwords |
| ML | TF-IDF vectorizer + Isolation Forest (scikit-learn) |
| Anomaly scores | Per-row, normalized 0–1 |
| Charts | Chart.js rendering **real DB data** |
| Pagination | Server-side, works on any file size |
| Filter by level | Server-side SQL query |
| Delete files | Full cascade DB delete |

## Tech Stack
- **Backend**: Flask + SQLite + scikit-learn
- **ML Model**: TF-IDF (500 features, bigrams) + Isolation Forest (10% contamination)
- **Frontend**: Vanilla JS + Chart.js (no React/Vue needed)
- **Preprocessing**: Python regex (re module) + custom stopword set

## Project Structure
```
logsentinel/
├── run.py           # Start here
├── app.py           # All backend logic + ML
├── logsentinel.db   # Auto-created
└── templates/
    └── index.html   # Full SPA frontend
```
