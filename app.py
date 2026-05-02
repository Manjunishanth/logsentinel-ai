"""
LogSentinel AI — Real Backend + Authentication
Flask + SQLite + Scikit-learn (TF-IDF + Isolation Forest) + Flask-Login

Run:
    pip install flask flask-login werkzeug pandas scikit-learn itsdangerous
    python app.py
"""
import os, re, json, csv, io, math, time, sqlite3, hashlib
from datetime import datetime, timedelta
from collections import Counter
from flask import (
    Flask, request, jsonify, render_template, g,
    redirect, url_for, flash
)
from flask_login import (
    LoginManager, UserMixin, login_user, login_required,
    logout_user, current_user
)
from werkzeug.security import generate_password_hash, check_password_hash

# ── NEW IMPORTS FOR PHASE 2 (Password Reset) ──────────────────────────────
# itsdangerous: creates secure, time-limited tokens
# smtplib + email: built-in Python libraries for sending email
from itsdangerous import URLSafeTimedSerializer, SignatureExpired, BadSignature
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
# ──────────────────────────────────────────────────────────────────────────

from module4_validation import validation_bp
from module5_bert import bert_bp
from module6_rca import rca_bp
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import IsolationForest
import warnings
warnings.filterwarnings('ignore')

# ── APP ────────────────────────────────────────────────────────────────────
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024
app.config['SECRET_KEY'] = os.environ.get(
    'SECRET_KEY', 'change-me-to-a-real-secret-in-production'
)

# ── EMAIL CONFIG (PHASE 2) ─────────────────────────────────────────────────
# HOW TO SET UP GMAIL:
#   1. Go to myaccount.google.com → Security → 2-Step Verification (turn ON)
#   2. Search "App Passwords" → Create one for "Mail" → copy the 16-char code
#   3. Replace the values below with your Gmail and that App Password
#
# For production, use environment variables instead of hardcoding:
#   export MAIL_USERNAME="you@gmail.com"
#   export MAIL_PASSWORD="your-16-char-app-password"
app.config['MAIL_USERNAME'] = 'manjunishanth.03@gmail.com'
app.config['MAIL_PASSWORD'] = 'iitwqqizdxwbggbb'
app.config['MAIL_DEFAULT_SENDER'] = os.environ.get('MAIL_USERNAME', 'manjunishanth.03@gmail.com')

# This creates the "serializer" used to make secure tokens.
# It uses your SECRET_KEY, so tokens can't be faked.
# The salt is just a string that makes password-reset tokens different
# from any other tokens your app might create.
def get_serializer():
    return URLSafeTimedSerializer(app.config['SECRET_KEY'])

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'
login_manager.login_message = 'Please log in to access the dashboard.'
login_manager.login_message_category = 'info'

app.register_blueprint(validation_bp)
app.register_blueprint(bert_bp)
app.register_blueprint(rca_bp)

DB_PATH = os.path.join(os.path.dirname(__file__), 'logsentinel.db')

# ── PATTERNS ───────────────────────────────────────────────────────────────
P = {
    'ts_iso':   re.compile(r'\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})?'),
    'ts_std':   re.compile(r'\w{3}\s+\d{1,2}\s+\d{2}:\d{2}:\d{2}'),
    'ts_epoch': re.compile(r'\b\d{10,13}\b'),
    'ip':       re.compile(r'\b\d{1,3}(?:\.\d{1,3}){3}(?::\d+)?\b'),
    'url':      re.compile(r'https?://\S+'),
    'fpath':    re.compile(r'(?:/[\w.\-]+){2,}(?::\d+)?'),
    'hex':      re.compile(r'\b[0-9a-fA-F]{8,}\b'),
    'errcode':  re.compile(r'\b[A-Z]{2,6}-\d{3,6}\b'),
    'level':    re.compile(r'\b(ERROR|WARN(?:ING)?|INFO|DEBUG|CRITICAL|FATAL|TRACE|NOTICE)\b', re.I),
    'stack':    re.compile(r'at\s+[\w\.\$<>]+\([\w\.]+:\d+\)'),
    'memsize':  re.compile(r'\b\d+(?:\.\d+)?\s*(?:KB|MB|GB|TB|bytes?)\b', re.I),
    'duration': re.compile(r'\b\d+(?:\.\d+)?\s*(?:ms|seconds?|minutes?|hours?)\b', re.I),
    'special':  re.compile(r'[^\w\s]'),
    'ws':       re.compile(r'\s+'),
}
LEVEL_MAP = {'error':'ERROR','err':'ERROR','warning':'WARN','warn':'WARN',
             'info':'INFO','information':'INFO','debug':'DEBUG','dbg':'DEBUG',
             'critical':'CRITICAL','crit':'CRITICAL','fatal':'FATAL',
             'trace':'TRACE','notice':'INFO'}
STOPWORDS = {'the','a','an','is','it','in','on','at','to','of','and','or','not',
    'with','for','from','by','be','as','this','that','these','those','was','were',
    'has','have','had','will','would','could','should','are','am','been','being',
    'do','does','did','into','out','up','down','but','so','if','then','than',
    'when','while','after','before','about','above','below','between','through',
    'java','scala','python','com','org','net','http','https','www','null','none',
    'true','false','undefined','nan','class','method','line','thread','exception',
    'caused','stack','trace','main','object','string','int','void'}

# ── DB ─────────────────────────────────────────────────────────────────────
def get_db():
    if 'db' not in g:
        g.db = sqlite3.connect(DB_PATH, detect_types=sqlite3.PARSE_DECLTYPES)
        g.db.row_factory = sqlite3.Row
    return g.db

@app.teardown_appcontext
def close_db(e=None):
    db = g.pop('db', None)
    if db: db.close()

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.executescript('''
        CREATE TABLE IF NOT EXISTS users (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            username      TEXT    UNIQUE NOT NULL,
            email         TEXT    UNIQUE NOT NULL,
            password_hash TEXT    NOT NULL,
            created_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS uploads (
            id TEXT PRIMARY KEY, file_name TEXT, file_size INTEGER,
            total_lines INTEGER DEFAULT 0, error_count INTEGER DEFAULT 0,
            warn_count INTEGER DEFAULT 0, info_count INTEGER DEFAULT 0,
            uploaded_at DATETIME DEFAULT CURRENT_TIMESTAMP, status TEXT DEFAULT 'pending'
        );
        CREATE TABLE IF NOT EXISTS raw_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            file_id TEXT, file_name TEXT, line_number INTEGER,
            raw_text TEXT, timestamp TEXT, level TEXT DEFAULT 'INFO',
            source TEXT, message TEXT, ip_address TEXT, error_code TEXT,
            has_stack_trace INTEGER DEFAULT 0,
            uploaded_at DATETIME DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS processed_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            raw_id INTEGER, cleaned_text TEXT, tokens TEXT,
            token_count INTEGER, removed_count INTEGER,
            anomaly_score REAL DEFAULT 0.0, is_anomaly INTEGER DEFAULT 0,
            processed_at DATETIME DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS validation_reports (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            file_id TEXT NOT NULL UNIQUE,
            total_logs INTEGER, normal_count INTEGER, anomaly_count INTEGER,
            class_ratio REAL, train_size INTEGER, test_size INTEGER,
            cv_mean_acc REAL, cv_std_acc REAL, cv_scores TEXT,
            top_tokens TEXT, vocab_size INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS validated_splits (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            file_id TEXT NOT NULL, raw_id INTEGER NOT NULL,
            split TEXT NOT NULL, fold INTEGER DEFAULT 0,
            label INTEGER NOT NULL, tokens TEXT
        );
        CREATE TABLE IF NOT EXISTS bert_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            file_id TEXT NOT NULL UNIQUE, mode TEXT,
            accuracy REAL, precision_s REAL, recall REAL, f1_score REAL,
            confusion_matrix TEXT, epochs INTEGER, train_losses TEXT,
            model_path TEXT, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS bert_predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            file_id TEXT NOT NULL, raw_id INTEGER, cleaned_text TEXT,
            prediction INTEGER, confidence REAL, bert_score REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS rca_reports (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            file_id TEXT NOT NULL UNIQUE,
            total_predicted INTEGER, correct_preds INTEGER, wrong_preds INTEGER,
            match_accuracy REAL, health_score REAL, root_causes TEXT,
            outcome_summary TEXT, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS root_cause_entries (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            file_id TEXT NOT NULL, rank INTEGER, cause_id TEXT,
            cause_label TEXT, frequency INTEGER, confidence REAL,
            severity TEXT, evidence TEXT, first_seen TEXT, last_seen TEXT
        );
    ''')
    conn.commit()

    # Migrations for existing DBs
    migrations = [
        ("validated_splits", "fold",         "ALTER TABLE validated_splits ADD COLUMN fold INTEGER DEFAULT 0"),
        ("validated_splits", "tokens",        "ALTER TABLE validated_splits ADD COLUMN tokens TEXT"),
        ("bert_results",     "epochs",        "ALTER TABLE bert_results ADD COLUMN epochs INTEGER"),
        ("bert_results",     "train_losses",  "ALTER TABLE bert_results ADD COLUMN train_losses TEXT"),
        ("bert_results",     "model_path",    "ALTER TABLE bert_results ADD COLUMN model_path TEXT"),
        ("bert_results",     "created_at",    "ALTER TABLE bert_results ADD COLUMN created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP"),
        ("bert_predictions", "cleaned_text",  "ALTER TABLE bert_predictions ADD COLUMN cleaned_text TEXT"),
        ("bert_predictions", "bert_score",    "ALTER TABLE bert_predictions ADD COLUMN bert_score REAL"),
        ("bert_predictions", "created_at",    "ALTER TABLE bert_predictions ADD COLUMN created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP"),
        ("validation_reports","total_logs",   "ALTER TABLE validation_reports ADD COLUMN total_logs INTEGER"),
        ("validation_reports","normal_count", "ALTER TABLE validation_reports ADD COLUMN normal_count INTEGER"),
        ("validation_reports","anomaly_count","ALTER TABLE validation_reports ADD COLUMN anomaly_count INTEGER"),
        ("validation_reports","class_ratio",  "ALTER TABLE validation_reports ADD COLUMN class_ratio REAL"),
        ("validation_reports","cv_scores",    "ALTER TABLE validation_reports ADD COLUMN cv_scores TEXT"),
        ("validation_reports","vocab_size",   "ALTER TABLE validation_reports ADD COLUMN vocab_size INTEGER"),
    ]
    for table, col, sql in migrations:
        try:
            conn.execute(sql)
        except Exception:
            pass
    conn.commit()
    conn.close()

# ── USER MODEL ─────────────────────────────────────────────────────────────
class User(UserMixin):
    def __init__(self, row):
        self.id = row['id']
        self.username = row['username']
        self.email = row['email']
        self.password_hash = row['password_hash']

    @staticmethod
    def get(user_id):
        row = get_db().execute(
            "SELECT * FROM users WHERE id = ?", (user_id,)
        ).fetchone()
        return User(row) if row else None

    @staticmethod
    def find(identifier):
        """Look up by username OR email."""
        row = get_db().execute(
            "SELECT * FROM users WHERE username = ? OR email = ?",
            (identifier, identifier),
        ).fetchone()
        return User(row) if row else None

    @staticmethod
    def find_by_email(email):
        """Look up by email only — used in password reset."""
        row = get_db().execute(
            "SELECT * FROM users WHERE email = ?", (email,)
        ).fetchone()
        return User(row) if row else None

    def check_password(self, pw):
        return check_password_hash(self.password_hash, pw)


@login_manager.user_loader
def load_user(user_id):
    return User.get(user_id)

# ── VALIDATION HELPERS ─────────────────────────────────────────────────────
EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
USERNAME_RE = re.compile(r"^[A-Za-z0-9_]{3,20}$")

def validate_signup(username, email, password, confirm):
    errs = []
    if not USERNAME_RE.match(username or ""):
        errs.append("Username must be 3–20 characters (letters, numbers, underscore).")
    if not EMAIL_RE.match(email or ""):
        errs.append("Please enter a valid email address.")
    if not password or len(password) < 8:
        errs.append("Password must be at least 8 characters.")
    if password != confirm:
        errs.append("Passwords do not match.")
    return errs

# ── PREPROCESS ─────────────────────────────────────────────────────────────
def extract_fields(line):
    r = {'raw_text': line.strip(), 'timestamp': None, 'level': 'INFO',
         'source': None, 'message': line.strip(), 'ip_address': None,
         'error_code': None, 'has_stack_trace': 0}
    for k in ('ts_iso', 'ts_std'):
        m = P[k].search(line)
        if m: r['timestamp'] = m.group(0); break
    m = P['level'].search(line)
    if m: r['level'] = LEVEL_MAP.get(m.group(0).lower(), 'INFO')
    m = P['ip'].search(line)
    if m: r['ip_address'] = m.group(0)
    m = P['errcode'].search(line)
    if m: r['error_code'] = m.group(0)
    if P['stack'].search(line): r['has_stack_trace'] = 1
    m = re.search(r'\[([^\]]{2,40})\]', line)
    if m: r['source'] = m.group(1)
    m = re.search(r'(?:ERROR|WARN(?:ING)?|INFO|DEBUG|CRITICAL|FATAL)\s*[:\-]?\s*(.+)', line, re.I)
    if m: r['message'] = m.group(1).strip()[:300]
    return r

def preprocess_text(text):
    orig = len(text.split())
    for k in ('ts_iso','ts_std','ts_epoch','ip','url','fpath','hex','memsize','duration','stack','level'):
        text = P[k].sub(' ', text)
    text = P['special'].sub(' ', text.lower())
    tokens = [t for t in text.split() if len(t) > 2 and t not in STOPWORDS and not t.isdigit()]
    return ' '.join(tokens), tokens, orig - len(tokens)

def run_anomaly_detection(texts):
    if len(texts) < 3:
        return [0.0]*len(texts), [0]*len(texts)
    try:
        vect = TfidfVectorizer(max_features=500, ngram_range=(1,2), min_df=1, sublinear_tf=True)
        X = vect.fit_transform(texts)
        clf = IsolationForest(n_estimators=min(100,max(10,len(texts)//5)),
                              contamination=0.1, random_state=42)
        clf.fit(X)
        scores = clf.decision_function(X).tolist()
        preds = [1 if p==-1 else 0 for p in clf.predict(X).tolist()]
        mn, mx = min(scores), max(scores)
        rng = mx-mn if mx!=mn else 1
        norm = [round((s-mn)/rng, 4) for s in scores]
        return norm, preds
    except:
        return [0.0]*len(texts), [0]*len(texts)

def parse_file(content, fname):
    lines = []
    if fname.endswith('.csv'):
        try:
            reader = csv.DictReader(io.StringIO(content))
            for row in reader:
                lines.append(' '.join(str(v) for v in row.values()))
        except: lines = content.splitlines()
    else:
        lines = content.splitlines()
    entries = []
    for i, line in enumerate(lines):
        line = line.strip()
        if not line or len(line) < 5: continue
        f = extract_fields(line); f['line_number'] = i+1
        entries.append(f)
    return entries

# ══════════════════════════════════════════════════════════════════════════
# EMAIL HELPER (PHASE 2)
# ══════════════════════════════════════════════════════════════════════════
def send_reset_email(to_email, reset_url):
    """
    Sends the password reset email via Gmail SMTP.
    Returns True if successful, False if it fails.

    HOW THIS WORKS:
    - We connect to Gmail's server (smtp.gmail.com) on port 587
    - We "say hello" with EHLO
    - We upgrade the connection to be encrypted (starttls)
    - We log in with Gmail + App Password
    - We send the email
    - We log out
    """
    sender = app.config['MAIL_USERNAME']
    password = app.config['MAIL_PASSWORD']

    # If email isn't configured, don't crash — just return False
    if sender == 'your-gmail@gmail.com' or not sender or not password:
        return False

    # Build a nice HTML email
    msg = MIMEMultipart('alternative')
    msg['Subject'] = 'LogSentinel AI — Password Reset Request'
    msg['From'] = f'LogSentinel AI <{sender}>'
    msg['To'] = to_email

    # Plain text version (fallback for old email clients)
    text_body = f"""
Hello,

You requested a password reset for your LogSentinel AI account.

Click this link to reset your password (expires in 1 hour):
{reset_url}

If you did not request this, simply ignore this email.

— LogSentinel AI Security Team
"""

    # HTML version (looks nice in modern email clients)
    html_body = f"""
<!DOCTYPE html>
<html>
<head>
<style>
  body {{ font-family: Arial, sans-serif; background: #040710; color: #e2e8f0; margin: 0; padding: 20px; }}
  .container {{ max-width: 520px; margin: 0 auto; background: #0b1221; border: 1px solid rgba(0,245,196,0.15); border-radius: 12px; padding: 36px; }}
  .logo {{ font-size: 1.2rem; font-weight: bold; color: #00f5c4; margin-bottom: 24px; }}
  h2 {{ color: #fff; margin-bottom: 12px; }}
  p {{ color: #94a3b8; line-height: 1.6; }}
  .btn {{ display: inline-block; background: #00f5c4; color: #000; font-weight: bold; padding: 14px 28px; border-radius: 8px; text-decoration: none; margin: 20px 0; }}
  .footer {{ margin-top: 24px; font-size: 0.75rem; color: #475569; border-top: 1px solid rgba(0,245,196,0.1); padding-top: 16px; }}
  .url {{ word-break: break-all; font-size: 0.75rem; color: #64748b; }}
</style>
</head>
<body>
  <div class="container">
    <div class="logo">⬡ LogSentinel AI</div>
    <h2>Password Reset Request</h2>
    <p>Someone requested a password reset for your account. If this was you, click the button below to set a new password.</p>
    <p><strong>⏱ This link expires in 1 hour.</strong></p>
    <a href="{reset_url}" class="btn">⏵ Reset My Password</a>
    <p>Or copy this link into your browser:</p>
    <p class="url">{reset_url}</p>
    <div class="footer">
      If you did not request a password reset, ignore this email — your account is safe.<br>
      © LogSentinel AI Security Team
    </div>
  </div>
</body>
</html>
"""

    msg.attach(MIMEText(text_body, 'plain'))
    msg.attach(MIMEText(html_body, 'html'))

    try:
        with smtplib.SMTP('smtp.gmail.com', 587) as server:
            server.ehlo()           # Say hello to the server
            server.starttls()       # Encrypt the connection
            server.ehlo()           # Say hello again (required after starttls)
            server.login(sender, password)
            server.sendmail(sender, to_email, msg.as_string())
        return True
    except Exception as e:
        print(f"[EMAIL ERROR] {e}")
        return False


# ══════════════════════════════════════════════════════════════════════════
# AUTH ROUTES
# ══════════════════════════════════════════════════════════════════════════
@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('index'))

    if request.method == 'POST':
        identifier = (request.form.get('identifier') or '').strip()
        password   = request.form.get('password') or ''
        remember   = bool(request.form.get('remember'))

        if not identifier or not password:
            flash('Please enter both username/email and password.', 'err')
            return render_template('login.html'), 400

        user = User.find(identifier)
        if user and user.check_password(password):
            login_user(user, remember=remember)
            flash(f'Welcome back, {user.username}.', 'ok')
            return redirect(request.args.get('next') or url_for('index'))

        flash('Invalid credentials. Please try again.', 'err')
        return render_template('login.html'), 401

    return render_template('login.html')


# ══════════════════════════════════════════════════════════════════════════
# FORGOT PASSWORD — Step 1: User enters their email
# ══════════════════════════════════════════════════════════════════════════
@app.route('/forgot-password', methods=['GET', 'POST'])
def forgot_password():
    if request.method == 'POST':
        email = (request.form.get('email') or '').strip().lower()

        if not EMAIL_RE.match(email):
            flash('Please enter a valid email address.', 'err')
            return render_template('forgot_password.html')

        user = User.find_by_email(email)

        # ── EMAIL NOT FOUND: stop here and tell the user ──────────────
        if not user:
            flash('No account found with that email address. Please check and try again.', 'err')
            return render_template('forgot_password.html')

        # ── EMAIL FOUND: generate token and send/show reset link ──────
        s = get_serializer()
        token = s.dumps(email, salt='password-reset-salt')
        reset_url = url_for('reset_password', token=token, _external=True)

        email_sent = send_reset_email(email, reset_url)

        if not email_sent:
            flash('Failed to send email. Please try again later.', 'err')
            return render_template('forgot_password.html')

        # Email was sent successfully
        flash(
            'Reset link sent! Check your inbox (and spam folder).',
            'ok'
        )
        return render_template('forgot_password.html')

    return render_template('forgot_password.html')


# ══════════════════════════════════════════════════════════════════════════
# RESET PASSWORD — Step 2: User clicks link, sets new password
# ══════════════════════════════════════════════════════════════════════════
@app.route('/reset-password/<token>', methods=['GET', 'POST'])
def reset_password(token):
    """
    WHAT THIS ROUTE DOES:
    - The <token> in the URL is the secure token we generated in forgot_password
    - First, we verify the token is valid and not expired (max_age=3600 = 1 hour)
    - If valid: show the "Set New Password" form (GET) or update the password (POST)
    - If expired or invalid: redirect back to forgot_password with an error message
    """

    # ── STEP 1: Validate the token ─────────────────────────────────────────
    s = get_serializer()
    try:
        # This line does THREE things at once:
        # 1. Verifies the token hasn't been tampered with (uses SECRET_KEY)
        # 2. Verifies it hasn't expired (max_age = 3600 seconds = 1 hour)
        # 3. Extracts the email that was encoded inside the token
        email = s.loads(token, salt='password-reset-salt', max_age=3600)

    except SignatureExpired:
        # Token is valid but older than 1 hour
        flash('That reset link has expired. Please request a new one.', 'err')
        return redirect(url_for('forgot_password'))

    except BadSignature:
        # Token was tampered with, is fake, or just wrong
        flash('That reset link is invalid. Please request a new one.', 'err')
        return redirect(url_for('forgot_password'))

    # ── STEP 2: Find the user this token belongs to ─────────────────────────
    user = User.find_by_email(email)
    if not user:
        flash('No account found for this reset link.', 'err')
        return redirect(url_for('forgot_password'))

    # ── STEP 3: Handle the form submission (POST) ──────────────────────────
    if request.method == 'POST':
        new_password = request.form.get('password') or ''
        confirm      = request.form.get('confirm') or ''

        # Validate the new password
        if len(new_password) < 8:
            flash('Password must be at least 8 characters.', 'err')
            return render_template('reset_password.html', token=token)

        if new_password != confirm:
            flash('Passwords do not match.', 'err')
            return render_template('reset_password.html', token=token)

        # Hash the new password and save it to the database.
        # We ONLY update THIS specific user (by their email/id), not all users.
        new_hash = generate_password_hash(new_password)
        db = get_db()
        db.execute(
            "UPDATE users SET password_hash = ? WHERE id = ?",
            (new_hash, user.id)
        )
        db.commit()

        flash('Password updated successfully! Please sign in with your new password.', 'ok')
        return redirect(url_for('login'))

    # ── STEP 4: Show the "Set New Password" form (GET) ────────────────────
    # We pass the token into the template so the form can POST back to the
    # same URL (which includes the token).
    return render_template('reset_password.html', token=token)


@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if current_user.is_authenticated:
        return redirect(url_for('index'))

    if request.method == 'POST':
        username = (request.form.get('username') or '').strip()
        email    = (request.form.get('email') or '').strip().lower()
        password = request.form.get('password') or ''
        confirm  = request.form.get('confirm') or ''

        errs = validate_signup(username, email, password, confirm)
        if errs:
            for e in errs:
                flash(e, 'err')
            return render_template('signup.html', username=username, email=email), 400

        db = get_db()
        existing = db.execute(
            "SELECT 1 FROM users WHERE username = ? OR email = ?",
            (username, email),
        ).fetchone()
        if existing:
            flash('That username or email is already registered.', 'err')
            return render_template('signup.html', username=username, email=email), 409

        db.execute(
            "INSERT INTO users (username, email, password_hash) VALUES (?, ?, ?)",
            (username, email, generate_password_hash(password)),
        )
        db.commit()

        user = User.find(username)
        login_user(user)
        flash(f'Account created. Welcome aboard, {username}.', 'ok')
        return redirect(url_for('index'))

    return render_template('signup.html')


@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('You have been logged out.', 'info')
    return redirect(url_for('login'))

# ══════════════════════════════════════════════════════════════════════════
# DASHBOARD + API ROUTES (all protected)
# ══════════════════════════════════════════════════════════════════════════
@app.route('/')
@login_required
def index():
    return render_template('index.html')

@app.route('/api/upload', methods=['POST'])
@login_required
def upload():
    if 'file' not in request.files: return jsonify({'error':'No file'}), 400
    f = request.files['file']
    if not f.filename: return jsonify({'error':'No filename'}), 400
    ext = os.path.splitext(f.filename)[1].lower()
    if ext not in {'.log','.txt','.csv'}: return jsonify({'error':f'{ext} not supported'}), 400
    raw = f.read()
    try: content = raw.decode('utf-8')
    except: content = raw.decode('latin-1')
    fid = hashlib.md5(f"{f.filename}{time.time()}".encode()).hexdigest()[:12]
    db = get_db()
    db.execute("INSERT INTO uploads (id,file_name,file_size,status) VALUES (?,?,?,'processing')",
               (fid, f.filename, len(raw)))
    db.commit()
    entries = parse_file(content, f.filename)
    if not entries:
        db.execute("UPDATE uploads SET status='error' WHERE id=?", (fid,))
        db.commit()
        return jsonify({'error':'No valid log entries found'}), 400
    lc = Counter(e['level'] for e in entries)
    rows = [(fid,f.filename,e['line_number'],e['raw_text'],e['timestamp'],e['level'],
             e['source'],e['message'],e['ip_address'],e['error_code'],e['has_stack_trace']) for e in entries]
    db.executemany('''INSERT INTO raw_logs
        (file_id,file_name,line_number,raw_text,timestamp,level,source,message,ip_address,error_code,has_stack_trace)
        VALUES (?,?,?,?,?,?,?,?,?,?,?)''', rows)
    db.execute('''UPDATE uploads SET total_lines=?,error_count=?,warn_count=?,info_count=?,status='ready' WHERE id=?''',
               (len(entries), lc.get('ERROR',0)+lc.get('CRITICAL',0)+lc.get('FATAL',0),
                lc.get('WARN',0), lc.get('INFO',0)+lc.get('DEBUG',0), fid))
    db.commit()
    return jsonify({'file_id':fid,'file_name':f.filename,'total_entries':len(entries),'level_counts':dict(lc)})

@app.route('/api/logs/<fid>')
@login_required
def get_logs(fid):
    page = int(request.args.get('page',1))
    pp = int(request.args.get('per_page',50))
    lf = request.args.get('level','')
    db = get_db()
    q = "SELECT * FROM raw_logs WHERE file_id=?"
    params = [fid]
    if lf: q += " AND level=?"; params.append(lf.upper())
    total = db.execute("SELECT COUNT(*) FROM raw_logs WHERE file_id=?",[fid]).fetchone()[0]
    rows = db.execute(q+f" ORDER BY line_number LIMIT {pp} OFFSET {(page-1)*pp}", params).fetchall()
    return jsonify({'total':total,'page':page,'logs':[dict(r) for r in rows]})

@app.route('/api/preprocess/<fid>', methods=['POST'])
@login_required
def preprocess(fid):
    db = get_db()
    raws = db.execute("SELECT id,raw_text FROM raw_logs WHERE file_id=?",(fid,)).fetchall()
    if not raws: return jsonify({'error':'No logs found'}), 404
    db.execute("DELETE FROM processed_logs WHERE raw_id IN (SELECT id FROM raw_logs WHERE file_id=?)",(fid,))
    processed, texts = [], []
    for row in raws:
        cl, toks, rem = preprocess_text(row['raw_text'])
        texts.append(cl or 'unknown')
        processed.append({'raw_id':row['id'],'cleaned_text':cl,'tokens':json.dumps(toks),'token_count':len(toks),'removed_count':rem})
    scores, anomalies = run_anomaly_detection(texts)
    db.executemany('INSERT INTO processed_logs (raw_id,cleaned_text,tokens,token_count,removed_count,anomaly_score,is_anomaly) VALUES (?,?,?,?,?,?,?)',
                   [(p['raw_id'],p['cleaned_text'],p['tokens'],p['token_count'],p['removed_count'],scores[i],anomalies[i]) for i,p in enumerate(processed)])
    db.commit()
    all_toks = []
    for p in processed:
        all_toks.extend(json.loads(p['tokens']))
    return jsonify({'processed':len(processed),'anomaly_count':sum(anomalies),
                    'normal_count':len(processed)-sum(anomalies),
                    'top_tokens':Counter(all_toks).most_common(20),
                    'sample_before':raws[0]['raw_text'][:200],
                    'sample_after':processed[0]['cleaned_text'][:200]})

@app.route('/api/processed/<fid>')
@login_required
def get_processed(fid):
    page = int(request.args.get('page',1)); pp=30
    db = get_db()
    rows = db.execute('''SELECT p.*,r.raw_text,r.level,r.timestamp,r.source FROM processed_logs p
        JOIN raw_logs r ON r.id=p.raw_id WHERE r.file_id=? ORDER BY p.id LIMIT ? OFFSET ?''',
        (fid,pp,(page-1)*pp)).fetchall()
    total = db.execute('SELECT COUNT(*) FROM processed_logs p JOIN raw_logs r ON r.id=p.raw_id WHERE r.file_id=?',(fid,)).fetchone()[0]
    return jsonify({'total':total,'page':page,'logs':[dict(r) for r in rows]})

@app.route('/api/visualize/<fid>')
@login_required
def visualize(fid):
    db = get_db()
    levels = db.execute("SELECT level,COUNT(*) cnt FROM raw_logs WHERE file_id=? GROUP BY level ORDER BY cnt DESC",(fid,)).fetchall()
    rows = db.execute("SELECT timestamp,line_number FROM raw_logs WHERE file_id=? ORDER BY line_number",(fid,)).fetchall()
    total = len(rows); hourly = {}
    for row in rows:
        ts = row['timestamp']
        if ts:
            m = re.search(r'T?(\d{2}):\d{2}', ts)
            if m: h=int(m.group(1)); hourly[h]=hourly.get(h,0)+1
        else:
            b=min(int((row['line_number']/max(total,1))*24),23); hourly[b]=hourly.get(b,0)+1
    errs = db.execute("SELECT message FROM raw_logs WHERE file_id=? AND level IN ('ERROR','CRITICAL','FATAL')",(fid,)).fetchall()
    ewords = Counter()
    for r in errs:
        if r['message']:
            _,toks,_ = preprocess_text(r['message']); ewords.update(toks[:5])
    token_rows = db.execute('SELECT p.tokens FROM processed_logs p JOIN raw_logs r ON r.id=p.raw_id WHERE r.file_id=?',(fid,)).fetchall()
    all_toks = []
    for r in token_rows:
        if r['tokens']:
            try: all_toks.extend(json.loads(r['tokens']))
            except: pass
    anom = db.execute('SELECT COUNT(*) total, SUM(is_anomaly) anomalies FROM processed_logs p JOIN raw_logs r ON r.id=p.raw_id WHERE r.file_id=?',(fid,)).fetchone()
    sources = db.execute("SELECT COALESCE(source,'unknown') src,COUNT(*) cnt FROM raw_logs WHERE file_id=? GROUP BY src ORDER BY cnt DESC LIMIT 8",(fid,)).fetchall()
    top_anom = db.execute('''SELECT r.raw_text,r.level,p.anomaly_score FROM processed_logs p
        JOIN raw_logs r ON r.id=p.raw_id WHERE r.file_id=? AND p.is_anomaly=1
        ORDER BY p.anomaly_score ASC LIMIT 10''',(fid,)).fetchall()
    return jsonify({
        'level_distribution':{r['level']:r['cnt'] for r in levels},
        'hourly_trend':[{'hour':h,'count':hourly.get(h,0)} for h in range(24)],
        'top_errors':[{'word':w,'count':c} for w,c in ewords.most_common(10)],
        'word_frequency':[{'word':w,'count':c} for w,c in Counter(all_toks).most_common(25)],
        'anomaly_summary':{'total':anom['total'] or 0,'anomalies':anom['anomalies'] or 0,'normal':(anom['total'] or 0)-(anom['anomalies'] or 0)},
        'source_distribution':[dict(s) for s in sources],
        'top_anomalies':[dict(a) for a in top_anom],
    })

@app.route('/api/stats/<fid>')
@login_required
def stats(fid):
    db = get_db()
    u = db.execute("SELECT * FROM uploads WHERE id=?",(fid,)).fetchone()
    if not u: return jsonify({'error':'Not found'}),404
    pc = db.execute('SELECT COUNT(*) FROM processed_logs p JOIN raw_logs r ON r.id=p.raw_id WHERE r.file_id=?',(fid,)).fetchone()[0]
    return jsonify({**dict(u),'processed_count':pc})

@app.route('/api/uploads')
@login_required
def list_uploads():
    db = get_db()
    rows = db.execute("SELECT * FROM uploads ORDER BY uploaded_at DESC LIMIT 20").fetchall()
    return jsonify([dict(r) for r in rows])

@app.route('/api/delete/<fid>', methods=['DELETE'])
@login_required
def delete_file(fid):
    db = get_db()
    db.execute("DELETE FROM processed_logs WHERE raw_id IN (SELECT id FROM raw_logs WHERE file_id=?)",(fid,))
    db.execute("DELETE FROM raw_logs WHERE file_id=?",(fid,))
    db.execute("DELETE FROM uploads WHERE id=?",(fid,))
    db.commit()
    return jsonify({'success':True})

@app.route('/api/generate-sample', methods=['POST'])
@login_required
def generate_sample():
    import random
    levels_pool = ['INFO']*60+['WARN']*20+['ERROR']*15+['CRITICAL']*4+['DEBUG']*1
    sources = ['app-server-01','app-server-02','db-node-1','db-node-2','nginx-proxy',
               'auth-service','payment-svc','cache-layer','queue-worker','scheduler']
    messages = {
        'INFO':  ['User login successful user_id={}','Cache hit ratio: {:.1f}%',
                  'Request processed in {}ms','Health check passed uptime {}s','Batch job completed {} records'],
        'WARN':  ['Slow query detected {}ms exceeds threshold','Memory usage at {:.1f}% of limit',
                  'Retry attempt {} of 3 for request','Connection pool utilization {:.0f}%'],
        'ERROR': ['Connection timeout to host 192.168.{}.{}','NullPointerException at Service.java:{}',
                  'Failed to acquire DB lock after {}ms','SSL certificate verification failed NET-504',
                  'Queue consumer crashed restarting','Disk I/O error code IO-28'],
        'CRITICAL':['Disk usage {}% emergency cleanup required','DB replication lag {}s data loss risk'],
        'DEBUG': ['Processing request id={}','SQL query executed {} rows returned'],
    }
    lines = []
    base = datetime.now() - timedelta(hours=23)
    for i in range(500):
        ts = base + timedelta(minutes=i*2, seconds=random.randint(0,59))
        lv = random.choice(levels_pool)
        src = random.choice(sources)
        tpl = random.choice(messages[lv])
        try:
            msg = tpl.format(*[random.randint(1,999) for _ in range(5)], *[random.uniform(1,99) for _ in range(2)])
        except: msg = tpl
        line = f"{ts.strftime('%Y-%m-%dT%H:%M:%S.000Z')} [{lv}] [{src}] {msg}"
        if lv in ('ERROR','CRITICAL') and random.random()<0.3:
            line += f" at com.app.{src.replace('-','.')}.Service(Service.java:{random.randint(10,200)})"
        lines.append(line)
    content = '\n'.join(lines)
    fid = hashlib.md5(f"sample{time.time()}".encode()).hexdigest()[:12]
    db = get_db()
    db.execute("INSERT INTO uploads (id,file_name,file_size,status) VALUES (?,?,?,'processing')",(fid,'sample_logs.log',len(content)))
    db.commit()
    entries = parse_file(content,'sample_logs.log')
    lc = Counter(e['level'] for e in entries)
    rows = [(fid,'sample_logs.log',e['line_number'],e['raw_text'],e['timestamp'],e['level'],
             e['source'],e['message'],e['ip_address'],e['error_code'],e['has_stack_trace']) for e in entries]
    db.executemany('''INSERT INTO raw_logs
        (file_id,file_name,line_number,raw_text,timestamp,level,source,message,ip_address,error_code,has_stack_trace)
        VALUES (?,?,?,?,?,?,?,?,?,?,?)''', rows)
    db.execute('''UPDATE uploads SET total_lines=?,error_count=?,warn_count=?,info_count=?,status='ready' WHERE id=?''',
               (len(entries),lc.get('ERROR',0)+lc.get('CRITICAL',0),lc.get('WARN',0),lc.get('INFO',0),fid))
    db.commit()
    return jsonify({'file_id':fid,'file_name':'sample_logs.log','total_entries':len(entries),'level_counts':dict(lc)})

@app.route('/api/run-pipeline/<fid>', methods=['POST'])
@login_required
def run_pipeline(fid):
    """Run all 3 modules in sequence: Validate → BERT → RCA"""
    from module4_validation import run_validation
    from module5_bert import run_bert_training
    from module6_rca import run_rca

    v = run_validation(fid)
    if "error" in v: return jsonify({"step":"validation","error":v["error"]}), 400

    b = run_bert_training(fid)
    if "error" in b: return jsonify({"step":"bert","error":b["error"]}), 400

    r = run_rca(fid)
    if "error" in r: return jsonify({"step":"rca","error":r["error"]}), 400

    return jsonify({
        "status":  "success",
        "file_id": fid,
        "summary": {
            "cv_accuracy":   v.get("cv_mean_acc"),
            "bert_accuracy": b.get("accuracy"),
            "bert_f1":       b.get("f1_score"),
            "health_score":  r["outcome_summary"].get("health_score"),
            "top_root_cause":r["outcome_summary"].get("top_root_cause"),
            "anomaly_rate":  r["outcome_summary"].get("anomaly_rate"),
        }
    })

if __name__ == '__main__':
    with app.app_context():
        init_db()
    app.run(debug=True, host='0.0.0.0', port=5000)
