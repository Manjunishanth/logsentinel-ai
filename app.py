"""
LogSentinel AI — Real Backend
Flask + SQLite + Scikit-learn (TF-IDF + Isolation Forest)
"""
import os, re, json, csv, io, math, time, sqlite3, hashlib
from datetime import datetime, timedelta
from collections import Counter
from flask import Flask, request, jsonify, render_template, g
from module4_validation import validation_bp
from module5_bert import bert_bp
from module6_rca import rca_bp
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import IsolationForest
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024
app.register_blueprint(validation_bp)
app.register_blueprint(bert_bp)
app.register_blueprint(rca_bp)
DB_PATH = os.path.join(os.path.dirname(__file__), 'logsentinel.db')

# ── PATTERNS ──
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

# ── DB ──
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
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            file_id      TEXT NOT NULL UNIQUE,
            total_logs   INTEGER,
            normal_count INTEGER,
            anomaly_count INTEGER,
            class_ratio  REAL,
            train_size   INTEGER,
            test_size    INTEGER,
            cv_mean_acc  REAL,
            cv_std_acc   REAL,
            cv_scores    TEXT,
            top_tokens   TEXT,
            vocab_size   INTEGER,
            created_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS validated_splits (
            id       INTEGER PRIMARY KEY AUTOINCREMENT,
            file_id  TEXT NOT NULL,
            raw_id   INTEGER NOT NULL,
            split    TEXT NOT NULL,
            fold     INTEGER DEFAULT 0,
            label    INTEGER NOT NULL,
            tokens   TEXT
        );
        CREATE TABLE IF NOT EXISTS bert_results (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            file_id      TEXT NOT NULL UNIQUE,
            mode         TEXT,
            accuracy     REAL,
            precision_s  REAL,
            recall       REAL,
            f1_score     REAL,
            confusion_matrix TEXT,
            epochs       INTEGER,
            train_losses TEXT,
            model_path   TEXT,
            created_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS bert_predictions (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            file_id      TEXT NOT NULL,
            raw_id       INTEGER,
            cleaned_text TEXT,
            prediction   INTEGER,
            confidence   REAL,
            bert_score   REAL,
            created_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS rca_reports (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            file_id TEXT NOT NULL UNIQUE,
            total_predicted INTEGER,
            correct_preds INTEGER,
            wrong_preds INTEGER,
            match_accuracy REAL,
            health_score REAL,
            root_causes TEXT,
            outcome_summary TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS root_cause_entries (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            file_id TEXT NOT NULL,
            rank INTEGER,
            cause_id TEXT,
            cause_label TEXT,
            frequency INTEGER,
            confidence REAL,
            severity TEXT,
            evidence TEXT,
            first_seen TEXT,
            last_seen TEXT
        );
    ''')
    conn.commit()

    # ── Migrate existing DB: add missing columns if tables already existed ──
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
        ("validation_reports","total_logs",    "ALTER TABLE validation_reports ADD COLUMN total_logs INTEGER"),
        ("validation_reports","normal_count",  "ALTER TABLE validation_reports ADD COLUMN normal_count INTEGER"),
        ("validation_reports","anomaly_count", "ALTER TABLE validation_reports ADD COLUMN anomaly_count INTEGER"),
        ("validation_reports","class_ratio",   "ALTER TABLE validation_reports ADD COLUMN class_ratio REAL"),
        ("validation_reports","cv_scores",     "ALTER TABLE validation_reports ADD COLUMN cv_scores TEXT"),
        ("validation_reports","vocab_size",    "ALTER TABLE validation_reports ADD COLUMN vocab_size INTEGER"),
    ]
    for table, col, sql in migrations:
        try:
            conn.execute(sql)
        except Exception:
            pass  # column already exists
    conn.commit()
    conn.close()


# ── PREPROCESS ──
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

# ── ROUTES ──
@app.route('/')
def index(): return render_template('index.html')

@app.route('/api/upload', methods=['POST'])
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
def get_processed(fid):
    page = int(request.args.get('page',1)); pp=30
    db = get_db()
    rows = db.execute('''SELECT p.*,r.raw_text,r.level,r.timestamp,r.source FROM processed_logs p
        JOIN raw_logs r ON r.id=p.raw_id WHERE r.file_id=? ORDER BY p.id LIMIT ? OFFSET ?''',
        (fid,pp,(page-1)*pp)).fetchall()
    total = db.execute('SELECT COUNT(*) FROM processed_logs p JOIN raw_logs r ON r.id=p.raw_id WHERE r.file_id=?',(fid,)).fetchone()[0]
    return jsonify({'total':total,'page':page,'logs':[dict(r) for r in rows]})

@app.route('/api/visualize/<fid>')
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
def stats(fid):
    db = get_db()
    u = db.execute("SELECT * FROM uploads WHERE id=?",(fid,)).fetchone()
    if not u: return jsonify({'error':'Not found'}),404
    pc = db.execute('SELECT COUNT(*) FROM processed_logs p JOIN raw_logs r ON r.id=p.raw_id WHERE r.file_id=?',(fid,)).fetchone()[0]
    return jsonify({**dict(u),'processed_count':pc})

@app.route('/api/uploads')
def list_uploads():
    db = get_db()
    rows = db.execute("SELECT * FROM uploads ORDER BY uploaded_at DESC LIMIT 20").fetchall()
    return jsonify([dict(r) for r in rows])

@app.route('/api/delete/<fid>', methods=['DELETE'])
def delete_file(fid):
    db = get_db()
    db.execute("DELETE FROM processed_logs WHERE raw_id IN (SELECT id FROM raw_logs WHERE file_id=?)",(fid,))
    db.execute("DELETE FROM raw_logs WHERE file_id=?",(fid,))
    db.execute("DELETE FROM uploads WHERE id=?",(fid,))
    db.commit()
    return jsonify({'success':True})

@app.route('/api/generate-sample', methods=['POST'])
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

if __name__ == '__main__':
    with app.app_context(): init_db()
    app.run(debug=True, host='0.0.0.0', port=5000)
