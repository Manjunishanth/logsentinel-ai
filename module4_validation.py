"""
=============================================================
MODULE 4 — DATASET VALIDATION  (Fixed for LogSentinel app.py)
=============================================================
FIXES APPLIED vs original:
  1. No file_id in processed_logs → JOIN with raw_logs to filter
  2. Column is_anomaly (not anomaly_flag)
  3. raw_id used as log identity (not id)
  4. DB_PATH matches app.py absolute path style
=============================================================
"""

import json, math, random, sqlite3, os
from collections import Counter
from flask import Blueprint, jsonify

validation_bp = Blueprint("validation", __name__)

# ── Match app.py DB path style ──
DB_PATH = os.path.join(os.path.dirname(__file__), 'logsentinel.db')

# ─────────────────────────────────────────────
# DB HELPER  (standalone — does NOT use Flask g)
# ─────────────────────────────────────────────
def _db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def ensure_validation_tables():
    conn = _db()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS validated_splits (
            id       INTEGER PRIMARY KEY AUTOINCREMENT,
            file_id  TEXT NOT NULL,
            raw_id   INTEGER NOT NULL,
            split    TEXT NOT NULL,
            fold     INTEGER DEFAULT 0,
            label    INTEGER NOT NULL,
            tokens   TEXT
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
    """)
    conn.commit(); conn.close()


# ─────────────────────────────────────────────
# LOAD — JOIN processed_logs + raw_logs
# fixes: no file_id column in processed_logs
#        column name is is_anomaly not anomaly_flag
# ─────────────────────────────────────────────
def load_processed_logs(file_id: str):
    conn = _db()
    rows = conn.execute("""
        SELECT p.id        AS proc_id,
               p.raw_id    AS raw_id,
               p.tokens    AS tokens,
               p.is_anomaly AS label
        FROM processed_logs p
        JOIN raw_logs r ON r.id = p.raw_id
        WHERE r.file_id = ?
        ORDER BY p.id
    """, (file_id,)).fetchall()
    conn.close()
    return [dict(r) for r in rows]


# ─────────────────────────────────────────────
# STRATIFIED 80/20 SPLIT
# ─────────────────────────────────────────────
def stratified_split(rows, test_ratio=0.2, seed=42):
    random.seed(seed)
    normal  = [r for r in rows if r["label"] == 0]
    anomaly = [r for r in rows if r["label"] == 1]
    random.shuffle(normal); random.shuffle(anomaly)

    n_test_n = max(1, int(len(normal)  * test_ratio))
    n_test_a = max(1, int(len(anomaly) * test_ratio))

    test  = normal[:n_test_n]  + anomaly[:n_test_a]
    train = normal[n_test_n:]  + anomaly[n_test_a:]
    random.shuffle(train); random.shuffle(test)
    return train, test


# ─────────────────────────────────────────────
# PURE-PYTHON TF-IDF VOCAB
# ─────────────────────────────────────────────
def get_tokens(row):
    try:
        t = json.loads(row["tokens"]) if row["tokens"] else []
        return t if isinstance(t, list) else []
    except Exception:
        return str(row["tokens"]).split() if row["tokens"] else []


def build_tfidf_vocab(rows, max_features=500):
    docs = [get_tokens(r) for r in rows]
    N    = len(docs)
    df   = Counter()
    for toks in docs:
        df.update(set(toks))
    idf = {t: math.log((1 + N) / (1 + df[t])) + 1.0 for t in df}
    top = sorted(idf, key=idf.get, reverse=True)[:max_features]
    vocab = {t: i for i, t in enumerate(top)}
    return vocab, idf


def tfidf_vec(toks, vocab, idf):
    tf  = Counter(toks)
    tot = sum(tf.values()) or 1
    vec = [0.0] * len(vocab)
    for t, idx in vocab.items():
        if t in tf:
            vec[idx] = (tf[t] / tot) * idf.get(t, 1.0)
    norm = math.sqrt(sum(v*v for v in vec)) or 1.0
    return [v / norm for v in vec]


# ─────────────────────────────────────────────
# LIGHTWEIGHT ISOLATION FOREST (pure Python)
# ─────────────────────────────────────────────
class _IsoTree:
    __slots__ = ["feat","thresh","left","right","size","leaf"]
    def __init__(self): self.feat=self.thresh=self.left=self.right=None; self.size=0; self.leaf=False

def _build(X, max_d, d=0):
    nd = _IsoTree(); nd.size = len(X)
    if d >= max_d or len(X) <= 1: nd.leaf = True; return nd
    nf = len(X[0])
    f  = random.randint(0, nf-1)
    vals = [x[f] for x in X]
    lo, hi = min(vals), max(vals)
    if lo == hi: nd.leaf = True; return nd
    sp = random.uniform(lo, hi)
    nd.feat = f; nd.thresh = sp
    nd.left  = _build([x for x in X if x[f] <  sp], max_d, d+1)
    nd.right = _build([x for x in X if x[f] >= sp], max_d, d+1)
    return nd

def _path(nd, x, c=0):
    if nd.leaf or nd.feat is None:
        n = nd.size
        if n <= 1: return c
        h = math.log(n-1) + 0.5772156649
        return c + 2*h - 2*(n-1)/n
    return _path(nd.left if x[nd.feat] < nd.thresh else nd.right, x, c+1)

class LightIF:
    def __init__(self, n=50, s=256, cont=0.1, seed=42):
        self.n=n; self.s=s; self.cont=cont; self.seed=seed
        self.trees=[]; self.thresh=None
    def fit(self, X):
        random.seed(self.seed)
        md = math.ceil(math.log2(self.s)) if self.s > 1 else 1
        for _ in range(self.n):
            samp = random.sample(X, min(self.s, len(X)))
            self.trees.append(_build(samp, md))
        sc = self._scores(X)
        self.thresh = sorted(sc)[max(1, int(len(X)*self.cont))]
        return self
    def _c(self, n):
        return 2*(math.log(n-1)+0.5772156649) - 2*(n-1)/n if n > 1 else 1.0
    def _scores(self, X):
        c = self._c(self.s)
        return [round(2**(-sum(_path(t,x) for t in self.trees)/len(self.trees)/c), 4) for x in X]
    def predict(self, X):
        if self.thresh is None: raise ValueError("fit() first")
        return [1 if s > self.thresh else 0 for s in self._scores(X)]


# ─────────────────────────────────────────────
# 5-FOLD CROSS VALIDATION
# ─────────────────────────────────────────────
def cross_validate(rows, vocab, idf, folds=5):
    random.seed(42)
    data = rows[:]
    random.shuffle(data)
    fs   = len(data) // folds
    scores = []
    for k in range(folds):
        te = data[k*fs:(k+1)*fs]
        tr = data[:k*fs] + data[(k+1)*fs:]
        try:
            Xtr = [tfidf_vec(get_tokens(r), vocab, idf) for r in tr]
            Xte = [tfidf_vec(get_tokens(r), vocab, idf) for r in te]
            ytr = [r["label"] for r in tr]   # not used by unsupervised IF
            yte = [r["label"] for r in te]
            m   = LightIF(n=30, s=min(128, len(Xtr)), cont=0.1)
            m.fit(Xtr)
            preds = m.predict(Xte)
            acc   = sum(p==y for p,y in zip(preds, yte)) / len(yte)
            scores.append(round(acc, 4))
        except Exception as e:
            scores.append(0.0)
    return scores


# ─────────────────────────────────────────────
# MAIN VALIDATION
# ─────────────────────────────────────────────
def run_validation(file_id: str) -> dict:
    ensure_validation_tables()

    rows = load_processed_logs(file_id)
    if not rows:
        return {"error": f"No processed logs for file_id={file_id}. Run /api/preprocess/{file_id} first."}

    total   = len(rows)
    normal  = sum(1 for r in rows if r["label"] == 0)
    anomaly = sum(1 for r in rows if r["label"] == 1)
    ratio   = round(anomaly / total * 100, 2) if total else 0

    print(f"[Module 4] Total={total} | Normal={normal} | Anomaly={anomaly} ({ratio}%)")

    train_rows, test_rows = stratified_split(rows, test_ratio=0.2)
    vocab, idf = build_tfidf_vocab(rows, max_features=500)
    vocab_size = len(vocab)

    print(f"[Module 4] Train={len(train_rows)} | Test={len(test_rows)} | Vocab={vocab_size}")
    print(f"[Module 4] Running 5-fold cross-validation...")

    cv_scores = cross_validate(rows, vocab, idf, folds=5)
    cv_mean   = round(sum(cv_scores)/len(cv_scores), 4)
    cv_std    = round(math.sqrt(sum((s-cv_mean)**2 for s in cv_scores)/len(cv_scores)), 4)

    print(f"[Module 4] CV Accuracy: {cv_mean:.4f} ± {cv_std:.4f}  |  Folds: {cv_scores}")

    all_toks = Counter()
    for r in rows:
        all_toks.update(get_tokens(r))
    top_tokens = dict(all_toks.most_common(20))

    # Save splits — use raw_id as the log identity
    conn = _db()
    conn.execute("DELETE FROM validated_splits WHERE file_id=?", (file_id,))
    insert_data = []
    for r in train_rows:
        insert_data.append((file_id, r["raw_id"], "train", 0, r["label"], r["tokens"]))
    for r in test_rows:
        insert_data.append((file_id, r["raw_id"], "test",  0, r["label"], r["tokens"]))
    conn.executemany(
        "INSERT INTO validated_splits (file_id,raw_id,split,fold,label,tokens) VALUES (?,?,?,?,?,?)",
        insert_data
    )
    conn.execute("DELETE FROM validation_reports WHERE file_id=?", (file_id,))
    conn.execute("""
        INSERT INTO validation_reports
            (file_id,total_logs,normal_count,anomaly_count,class_ratio,
             train_size,test_size,cv_mean_acc,cv_std_acc,cv_scores,top_tokens,vocab_size)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
    """, (file_id,total,normal,anomaly,ratio,
          len(train_rows),len(test_rows),
          cv_mean,cv_std,json.dumps(cv_scores),
          json.dumps(top_tokens),vocab_size))
    conn.commit(); conn.close()

    print(f"[Module 4] ✅ Saved to DB")
    return {
        "file_id":file_id,"total_logs":total,"normal_count":normal,
        "anomaly_count":anomaly,"class_ratio":ratio,
        "train_size":len(train_rows),"test_size":len(test_rows),
        "cv_mean_acc":cv_mean,"cv_std_acc":cv_std,
        "cv_scores":cv_scores,"top_tokens":top_tokens,"vocab_size":vocab_size,
    }


# ─────────────────────────────────────────────
# REST ENDPOINTS
# ─────────────────────────────────────────────
@validation_bp.route("/api/validate/<file_id>", methods=["POST"])
def api_validate(file_id):
    r = run_validation(file_id)
    if "error" in r: return jsonify(r), 400
    return jsonify({"status":"success","report":r}), 200

@validation_bp.route("/api/validation-report/<file_id>", methods=["GET"])
def api_validation_report(file_id):
    conn = _db()
    row  = conn.execute("SELECT * FROM validation_reports WHERE file_id=?", (file_id,)).fetchone()
    conn.close()
    if not row: return jsonify({"error":"No report. Run POST /api/validate/<file_id> first."}), 404
    r = dict(row)
    r["cv_scores"]  = json.loads(r["cv_scores"]  or "[]")
    r["top_tokens"] = json.loads(r["top_tokens"] or "{}")
    return jsonify(r), 200

@validation_bp.route("/api/validated-splits/<file_id>", methods=["GET"])
def api_splits(file_id):
    conn = _db()
    def cnt(split, label=None):
        q = "SELECT COUNT(*) FROM validated_splits WHERE file_id=? AND split=?"
        p = [file_id, split]
        if label is not None: q += " AND label=?"; p.append(label)
        return conn.execute(q, p).fetchone()[0]
    result = {
        "file_id": file_id,
        "train_total":   cnt("train"),
        "train_anomaly": cnt("train", 1),
        "train_normal":  cnt("train", 0),
        "test_total":    cnt("test"),
        "test_anomaly":  cnt("test",  1),
        "test_normal":   cnt("test",  0),
    }
    conn.close()
    return jsonify(result), 200
