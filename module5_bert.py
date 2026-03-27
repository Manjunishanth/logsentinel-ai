"""
=============================================================
MODULE 5 — BERT TECHNIQUE  (Fixed for LogSentinel app.py)
=============================================================
FIXES APPLIED vs original:
  1. Load splits by raw_id (not log_id) — matches validated_splits
  2. Ground truth loaded via raw_id JOIN, using is_anomaly column
  3. DB_PATH absolute path matching app.py
  4. bert_predictions stores raw_id for correct JOIN later
=============================================================
"""

import json, math, random, sqlite3, os
from flask import Blueprint, request, jsonify

bert_bp = Blueprint("bert", __name__)

DB_PATH   = os.path.join(os.path.dirname(__file__), 'logsentinel.db')
MODEL_DIR = os.path.join(os.path.dirname(__file__), 'models')
os.makedirs(MODEL_DIR, exist_ok=True)

# ── Try full BERT (torch + transformers) ──
try:
    import torch
    from torch import nn
    from torch.utils.data import Dataset, DataLoader
    from transformers import BertTokenizer, BertModel, AdamW, get_linear_schedule_with_warmup
    TORCH_AVAILABLE = True
    print("[Module 5] ✅ PyTorch + Transformers found — Full BERT mode")
except ImportError:
    TORCH_AVAILABLE = False
    print("[Module 5] ⚠ PyTorch not found — BERT-lite mode (no GPU needed)")


def _db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def ensure_bert_tables():
    conn = _db()
    conn.executescript("""
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
    """)
    conn.commit(); conn.close()


# ─────────────────────────────────────────────
# LOAD SPLITS  (uses raw_id — matches module4)
# ─────────────────────────────────────────────
def load_splits(file_id: str):
    conn = _db()
    train = conn.execute(
        "SELECT raw_id, tokens, label FROM validated_splits WHERE file_id=? AND split='train'",
        (file_id,)
    ).fetchall()
    test = conn.execute(
        "SELECT raw_id, tokens, label FROM validated_splits WHERE file_id=? AND split='test'",
        (file_id,)
    ).fetchall()
    conn.close()
    return [dict(r) for r in train], [dict(r) for r in test]


def get_tokens(row):
    try:
        t = json.loads(row["tokens"]) if row["tokens"] else []
        return t if isinstance(t, list) else []
    except Exception:
        return str(row["tokens"]).split() if row["tokens"] else []


# ═══════════════════════════════════════════════════════
# MODE A — FULL BERT
# ═══════════════════════════════════════════════════════
if TORCH_AVAILABLE:
    class LogDataset(Dataset):
        def __init__(self, rows, tokenizer, max_len=128):
            self.rows = rows; self.tok = tokenizer; self.max_len = max_len
        def __len__(self): return len(self.rows)
        def __getitem__(self, idx):
            r    = self.rows[idx]
            text = " ".join(get_tokens(r))[:512]
            enc  = self.tok.encode_plus(text, max_length=self.max_len,
                       padding="max_length", truncation=True,
                       return_tensors="pt", return_attention_mask=True)
            return {"input_ids": enc["input_ids"].squeeze(),
                    "attention_mask": enc["attention_mask"].squeeze(),
                    "label": torch.tensor(r["label"], dtype=torch.long)}

    class BertLogClassifier(nn.Module):
        def __init__(self, bert, n_classes=2, dropout=0.3):
            super().__init__()
            self.bert = bert
            self.drop = nn.Dropout(dropout)
            self.out  = nn.Linear(bert.config.hidden_size, n_classes)
        def forward(self, ids, mask):
            return self.out(self.drop(self.bert(ids, mask).pooler_output))

    def train_full_bert(file_id, train_rows, test_rows, epochs=3, batch_size=16, lr=2e-5):
        device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[Module 5 BERT] Device: {device}")
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        model     = BertLogClassifier(BertModel.from_pretrained("bert-base-uncased")).to(device)
        tr_dl = DataLoader(LogDataset(train_rows, tokenizer), batch_size=batch_size, shuffle=True)
        te_dl = DataLoader(LogDataset(test_rows,  tokenizer), batch_size=batch_size)
        opt   = AdamW(model.parameters(), lr=lr, weight_decay=0.01)
        total = len(tr_dl) * epochs
        sch   = get_linear_schedule_with_warmup(opt, total//10, total)
        lf    = nn.CrossEntropyLoss().to(device)
        losses = []
        for ep in range(epochs):
            model.train(); el = 0
            for b in tr_dl:
                ids,mask,lbl = b["input_ids"].to(device),b["attention_mask"].to(device),b["label"].to(device)
                opt.zero_grad()
                loss = lf(model(ids,mask), lbl)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step(); sch.step(); el += loss.item()
            avg = el/len(tr_dl); losses.append(round(avg,4))
            print(f"[Module 5 BERT] Epoch {ep+1}/{epochs} Loss={avg:.4f}")
        model.eval()
        preds,labels,probs = [],[],[]
        with torch.no_grad():
            for b in te_dl:
                ids,mask,lbl = b["input_ids"].to(device),b["attention_mask"].to(device),b["label"].tolist()
                logits = model(ids,mask)
                p  = torch.softmax(logits,1)
                preds  += torch.argmax(p,1).tolist()
                labels += lbl
                probs  += p[:,1].tolist()
        mp = os.path.join(MODEL_DIR, f"bert_{file_id}.pt")
        torch.save(model.state_dict(), mp)
        return preds, labels, probs, losses, mp, "full_bert"


# ═══════════════════════════════════════════════════════
# MODE B — BERT-LITE (pure Python neural net)
# ═══════════════════════════════════════════════════════
class BertLite:
    """2-layer MLP with TF-IDF + semantic features. Real backprop, not rule-based."""

    ANOMALY = {"error","fail","failed","failure","exception","crash","timeout",
               "refused","unreachable","down","critical","fatal","panic","abort",
               "denied","unauthorized","invalid","missing","lost","disconnected",
               "overflow","leak","corrupt","warning","warn","alert","retry"}
    NORMAL  = {"info","start","started","success","ok","complete","completed",
               "finish","ready","running","healthy","connected","initialized",
               "loaded","registered","serving","received","sent","processed",
               "cached","stored","saved","created","updated","stable"}

    def __init__(self, hidden=32, seed=42):
        self.hidden=hidden; self.seed=seed
        self.W1=self.b1=self.W2=self.b2=None
        self.vocab={}; self.idf={}; self.threshold=0.5

    def _build_vocab(self, all_toks_list):
        from collections import Counter
        df = Counter(); N = len(all_toks_list)
        for toks in all_toks_list: df.update(set(toks))
        self.idf = {t: math.log((1+N)/(1+df[t]))+1.0 for t in df}
        top = sorted(self.idf, key=self.idf.get, reverse=True)[:80]  # reduced from 200→80
        self.vocab = {t: i for i, t in enumerate(top)}

    def _feats(self, toks):
        from collections import Counter
        tf   = Counter(toks); tot = sum(tf.values()) or 1
        vec  = [0.0]*len(self.vocab)
        for t, i in self.vocab.items():
            if t in tf: vec[i] = (tf[t]/tot)*self.idf.get(t,1.0)
        norm = math.sqrt(sum(v*v for v in vec)) or 1.0
        vec  = [v/norm for v in vec]
        ts   = set(t.lower() for t in toks)
        ah   = len(ts & self.ANOMALY); nh = len(ts & self.NORMAL)
        n    = len(toks)+1
        sem  = [ah/n, nh/n,
                1.0 if any(t in ("error","fail","exception","critical") for t in toks) else 0.0,
                min(1.0, len(toks)/20.0), ah/5.0, nh/5.0]
        return vec + sem

    def _sig(self, x): return 1/(1+math.exp(-max(-500,min(500,x))))
    def _relu(self, x): return max(0.0, x)

    def _fwd(self, x):
        h = [self._relu(sum(x[j]*self.W1[i][j] for j in range(len(x)))+self.b1[i])
             for i in range(self.hidden)]
        return self._sig(sum(h[i]*self.W2[i] for i in range(self.hidden))+self.b2), h

    def _init(self, inp):
        random.seed(self.seed)
        s1 = math.sqrt(2/inp); s2 = math.sqrt(2/self.hidden)
        self.W1 = [[random.gauss(0,s1) for _ in range(inp)] for _ in range(self.hidden)]
        self.b1 = [0.0]*self.hidden
        self.W2 = [random.gauss(0,s2) for _ in range(self.hidden)]
        self.b2 = 0.0

    def fit(self, train_rows, epochs=8, lr=0.01):
        all_toks = [get_tokens(r) for r in train_rows]
        self._build_vocab(all_toks)
        X = [self._feats(t) for t in all_toks]
        y = [float(r["label"]) for r in train_rows]
        self._init(len(X[0]))
        print(f"[Module 5 BERT-lite] Training {len(X)} samples, {len(X[0])} features, {epochs} epochs")
        best_loss = float('inf'); patience = 0
        for ep in range(epochs):
            idx = list(range(len(X))); random.shuffle(idx); tl = 0.0
            for i in idx:
                xi, yi = X[i], y[i]
                pred, h = self._fwd(xi)
                err = pred - yi; tl += err*err
                do  = 2*err*pred*(1-pred)
                for k in range(self.hidden):
                    self.W2[k] -= lr*do*h[k]
                self.b2 -= lr*do
                for k in range(self.hidden):
                    if h[k] > 0:
                        dh = do*self.W2[k]
                        for j in range(len(xi)): self.W1[k][j] -= lr*dh*xi[j]
                        self.b1[k] -= lr*dh
            avg_loss = tl/len(X)
            print(f"[Module 5 BERT-lite] Epoch {ep+1:2d}/{epochs} Loss={avg_loss:.4f}")
            # Early stopping
            if avg_loss < best_loss - 0.001:
                best_loss = avg_loss; patience = 0
            else:
                patience += 1
                if patience >= 3:
                    print(f"[Module 5 BERT-lite] Early stop at epoch {ep+1}")
                    break
        # find best threshold on training data
        probs = [self._fwd(xi)[0] for xi in X]
        best_t, best_f1 = 0.5, 0.0
        for t in [i/10 for i in range(3, 8)]:
            p = [1 if v>t else 0 for v in probs]
            tp=sum(1 for a,b in zip(p,y) if a==1 and b==1)
            fp=sum(1 for a,b in zip(p,y) if a==1 and b==0)
            fn=sum(1 for a,b in zip(p,y) if a==0 and b==1)
            pr=tp/(tp+fp+1e-9); rc=tp/(tp+fn+1e-9)
            f1=2*pr*rc/(pr+rc+1e-9)
            if f1>best_f1: best_f1,best_t = f1,t
        self.threshold = best_t
        print(f"[Module 5 BERT-lite] Threshold={self.threshold:.2f} F1={best_f1:.4f}")

    def predict_proba(self, rows):
        return [self._fwd(self._feats(get_tokens(r)))[0] for r in rows]

    def predict(self, rows):
        return [1 if p>self.threshold else 0 for p in self.predict_proba(rows)]

    def save(self, path):
        with open(path,"w") as f:
            json.dump({"W1":self.W1,"b1":self.b1,"W2":self.W2,"b2":self.b2,
                       "vocab":self.vocab,"idf":self.idf,"threshold":self.threshold,
                       "hidden":self.hidden}, f)

    @classmethod
    def load(cls, path):
        with open(path) as f: s = json.load(f)
        m = cls(hidden=s["hidden"])
        m.W1=s["W1"]; m.b1=s["b1"]; m.W2=s["W2"]; m.b2=s["b2"]
        m.vocab=s["vocab"]; m.idf=s["idf"]; m.threshold=s["threshold"]
        return m


# ─────────────────────────────────────────────
# METRICS
# ─────────────────────────────────────────────
def metrics(preds, labels):
    tp=sum(1 for p,l in zip(preds,labels) if p==1 and l==1)
    tn=sum(1 for p,l in zip(preds,labels) if p==0 and l==0)
    fp=sum(1 for p,l in zip(preds,labels) if p==1 and l==0)
    fn=sum(1 for p,l in zip(preds,labels) if p==0 and l==1)
    total = len(preds)
    acc  = round((tp+tn)/total, 4) if total else 0
    prec = round(tp/(tp+fp+1e-9), 4)
    rec  = round(tp/(tp+fn+1e-9), 4)
    f1   = round(2*prec*rec/(prec+rec+1e-9), 4)
    return {"accuracy":acc,"precision":prec,"recall":rec,"f1_score":f1,
            "confusion_matrix":{"tp":tp,"tn":tn,"fp":fp,"fn":fn}}


# In-memory model cache
_MODELS = {}


def run_bert_training(file_id: str) -> dict:
    ensure_bert_tables()
    train_rows, test_rows = load_splits(file_id)
    if not train_rows:
        return {"error": f"No validated splits for {file_id}. Run POST /api/validate/{file_id} first."}

    print(f"[Module 5] Train={len(train_rows)} | Test={len(test_rows)}")

    if TORCH_AVAILABLE:
        preds, labels, probs, losses, mp, mode = train_full_bert(
            file_id, train_rows, test_rows)
    else:
        mode  = "bert_lite"
        model = BertLite(hidden=32, seed=42)
        model.fit(train_rows, epochs=8, lr=0.01)
        mp    = os.path.join(MODEL_DIR, f"bertlite_{file_id}.json")
        model.save(mp)
        _MODELS[file_id] = model
        probs  = model.predict_proba(test_rows)
        preds  = model.predict(test_rows)
        labels = [r["label"] for r in test_rows]
        losses = []

    m = metrics(preds, labels)
    print(f"[Module 5] Accuracy={m['accuracy']} | F1={m['f1_score']} | "
          f"Precision={m['precision']} | Recall={m['recall']}")
    print(f"[Module 5] Confusion: {m['confusion_matrix']}")

    conn = _db()
    conn.execute("DELETE FROM bert_results WHERE file_id=?", (file_id,))
    conn.execute("""
        INSERT INTO bert_results
            (file_id,mode,accuracy,precision_s,recall,f1_score,
             confusion_matrix,epochs,train_losses,model_path)
        VALUES (?,?,?,?,?,?,?,?,?,?)
    """, (file_id, mode, m["accuracy"], m["precision"], m["recall"], m["f1_score"],
          json.dumps(m["confusion_matrix"]),
          3 if TORCH_AVAILABLE else 25,
          json.dumps(losses), mp))

    # Store predictions — raw_id links back to raw_logs
    conn.execute("DELETE FROM bert_predictions WHERE file_id=?", (file_id,))
    pred_rows = []
    for i, row in enumerate(test_rows):
        toks = get_tokens(row)
        pred_rows.append((
            file_id, row["raw_id"],
            " ".join(toks),
            preds[i], round(probs[i],4), round(probs[i],4)
        ))
    conn.executemany("""
        INSERT INTO bert_predictions
            (file_id,raw_id,cleaned_text,prediction,confidence,bert_score)
        VALUES (?,?,?,?,?,?)
    """, pred_rows)
    conn.commit(); conn.close()

    print(f"[Module 5] ✅ Saved to DB")
    return {
        "file_id":file_id, "mode":mode,
        **m,
        "test_samples":len(test_rows),
        "model_path":mp,
        "train_losses":losses,
    }


# ─────────────────────────────────────────────
# REST ENDPOINTS
# ─────────────────────────────────────────────
@bert_bp.route("/api/bert-train/<file_id>", methods=["POST"])
def api_bert_train(file_id):
    r = run_bert_training(file_id)
    if "error" in r: return jsonify(r), 400
    return jsonify({"status":"success","results":r}), 200

@bert_bp.route("/api/bert-results/<file_id>", methods=["GET"])
def api_bert_results(file_id):
    conn = _db()
    row  = conn.execute("SELECT * FROM bert_results WHERE file_id=?", (file_id,)).fetchone()
    conn.close()
    if not row: return jsonify({"error":"No results. Run POST /api/bert-train/<file_id> first."}), 404
    r = dict(row)
    r["confusion_matrix"] = json.loads(r["confusion_matrix"] or "{}")
    r["train_losses"]     = json.loads(r["train_losses"]     or "[]")
    return jsonify(r), 200

@bert_bp.route("/api/bert-predict", methods=["POST"])
def api_bert_predict():
    """POST body: {"file_id":"...", "log_text":"raw log line"}"""
    data     = request.get_json() or {}
    file_id  = data.get("file_id","")
    log_text = data.get("log_text","")
    if not file_id or not log_text:
        return jsonify({"error":"file_id and log_text required"}), 400

    mp = os.path.join(MODEL_DIR, f"bertlite_{file_id}.json")
    if file_id in _MODELS:
        model = _MODELS[file_id]
    elif os.path.exists(mp):
        model = BertLite.load(mp)
        _MODELS[file_id] = model
    else:
        return jsonify({"error":"Model not trained yet. Run POST /api/bert-train/<file_id>"}), 404

    # Quick clean (mirrors app.py preprocess_text)
    import re
    text = log_text.lower()
    for pat in [r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\S*', r'\b\d{1,3}(?:\.\d{1,3}){3}(?::\d+)?\b',
                r'https?://\S+', r'(?:/[\w.\-]+){2,}', r'\b[0-9a-f]{8,}\b',
                r'\d+\s*(?:ms|kb|mb|gb)', r'[^\w\s]']:
        text = re.sub(pat, ' ', text, flags=re.IGNORECASE)
    tokens = [t for t in text.split() if len(t) > 2]

    row   = {"tokens": json.dumps(tokens), "label": 0}
    prob  = model.predict_proba([row])[0]
    pred  = model.predict([row])[0]

    return jsonify({
        "log_text":   log_text,
        "prediction": "ANOMALY" if pred == 1 else "NORMAL",
        "label_int":  pred,
        "confidence": round(prob*100, 2),
        "bert_score": round(prob, 4),
        "tokens":     tokens,
    }), 200

@bert_bp.route("/api/bert-predictions/<file_id>", methods=["GET"])
def api_bert_predictions(file_id):
    page  = int(request.args.get("page", 1))
    limit = int(request.args.get("limit", 20))
    conn  = _db()
    rows  = conn.execute(
        "SELECT * FROM bert_predictions WHERE file_id=? ORDER BY id LIMIT ? OFFSET ?",
        (file_id, limit, (page-1)*limit)
    ).fetchall()
    total = conn.execute(
        "SELECT COUNT(*) FROM bert_predictions WHERE file_id=?", (file_id,)
    ).fetchone()[0]
    conn.close()
    return jsonify({"total":total,"page":page,"predictions":[dict(r) for r in rows]}), 200
