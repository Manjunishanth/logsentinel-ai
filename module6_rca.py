"""
=============================================================
MODULE 6 — MATCHING PREDICTION WITH OUTCOME + RCA
           (Fixed for LogSentinel app.py)
=============================================================
FIXES APPLIED vs original:
  1. bert_predictions uses raw_id (not log_id)
  2. raw_logs joined by raw_id to get raw_text, level, source, timestamp
  3. Ground truth fetched from processed_logs.is_anomaly (not anomaly_flag)
     via JOIN raw_logs → processed_logs on raw_id
  4. DB_PATH absolute path matching app.py
=============================================================
"""

import json, sqlite3, re, os
from collections import Counter, defaultdict
from flask import Blueprint, request, jsonify

rca_bp = Blueprint("rca", __name__)

DB_PATH = os.path.join(os.path.dirname(__file__), 'logsentinel.db')


def _db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def ensure_rca_tables():
    conn = _db()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS rca_reports (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            file_id         TEXT NOT NULL UNIQUE,
            total_predicted INTEGER,
            correct_preds   INTEGER,
            wrong_preds     INTEGER,
            match_accuracy  REAL,
            health_score    REAL,
            root_causes     TEXT,
            outcome_summary TEXT,
            created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS root_cause_entries (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            file_id     TEXT NOT NULL,
            rank        INTEGER,
            cause_id    TEXT,
            cause_label TEXT,
            frequency   INTEGER,
            confidence  REAL,
            severity    TEXT,
            evidence    TEXT,
            first_seen  TEXT,
            last_seen   TEXT
        );
    """)
    conn.commit(); conn.close()


# ─────────────────────────────────────────────
# LOAD — uses raw_id (fixes log_id mismatch)
# ─────────────────────────────────────────────
def load_predictions_with_raw(file_id: str):
    """
    Load bert_predictions joined with raw_logs
    so we have raw_text, level, source, timestamp.
    Also loads ground truth is_anomaly from processed_logs.
    """
    conn = _db()
    rows = conn.execute("""
        SELECT
            bp.id           AS pred_id,
            bp.raw_id       AS raw_id,
            bp.prediction   AS prediction,
            bp.confidence   AS confidence,
            r.raw_text      AS raw_text,
            r.level         AS level,
            r.source        AS source,
            r.timestamp     AS timestamp,
            p.is_anomaly    AS actual_label
        FROM bert_predictions bp
        JOIN raw_logs        r  ON r.id  = bp.raw_id
        JOIN processed_logs  p  ON p.raw_id = bp.raw_id
        WHERE bp.file_id = ?
        ORDER BY bp.id
    """, (file_id,)).fetchall()
    conn.close()
    return [dict(r) for r in rows]


# ─────────────────────────────────────────────
# RCA PATTERN DEFINITIONS
# ─────────────────────────────────────────────
RCA_PATTERNS = [
    {"id":"NET_TIMEOUT",    "label":"Network Timeout / Connection Failure",
     "severity":"HIGH",     "keywords":{"timeout","connection","refused","unreachable",
                                         "reset","broken","pipe","socket","network","host"}, "weight":1.4},
    {"id":"DB_ERROR",       "label":"Database Error / Query Failure",
     "severity":"HIGH",     "keywords":{"database","db","sql","query","deadlock","lock",
                                         "transaction","rollback","constraint","duplicate"}, "weight":1.5},
    {"id":"AUTH_FAILURE",   "label":"Authentication / Authorization Failure",
     "severity":"CRITICAL", "keywords":{"auth","unauthorized","forbidden","permission",
                                         "denied","credential","token","session","login"}, "weight":1.6},
    {"id":"MEM_ERROR",      "label":"Memory / Resource Exhaustion",
     "severity":"CRITICAL", "keywords":{"memory","oom","heap","stack","overflow",
                                         "outofmemory","leak","exhausted","limit","gc"}, "weight":1.7},
    {"id":"DISK_IO",        "label":"Disk I/O / Storage Failure",
     "severity":"HIGH",     "keywords":{"disk","io","read","write","file","corrupt",
                                         "space","storage","filesystem","directory"}, "weight":1.3},
    {"id":"SERVICE_CRASH",  "label":"Service Crash / Process Failure",
     "severity":"CRITICAL", "keywords":{"crash","killed","exit","abort","panic",
                                         "segfault","core","dump","signal","terminated"}, "weight":1.8},
    {"id":"CONFIG_ERROR",   "label":"Configuration / Startup Error",
     "severity":"MEDIUM",   "keywords":{"config","configuration","property","setting",
                                         "invalid","missing","not","found","load"}, "weight":1.1},
    {"id":"HTTP_ERROR",     "label":"HTTP / API Request Failure",
     "severity":"MEDIUM",   "keywords":{"http","request","response","status","error",
                                         "retry","endpoint","api","rest"}, "weight":1.2},
    {"id":"UNKNOWN_ANOMALY","label":"Unknown / Unclassified Anomaly",
     "severity":"LOW",      "keywords":set(), "weight":0.5},
]
SEV = {"CRITICAL":4, "HIGH":3, "MEDIUM":2, "LOW":1}


def match_pattern(tokens):
    ts = set(t.lower() for t in tokens)
    best_pat, best_score = None, 0.0
    for pat in RCA_PATTERNS[:-1]:
        hits  = len(ts & pat["keywords"])
        if hits == 0: continue
        score = (hits / len(pat["keywords"])) * pat["weight"]
        if score > best_score: best_score, best_pat = score, pat
    if best_pat is None or best_score < 0.05:
        best_pat, best_score = RCA_PATTERNS[-1], 0.1
    return best_pat["id"], best_pat["label"], best_pat["severity"], round(best_score,4)


def rank_root_causes(anomaly_rows):
    buckets = defaultdict(lambda: {
        "count":0,"conf_sum":0.0,"severity":"LOW","label":"","evidence":[],"ts":[]
    })
    for row in anomaly_rows:
        text = row.get("raw_text","") or row.get("cleaned_text","")
        text = re.sub(r"[^\w\s]"," ", text.lower())
        toks = text.split()
        cid, label, sev, score = match_pattern(toks)
        b = buckets[cid]
        b["label"]    = label; b["severity"] = sev
        b["count"]   += 1
        b["conf_sum"] += score * row.get("confidence",0.5)
        b["ts"].append(row.get("timestamp",""))
        if len(b["evidence"]) < 3:
            b["evidence"].append({
                "raw_id":     row.get("raw_id"),
                "text":       text[:200],
                "confidence": row.get("confidence",0),
            })
    ranked = []
    for cid, d in buckets.items():
        avg_c = d["conf_sum"] / (d["count"] or 1)
        score = d["count"] * avg_c * SEV.get(d["severity"],1)
        ts    = [t for t in d["ts"] if t]
        ranked.append({
            "cause_id":    cid,
            "cause_label": d["label"],
            "severity":    d["severity"],
            "frequency":   d["count"],
            "confidence":  round(avg_c*100,2),
            "rca_score":   round(score,4),
            "evidence":    d["evidence"],
            "first_seen":  min(ts) if ts else "N/A",
            "last_seen":   max(ts) if ts else "N/A",
        })
    ranked.sort(key=lambda x:(SEV.get(x["severity"],0), x["rca_score"]), reverse=True)
    for i, r in enumerate(ranked): r["rank"] = i+1
    return ranked


def health_score(total, n_anomaly, match_acc, root_causes):
    if total == 0: return 100.0
    score = 100.0
    score -= min(40, (n_anomaly/total)*200)
    score -= sum(8 for c in root_causes if c["severity"]=="CRITICAL")
    score -= sum(4 for c in root_causes if c["severity"]=="HIGH")
    score += match_acc * 10
    return round(max(0.0, min(100.0, score)), 1)


def match_predictions(rows):
    """Compare BERT prediction vs ground truth is_anomaly."""
    correct = wrong = 0
    mismatches = []
    for r in rows:
        actual = r.get("actual_label")
        pred   = r.get("prediction")
        if actual is None: continue
        if pred == actual:
            correct += 1
        else:
            wrong += 1
            if len(mismatches) < 10:
                mismatches.append({
                    "raw_id":    r.get("raw_id"),
                    "predicted": pred,
                    "actual":    actual,
                    "text":      (r.get("raw_text",""))[:100],
                })
    total     = correct + wrong
    match_acc = round(correct/(total+1e-9), 4)
    return {"total":total,"correct":correct,"wrong":wrong,
            "match_accuracy":match_acc,"mismatches":mismatches}


# ─────────────────────────────────────────────
# MAIN RCA
# ─────────────────────────────────────────────
def run_rca(file_id: str) -> dict:
    ensure_rca_tables()
    rows = load_predictions_with_raw(file_id)
    if not rows:
        return {"error": f"No BERT predictions for {file_id}. Run POST /api/bert-train/{file_id} first."}

    print(f"[Module 6] Loaded {len(rows)} predictions")

    match = match_predictions(rows)
    print(f"[Module 6] Match accuracy: {match['match_accuracy']:.4f} "
          f"({match['correct']}/{match['total']})")

    anomaly_rows = [r for r in rows if r.get("prediction") == 1]
    print(f"[Module 6] Anomalies: {len(anomaly_rows)}")

    causes  = rank_root_causes(anomaly_rows)
    health  = health_score(len(rows), len(anomaly_rows), match["match_accuracy"], causes)

    print(f"[Module 6] Root causes: {len(causes)}")
    for c in causes[:3]:
        print(f"           #{c['rank']} [{c['severity']}] {c['cause_label']} freq={c['frequency']}")
    print(f"[Module 6] Health Score: {health}/100")

    summary = {
        "total_logs_analyzed": len(rows),
        "anomalies_detected":  len(anomaly_rows),
        "normal_logs":         len(rows) - len(anomaly_rows),
        "anomaly_rate":        round(len(anomaly_rows)/len(rows)*100,2) if rows else 0,
        "prediction_accuracy": round(match["match_accuracy"]*100,2),
        "health_score":        health,
        "top_root_cause":      causes[0]["cause_label"] if causes else "None",
        "critical_issues":     sum(1 for c in causes if c["severity"]=="CRITICAL"),
        "high_issues":         sum(1 for c in causes if c["severity"]=="HIGH"),
    }

    conn = _db()
    conn.execute("DELETE FROM rca_reports       WHERE file_id=?", (file_id,))
    conn.execute("DELETE FROM root_cause_entries WHERE file_id=?", (file_id,))
    conn.execute("""
        INSERT INTO rca_reports
            (file_id,total_predicted,correct_preds,wrong_preds,
             match_accuracy,health_score,root_causes,outcome_summary)
        VALUES (?,?,?,?,?,?,?,?)
    """, (file_id, match["total"], match["correct"], match["wrong"],
          match["match_accuracy"], health,
          json.dumps(causes), json.dumps(summary)))
    for c in causes:
        conn.execute("""
            INSERT INTO root_cause_entries
                (file_id,rank,cause_id,cause_label,frequency,
                 confidence,severity,evidence,first_seen,last_seen)
            VALUES (?,?,?,?,?,?,?,?,?,?)
        """, (file_id, c["rank"], c["cause_id"], c["cause_label"],
              c["frequency"], c["confidence"], c["severity"],
              json.dumps(c["evidence"]), c["first_seen"], c["last_seen"]))
    conn.commit(); conn.close()

    print(f"[Module 6] ✅ Saved to DB")
    return {"file_id":file_id,"outcome_summary":summary,
            "root_causes":causes,"match_result":match}


# ─────────────────────────────────────────────
# REST ENDPOINTS
# ─────────────────────────────────────────────
@rca_bp.route("/api/rca/<file_id>", methods=["GET"])
def api_rca(file_id):
    r = run_rca(file_id)
    if "error" in r: return jsonify(r), 400
    return jsonify({"status":"success", **r}), 200

@rca_bp.route("/api/rca-report/<file_id>", methods=["GET"])
def api_rca_report(file_id):
    conn = _db()
    row  = conn.execute("SELECT * FROM rca_reports WHERE file_id=?", (file_id,)).fetchone()
    conn.close()
    if not row: return jsonify({"error":"No report. Run GET /api/rca/<file_id> first."}), 404
    r = dict(row)
    r["root_causes"]     = json.loads(r["root_causes"]     or "[]")
    r["outcome_summary"] = json.loads(r["outcome_summary"] or "{}")
    return jsonify(r), 200

@rca_bp.route("/api/root-causes/<file_id>", methods=["GET"])
def api_root_causes(file_id):
    limit = int(request.args.get("limit", 10))
    conn  = _db()
    rows  = conn.execute(
        "SELECT * FROM root_cause_entries WHERE file_id=? ORDER BY rank LIMIT ?",
        (file_id, limit)
    ).fetchall()
    conn.close()
    result = [{**dict(r), "evidence": json.loads(r["evidence"] or "[]")} for r in rows]
    return jsonify({"file_id":file_id,"root_causes":result}), 200

@rca_bp.route("/api/health-score/<file_id>", methods=["GET"])
def api_health_score(file_id):
    conn = _db()
    row  = conn.execute(
        "SELECT health_score, outcome_summary FROM rca_reports WHERE file_id=?", (file_id,)
    ).fetchone()
    conn.close()
    if not row: return jsonify({"error":"No report found."}), 404
    s = json.loads(row["outcome_summary"] or "{}")
    return jsonify({
        "file_id":         file_id,
        "health_score":    row["health_score"],
        "status":          "HEALTHY"  if row["health_score"]>=80 else
                           "WARNING"  if row["health_score"]>=50 else "CRITICAL",
        "anomaly_rate":    s.get("anomaly_rate",0),
        "critical_issues": s.get("critical_issues",0),
    }), 200

@rca_bp.route("/api/pipeline-status/<file_id>", methods=["GET"])
def api_pipeline_status(file_id):
    conn = _db()
    def exists(table, col, val, extra=""):
        q = f"SELECT 1 FROM {table} WHERE {col}=?{extra} LIMIT 1"
        return bool(conn.execute(q, (val,)).fetchone())
    status = {
        "module1_collection":  exists("uploads","id",file_id),
        "module2_preprocessing": conn.execute(
            "SELECT 1 FROM processed_logs p JOIN raw_logs r ON r.id=p.raw_id WHERE r.file_id=? LIMIT 1",
            (file_id,)).fetchone() is not None,
        "module3_visualization": conn.execute(
            "SELECT 1 FROM processed_logs p JOIN raw_logs r ON r.id=p.raw_id WHERE r.file_id=? LIMIT 1",
            (file_id,)).fetchone() is not None,
        "module4_validation":  exists("validation_reports","file_id",file_id),
        "module5_bert":        exists("bert_results","file_id",file_id),
        "module6_rca":         exists("rca_reports","file_id",file_id),
    }
    conn.close()
    done = sum(1 for v in status.values() if v)
    return jsonify({"file_id":file_id,"modules":status,
                    "completed":done,"total":6,"progress":round(done/6*100)}), 200

@rca_bp.route("/api/full-report/<file_id>", methods=["GET"])
def api_full_report(file_id):
    conn = _db()
    upload  = conn.execute("SELECT * FROM uploads WHERE id=?", (file_id,)).fetchone()
    if not upload: conn.close(); return jsonify({"error":"File not found"}), 404
    val     = conn.execute("SELECT * FROM validation_reports WHERE file_id=?", (file_id,)).fetchone()
    bert    = conn.execute("SELECT * FROM bert_results        WHERE file_id=?", (file_id,)).fetchone()
    rca     = conn.execute("SELECT * FROM rca_reports         WHERE file_id=?", (file_id,)).fetchone()
    rcs     = conn.execute(
        "SELECT * FROM root_cause_entries WHERE file_id=? ORDER BY rank LIMIT 10", (file_id,)
    ).fetchall()
    conn.close()
    return jsonify({
        "file_id": file_id,
        "upload":  dict(upload),
        "validation": {
            "cv_accuracy": val["cv_mean_acc"] if val else None,
            "cv_std":      val["cv_std_acc"]  if val else None,
            "train_size":  val["train_size"]  if val else None,
            "test_size":   val["test_size"]   if val else None,
            "top_tokens":  json.loads(val["top_tokens"] or "{}") if val else {},
        },
        "bert": {
            "mode":      bert["mode"]        if bert else None,
            "accuracy":  bert["accuracy"]    if bert else None,
            "precision": bert["precision_s"] if bert else None,
            "recall":    bert["recall"]      if bert else None,
            "f1_score":  bert["f1_score"]    if bert else None,
            "confusion_matrix": json.loads(bert["confusion_matrix"] or "{}") if bert else {},
        },
        "rca": {
            "health_score":    rca["health_score"]  if rca else None,
            "match_accuracy":  rca["match_accuracy"] if rca else None,
            "outcome_summary": json.loads(rca["outcome_summary"] or "{}") if rca else {},
            "root_causes": [{**dict(r),"evidence":json.loads(r["evidence"] or "[]")} for r in rcs],
        }
    }), 200
