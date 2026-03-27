# ============================================================
# ADD THESE LINES TO YOUR app.py
# ============================================================
# 
# STEP 1 — Add these 3 imports at the TOP of app.py
#           (paste right after your existing imports section)
#
from module4_validation import validation_bp
from module5_bert        import bert_bp
from module6_rca         import rca_bp

#
# STEP 2 — Add these 3 lines AFTER  app = Flask(__name__)
#           (paste right after the line: app.config['MAX_CONTENT_LENGTH'] = ...)
#
app.register_blueprint(validation_bp)
app.register_blueprint(bert_bp)
app.register_blueprint(rca_bp)

#
# STEP 3 — Add this ONE endpoint at the BOTTOM of app.py
#           (paste just before the  if __name__ == '__main__':  block)
#

from module4_validation import run_validation
from module5_bert        import run_bert_training
from module6_rca         import run_rca

@app.route('/api/run-pipeline/<fid>', methods=['POST'])
def run_pipeline(fid):
    """Run all 3 new modules in sequence: Validate → BERT → RCA"""
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
    }), 200
