"""Private SQLite ledger: idempotent decisions, later outcomes, honest accounting."""
from __future__ import annotations
from dataclasses import asdict
from contextlib import closing, contextmanager
from datetime import datetime, timezone
from pathlib import Path
import json
import os
import sqlite3
from .protocol import DecisionError, digest, encode, safe, request_hashes

def private_dir(path):
    path = Path(path)
    if path.is_symlink():
        raise DecisionError("UNSAFE_LEDGER_PATH")
    if not path.exists():
        private_dir(path.parent)
        path.mkdir(mode=0o700, exist_ok=True)
    if not path.is_dir():
        raise DecisionError("UNSAFE_LEDGER_PATH")

class Ledger:
    def __init__(self, root):
        self.root = Path(root)

    @contextmanager
    def connect(self, agent, capacity):
        with closing(self._connect(agent, capacity)) as db:
            with db:
                yield db

    def _connect(self, agent, capacity):
        from .protocol import ID
        if not isinstance(agent,str) or not ID.fullmatch(agent) or capacity not in ("business","personal"):
            raise DecisionError("INVALID_LEDGER_SCOPE")
        private_dir(self.root)
        folder = self.root / capacity / agent.upper()
        private_dir(folder)
        os.chmod(folder, 0o700)
        path = folder / "decisions.sqlite3"
        if path.is_symlink():
            raise DecisionError("UNSAFE_LEDGER_PATH")
        fd = os.open(path, os.O_CREAT | os.O_RDWR | getattr(os,"O_NOFOLLOW",0), 0o600)
        os.close(fd)
        os.chmod(path, 0o600)
        connection = sqlite3.connect(path, timeout=3)
        connection.execute("PRAGMA journal_mode=DELETE")
        connection.execute("""CREATE TABLE IF NOT EXISTS decisions(
            decision_id TEXT PRIMARY KEY, timestamp TEXT NOT NULL, request_hash TEXT NOT NULL,
            state_hash TEXT NOT NULL, decision_type TEXT NOT NULL, stakes TEXT NOT NULL,
            result TEXT, outcome TEXT, action_taken TEXT)""")
        connection.execute("""CREATE TABLE IF NOT EXISTS consultation_reviews(
            decision_id TEXT PRIMARY KEY, review TEXT NOT NULL, timestamp TEXT NOT NULL)""")
        connection.commit()
        return connection

    def reserve(self, request, state_hash):
        fingerprint = digest(asdict(request))
        with self.connect(request.requesting_agent, request.capacity) as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT request_hash,result FROM decisions WHERE decision_id=?",
                             (request.decision_id,)).fetchone()
            if row:
                if row[0] not in request_hashes(request):
                    raise DecisionError("DECISION_ID_CONFLICT")
                if row[1]:
                    result = json.loads(row[1])
                    result["replayed"] = True
                    return result
                raise DecisionError("DECISION_PENDING_RECONCILIATION")
            db.execute("INSERT INTO decisions VALUES(?,?,?,?,?,?,NULL,NULL,NULL)",
                       (request.decision_id, datetime.now(timezone.utc).isoformat(), fingerprint,
                        state_hash, request.decision_type, request.stakes))
        return None

    def finish(self, request, result):
        with self.connect(request.requesting_agent, request.capacity) as db:
            db.execute("UPDATE decisions SET result=? WHERE decision_id=?",
                       (encode(result).decode(), request.decision_id))

    @contextmanager
    def reader(self, agent, capacity):
        from .protocol import ID
        if not isinstance(agent, str) or not ID.fullmatch(agent) or capacity not in ("business", "personal"):
            raise DecisionError("INVALID_LEDGER_SCOPE")
        path = self.root / capacity / agent.upper() / "decisions.sqlite3"
        if path.is_symlink():
            raise DecisionError("UNSAFE_LEDGER_PATH")
        if not path.is_file():
            raise DecisionError("DECISION_NOT_FOUND")
        with closing(sqlite3.connect(path.absolute().as_uri() + "?mode=ro", uri=True, timeout=3)) as db:
            yield db

    def inspect(self, decision_id, agent, capacity):
        with self.reader(agent, capacity) as db:
            row = db.execute("SELECT timestamp,state_hash,decision_type,stakes,result,outcome,action_taken,request_hash FROM decisions WHERE decision_id=?",(decision_id,)).fetchone()
        if row is None:
            raise DecisionError("DECISION_NOT_FOUND")
        return dict(zip(("timestamp","state_hash","decision_type","stakes","result","outcome","action_taken","request_hash"),
                        (row[0],row[1],row[2],row[3],json.loads(row[4]) if row[4] else None,
                         json.loads(row[5]) if row[5] else None,row[6],row[7])))

    def get_review(self, decision_id, agent, capacity):
        with self.reader(agent, capacity) as db:
            if not db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='consultation_reviews'").fetchone():
                return None
            row = db.execute("SELECT review FROM consultation_reviews WHERE decision_id=?", (decision_id,)).fetchone()
        return json.loads(row[0]) if row else None

    def attach_review(self, decision_id, agent, capacity, review):
        safe(review)
        encoded = encode(review).decode()
        with self.connect(agent, capacity) as db:
            db.execute("BEGIN IMMEDIATE")
            record = db.execute("SELECT request_hash,result FROM decisions WHERE decision_id=?", (decision_id,)).fetchone()
            if not record or not record[1] or record[0] != review.get("request_hash"):
                raise DecisionError("REVIEW_REQUEST_MISMATCH")
            existing = db.execute("SELECT review FROM consultation_reviews WHERE decision_id=?", (decision_id,)).fetchone()
            if existing:
                if existing[0] != encoded:
                    raise DecisionError("REVIEW_ALREADY_ATTACHED")
                return
            db.execute("INSERT INTO consultation_reviews VALUES(?,?,?)",
                       (decision_id, encoded, datetime.now(timezone.utc).isoformat()))

    def attach_outcome(self, decision_id, agent, capacity, outcome, action_taken=None):
        safe(outcome)
        safe(action_taken)
        if not isinstance(outcome,dict) or len(encode(outcome)) > 2000:
            raise DecisionError("INVALID_OUTCOME")
        if action_taken is not None and (not isinstance(action_taken,str) or len(action_taken)>500):
            raise DecisionError("INVALID_ACTION")
        with self.connect(agent,capacity) as db:
            row = db.execute("SELECT result FROM decisions WHERE decision_id=?",(decision_id,)).fetchone()
            if not row or not row[0]:
                raise DecisionError("DECISION_NOT_COMPLETE")
            existing = db.execute("SELECT outcome FROM decisions WHERE decision_id=?",(decision_id,)).fetchone()[0]
            if existing and existing != encode(outcome).decode():
                raise DecisionError("OUTCOME_ALREADY_ATTACHED")
            db.execute("UPDATE decisions SET outcome=?,action_taken=? WHERE decision_id=?",
                       (encode(outcome).decode(),action_taken,decision_id))
        return {"decision_id":decision_id,"outcome_attached":True}

    def metrics(self, agent, capacity):
        with self.connect(agent,capacity) as db:
            rows = db.execute("SELECT decision_type,result,outcome FROM decisions WHERE result IS NOT NULL").fetchall()
        total = {"decisions":len(rows),"provider_calls":0,"reasoning_calls":0,
                 "completed_without_reasoning":0,"input_tokens":0,"output_tokens":0,
                 "known_estimated_cost_usd":0.0,"unknown_usage_calls":0,"unknown_cost_calls":0}
        calibration = {}
        for category, raw, outcome in rows:
            result = json.loads(raw)
            calls = result.get("provider_results",[])
            total["provider_calls"] += sum(c.get("provider_called",True) for c in calls)
            total["reasoning_calls"] += sum(c["tier"]==4 and c.get("provider_called",True) for c in calls)
            total["completed_without_reasoning"] += int(not result["abstain"] and not any(c["tier"]==4 for c in calls))
            for call in calls:
                if not call.get("provider_called",True):
                    continue
                for key in ("input_tokens","output_tokens"):
                    total[key] += call.get(key) or 0
                total["unknown_usage_calls"] += int(call.get("input_tokens") is None or call.get("output_tokens") is None)
                cost = call.get("cost_usd")
                total["unknown_cost_calls"] += int(cost is None)
                total["known_estimated_cost_usd"] += cost or 0
                observed = json.loads(outcome) if outcome else {}
                if "correct_decision" in observed and not call.get("abstain",True):
                    key = call["provider"] + ":" + category
                    item = calibration.setdefault(key, {"n":0,"correct":0,"mean_confidence":0.0,"brier_binary_proxy":0.0})
                    correct = call["decision"] == observed["correct_decision"]
                    item["n"] += 1
                    item["correct"] += int(correct)
                    item["mean_confidence"] += call["confidence"]
                    item["brier_binary_proxy"] += (call["confidence"]-int(correct))**2
        for item in calibration.values():
            item["mean_confidence"] /= item["n"]
            item["brier_binary_proxy"] /= item["n"]
        total["calibration"] = calibration
        total["savings_basis"] = "completed_without_reasoning is observed; avoided calls and money require a counterfactual baseline"
        return total
