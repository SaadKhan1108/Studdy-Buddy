"""Progress tracking in SQLite: quiz attempts, confusions, weak-spot scores, study plan."""
from __future__ import annotations

import sqlite3
import time
from contextlib import contextmanager
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

DB_PATH = Path("data/progress.db")
HALF_LIFE_DAYS = 7  # older answers count less
SCHEMA = """
CREATE TABLE IF NOT EXISTS attempts(id INTEGER PRIMARY KEY, scope TEXT, topic TEXT,
    question TEXT, correct INTEGER, ts REAL);
CREATE TABLE IF NOT EXISTS confusions(id INTEGER PRIMARY KEY, scope TEXT, topic TEXT,
    question TEXT, ts REAL);
"""


@contextmanager
def _db():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    try:
        conn.executescript(SCHEMA)
        yield conn
        conn.commit()
    finally:
        conn.close()


def log_attempt(scope, topic, question, correct, ts=None):
    with _db() as c:
        c.execute("INSERT INTO attempts(scope, topic, question, correct, ts) VALUES (?,?,?,?,?)",
                  (scope, topic, question, int(correct), ts or time.time()))


def log_confusion(scope, topic, question, ts=None):
    with _db() as c:
        c.execute("INSERT INTO confusions(scope, topic, question, ts) VALUES (?,?,?,?)",
                  (scope, topic, question, ts or time.time()))


def reset(scope):
    with _db() as c:
        c.execute("DELETE FROM attempts WHERE scope=?", (scope,))
        c.execute("DELETE FROM confusions WHERE scope=?", (scope,))


def topic_stats(scope, topics, now=None) -> pd.DataFrame:
    """One row per topic. Mastery = recency-weighted accuracy with a neutral prior (0.5);
    weakness = 1 - mastery, nudged up by recent 'I'm confused' questions."""
    now = now or time.time()
    with _db() as c:
        att = c.execute("SELECT topic, correct, ts FROM attempts WHERE scope=?", (scope,)).fetchall()
        con = c.execute("SELECT topic, ts FROM confusions WHERE scope=?", (scope,)).fetchall()
    rows = []
    for t in topics:
        a = [(ok, ts) for tp, ok, ts in att if tp == t]
        w = [0.5 ** ((now - ts) / 86400 / HALF_LIFE_DAYS) for _, ts in a]
        mastery = (sum(wi * ok for wi, (ok, _) in zip(w, a)) + 1) / (sum(w) + 2)
        recent_conf = sum(1 for tp, ts in con if tp == t and now - ts < 14 * 86400)
        status = ("Not tested" if not a else
                  "Weak" if mastery < 0.5 else "Learning" if mastery < 0.75 else "Strong")
        rows.append({
            "Topic": t,
            "Attempts": len(a),
            "Accuracy %": round(100 * sum(ok for ok, _ in a) / len(a)) if a else None,
            "Mastery %": round(100 * mastery),
            "Confusions": sum(1 for tp, _ in con if tp == t),
            "Status": status,
            "weakness": (1 - mastery) + 0.04 * min(recent_conf, 5),
        })
    return pd.DataFrame(rows)


def make_plan(stats: pd.DataFrame, exam_date: date, today: date | None = None):
    """Spread study days over topics (weaker topics get more days, revisited in rotation).
    The last day before the exam is a mixed mock quiz."""
    today = today or date.today()
    days = (exam_date - today).days
    if days < 1 or stats.empty:
        return []
    study_days = days - 1 if days >= 3 else days
    order = stats.sort_values("weakness", ascending=False).reset_index(drop=True)
    w = [(x + 0.1) ** 2 for x in order["weakness"]]
    raw = [x / sum(w) * study_days for x in w]
    alloc = [int(x) for x in raw]
    for i in sorted(range(len(raw)), key=lambda i: raw[i] - alloc[i], reverse=True)[: study_days - sum(alloc)]:
        alloc[i] += 1
    seq, visits = [], {}
    while any(alloc):  # round-robin so topics are revisited (spaced repetition)
        for i, topic in enumerate(order["Topic"]):
            if alloc[i] > 0:
                alloc[i] -= 1
                visits[topic] = visits.get(topic, 0) + 1
                seq.append((topic, "Read your notes, then take a 5-question quiz" if visits[topic] == 1
                            else "Review past mistakes, then take a new quiz"))
    plan = [{"Date": (today + timedelta(days=i)).strftime("%a %d %b"), "Focus": t, "Activity": a}
            for i, (t, a) in enumerate(seq, start=1)]
    if days >= 3:
        plan.append({"Date": (today + timedelta(days=days)).strftime("%a %d %b"),
                     "Focus": "All topics", "Activity": "Mixed mock quiz; revisit your 2-3 weakest topics"})
    return plan
