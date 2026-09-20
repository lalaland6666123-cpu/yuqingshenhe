"""Create a compact spreadsheet-friendly review queue from test.jsonl."""
from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
rows = [json.loads(x) for x in (ROOT / "test.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
fields = ["id", "scenario", "event_description", "draft_response", "human_crisis_types", "human_legal", "human_business", "human_reputation", "human_actions", "notes"]
with (ROOT / "review_queue.csv").open("w", encoding="utf-8-sig", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    for row in rows:
        writer.writerow({
            "id": row["id"],
            "scenario": row["scenario"],
            "event_description": row["event_description"],
            "draft_response": row["draft_response"],
            "human_crisis_types": "",
            "human_legal": "",
            "human_business": "",
            "human_reputation": "",
            "human_actions": "",
            "notes": "",
        })
print(f"exported {len(rows)} rows to {ROOT / 'review_queue.csv'}")
