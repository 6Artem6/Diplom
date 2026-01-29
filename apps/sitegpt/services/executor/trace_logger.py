import json
from datetime import datetime
from pathlib import Path


class TraceLogger:
    def __init__(self, path: str):
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)

    def log(self, trace: dict):
        filename = self.path / f"trace_{datetime.utcnow().isoformat()}.json"
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(trace, f, indent=2, ensure_ascii=False)
