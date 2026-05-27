import json
import os
import time
from collections import deque
from .types import STMEntry

STM_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "storage", "stm")


class ShortTermMemory:
    """Sliding window short-term memory with JSON file persistence."""

    def __init__(self, session_id: str, max_messages: int = 20):
        self.session_id = session_id
        self.max_messages = max_messages
        self.messages: deque[STMEntry] = deque(maxlen=max_messages)
        self._filepath = os.path.join(STM_DIR, f"{session_id}.json")
        self._load()

    def add(self, role: str, content: str):
        entry = STMEntry(role=role, content=content)
        self.messages.append(entry)
        self._save()

    def get_context(self, last_n: int = None) -> list[dict]:
        n = last_n or self.max_messages
        items = list(self.messages)[-n:]
        return [{"role": m.role, "content": m.content} for m in items]

    def get_all(self) -> list[STMEntry]:
        return list(self.messages)

    def clear(self):
        self.messages.clear()
        self._save()

    def _save(self):
        os.makedirs(STM_DIR, exist_ok=True)
        data = [
            {"role": m.role, "content": m.content, "timestamp": m.timestamp}
            for m in self.messages
        ]
        with open(self._filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def _load(self):
        if not os.path.exists(self._filepath):
            return
        with open(self._filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        for item in data[-self.max_messages:]:
            self.messages.append(STMEntry(
                role=item["role"],
                content=item["content"],
                timestamp=item.get("timestamp", time.time()),
            ))
