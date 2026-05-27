from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
import time


class MemoryType(str, Enum):
    USER = "user"
    FEEDBACK = "feedback"
    PROJECT = "project"
    REFERENCE = "reference"


class RelationType(str, Enum):
    CONTRADICTS = "contradicts"    # 矛盾
    EXTENDS = "extends"            # 扩展
    SUPPORTS = "supports"          # 佐证


@dataclass
class Relation:
    """Typed relationship between two memories."""
    id: str                       # "{mem_a}--{mem_b}"
    source_id: str
    target_id: str
    relation_type: RelationType
    description: str = ""         # LLM-generated explanation of the relationship
    confidence: float = 0.7       # 0-1, LLM-estimated confidence in this relation
    created: float = field(default_factory=time.time)

    def to_dict(self):
        return {
            "id": self.id,
            "source_id": self.source_id,
            "target_id": self.target_id,
            "relation_type": self.relation_type.value,
            "description": self.description,
            "confidence": self.confidence,
            "created": self.created,
        }


@dataclass
class Memory:
    id: str
    name: str
    description: str
    memory_type: MemoryType
    content: str
    tags: list[str] = field(default_factory=list)
    references: list[str] = field(default_factory=list)
    importance: int = 3            # 1-5, scored by LLM during extraction
    confidence: float = 0.7        # 0-1, computed from signals (access, recency, contradictions)
    mention_count: int = 1         # times this fact was mentioned by the user
    access_count: int = 0          # times this memory was retrieved
    last_accessed: float = 0.0     # timestamp of last retrieval hit
    created: float = field(default_factory=time.time)
    updated: float = field(default_factory=time.time)

    def to_frontmatter(self) -> str:
        import yaml
        meta = {
            "name": self.name,
            "description": self.description,
            "type": self.memory_type.value,
            "tags": self.tags,
            "references": self.references,
            "importance": self.importance,
            "confidence": round(self.confidence, 3),
            "mention_count": self.mention_count,
            "access_count": self.access_count,
            "last_accessed": self.last_accessed,
            "created": self.created,
            "updated": self.updated,
        }
        return f"---\n{yaml.dump(meta, allow_unicode=True, sort_keys=False)}---\n\n{self.content}"

    @classmethod
    def from_file(cls, filepath: str, content: str) -> "Memory":
        import yaml, os, re
        match = re.match(r"^---\n(.*?)\n---\n\n(.*)", content, re.DOTALL)
        if match:
            meta = yaml.safe_load(match.group(1)) or {}
            body = match.group(2)
        else:
            meta = {}
            body = content

        filename = os.path.basename(filepath).replace(".md", "")
        return cls(
            id=filename,
            name=meta.get("name", filename),
            description=meta.get("description", ""),
            memory_type=MemoryType(meta.get("type", "reference")),
            content=body.strip(),
            tags=meta.get("tags", []),
            references=meta.get("references", []),
            importance=meta.get("importance", 3),
            confidence=meta.get("confidence", 0.7),
            mention_count=meta.get("mention_count", 1),
            access_count=meta.get("access_count", 0),
            last_accessed=meta.get("last_accessed", 0.0),
            created=meta.get("created", time.time()),
            updated=meta.get("updated", time.time()),
        )


@dataclass
class STMEntry:
    role: str
    content: str
    timestamp: float = field(default_factory=time.time)


@dataclass
class RetrievalResult:
    memory: Memory
    score: float
    method: str

    def to_dict(self):
        return {
            "memory_id": self.memory.id,
            "name": self.memory.name,
            "description": self.memory.description,
            "type": self.memory.memory_type.value,
            "score": round(self.score, 4),
            "method": self.method,
        }
