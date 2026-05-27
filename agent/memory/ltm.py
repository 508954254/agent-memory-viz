import json
import os
import re
import time
from .types import Memory, MemoryType, Relation, RelationType

LTM_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "storage", "ltm")
STORAGE_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "storage")
RELATIONS_FILE = os.path.join(STORAGE_DIR, "relations.json")


class LongTermMemory:
    """File-based long-term memory with relation management and confidence scoring."""

    def __init__(self):
        os.makedirs(LTM_DIR, exist_ok=True)
        self._relations: dict[str, Relation] = {}
        self._load_relations()

    # ===== Relations =====

    def _load_relations(self):
        if os.path.exists(RELATIONS_FILE):
            with open(RELATIONS_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            for item in data:
                rel = Relation(
                    id=item["id"],
                    source_id=item["source_id"],
                    target_id=item["target_id"],
                    relation_type=RelationType(item["relation_type"]),
                    description=item.get("description", ""),
                    confidence=item.get("confidence", 0.7),
                    created=item.get("created", time.time()),
                )
                self._relations[rel.id] = rel

    def _save_relations(self):
        data = [r.to_dict() for r in self._relations.values()]
        with open(RELATIONS_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def add_relation(self, source_id: str, target_id: str,
                     relation_type: RelationType, description: str = "",
                     confidence: float = 0.7) -> Relation | None:
        """Create or update a typed relationship between two memories."""
        if source_id == target_id:
            return None
        rid = f"{source_id}--{target_id}"
        # Check reverse
        rid_rev = f"{target_id}--{source_id}"
        if rid_rev in self._relations:
            existing = self._relations[rid_rev]
            existing.relation_type = relation_type
            existing.description = description or existing.description
            existing.confidence = confidence
            existing.created = time.time()
            self._save_relations()
            return existing
        if rid in self._relations:
            existing = self._relations[rid]
            existing.relation_type = relation_type
            existing.description = description or existing.description
            existing.confidence = confidence
            existing.created = time.time()
            self._save_relations()
            return existing
        rel = Relation(id=rid, source_id=source_id, target_id=target_id,
                       relation_type=relation_type, description=description,
                       confidence=confidence)
        self._relations[rid] = rel
        self._save_relations()
        return rel

    def get_relations(self, mem_id: str = None) -> list[Relation]:
        """Get all relations, or relations involving a specific memory."""
        if mem_id is None:
            return list(self._relations.values())
        return [r for r in self._relations.values()
                if r.source_id == mem_id or r.target_id == mem_id]

    def delete_relation(self, source_id: str, target_id: str):
        rid = f"{source_id}--{target_id}"
        rid_rev = f"{target_id}--{source_id}"
        for r in [rid, rid_rev]:
            if r in self._relations:
                del self._relations[r]
                self._save_relations()
                return

    def get_all_relations(self) -> list[Relation]:
        return list(self._relations.values())

    # ===== CRUD =====

    def create(self, name: str, description: str, memory_type: MemoryType,
               content: str, tags: list[str] = None,
               references: list[str] = None, importance: int = 3,
               mention_count: int = 1) -> Memory:
        tags = tags or []
        references = references or []

        # Dedup: check if a memory with similar name already exists
        existing = self._find_similar(name)
        if existing:
            merged_tags = list(set(existing.tags + tags))
            merged_refs = list(set(existing.references + references))
            if content.strip() != existing.content.strip():
                merged_content = content + "\n\n---\n历史记录:\n" + existing.content
            else:
                merged_content = content
            return self.update(existing.id,
                              name=name,
                              description=description,
                              content=merged_content,
                              tags=merged_tags,
                              references=merged_refs,
                              importance=importance,
                              mention_count=existing.mention_count + 1)

        mem_id = self._name_to_id(name)
        mem = Memory(
            id=mem_id,
            name=name,
            description=description,
            memory_type=memory_type,
            content=content,
            tags=tags,
            references=references,
            importance=importance,
            mention_count=mention_count,
        )
        self._recalc_confidence(mem)
        self._write(mem)
        return mem

    def _find_similar(self, name: str) -> Memory | None:
        name_chars = set(name.replace(" ", ""))
        if not name_chars:
            return None
        for mem in self.list_all():
            mem_chars = set(mem.name.replace(" ", ""))
            if not mem_chars:
                continue
            jaccard = len(name_chars & mem_chars) / len(name_chars | mem_chars)
            if jaccard > 0.4:
                return mem
        return None

    def deduplicate(self) -> list[dict]:
        merges = []
        all_mems = self.list_all()
        for i in range(len(all_mems)):
            for j in range(i + 1, len(all_mems)):
                a, b = all_mems[i], all_mems[j]
                a_chars = set(a.name.replace(" ", ""))
                b_chars = set(b.name.replace(" ", ""))
                if not a_chars or not b_chars:
                    continue
                jaccard = len(a_chars & b_chars) / len(a_chars | b_chars)
                if jaccard <= 0.4:
                    continue
                if len(b.content) > len(a.content):
                    keeper, candidate = b, a
                else:
                    keeper, candidate = a, b
                kp = os.path.join(LTM_DIR, f"{keeper.id}.md")
                cp = os.path.join(LTM_DIR, f"{candidate.id}.md")
                if not os.path.exists(kp) or not os.path.exists(cp):
                    continue
                merged_tags = list(set(keeper.tags + candidate.tags))
                merged_refs = list(set(keeper.references + candidate.references))
                if candidate.content.strip() != keeper.content.strip():
                    merged_content = keeper.content + "\n\n---\n历史记录:\n" + candidate.content
                else:
                    merged_content = keeper.content
                self.update(keeper.id, tags=merged_tags, references=merged_refs,
                           content=merged_content, mention_count=keeper.mention_count + 1)
                self.delete(candidate.id)
                merges.append({"kept": keeper.name, "merged": candidate.name})
        return merges

    def record_access(self, mem_id: str):
        mem = self.get(mem_id)
        if mem:
            mem.access_count += 1
            mem.last_accessed = time.time()
            self._recalc_confidence(mem)
            self._write(mem)

    def record_mention(self, mem_id: str):
        """Called when user mentions a fact again — boosts confidence."""
        mem = self.get(mem_id)
        if mem:
            mem.mention_count += 1
            self._recalc_confidence(mem)
            self._write(mem)

    def _recalc_confidence(self, mem: Memory):
        """Compute confidence from importance, mentions, accesses, and relations.

        Wide dynamic range: low-importance unaccessed memories ~0.25,
        high-importance frequently-accessed memories ~0.95.
        Incorporates time decay for stale memories and relation signals."""
        base = 0.15
        base += mem.importance * 0.12           # imp 1→0.12, imp 5→0.60
        base += min(mem.mention_count, 10) * 0.04  # 10 mentions → +0.40
        base += min(mem.access_count, 25) * 0.012  # 25 accesses → +0.30

        # Time decay: memories not accessed for 7+ days lose confidence gradually
        elapsed = time.time() - max(mem.last_accessed, mem.updated, mem.created)
        half_life = 14 * 24 * 3600  # 14 days
        if elapsed > half_life:
            decay = 0.5 + 0.5 * (half_life / elapsed)  # approaches 0.5 as time passes
            base *= decay

        # Relation signals
        rels = self.get_relations(mem.id)
        contradictions = [r for r in rels if r.relation_type == RelationType.CONTRADICTS]
        base -= len(contradictions) * 0.15
        supports = [r for r in rels if r.relation_type == RelationType.SUPPORTS]
        base += min(len(supports), 5) * 0.04       # up to +0.20 from supporting evidence
        extends = [r for r in rels if r.relation_type == RelationType.EXTENDS]
        base += min(len(extends), 5) * 0.02         # up to +0.10 from related extensions

        mem.confidence = round(max(0.05, min(base, 0.98)), 3)

    def find_by_name(self, name: str) -> Memory | None:
        for mem in self.list_all():
            if mem.name == name:
                return mem
        return None

    def update(self, mem_id: str, **kwargs) -> Memory | None:
        mem = self.get(mem_id)
        if not mem:
            return None
        for key, value in kwargs.items():
            if hasattr(mem, key):
                setattr(mem, key, value)
        mem.updated = time.time()
        self._recalc_confidence(mem)
        self._write(mem)
        return mem

    def get(self, mem_id: str) -> Memory | None:
        filepath = os.path.join(LTM_DIR, f"{mem_id}.md")
        if not os.path.exists(filepath):
            return None
        with open(filepath, "r", encoding="utf-8") as f:
            return Memory.from_file(filepath, f.read())

    def list_all(self) -> list[Memory]:
        memories = []
        if not os.path.exists(LTM_DIR):
            return memories
        for filename in sorted(os.listdir(LTM_DIR)):
            if not filename.endswith(".md"):
                continue
            filepath = os.path.join(LTM_DIR, filename)
            with open(filepath, "r", encoding="utf-8") as f:
                mem = Memory.from_file(filepath, f.read())
                # Recalc confidence each load to account for relation changes
                self._recalc_confidence(mem)
                memories.append(mem)
        return memories

    def delete(self, mem_id: str) -> bool:
        filepath = os.path.join(LTM_DIR, f"{mem_id}.md")
        if os.path.exists(filepath):
            os.remove(filepath)
            # Also remove relations involving this memory
            for r in list(self._relations.values()):
                if r.source_id == mem_id or r.target_id == mem_id:
                    del self._relations[r.id]
            self._save_relations()
            return True
        return False

    def search_by_tags(self, tags: list[str]) -> list[Memory]:
        return [m for m in self.list_all() if any(t in m.tags for t in tags)]

    def search_by_type(self, memory_type: MemoryType) -> list[Memory]:
        return [m for m in self.list_all() if m.memory_type == memory_type]

    def get_stale_memories(self, confidence_threshold: float = 0.35,
                           days_threshold: int = 14) -> list[Memory]:
        """Find memories that are candidates for cleanup/forgetting.

        A memory is stale if its confidence is below threshold AND it hasn't been
        accessed in `days_threshold` days. These are candidates for archiving or deletion."""
        cutoff = time.time() - days_threshold * 24 * 3600
        stale = []
        for m in self.list_all():
            if m.confidence < confidence_threshold and m.last_accessed < cutoff:
                stale.append(m)
        return sorted(stale, key=lambda m: m.confidence)

    def get_all_contents(self) -> list[tuple[str, str]]:
        results = []
        for mem in self.list_all():
            text = f"{mem.name}\n{mem.description}\n{mem.content}"
            results.append((mem.id, text))
        return results

    def _write(self, mem: Memory):
        filepath = os.path.join(LTM_DIR, f"{mem.id}.md")
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(mem.to_frontmatter())

    @staticmethod
    def _name_to_id(name: str) -> str:
        slug = re.sub(r"[^\w一-鿿-]", "_", name.strip()).strip("_")
        return slug or "untitled"
