import os
import json
from openai import OpenAI
from .memory.types import MemoryType, RelationType, RetrievalResult
from .memory.stm import ShortTermMemory
from .memory.ltm import LongTermMemory
from .memory.extraction import extract_memories, summarize_dialogue
from .memory.retrieval import MemoryRetriever


SYSTEM_PROMPT = """你是一个有记忆能力的AI助手。每次对话，系统会从长期记忆中检索相关信息，标注为【已激活记忆】。

和用户交流时:
- 如果检索到的记忆与当前话题相关，自然地引用它们
- 用户告诉你的事会被自动记录到长期记忆中
- 用中文回复用户"""


class MemoryAgent:
    """Agent with short-term + long-term memory, retrieval, extraction, summarization."""

    def __init__(self, api_key: str = None, base_url: str = None,
                 model: str = "gpt-4o-mini", embedding_model: str = "text-embedding-3-small"):
        api_key = api_key or os.getenv("OPENAI_API_KEY")
        base_url = base_url or os.getenv("OPENAI_BASE_URL")
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.embedding_model = embedding_model

        self.session_id = "default"
        self.stm = ShortTermMemory(self.session_id)
        self.ltm = LongTermMemory()
        self.retriever = MemoryRetriever(self.ltm, self.client, embedding_model)

        self._exchange_count = 0
        self._extract_every = 1       # configurable
        self._summarize_threshold = 16  # trigger STM→LTM summarization at 16/20 msgs
        self._retrieve_top_k = 5      # how many LTM memories to retrieve per query

        # Event hooks for visualization
        self._on_retrieve = None
        self._on_extract = None
        self._on_summarize = None

    def on_retrieve(self, callback):
        self._on_retrieve = callback

    def on_extract(self, callback):
        self._on_extract = callback

    def on_summarize(self, callback):
        self._on_summarize = callback

    def set_settings(self, extract_every: int = None, summarize_threshold: int = None,
                     retrieve_top_k: int = None):
        if extract_every is not None:
            self._extract_every = max(1, extract_every)
        if summarize_threshold is not None:
            self._summarize_threshold = max(5, min(summarize_threshold, self.stm.max_messages - 2))
        if retrieve_top_k is not None:
            self._retrieve_top_k = max(1, min(retrieve_top_k, 20))

    def get_settings(self) -> dict:
        return {
            "extract_every": self._extract_every,
            "summarize_threshold": self._summarize_threshold,
            "retrieve_top_k": self._retrieve_top_k,
            "stm_max": self.stm.max_messages,
        }

    def chat(self, user_message: str) -> dict:
        # 1. Save user message to STM
        self.stm.add("user", user_message)

        # 2. Retrieve relevant LTM memories
        retrieved = self.retriever.retrieve(user_message, top_k=self._retrieve_top_k)

        # Track access
        for r in retrieved:
            self.ltm.record_access(r.memory.id)

        if self._on_retrieve:
            self._on_retrieve(user_message, retrieved)

        # 3. Build messages for LLM
        messages = [{"role": "system", "content": self._build_system_prompt(retrieved)}]
        stm_context = self.stm.get_context(last_n=15)
        messages.extend(stm_context)

        # 4. Call LLM
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.7,
                max_tokens=2000,
            )
            reply = response.choices[0].message.content or ""
        except Exception as e:
            print(f"[Chat] LLM error: {e}")
            reply = f"API错误: {str(e)[:200]}"

        # 5. Save assistant reply to STM
        self.stm.add("assistant", reply)

        # 6. Check STM capacity → auto-summarize if near full
        summarized = False
        if len(self.stm.messages) >= self._summarize_threshold:
            summarized = self._maybe_summarize()

        # 7. Periodic memory extraction
        self._exchange_count += 1
        extracted = []
        if self._exchange_count % self._extract_every == 0:
            extracted = self._maybe_extract()

        return {
            "reply": reply,
            "retrieved": [r.to_dict() for r in retrieved],
            "extracted": extracted,
            "summarized": summarized,
        }

    def _build_system_prompt(self, retrieved: list[RetrievalResult]) -> str:
        if not retrieved:
            return SYSTEM_PROMPT

        lines = [SYSTEM_PROMPT, "", "## 已激活记忆 (从长期记忆中检索到以下相关信息)"]
        for i, r in enumerate(retrieved, 1):
            lines.append(f"{i}. [{r.memory.memory_type.value}] {r.memory.name}")
            lines.append(f"   {r.memory.description}")
            if len(r.memory.content) > 200:
                lines.append(f"   {r.memory.content[:200]}...")
            else:
                lines.append(f"   {r.memory.content}")
        lines.append("")
        return "\n".join(lines)

    def _maybe_extract(self) -> list[dict]:
        dialogue_parts = []
        for m in self.stm.get_all():
            dialogue_parts.append(f"[{m.role}]: {m.content}")
        dialogue = "\n".join(dialogue_parts[-20:])

        all_mems = self.ltm.list_all()
        existing_summary = [
            {"name": m.name, "description": m.description, "type": m.memory_type.value}
            for m in all_mems
        ]

        memories, relations = extract_memories(self.client, dialogue, self.model, existing_summary)

        saved = []
        for item in memories:
            importance = item.get("importance", 3)
            action = item.get("action", "create")

            if action == "update":
                target_name = item.get("update_target", "")
                target = self.ltm.find_by_name(target_name) if target_name else None
                if target:
                    merged_tags = list(set(target.tags + item.get("tags", [])))
                    if item["content"].strip() != target.content.strip():
                        merged_content = item["content"] + "\n\n---\n历史记录:\n" + target.content
                    else:
                        merged_content = item["content"]
                    self.ltm.update(target.id,
                                    name=item["name"],
                                    description=item["description"],
                                    content=merged_content,
                                    tags=merged_tags,
                                    importance=importance,
                                    mention_count=target.mention_count + 1)
                    saved.append({"name": item["name"], "type": item["type"], "id": target.id,
                                  "action": "update"})
                    continue

            mem = self.ltm.create(
                name=item["name"],
                description=item["description"],
                memory_type=MemoryType(item["type"]),
                content=item["content"],
                tags=item.get("tags", []),
                importance=importance,
            )
            saved.append({"name": mem.name, "type": mem.memory_type.value, "id": mem.id,
                          "action": "create"})

        # Process relations from extraction
        for rel in relations:
            source_name = rel.get("source", "")
            target_name = rel.get("target", "")
            rel_type = rel.get("relation", "extends")
            desc = rel.get("description", "")
            conf = rel.get("confidence", 0.7)

            source_mem = self.ltm.find_by_name(source_name)
            target_mem = self.ltm.find_by_name(target_name)
            if source_mem and target_mem and source_mem.id != target_mem.id:
                try:
                    rt = RelationType(rel_type)
                except ValueError:
                    continue
                self.ltm.add_relation(source_mem.id, target_mem.id, rt, desc, conf)

        if saved and self._on_extract:
            self._on_extract(saved)
        if saved:
            self.retriever.rebuild_cache()

        return saved

    def _maybe_summarize(self) -> bool:
        """Summarize oldest STM messages into LTM when nearing capacity."""
        all_msgs = self.stm.get_all()
        if len(all_msgs) < 10:
            return False

        # Take oldest ~8 messages to summarize
        old = all_msgs[:8]
        dialogue_parts = [f"[{m.role}]: {m.content}" for m in old]
        dialogue = "\n".join(dialogue_parts)

        summary = summarize_dialogue(self.client, dialogue, self.model)
        if summary.strip():
            self.ltm.create(
                name="对话摘要",
                description="自动生成的对话摘要",
                memory_type=MemoryType.REFERENCE,
                content=summary.strip(),
                tags=["摘要", "自动生成"],
                importance=2,
            )
            # Remove summarized messages from STM
            for m in old:
                if m in self.stm.messages:
                    self.stm.messages.remove(m)
            self.stm._save()

            if self._on_summarize:
                self._on_summarize({"summary": summary.strip()})

            self.retriever.rebuild_cache()
            return True
        return False

    def reset_session(self):
        self.session_id = "default"
        self.stm.clear()
        self._exchange_count = 0

    def get_stats(self) -> dict:
        all_mems = self.ltm.list_all()
        types_count = {}
        for m in all_mems:
            types_count[m.memory_type.value] = types_count.get(m.memory_type.value, 0) + 1
        return {
            "total_ltm": len(all_mems),
            "by_type": types_count,
            "stm_messages": len(self.stm.messages),
        }
