import asyncio
import json
import os
import time
from collections import deque
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from agent.core import MemoryAgent
from agent.memory.types import MemoryType, RelationType
from agent.memory.extraction import detect_relations_between_memories

app = FastAPI(title="Agent Memory System")

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# SSE event queue (in-memory pub/sub)
_event_queues: list[asyncio.Queue] = []


def _emit(event_type: str, data: dict):
    """Push an event to all connected SSE clients."""
    payload = f"event: {event_type}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"
    for q in _event_queues:
        try:
            q.put_nowait(payload)
        except asyncio.QueueFull:
            pass


# Initialize agent
agent = MemoryAgent(
    api_key="",          # 你的API Key
    base_url="",    # API地址
    model="gpt-4.1",       # 对话模型
    embedding_model="text-embedding-3-small"  # 嵌入模型
)

# Wire agent events to SSE
def on_retrieve(query: str, results):
    _emit("retrieve", {
        "query": query,
        "hits": [r.to_dict() for r in results],
        "timestamp": time.time(),
    })

def on_extract(memories: list[dict]):
    _emit("extract", {
        "memories": memories,
        "timestamp": time.time(),
    })

agent.on_retrieve(on_retrieve)
agent.on_extract(on_extract)


# --- REST API ---

class ChatRequest(BaseModel):
    message: str

class MemoryCreate(BaseModel):
    name: str
    description: str
    memory_type: str
    content: str
    tags: list[str] = []
    importance: int = 3

@app.post("/api/chat")
async def chat(req: ChatRequest):
    """Send a message to the agent. Returns reply + memory hits + extractions."""
    try:
        result = agent.chat(req.message)
        return result
    except Exception as e:
        return {"reply": f"API调用失败: {str(e)[:200]}", "retrieved": [], "extracted": []}

@app.get("/api/memories")
async def list_memories():
    """List all long-term memories."""
    memories = agent.ltm.list_all()
    return [
        {
            "id": m.id,
            "name": m.name,
            "description": m.description,
            "type": m.memory_type.value,
            "content": m.content[:300] + ("..." if len(m.content) > 300 else ""),
            "tags": m.tags,
            "references": m.references,
            "importance": m.importance,
            "confidence": m.confidence,
            "mention_count": m.mention_count,
            "access_count": m.access_count,
            "last_accessed": m.last_accessed,
            "created": m.created,
            "updated": m.updated,
        }
        for m in memories
    ]

@app.get("/api/memories/{mem_id}")
async def get_memory(mem_id: str):
    """Get full content of a single memory."""
    mem = agent.ltm.get(mem_id)
    if not mem:
        raise HTTPException(status_code=404, detail="Memory not found")
    return {
        "id": mem.id,
        "name": mem.name,
        "description": mem.description,
        "type": mem.memory_type.value,
        "content": mem.content,
        "tags": mem.tags,
        "references": mem.references,
        "importance": mem.importance,
        "confidence": mem.confidence,
        "mention_count": mem.mention_count,
        "access_count": mem.access_count,
        "last_accessed": mem.last_accessed,
        "created": mem.created,
        "updated": mem.updated,
    }

@app.post("/api/memories")
async def create_memory(req: MemoryCreate):
    """Manually create a memory."""
    mem = agent.ltm.create(
        name=req.name,
        description=req.description,
        memory_type=MemoryType(req.memory_type),
        content=req.content,
        tags=req.tags,
        importance=req.importance,
    )
    agent.retriever.rebuild_cache()
    _emit("extract", {"memories": [{"name": mem.name, "type": mem.memory_type.value, "id": mem.id}], "timestamp": time.time()})
    return {"id": mem.id, "name": mem.name}

@app.delete("/api/memories/{mem_id}")
async def delete_memory(mem_id: str):
    """Delete a memory."""
    ok = agent.ltm.delete(mem_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Memory not found")
    agent.retriever.rebuild_cache()
    return {"deleted": mem_id}

@app.post("/api/memories/dedup")
async def deduplicate_memories():
    """Merge near-duplicate memories retroactively."""
    merges = agent.ltm.deduplicate()
    if merges:
        agent.retriever.rebuild_cache()
    return {"merges": merges, "count": len(merges)}


@app.post("/api/memories/cleanup")
async def cleanup_stale_memories(confidence_threshold: float = 0.35, days_threshold: int = 14, dry_run: bool = True):
    """Find and optionally delete stale (low-confidence, unaccessed) memories."""
    stale = agent.ltm.get_stale_memories(confidence_threshold, days_threshold)
    deleted = []
    if not dry_run:
        for m in stale:
            agent.ltm.delete(m.id)
            deleted.append({"id": m.id, "name": m.name, "confidence": m.confidence})
        if deleted:
            agent.retriever.rebuild_cache()
    return {
        "stale": [{"id": m.id, "name": m.name, "confidence": m.confidence,
                    "last_accessed": m.last_accessed, "created": m.created} for m in stale],
        "deleted": deleted,
        "count": len(stale),
        "dry_run": dry_run,
    }


# --- Relations ---

@app.get("/api/relations")
async def get_relations():
    """Get all typed relations between memories."""
    return [
        {
            "id": r.id,
            "source_id": r.source_id,
            "target_id": r.target_id,
            "relation_type": r.relation_type.value,
            "description": r.description,
            "confidence": r.confidence,
            "created": r.created,
        }
        for r in agent.ltm.get_all_relations()
    ]


@app.post("/api/relations")
async def add_relation(source_id: str, target_id: str, relation_type: str, description: str = "", confidence: float = 0.7):
    """Add a typed relation between two memories."""
    try:
        rt = RelationType(relation_type)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid relation type: {relation_type}")
    rel = agent.ltm.add_relation(source_id, target_id, rt, description, confidence)
    return rel.to_dict()


@app.post("/api/relations/detect")
async def detect_relations():
    """Auto-detect semantic relations among all existing memories using LLM."""
    all_mems = agent.ltm.list_all()
    mem_dicts = [
        {"name": m.name, "description": m.description, "type": m.memory_type.value}
        for m in all_mems
    ]
    if len(mem_dicts) < 2:
        return {"relations": [], "count": 0}

    detected = detect_relations_between_memories(agent.client, mem_dicts, agent.model)
    saved = []
    for rel in detected:
        source_name = rel.get("source", "")
        target_name = rel.get("target", "")
        rel_type = rel.get("relation", "extends")
        desc = rel.get("description", "")
        conf = rel.get("confidence", 0.7)

        source_mem = agent.ltm.find_by_name(source_name)
        target_mem = agent.ltm.find_by_name(target_name)
        if not source_mem or not target_mem or source_mem.id == target_mem.id:
            continue
        try:
            rt = RelationType(rel_type)
        except ValueError:
            continue
        r = agent.ltm.add_relation(source_mem.id, target_mem.id, rt, desc, conf)
        if r:
            saved.append(r.to_dict())

    _emit("extract", {"memories": [], "timestamp": time.time()})
    return {"relations": saved, "count": len(saved)}


@app.delete("/api/relations/{source_id}/{target_id}")
async def delete_relation(source_id: str, target_id: str):
    """Delete a relation between two memories."""
    agent.ltm.delete_relation(source_id, target_id)
    return {"deleted": f"{source_id}--{target_id}"}


# --- Compare ---

@app.get("/api/memories/compare/{id1}/{id2}")
async def compare_memories(id1: str, id2: str):
    """Compare two memories side by side."""
    m1 = agent.ltm.get(id1)
    m2 = agent.ltm.get(id2)
    if not m1 or not m2:
        raise HTTPException(status_code=404, detail="One or both memories not found")
    # Find relations between them
    rels = [r for r in agent.ltm.get_all_relations()
            if (r.source_id == id1 and r.target_id == id2) or
               (r.source_id == id2 and r.target_id == id1)]
    return {
        "memory_a": {
            "id": m1.id, "name": m1.name, "description": m1.description,
            "type": m1.memory_type.value, "content": m1.content, "tags": m1.tags,
            "importance": m1.importance, "confidence": m1.confidence,
            "mention_count": m1.mention_count, "access_count": m1.access_count,
            "created": m1.created, "updated": m1.updated,
        },
        "memory_b": {
            "id": m2.id, "name": m2.name, "description": m2.description,
            "type": m2.memory_type.value, "content": m2.content, "tags": m2.tags,
            "importance": m2.importance, "confidence": m2.confidence,
            "mention_count": m2.mention_count, "access_count": m2.access_count,
            "created": m2.created, "updated": m2.updated,
        },
        "relations": [r.to_dict() for r in rels],
    }


# --- Settings ---

@app.get("/api/settings")
async def get_settings():
    """Get agent settings."""
    return agent.get_settings()


class SettingsUpdate(BaseModel):
    extract_every: int | None = None
    summarize_threshold: int | None = None
    retrieve_top_k: int | None = None


@app.post("/api/settings")
async def update_settings(req: SettingsUpdate):
    """Update agent settings."""
    agent.set_settings(extract_every=req.extract_every, summarize_threshold=req.summarize_threshold,
                       retrieve_top_k=req.retrieve_top_k)
    return agent.get_settings()


# --- Summarize ---

@app.post("/api/summarize")
async def trigger_summarize():
    """Manually trigger STM summarization."""
    ok = agent._maybe_summarize()
    return {"summarized": ok}


@app.get("/api/stats")
async def get_stats():
    """Get memory system stats."""
    return agent.get_stats()

@app.get("/api/events")
async def sse_events():
    """Server-Sent Events stream for real-time memory visualization."""
    queue: asyncio.Queue = asyncio.Queue(maxsize=100)
    _event_queues.append(queue)
    try:
        async def generate():
            # Send initial connection event with all current memories
            memories = agent.ltm.list_all()
            init_data = json.dumps({
                "type": "init",
                "memories": [
                    {
                        "id": m.id, "name": m.name, "description": m.description,
                        "type": m.memory_type.value, "tags": m.tags,
                        "references": m.references,
                        "importance": m.importance, "confidence": m.confidence,
                        "mention_count": m.mention_count, "access_count": m.access_count,
                        "last_accessed": m.last_accessed, "created": m.created,
                    }
                    for m in memories
                ],
                "stats": agent.get_stats(),
            }, ensure_ascii=False)
            yield f"event: init\ndata: {init_data}\n\n"

            while True:
                try:
                    data = await asyncio.wait_for(queue.get(), timeout=30)
                    yield data
                except asyncio.TimeoutError:
                    yield ": heartbeat\n\n"
        return StreamingResponse(generate(), media_type="text/event-stream")
    finally:
        _event_queues.remove(queue)


# Serve static frontend
static_dir = Path(__file__).parent / "static"
if static_dir.exists():
    @app.get("/")
    async def index():
        content = (static_dir / "index.html").read_text(encoding="utf-8")
        return HTMLResponse(content, headers={"Cache-Control": "no-cache, no-store, must-revalidate"})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8765)
