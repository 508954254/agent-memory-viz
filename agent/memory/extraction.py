import json
from openai import OpenAI
from .types import MemoryType


EXTRACTION_PROMPT = """你是一个记忆提取器。分析下面的对话，提取出值得长期记住的信息。

## 记忆类型
- user: 关于用户的信息（角色、偏好、习惯、知识背景）
- feedback: 用户对你行为的反馈（什么做得好、什么不要做）
- project: 项目相关的进展、决定、计划
- reference: 外部资源或工具的引用

## 提取字段
- name: 简短记忆名称（中文，5-15字）
- description: 一句话描述（中文，15-40字）
- type: 类型（user/feedback/project/reference）
- content: 详细内容（中文，2-5句话）
- importance: 1-5（5=姓名身份，4=重要偏好，3=一般信息，2=琐碎，1=可忽略）
- tags: 标签列表
- action: "create" 或 "update"
- update_target: 若 action="update"，填已有记忆名称

## 记忆关系检测
分析新提取的记忆与已有记忆之间是否存在以下关系：

- extends（扩展）: 新记忆为已有记忆补充了细节、例子或新角度。如"喜欢打篮球"和"喜欢看NBA"→ extends
- supports（佐证）: 新记忆从不同来源印证了已有记忆。如"网安专业"和"在学习渗透测试"→ supports
- contradicts（矛盾）: 新记忆与已有记忆内容直接冲突。如"就读广州大学"和"就读清华大学"→ contradicts

关系示例：
已有: [user] 喜欢打篮球 - 用户喜欢打篮球
新: [user] 喜欢看NBA - 用户喜欢看NBA
→ relation: extends, description: "从打球扩展到看NBA，同属篮球兴趣"

已有: [user] 就读广州大学 - 用户就读广州大学
新: [user] 就读清华大学 - 用户就读清华大学
→ relation: contradicts, description: "两所学校不同，存在信息矛盾"

返回 relations 数组，每条：
- source: 已有记忆名称（从已有记忆列表精确复制）
- target: 新记忆名称（本次提取的 memories 中 name 字段值，与 source 不同）
- relation: "extends" / "supports" / "contradicts"
- confidence: 0.0-1.0 表示对该关系的把握度（0.9=非常确定，0.6=有一定可能，0.3=猜测）
- description: 关系说明（中文，10-30字）

只返回确实存在语义关系的记忆对。没有就返回空数组[]。

## 当前已有记忆
{existing_memories}

## 对话
{dialogue}

直接返回JSON对象: {"memories": [...], "relations": [...]}，不要有其他内容:"""


def extract_memories(client: OpenAI, dialogue: str, model: str = "gpt-4o-mini",
                     existing_memories: list[dict] = None) -> tuple[list[dict], list[dict]]:
    """Extract memories and detect relationships from a conversation using LLM.
    Returns (memories, relations)."""
    if not dialogue.strip():
        return [], []

    if existing_memories:
        lines = []
        for m in existing_memories:
            lines.append(f"- [{m['type']}] {m['name']}: {m['description']}")
        existing_text = "\n".join(lines)
    else:
        existing_text = "（暂无已有记忆）"

    prompt = EXTRACTION_PROMPT.replace("{existing_memories}", existing_text).replace("{dialogue}", dialogue)
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=2000,
        )
        text = response.choices[0].message.content
        if not text:
            print("[Extraction] Empty response from LLM")
            return [], []
        text = text.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1].rsplit("\n", 1)[0]
        result = json.loads(text)
        if isinstance(result, list):
            return result, []
        return result.get("memories", []), result.get("relations", [])
    except json.JSONDecodeError as e:
        print(f"[Extraction] JSON parse error: {e}")
        print(f"[Extraction] Raw text (first 500 chars): {text[:500] if 'text' in dir() else 'N/A'}")
        return [], []
    except Exception as e:
        print(f"[Extraction] Error: {e}")
        return [], []


SUMMARIZE_PROMPT = """你是一个对话摘要器。将以下对话片段压缩成简短摘要。

摘要要求：
- 中文，30-80字
- 只记录有长期价值的信息（用户表达的观点、偏好、决定、事实）
- 纯闲聊内容忽略
- 格式：一句或两句话概括

对话:
{dialogue}

直接返回摘要文本，不要引号或其他格式:"""


def summarize_dialogue(client: OpenAI, dialogue: str, model: str = "gpt-4o-mini") -> str:
    """Summarize old STM messages into a brief memory-worthy note."""
    if not dialogue.strip():
        return ""
    prompt = SUMMARIZE_PROMPT.replace("{dialogue}", dialogue)
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=300,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"[Summarize] Error: {e}")
        return ""


RELATION_DETECT_PROMPT = """你是一个记忆关系分析器。分析下面所有已有记忆，找出存在语义关系的记忆对。

## 关系类型
- extends（扩展）: 一条记忆为另一条补充了细节、例子或具体化。如"喜欢打篮球"和"喜欢看NBA"→ extends
- supports（佐证）: 两条记忆从不同角度支撑同一个事实或目标。如"网安专业"和"在学习渗透测试"→ supports
- contradicts（矛盾）: 两条记忆内容直接冲突。如"就读广州大学"和"就读清华大学"→ contradicts

## 规则
- source 和 target 必须是不同的记忆
- 只返回确实有明确语义关系的记忆对
- 没有关系的记忆对不要列出来

## 所有记忆列表
{memories_list}

返回JSON: {"relations": [{"source": "记忆名称", "target": "记忆名称", "relation": "extends|supports|contradicts", "confidence": 0.0-1.0, "description": "中文10-30字"}]}
没有就返回 {"relations": []}。不要有其他内容:"""


def detect_relations_between_memories(client: OpenAI, memories: list[dict],
                                       model: str = "gpt-4.1") -> list[dict]:
    """Detect semantic relations among all existing memories using LLM."""
    if len(memories) < 2:
        return []

    lines = []
    for m in memories:
        lines.append(f"- [{m.get('type', '?')}] {m['name']}: {m.get('description', '')}")
    memories_text = "\n".join(lines)

    prompt = RELATION_DETECT_PROMPT.replace("{memories_list}", memories_text)
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=2000,
        )
        text = response.choices[0].message.content
        if not text:
            print("[RelationDetect] Empty response from LLM")
            return []
        text = text.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1].rsplit("\n", 1)[0]
        result = json.loads(text)
        return result.get("relations", [])
    except json.JSONDecodeError as e:
        print(f"[RelationDetect] JSON parse error: {e}")
        return []
    except Exception as e:
        print(f"[RelationDetect] Error: {e}")
        return []
