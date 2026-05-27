# Agent Memory System — AI 记忆系统

一个模拟人类记忆机制的双存储器 AI 系统，支持短期记忆滑动窗口、长期记忆持久化存储、LLM 驱动的自动记忆提取与关系检测，以及 D3.js 实时知识图谱可视化。

---

## 核心概念

### 双存储器架构

| 存储器 | 类比 | 容量 | 持久化 | 实现 |
|---|---|---|---|---|
| **STM** (短期记忆) | 工作记忆 | 最多 20 条消息 | JSON 文件 | `deque` 滑动窗口 |
| **LTM** (长期记忆) | 永久记忆 | 无限 | Markdown 文件 | 文件系统 + 关系图谱 |

STM 是对话的即时上下文窗口。当 STM 接近容量上限时，最早的对话会被 LLM 自动摘要并压缩到 LTM 中。每次对话还会自动提取新的记忆（用户信息、偏好、项目进展等）存入 LTM。

### 四种记忆类型

| 类型 | 含义 | 例子 |
|---|---|---|
| `user` | 用户信息 | 姓名、职业、偏好、习惯 |
| `feedback` | 用户反馈 | "不要用 emoji"、"代码别加注释" |
| `project` | 项目相关 | 功能决定、Bug 修复、里程碑 |
| `reference` | 外部引用 | API 文档链接、工具名称 |

### 三种语义关系

记忆之间可以建立有向的语义关系：

| 关系 | 含义 | 对置信度的影响 |
|---|---|---|
| `extends` | 新记忆扩展/补充了已有记忆 | +0.02/条 |
| `supports` | 不同来源佐证同一事实 | +0.04/条 |
| `contradicts` | 内容直接矛盾 | −0.15/条 |

---

## 系统架构

```
用户输入 → FastAPI → MemoryAgent.chat()
                        ├── STM.add()          写入 storage/stm/*.json
                        ├── Retriever.retrieve()   TF-IDF → 嵌入 → 加权排序
                        ├── LLM 对话            构建提示 + 激活记忆
                        ├── STM 摘要            旧消息 → LLM → LTM
                        └── 记忆提取            对话 → LLM → 新记忆 + 关系
                                  ↓
                        SSE 实时推送 → 前端 D3.js 图谱
```

### 检索流程（混合检索）

1. **TF-IDF 粗排** — 纯 Python 实现，结巴中文分词，从所有 LTM 中召回 top-K 候选
2. **嵌入精排** — OpenAI `text-embedding-3-small` 计算查询与候选的余弦相似度
3. **后处理加权** — 时间衰减（7 天半衰期）+ 重要性加权 → 最终排序

### 置信度计算

每个记忆的置信度由多因素综合决定：

```
base = 0.15
     + importance × 0.12          (1→0.12, 5→0.60)
     + min(mentions, 10) × 0.04   (10 次提及 → +0.40)
     + min(accesses, 25) × 0.012  (25 次访问 → +0.30)

× 时间衰减    (14 天半衰期, 最低 0.5)
+ supports    × 0.04 (最多 +0.20)
+ extends     × 0.02 (最多 +0.10)
− contradicts × 0.15 (每条)

最终裁剪到 [0.05, 0.98]
```

低置信度 + 长期未访问的记忆会被标记为"过期"，可通过 API 清理。

---

## 项目结构

```
memory-viz/
├── server.py                  # FastAPI 后端（REST + SSE）
├── requirements.txt           # Python 依赖
├── run.bat                    # Windows 一键启动
├── agent/
│   ├── core.py                # MemoryAgent 编排层
│   └── memory/
│       ├── types.py           # 数据类型定义
│       ├── stm.py             # 短期记忆
│       ├── ltm.py             # 长期记忆（文件存储 + 关系）
│       ├── extraction.py      # LLM 提取 / 摘要 / 关系检测
│       └── retrieval.py       # 混合检索引擎
├── static/
│   └── index.html             # 前端（D3.js 可视化 + 聊天界面）
└── storage/
    ├── stm/                   # STM 会话文件
    ├── ltm/                   # LTM 记忆文件（每个记忆一个 .md）
    └── relations.json          # 关系三元组
```

---

## API 端点

### 对话
| 方法 | 路径 | 说明 |
|---|---|---|
| POST | `/api/chat` | 发送消息，返回回复 + 检索命中 + 提取结果 |

### 记忆 CRUD
| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/api/memories` | 列出所有长期记忆 |
| GET | `/api/memories/{id}` | 获取单个记忆完整内容 |
| POST | `/api/memories` | 手动创建记忆 |
| DELETE | `/api/memories/{id}` | 删除记忆 |
| POST | `/api/memories/dedup` | 自动合并相似记忆 |
| POST | `/api/memories/cleanup` | 清理过期记忆（可 dry-run） |
| GET | `/api/memories/compare/{id1}/{id2}` | 对比两个记忆 |

### 关系
| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/api/relations` | 列出所有语义关系 |
| POST | `/api/relations` | 手动添加关系 |
| POST | `/api/relations/detect` | LLM 自动检测所有记忆间关系 |
| DELETE | `/api/relations/{src}/{tgt}` | 删除关系 |

### 系统
| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/api/stats` | 记忆系统统计 |
| GET/POST | `/api/settings` | 读取/修改提取频率等参数 |
| POST | `/api/summarize` | 手动触发 STM 摘要 |
| GET | `/api/events` | SSE 实时事件流 |

---

## 前端功能

- **三栏布局** — 聊天面板 + 知识图谱 + 记忆列表
- **D3 力导向图** — 节点大小/透明度反映重要性与置信度，颜色区分记忆类型
- **流动粒子** — 语义关系线上有光点流动（extends/supports/contradicts 分别用蓝/绿/红）
- **Canvas 时间线** — 按记忆类型分泳道，支持缩放平移，雷达扫描线背景
- **雷达扫描线** — 青色渐变水平线周期性扫过背景网格
- **增长回放** — 按创建时间逐个揭示记忆节点
- **实时推送** — SSE 连接，新记忆提取时自动弹出庆祝动画
- **搜索高亮** — 客户端搜索，命中节点脉冲高亮
- **关系管理** — 可视化关系线，支持手动添加/LLM 自动检测

---

## 启动方式

```bash
pip install -r requirements.txt
python server.py
# 打开 http://localhost:8765
```

或 Windows 下直接双击 `run.bat`。

配置 API Key：修改 `server.py` 中的 `api_key` 和 `base_url`，或设置环境变量 `OPENAI_API_KEY` / `OPENAI_BASE_URL`。

---

## 设计理念

1. **文件即数据库** — 每个记忆是一个 Markdown 文件，人类可读、Git 可追踪
2. **记忆会衰减** — 不常用的记忆置信度逐渐降低，模拟人类遗忘曲线
3. **关系即证据** — 被多个记忆佐证的事实更可信，矛盾的记忆互相削弱
4. **零前端依赖** — 除了 D3.js CDN，纯原生 JS，无构建步骤
