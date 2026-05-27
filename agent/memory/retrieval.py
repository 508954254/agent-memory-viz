import re
import math
import time
import jieba
from collections import Counter
import numpy as np
from openai import OpenAI
from .types import Memory, RetrievalResult
from .ltm import LongTermMemory


# Time decay constants (in seconds)
HALF_LIFE = 7 * 24 * 3600  # 7 days — memories older than this get half weight


def _tokenize(text: str) -> list[str]:
    """Tokenize Chinese+English mixed text using jieba for Chinese."""
    text = text.lower().strip()
    tokens = []
    # Use jieba for the full text (handles Chinese well, English passably)
    words = jieba.lcut(text)
    for w in words:
        w = w.strip()
        if not w:
            continue
        # Add unigram
        tokens.append(w)
        # Add bigram for multi-character words
        if len(w) >= 2:
            for i in range(len(w) - 1):
                tokens.append(w[i:i+2])
    # Also extract English/ASCII words that jieba might split
    ascii_words = re.findall(r"[a-z0-9]{2,}", text)
    tokens.extend(ascii_words)
    return tokens


def _time_decay(last_accessed: float) -> float:
    """Exponential decay based on time since last access. Returns weight in [0.5, 1.0]."""
    if last_accessed <= 0:
        return 1.0
    elapsed = time.time() - last_accessed
    return 0.5 + 0.5 * math.exp(-elapsed * math.log(2) / HALF_LIFE)


class TfidfRanker:
    """Pure-Python TF-IDF (no sklearn dependency)."""

    def __init__(self):
        self.idf: dict[str, float] = {}
        self.doc_count = 0

    def fit(self, docs: list[str]):
        self.idf.clear()
        self.doc_count = len(docs)
        tokenized = [_tokenize(d) for d in docs]
        for tokens in tokenized:
            seen = set(tokens)
            for t in seen:
                self.idf[t] = self.idf.get(t, 0) + 1
        for t, n in self.idf.items():
            self.idf[t] = math.log((self.doc_count + 1) / (n + 1)) + 1

    def transform_one(self, text: str) -> dict[str, float]:
        tokens = _tokenize(text)
        if not tokens:
            return {}
        tf = Counter(tokens)
        vec = {}
        norm = 0.0
        for t, f in tf.items():
            if t in self.idf:
                w = f * self.idf[t]
                vec[t] = w
                norm += w * w
        if norm > 0:
            norm = math.sqrt(norm)
            for t in vec:
                vec[t] /= norm
        return vec

    def fit_transform_rank(self, query: str, docs: list[str], top_k: int) -> list[tuple[int, float]]:
        """Fit on docs, transform query, return ranked (doc_index, score) pairs."""
        self.fit(docs)
        qvec = self.transform_one(query)
        if not qvec:
            return []
        scores = []
        for i, doc in enumerate(docs):
            dvec = self.transform_one(doc)
            if not dvec:
                scores.append((i, 0.0))
                continue
            dot = sum(qvec.get(t, 0) * w for t, w in dvec.items())
            scores.append((i, dot))
        scores.sort(key=lambda x: x[1], reverse=True)
        return [(i, s) for i, s in scores if s > 0][:top_k]


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))


class MemoryRetriever:
    """Hybrid retrieval: TF-IDF coarse rank -> Embedding fine rank.
    With time decay and importance weighting."""

    def __init__(self, ltm: LongTermMemory, client: OpenAI,
                 embedding_model: str = "text-embedding-3-small"):
        self.ltm = ltm
        self.client = client
        self.embedding_model = embedding_model
        self._ranker = TfidfRanker()

    def retrieve(self, query: str, top_k: int = 5, coarse_k: int = None) -> list[RetrievalResult]:
        if coarse_k is None:
            coarse_k = min(top_k * 3, 20)

        memories = self.ltm.list_all()
        if not memories:
            return []

        # Step 1: Pure-Python TF-IDF coarse ranking
        docs = [f"{m.name} {m.description} {m.content}" for m in memories]
        ranked = self._ranker.fit_transform_rank(query, docs, coarse_k)
        if not ranked:
            return []

        coarse_ids = [memories[i].id for i, _ in ranked]

        # Step 2: Embedding fine ranking (with fallback)
        try:
            results = self._embedding_rerank(query, memories, coarse_ids, top_k)
        except Exception:
            results = []
            for i, score in ranked[:top_k]:
                results.append(RetrievalResult(memory=memories[i], score=score, method="tfidf"))

        # Step 3: Apply time decay and importance weighting
        for r in results:
            decay = _time_decay(r.memory.last_accessed)
            imp_weight = 0.7 + 0.075 * r.memory.importance  # imp=1 -> 0.775, imp=5 -> 1.075
            final_score = r.score * decay * imp_weight
            r.score = round(min(final_score, 1.0), 4)
            r.method = "hybrid"

        # Re-sort after weighting
        results.sort(key=lambda r: r.score, reverse=True)
        return results[:top_k]

    def _embedding_rerank(self, query: str, all_memories: list[Memory],
                          candidate_ids: list[str], top_k: int) -> list[RetrievalResult]:
        candidates = [m for m in all_memories if m.id in candidate_ids]
        texts = [f"{m.name}: {m.description}" for m in candidates]

        response = self.client.embeddings.create(
            model=self.embedding_model,
            input=[query] + texts,
        )
        embeddings = [np.array(d.embedding) for d in response.data]
        query_emb = embeddings[0]
        doc_embs = embeddings[1:]

        results = []
        for i, doc_emb in enumerate(doc_embs):
            sim = _cosine_similarity(query_emb, doc_emb)
            results.append(RetrievalResult(
                memory=candidates[i], score=sim, method="embedding"
            ))

        results.sort(key=lambda r: r.score, reverse=True)
        return results[:top_k]

    def rebuild_cache(self):
        self._ranker = TfidfRanker()
