"""
07: Agentic RAG with DSPy
- ReActエージェントが検索ツールを使って必要な情報を収集し、回答を生成する例。
- 外部APIではなくローカルの簡易KBを使うので、そのまま実行可能。
"""

from __future__ import annotations

import re
from typing import List

import dspy
from dotenv import load_dotenv

load_dotenv()

# LMの設定
lm = dspy.LM("openai/gpt-4o-mini")
dspy.configure(lm=lm)


# --- 簡易ナレッジベース ---
DOCUMENTS = [
    {
        "id": "doc-1",
        "title": "DSPy Overview",
        "text": "DSPy is a declarative framework for building modular AI programs. It uses signatures, modules, and optimizers.",
    },
    {
        "id": "doc-2",
        "title": "RAG Basics",
        "text": "RAG combines retrieval and generation. It retrieves relevant documents and uses them as context for an LLM.",
    },
    {
        "id": "doc-3",
        "title": "Agentic RAG",
        "text": "Agentic RAG adds a decision loop: plan, retrieve, observe, and re-plan. It can call tools multiple times.",
    },
    {
        "id": "doc-4",
        "title": "ReAct Pattern",
        "text": "ReAct interleaves reasoning and tool use. The agent thinks, acts, observes, then decides next steps.",
    },
    {
        "id": "doc-5",
        "title": "Trade-offs",
        "text": "Agentic RAG improves accuracy but increases latency and cost due to iterative tool calls.",
    },
]


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[a-zA-Z0-9]+", text.lower())


def search_kb(query: str, k: int = 3) -> List[str]:
    """簡易検索ツール: キーワード一致でスコアリングして上位k件を返す。"""
    q_tokens = set(_tokenize(query))
    scored = []
    for doc in DOCUMENTS:
        d_tokens = set(_tokenize(doc["text"] + " " + doc["title"]))
        score = len(q_tokens & d_tokens)
        scored.append((score, doc))

    scored.sort(key=lambda x: x[0], reverse=True)
    top = [d for score, d in scored if score > 0][:k]

    if not top:
        return ["No relevant documents found."]

    return [f"[{d['title']}] {d['text']}" for d in top]


# --- Agentic RAG (ReAct) ---
agent = dspy.ReAct(
    "question -> answer",
    tools=[search_kb],
)

question = (
    "Agentic RAGとは何で、通常のRAGと何が違いますか？ "
    "短くまとめてください。"
)

print("=== Agentic RAG (DSPy + ReAct) ===")
result = agent(question=question)
print(result.answer)
