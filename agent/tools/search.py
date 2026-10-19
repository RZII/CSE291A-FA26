"""search tool: BM25 over a fixed list of passages; returns the top-k as text.

A passage is {"title": str, "text": str}. The real corpus (all HotpotQA
paragraphs of the chosen questions) is built in item 4; any list works here.
"""

import re

from rank_bm25 import BM25Okapi

from agent.config import ToolsConfig
from agent.tools import Tool


def tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


class BM25Index:
    def __init__(self, passages: list[dict]):
        self.passages = passages
        self.bm25 = BM25Okapi([tokenize(p["title"] + " " + p["text"]) for p in passages])

    def search(self, query: str, k: int) -> list[dict]:
        scores = self.bm25.get_scores(tokenize(query))
        # ties broken by corpus order, so results are deterministic
        ranked = sorted(range(len(scores)), key=lambda i: (-scores[i], i))
        return [self.passages[i] for i in ranked[:k] if scores[i] > 0]


def format_results(passages: list[dict]) -> str:
    if not passages:
        return "No results found."
    return "\n\n".join(f"[{i}] {p['title']}\n{p['text']}" for i, p in enumerate(passages, 1))


def make_search_tool(index: BM25Index, cfg: ToolsConfig) -> Tool:
    async def search(query: str) -> str:
        return format_results(index.search(query, cfg.search_top_k))

    return Tool(
        name="search",
        description="Search a corpus of Wikipedia paragraphs. Returns the best-matching paragraphs with their titles.",
        parameters={"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
        fn=search,
        extra_delay_s=cfg.search_delay_s,
    )
