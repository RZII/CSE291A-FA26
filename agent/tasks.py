"""Task sets: load the fixed questions, pick tools + system prompt per task type, grade answers.

    uv run python -m agent.tasks gsm8k 0      # run task #0 with one agent, print the trajectory + grade
    uv run python -m agent.tasks hotpotqa 3
    uv run python -m agent.tasks longqa 0
"""

import asyncio
import json
import re
import string
import sys
from collections import Counter
from functools import cache
from pathlib import Path

from agent.client import LLMClient
from agent.config import Config, load_config
from agent.loop import AgentResult, print_result, run_agent
from agent.tools import Tool
from agent.tools.python import make_python_tool
from agent.tools.read import make_read_tool
from agent.tools.search import BM25Index, make_search_tool

DATA = Path(__file__).resolve().parent.parent / "data"

GSM8K_PROMPT = (
    "You are a careful math assistant. Use the python tool for every calculation; do not compute in your head. "
    "When you have the result, give a one-sentence explanation and end with a line `Final answer: <number>`."
)
HOTPOTQA_PROMPTS = {
    "base": (
        "Answer the question using the search tool, which searches a collection of Wikipedia paragraphs. "
        "The question may need several searches: search for one thing at a time, read the results, "
        "then search again for what is still missing. Base your answer only on what you found. "
        "End with a line `Final answer: <short answer>`: just the answer, a few words at most."
    ),
    # item 8: baseline agents often stopped after one search and answered "unknown"
    "search_again": (
        "Answer the question using the search tool, which searches a collection of Wikipedia paragraphs. "
        "The question may need several searches: search for one thing at a time, read the results, "
        "then search again for what is still missing. If the results don't contain what you need, "
        "search again with a different query instead of answering unknown. Base your answer only on what you found. "
        "End with a line `Final answer: <short answer>`: just the answer, a few words at most."
    ),
}
# long-context task: the agent reads its own ~10K-word document page by page, so the context grows to 8-14K tokens
LONGQA_PROMPT = (
    "Answer the question using the read_page tool, which returns one page of a long document. The question says "
    "how many pages the document has. Read the pages one at a time, starting at page 1, and keep reading until you "
    "have found everything the question needs; the facts you need may be on different pages. Base your answer "
    "only on the document. End with a line `Final answer: <short answer>`: just the answer, a few words at most."
)


# tasks.response_length = "long": the same tasks, but the final reply carries a long explanation, so the workload
# has long responses (decode-heavy steps whose KV grows while generating). The answer line is unchanged.
GSM8K_LONG = (
    "give a one-sentence explanation",
    "write a detailed step-by-step explanation for a student (at least 500 words: what each quantity means "
    "and why each operation is needed),",
)
HOTPOTQA_LONG = (
    "End with a line",
    "Before the answer line, write a detailed explanation of at least 500 words: which paragraphs you found, "
    "what each one says, and how they combine into the answer. End with a line",
)


def long_answer(task: dict, cfg: Config) -> bool:
    """"mixed": every other task by its index in the task file, so both types get both lengths."""
    mode = cfg.tasks.response_length
    if mode not in ("short", "long", "mixed"):
        raise ValueError(f"unknown tasks.response_length {mode!r}")
    return mode == "long" or (mode == "mixed" and task.get("index", 0) % 2 == 1)


def system_prompt(kind: str, cfg: Config, long: bool = False) -> str:
    prompt = {"gsm8k": GSM8K_PROMPT, "longqa": LONGQA_PROMPT}.get(kind) or HOTPOTQA_PROMPTS[cfg.tasks.hotpotqa_prompt]
    return prompt.replace(*(GSM8K_LONG if kind == "gsm8k" else HOTPOTQA_LONG)) if long else prompt


def load_tasks(kind: str) -> list[dict]:
    return [json.loads(line) for line in open(DATA / f"{kind}_50.jsonl")]


def select_tasks(specs: list[str]) -> list[dict]:
    """["gsm8k:0-4", "hotpotqa:13"] -> those tasks, in that order."""
    tasks = []
    for spec in specs:
        kind, rng = spec.split(":")
        lo, _, hi = rng.partition("-")
        all_tasks = load_tasks(kind)
        tasks += [all_tasks[i] | {"index": i} for i in range(int(lo), int(hi or lo) + 1)]
    return tasks


@cache
def hotpotqa_index(corpus_file: str = "hotpotqa_corpus.jsonl") -> BM25Index:
    """Built once per process; read-only, so sharing it across agents shares no state."""
    return BM25Index([json.loads(line) for line in open(DATA / corpus_file)])


def make_tools(kind: str, cfg: Config, task: dict | None = None) -> list[Tool]:
    """`task` is needed for longqa only: its tool reads that task's own document."""
    if kind == "gsm8k":
        return [make_python_tool(cfg.tools)]
    if kind == "hotpotqa":
        return [make_search_tool(hotpotqa_index(cfg.tasks.hotpotqa_corpus), cfg.tools)]
    if kind == "longqa":
        return [make_read_tool(task["pages"], cfg.tools)]
    raise ValueError(f"unknown task type {kind!r}")


# ---- grading ----


def extract_final(text: str) -> str:
    """The text after the last `Final answer:` (to end of line), else the whole reply."""
    matches = re.findall(r"final answer\s*:\s*(.*)", text, flags=re.IGNORECASE)
    return matches[-1].strip().strip("`*").strip() if matches else text


NUMBER = re.compile(r"-?\d[\d,]*(?:\.\d+)?|-?\.\d+")


def parse_number(text: str, last: bool) -> float | None:
    nums = NUMBER.findall(text)
    if not nums:
        return None
    return float(nums[-1 if last else 0].replace(",", ""))


def grade_gsm8k(reply: str, gold: str) -> dict:
    has_final = re.search(r"final answer\s*:", reply, flags=re.IGNORECASE) is not None
    # "Final answer: 72 clips (48 + 24)" -> first number on that line; no such line -> last number in the reply
    pred = parse_number(extract_final(reply), last=not has_final)
    return {"pred": pred, "correct": pred is not None and abs(pred - float(gold)) < 1e-6}


def normalize_answer(s: str) -> str:
    """Official HotpotQA / SQuAD normalization."""
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def contains_tokens(pred: list[str], gold: list[str]) -> bool:
    n = len(gold)
    return n > 0 and any(pred[i : i + n] == gold for i in range(len(pred) - n + 1))


def f1_score(pred: str, gold: str) -> float:
    """Official HotpotQA F1 (yes/no/noanswer must match exactly)."""
    p, g = normalize_answer(pred), normalize_answer(gold)
    if (p in ("yes", "no", "noanswer") or g in ("yes", "no", "noanswer")) and p != g:
        return 0.0
    common = Counter(p.split()) & Counter(g.split())
    same = sum(common.values())
    if same == 0:
        return 0.0
    precision, recall = same / len(p.split()), same / len(g.split())
    return 2 * precision * recall / (precision + recall)


def grade_hotpotqa(reply: str, gold: str) -> dict:
    pred = extract_final(reply)
    return {
        "pred": pred,
        # token-sequence containment, so gold "no" does not match "not" / "know"
        "correct": contains_tokens(normalize_answer(pred).split(), normalize_answer(gold).split()),
        "em": normalize_answer(pred) == normalize_answer(gold),
        "f1": f1_score(pred, gold),
    }


def grade(task: dict, result: AgentResult) -> dict:
    """Only `answered` runs are graded; truncated / max_steps / error count as wrong."""
    if result.status != "answered":
        return {"pred": None, "correct": False}
    reply = result.final_answer or ""
    # longqa answers are HotpotQA answers (LongBench's HotpotQA subset), graded the same way
    return grade_gsm8k(reply, task["answer"]) if task["type"] == "gsm8k" else grade_hotpotqa(reply, task["answer"])


# ---- demo ----


async def _demo(kind: str, index: int):
    cfg = load_config()
    task = load_tasks(kind)[index]
    client = LLMClient(cfg.llm)
    try:
        prompt = system_prompt(kind, cfg, long_answer(task, cfg))
        res = await run_agent(task["question"], prompt, make_tools(kind, cfg, task), client, cfg.agent.max_steps)
    finally:
        await client.aclose()
    print_result(res)
    print(f"task={task['id']} gold={task['answer']!r} grade={grade(task, res)}")


if __name__ == "__main__":
    asyncio.run(_demo(sys.argv[1], int(sys.argv[2])))
