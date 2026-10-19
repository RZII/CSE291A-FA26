"""Build the fixed task sets in data/ (run once; the outputs are committed).

    uv run --group build python scripts/build_tasks.py

Downloads the raw datasets at pinned revisions into data/raw/ (gitignored), samples
N questions per dataset with a fixed seed, and writes:
    data/gsm8k_50.jsonl        {id, type, question, answer}
    data/hotpotqa_50.jsonl     {id, type, question, answer, qtype, supporting_titles}
    data/hotpotqa_corpus.jsonl {title, text}: every paragraph of the sampled HotpotQA
                               questions (2 gold + 8 distractors each), deduplicated by title
    data/hotpotqa_corpus_plus950.jsonl  the same + all paragraphs of 950 other dev questions
                               (extra distractors; a harder retrieval setting for item 8)
"""

import json
import random
import re
from pathlib import Path

import httpx
import pyarrow.parquet as pq

N = 50
SEED = 0
EXTRA_DISTRACTOR_QUESTIONS = 950
GSM8K_URL = (
    "https://raw.githubusercontent.com/openai/grade-school-math/"
    "3101c7d5072418e28b9008a6636bde82a006892c/grade_school_math/data/test.jsonl"
)
HOTPOT_URL = (
    "https://huggingface.co/datasets/hotpotqa/hotpot_qa/resolve/"
    "1908d6afbbead072334abe2965f91bd2709910ab/distractor/validation-00000-of-00001.parquet"
)

DATA = Path(__file__).resolve().parent.parent / "data"
RAW = DATA / "raw"


def download(url: str) -> Path:
    path = RAW / url.rsplit("/", 1)[-1]
    if not path.exists():
        RAW.mkdir(parents=True, exist_ok=True)
        with httpx.stream("GET", url, follow_redirects=True, timeout=60) as r:
            r.raise_for_status()
            with open(path, "wb") as f:
                for chunk in r.iter_bytes():
                    f.write(chunk)
    return path


def sample_indices(n_total: int) -> list[int]:
    return sorted(random.Random(SEED).sample(range(n_total), N))


def write_jsonl(path: Path, rows: list[dict]):
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows):>4} rows -> {path.relative_to(DATA.parent)}")


def build_gsm8k():
    rows = [json.loads(line) for line in open(download(GSM8K_URL))]
    tasks = []
    for i in sample_indices(len(rows)):
        answer = rows[i]["answer"].split("####")[-1].strip().replace(",", "")
        tasks.append({"id": f"gsm8k-test-{i:04d}", "type": "gsm8k", "question": rows[i]["question"], "answer": answer})
    write_jsonl(DATA / f"gsm8k_{N}.jsonl", tasks)


def add_paragraphs(row: dict, corpus: list[dict], seen: dict):
    for title, sentences in zip(row["context"]["title"], row["context"]["sentences"]):
        text = re.sub(r"\s+", " ", "".join(sentences)).strip()
        if title in seen:
            if seen[title] != text:
                print(f"  note: '{title}' appears twice with different text; keeping the first")
            continue
        seen[title] = text
        corpus.append({"title": title, "text": text})


def build_hotpotqa():
    rows = pq.read_table(download(HOTPOT_URL)).to_pylist()
    tasks, corpus, seen = [], [], {}
    sampled = sample_indices(len(rows))
    for i in sampled:
        r = rows[i]
        tasks.append({
            "id": f"hotpotqa-{r['id']}",
            "type": "hotpotqa",
            "question": r["question"],
            "answer": r["answer"],
            "qtype": r["type"],  # "bridge" | "comparison"
            "supporting_titles": sorted(set(r["supporting_facts"]["title"])),
        })
        add_paragraphs(r, corpus, seen)
    write_jsonl(DATA / f"hotpotqa_{N}.jsonl", tasks)
    write_jsonl(DATA / "hotpotqa_corpus.jsonl", corpus)

    sampled_set = set(sampled)
    others = [i for i in range(len(rows)) if i not in sampled_set]
    for i in sorted(random.Random(SEED + 1).sample(others, EXTRA_DISTRACTOR_QUESTIONS)):
        add_paragraphs(rows[i], corpus, seen)
    write_jsonl(DATA / f"hotpotqa_corpus_plus{EXTRA_DISTRACTOR_QUESTIONS}.jsonl", corpus)


if __name__ == "__main__":
    build_gsm8k()
    build_hotpotqa()
