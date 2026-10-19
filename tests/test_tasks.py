from agent.loop import AgentResult
from agent.tasks import (
    extract_final,
    f1_score,
    grade,
    grade_gsm8k,
    grade_hotpotqa,
    hotpotqa_index,
    load_tasks,
)

# ---- the committed task files ----


def test_task_files():
    for kind in ("gsm8k", "hotpotqa"):
        tasks = load_tasks(kind)
        assert len(tasks) == 50
        assert len({t["id"] for t in tasks}) == 50
        assert all(t["type"] == kind and t["question"] and t["answer"] for t in tasks)


def test_corpus_has_every_gold_paragraph_once():
    titles = [p["title"] for p in hotpotqa_index().passages]
    assert len(titles) == len(set(titles))
    gold = {title for t in load_tasks("hotpotqa") for title in t["supporting_titles"]}
    assert gold <= set(titles)


# ---- grading ----


def test_extract_final():
    assert extract_final("blah\nFinal answer: 72 clips") == "72 clips"
    assert extract_final("Final answer: **Paris**") == "Paris"
    assert extract_final("no marker here") == "no marker here"


def test_gsm8k_grading():
    assert grade_gsm8k("She sold 72.\nFinal answer: 72", "72")["correct"]
    assert grade_gsm8k("Final answer: $1,000.00", "1000")["correct"]
    assert grade_gsm8k("Final answer: 72 clips (48 + 24)", "72")["correct"]  # first number on the line
    assert grade_gsm8k("48 in April and 24 in May, 72 total", "72")["correct"]  # no marker: last number
    assert not grade_gsm8k("Final answer: 71", "72")["correct"]
    assert not grade_gsm8k("I don't know", "72")["correct"]


def test_hotpotqa_containment_is_token_level():
    assert grade_hotpotqa("Final answer: It is a minor basilica.", "a minor basilica")["correct"]
    assert grade_hotpotqa("Final answer: The Beatles", "Beatles")["correct"]
    assert not grade_hotpotqa("Final answer: I do not know", "no")["correct"]  # substring would say yes
    assert grade_hotpotqa("Final answer: No", "no")["correct"]


def test_hotpotqa_em_f1():
    g = grade_hotpotqa("Final answer: minor basilica", "a minor basilica")
    assert g["em"] and g["f1"] == 1.0
    assert abs(f1_score("minor basilica in Florence", "a minor basilica") - 2 / 3) < 1e-9  # P=2/4, R=2/2
    assert f1_score("yes it is", "yes") == 0.0  # official rule: yes/no answers must match exactly
    assert f1_score("no", "yes") == 0.0


def test_only_answered_runs_are_graded():
    task = {"type": "gsm8k", "answer": "72"}
    truncated = AgentResult("truncated", "72 72 72 72", [], [])
    assert grade(task, truncated) == {"pred": None, "correct": False}
    answered = AgentResult("answered", "Final answer: 72", [], [])
    assert grade(task, answered)["correct"]
