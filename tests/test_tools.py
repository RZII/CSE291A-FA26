import asyncio
import json
import time
from dataclasses import replace

import pytest

from agent.config import load_config
from agent.tools import Tool
from agent.tools.python import make_python_tool, run_python
from agent.tools.search import BM25Index, format_results, make_search_tool

CFG = load_config().tools


def py(code, timeout_s=CFG.python_timeout_s, max_chars=CFG.python_max_output_chars):
    return asyncio.run(run_python(code, timeout_s, max_chars))


# ---- python ----

def test_print_output():
    assert py("print(37 * 89 + 15)") == "3308\n"


def test_last_expression_value_is_returned():
    assert py("April = 48\nMay = April / 2\nApril + May") == "72.0\n"


def test_print_and_last_expression_both_shown():
    assert py("print('hi')\n'x' * 3") == "hi\n'xxx'\n"


def test_none_and_non_expression_last_line_print_nothing():
    assert py("x = 1") == ""
    assert py("None") == ""


def test_runtime_error_shows_only_model_code_in_traceback():
    out = py("a = 1\nb = a / 0")
    assert "ZeroDivisionError: division by zero" in out
    assert 'File "<code>", line 2' in out
    assert "<string>" not in out  # the runner's own frame is hidden


def test_syntax_error():
    out = py("x = (1 +")
    assert "SyntaxError" in out and "<string>" not in out


def test_timeout_kills_process():
    t0 = time.perf_counter()
    with pytest.raises(TimeoutError, match="timed out after 1s"):
        py("while True: pass", timeout_s=1)
    assert time.perf_counter() - t0 < 3


def test_output_is_capped():
    out = py("print('a' * 10000)", max_chars=100)
    assert out.startswith("a" * 100) and "[truncated: output was 10001 chars]" in out


def test_runs_in_temp_dir_not_repo():
    assert "291" not in py("import os; os.getcwd()")


# ---- extra_delay ----

def test_delay_is_not_in_schema_the_model_sees():
    tool = make_python_tool(replace(CFG, python_delay_s=1.5))
    assert tool.extra_delay_s == 1.5
    assert "delay" not in json.dumps(tool.schema())


def test_delay_applies_and_does_not_block_other_calls():
    async def echo(x: str) -> str:
        return x

    slow = Tool("echo", "", {}, echo, extra_delay_s=0.3)

    async def two_at_once():
        return await asyncio.gather(slow.run(x="a"), slow.run(x="b"))

    t0 = time.perf_counter()
    assert asyncio.run(two_at_once()) == ["a", "b"]
    elapsed = time.perf_counter() - t0
    assert 0.3 <= elapsed < 0.5  # both waited 0.3s, concurrently (not 0.6s in a row)


# ---- search ----

CORPUS = [
    {"title": "Paris", "text": "Paris is the capital and largest city of France."},
    {"title": "Berlin", "text": "Berlin is the capital of Germany."},
    {"title": "Eiffel Tower", "text": "The Eiffel Tower is a wrought-iron tower in Paris, France."},
    {"title": "Banana", "text": "A banana is an elongated, edible fruit."},
    # filler: BM25's idf = log((N - n + 0.5) / (n + 0.5)) is 0 for a term in exactly half the docs,
    # so a 4-doc corpus would score "capital" / "paris" as 0. A few unrelated docs fix that.
    {"title": "Oak", "text": "An oak is a tree or shrub in the genus Quercus."},
    {"title": "Violin", "text": "The violin is a wooden string instrument."},
    {"title": "Mercury", "text": "Mercury is the smallest planet in the Solar System."},
    {"title": "Chess", "text": "Chess is a board game for two players."},
]


def test_search_ranks_relevant_first_and_respects_k():
    hits = BM25Index(CORPUS).search("capital of Germany", k=2)
    assert hits[0]["title"] == "Berlin" and len(hits) == 2


def test_search_no_match():
    assert BM25Index(CORPUS).search("quantum chromodynamics", k=5) == []
    assert format_results([]) == "No results found."


def test_search_tool_output_format():
    tool = make_search_tool(BM25Index(CORPUS), replace(CFG, search_top_k=2))
    out = asyncio.run(tool.run(query="Eiffel Tower Paris"))
    assert out.startswith("[1] Eiffel Tower\nThe Eiffel Tower is")
    assert "\n\n[2] Paris\n" in out and "[3]" not in out
