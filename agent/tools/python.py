"""python tool: run model-written code in a subprocess, REPL-style.

If the last statement is an expression, its repr() is printed too, the way
Jupyter shows the last cell value (models often end with `total` instead of
`print(total)`). stdout and stderr are returned together so the model can see
its own tracebacks.

Isolation is light: `python -I` (ignores PYTHON* env vars and user site-packages),
a fresh temp dir as cwd, a timeout that kills the whole process group, and an
output cap. It guards against accidents, not against malicious code.

    uv run python -m agent.tools.python 'x = 48; x + x / 2'
"""

import asyncio
import os
import signal
import sys
import tempfile

from agent.config import ToolsConfig, load_config
from agent.tools import Tool

# Runs inside the subprocess; the model's code arrives on stdin.
RUNNER = r"""
import ast, sys, traceback
src = sys.stdin.read()
try:
    tree = ast.parse(src, "<code>")
    last = tree.body.pop() if tree.body and isinstance(tree.body[-1], ast.Expr) else None
    ns = {"__name__": "__main__"}
    exec(compile(tree, "<code>", "exec"), ns)
    if last is not None:
        value = eval(compile(ast.Expression(last.value), "<code>", "eval"), ns)
        if value is not None:
            print(repr(value))
except BaseException as e:
    # Hide this runner's own frame: show only the model's code (no frames at all for a SyntaxError).
    tb = None if isinstance(e, SyntaxError) else e.__traceback__.tb_next
    traceback.print_exception(type(e), e, tb)
"""


async def run_python(code: str, timeout_s: float, max_output_chars: int) -> str:
    with tempfile.TemporaryDirectory() as cwd:
        proc = await asyncio.create_subprocess_exec(
            sys.executable, "-I", "-c", RUNNER,
            stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT,
            cwd=cwd, start_new_session=True,  # own process group, so a timeout kills any children too
        )
        try:
            out, _ = await asyncio.wait_for(proc.communicate(code.encode()), timeout_s)
        except TimeoutError:
            os.killpg(proc.pid, signal.SIGKILL)
            await proc.wait()
            raise TimeoutError(f"timed out after {timeout_s:g}s") from None  # run_tool turns it into "Error: ..."

    text = out.decode(errors="replace")
    if len(text) > max_output_chars:
        text = text[:max_output_chars] + f"\n[truncated: output was {len(text)} chars]"
    return text


def make_python_tool(cfg: ToolsConfig) -> Tool:
    async def python(code: str) -> str:
        return await run_python(code, cfg.python_timeout_s, cfg.python_max_output_chars)

    return Tool(
        name="python",
        description="Run Python code and return its output: stdout, plus the value of the last line if it is an expression.",
        parameters={"type": "object", "properties": {"code": {"type": "string"}}, "required": ["code"]},
        fn=python,
        extra_delay_s=cfg.python_delay_s,
    )


if __name__ == "__main__":
    cfg = load_config().tools
    print(asyncio.run(run_python(sys.argv[1], cfg.python_timeout_s, cfg.python_max_output_chars)), end="")
