"""read_page tool: one page of the task's own long document per call (task type "longqa").

The document is split into pages when the task set is built (scripts/build_tasks.py), so every agent that
reads page n of a task sees the same text. Each page appended to the conversation grows the context by
~1.4K tokens, which is what makes these tasks long-context.
"""

from agent.config import ToolsConfig
from agent.tools import Tool


def make_read_tool(pages: list[str], cfg: ToolsConfig) -> Tool:
    async def read_page(page: int) -> str:
        if not 1 <= page <= len(pages):
            return f"There is no page {page}; the document has pages 1 to {len(pages)}."
        return f"[page {page} of {len(pages)}]\n{pages[page - 1]}"

    return Tool(
        name="read_page",
        description="Read one page of the document. Pages are numbered from 1.",
        parameters={"type": "object", "properties": {"page": {"type": "integer"}}, "required": ["page"]},
        fn=read_page,
        extra_delay_s=cfg.search_delay_s,  # the document tool shares the retrieval tool's delay knob
    )
