"""Smoke-test the MCP server by driving it as a real client over stdio.

Launches mcp_server/server.py as a subprocess, completes the MCP handshake,
lists the tools, and calls each one. This exercises the actual protocol path a
client such as Claude Desktop would use, rather than just importing the module.

    python scripts/test_mcp_server.py
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from mcp import ClientSession, StdioServerParameters  # noqa: E402
from mcp.client.stdio import stdio_client  # noqa: E402

SERVER = PROJECT_ROOT / "mcp_server" / "server.py"


def preview(text: str, limit: int = 400) -> str:
    text = text.strip()
    if len(text) <= limit:
        return text
    return text[:limit].rsplit(" ", 1)[0] + " ..."


async def main() -> int:
    params = StdioServerParameters(
        command=sys.executable,
        args=[str(SERVER)],
        env=None,
    )

    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            tools = (await session.list_tools()).tools
            print(f"\nConnected. {len(tools)} tools registered:\n")
            for tool in tools:
                first_line = (tool.description or "").strip().splitlines()[0]
                print(f"  {tool.name:<18} {first_line}")

            checks: list[tuple[str, dict]] = [
                ("list_documents", {"department": "engineering"}),
                ("search_database", {
                    "query": "why did robots hold charging dock reservations forever",
                    "top_k": 2,
                }),
                ("search_database", {
                    "query": "parental leave",
                    "department": "hr",
                    "top_k": 1,
                }),
                ("search_database", {"query": "", "top_k": 3}),
                ("search_database", {"query": "test", "department": "marketing"}),
                ("fetch_document", {
                    "source_file": "engineering/postmortem_INC-2024-017.txt",
                }),
                ("fetch_document", {"source_file": "../../secrets.txt"}),
                ("fetch_document", {"source_file": "nope/missing.txt"}),
                ("search_web", {"query": "ISO 3691-4 standard", "max_results": 2}),
            ]

            failures = 0
            for name, args in checks:
                shown = {k: v for k, v in args.items() if v not in (None, "")}
                print(f"\n{'=' * 72}\n{name}({shown})\n{'-' * 72}")
                try:
                    result = await session.call_tool(name, args)
                    text = "\n".join(
                        block.text for block in result.content
                        if getattr(block, "type", None) == "text"
                    )
                    print(preview(text))
                except Exception as exc:
                    failures += 1
                    print(f"FAILED: {type(exc).__name__}: {exc}")

            print(f"\n{'=' * 72}")
            print("All tool calls returned." if not failures
                  else f"{failures} call(s) raised.")
            return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
