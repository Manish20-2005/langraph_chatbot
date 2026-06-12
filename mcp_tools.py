"""
mcp_tools.py
Loads tools from external MCP (Model Context Protocol) servers and exposes
them as LangChain-compatible tools that can be bound to the agent alongside
the built-in tools in tools.py.

How it works
------------
MCP servers are declared in `mcp_config.json` (same shape as the Claude
Desktop / Cursor MCP config format):

    {
      "mcpServers": {
        "filesystem": {
          "transport": "stdio",
          "command": "npx",
          "args": ["-y", "@modelcontextprotocol/server-filesystem", "./mcp_data"]
        },
        "fetch": {
          "transport": "stdio",
          "command": "uvx",
          "args": ["mcp-server-fetch"]
        },
        "remote-example": {
          "transport": "streamable_http",
          "url": "https://example.com/mcp"
        }
      }
    }

Enable / disable
-----------------
Set `ENABLE_MCP=true` in `.env` to turn this on. If disabled, missing, or
no servers are configured, `get_mcp_tools()` simply returns an empty list
and the chatbot behaves exactly as before — MCP is fully optional.

Install dependencies:
    pip install langchain-mcp-adapters
"""

import asyncio
import json
import os
from pathlib import Path

MCP_CONFIG_PATH = Path("mcp_config.json")


def _load_config() -> dict:
    """Read mcp_config.json and return the `mcpServers` dict (or {})."""
    if not MCP_CONFIG_PATH.exists():
        return {}

    try:
        with open(MCP_CONFIG_PATH, "r", encoding="utf-8") as f:
            config = json.load(f)
        return config.get("mcpServers", {})
    except Exception as e:
        print(f"[MCP] Failed to read '{MCP_CONFIG_PATH}': {e}")
        return {}


async def _fetch_tools(servers: dict):
    """Connect to all configured MCP servers and collect their tools."""
    from langchain_mcp_adapters.client import MultiServerMCPClient

    client = MultiServerMCPClient(servers)
    tools = await client.get_tools()
    return tools


def get_mcp_tools() -> list:
    """
    Synchronously load and return all tools exposed by configured MCP
    servers.

    Returns:
        A list of LangChain tool objects (empty if MCP is disabled, no
        servers are configured, or connection fails).
    """
    if os.getenv("ENABLE_MCP", "false").lower() not in ("1", "true", "yes"):
        return []

    servers = _load_config()
    if not servers:
        print(
            "[MCP] ENABLE_MCP=true but no servers found in "
            f"'{MCP_CONFIG_PATH}'. Skipping."
        )
        return []

    try:
        tools = asyncio.run(_fetch_tools(servers))
        names = ", ".join(t.name for t in tools) if tools else "none"
        print(
            f"[MCP] ✅ Loaded {len(tools)} tool(s) from "
            f"{len(servers)} server(s): {names}"
        )
        return tools
    except Exception as e:
        print(f"[MCP] ⚠️  Failed to load MCP tools — continuing without them: {e}")
        return []