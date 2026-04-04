import asyncio
import mcp.types as types
from mcp.server import Server
from mcp.server.stdio import stdio_server
from .tools import TOOLS, handle_tool

server = Server("research-library")


@server.list_tools()
async def list_tools() -> list[types.Tool]:
    return TOOLS


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[types.TextContent]:
    return await handle_tool(name, arguments)


async def _run():
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options(),
        )


def main():
    asyncio.run(_run())


if __name__ == "__main__":
    main()
