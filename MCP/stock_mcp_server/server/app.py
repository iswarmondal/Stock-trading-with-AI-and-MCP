from dotenv import load_dotenv, find_dotenv

try:
    from mcp.server.fastmcp import FastMCP
except Exception:  # pragma: no cover
    from mcp.server.fastmcp import FastMCP

from stock_mcp_server.tools.recommend import get_stock_recommendations


def main() -> None:
    load_dotenv(find_dotenv(usecwd=True))

    server = FastMCP(
        name="stock-mcp-server",
        version="0.1.0",
        description="Provides stock momentum analysis and AI recommendations",
    )

    server.tool()(get_stock_recommendations)

    server.run()


if __name__ == "__main__":
    main()
