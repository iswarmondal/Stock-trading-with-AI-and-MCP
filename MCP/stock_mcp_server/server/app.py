from dotenv import load_dotenv, find_dotenv

try:
    from mcp.server.fastmcp import FastMCP
except Exception:  # pragma: no cover
    from mcp.server.fastmcp import FastMCP

from stock_mcp_server.tools.recommend import get_stock_recommendations


def create_server() -> FastMCP:
    load_dotenv(find_dotenv(usecwd=True))
    mcp_app = FastMCP(name="stock-mcp-server", host="127.0.0.1", port=8765)
    mcp_app.tool()(get_stock_recommendations)
    return mcp_app


app: FastMCP = create_server()
server: FastMCP = app


def main() -> None:
    app.run("sse")


if __name__ == "__main__":
    main()
