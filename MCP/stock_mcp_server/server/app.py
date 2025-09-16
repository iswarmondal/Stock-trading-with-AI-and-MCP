import os
from dotenv import load_dotenv, find_dotenv

try:
    from mcp.server.fastmcp import FastMCP
except Exception:  # pragma: no cover
    from mcp.server.fastmcp import FastMCP

from stock_mcp_server.tools.recommend import get_stock_recommendations


def create_server() -> FastMCP:
    load_dotenv(find_dotenv(usecwd=True))
    host = os.getenv("FASTMCP_HOST", os.getenv("HOST", "0.0.0.0"))
    port = int(os.getenv("FASTMCP_PORT", os.getenv("PORT", "8765")))
    mcp_app = FastMCP(name="stock-mcp-server", host=host, port=port)
    mcp_app.tool()(get_stock_recommendations)
    return mcp_app


app: FastMCP = create_server()
server: FastMCP = app


def main() -> None:
    app.run("sse")


if __name__ == "__main__":
    main()
