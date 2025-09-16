# Stock MCP Server Guide

This guide explains how the MCP server in `MCP/` works, how to configure environment variables, start the server, and how to use the exposed tool from MCP clients.

## What the server does

- Fetches OHLCV price data from Alpha Vantage
- Detects momentum events using:
  - Absolute percent change threshold (`threshold_abs_pct`, default 0.02)
  - Optional rolling z-score filter (`zscore_window`, default 50; threshold |z| > 3)
- Generates concise trading suggestions using:
  - Google Gemini (`GOOGLE_GEMINI_API_KEY`) or
  - OpenAI (`OPENAI_API_KEY`)
- If no AI keys are configured, it returns a neutral, risk-aware suggestion for each event

The MCP tool exposed is `get_stock_recommendations`.

## Project layout

- `stock_mcp_server/server/app.py`: MCP entrypoint (FastMCP) registering the tool
- `stock_mcp_server/tools/recommend.py`: Core logic (fetch, detect, AI suggestions)
- `pyproject.toml`: Package config and dependencies

## Requirements

- Python 3.9+
- Alpha Vantage API key is required for live data
- Optional Gemini/OpenAI API keys for AI suggestions

## Environment variables

Place a `.env` file in the repository root or inside `MCP/`:

```
ALPHAVANTAGE_API_KEY=your_alpha_vantage_key      # required
GOOGLE_GEMINI_API_KEY=your_gemini_key            # optional
GOOGLE_GEMINI_MODEL=gemini-1.5-flash             # optional (default)
OPENAI_API_KEY=your_openai_key                   # optional
OPENAI_MODEL=gpt-4o-mini                         # optional (default)
```

## Install

Create and activate a virtualenv, then install the package in editable mode:

```bash
cd MCP
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Start the server

Two main options via the `mcp` CLI (installed with this package):

### 1) Dev mode with Inspector (recommended for local testing)

```bash
mcp dev stock_mcp_server/server/app.py:main -e .
```

- Runs the server and opens the MCP Inspector in your browser
- `-e .` ensures the local package is used in editable mode

### 2) Direct run (stdio by default)

```bash
mcp run stock_mcp_server/server/app.py:main
```

- Suitable for embedding within a client that speaks MCP over stdio

## Using from a client (examples)

### From Claude Desktop (mcp install)

Install the server into Claude Desktop with environment configured:

```bash
# From the MCP directory
mcp install stock_mcp_server/server/app.py:main \
  -n "Stock Momentum MCP" \
  -e . \
  -f ../.env
```

- After installation, restart Claude Desktop. The server should appear in Settings → MCP Servers.

### From a custom MCP client

Call the tool `get_stock_recommendations` with parameters:

- `symbol` (e.g., `AAPL`)
- `function` (default `TIME_SERIES_INTRADAY`; for free demo testing use `TIME_SERIES_DAILY`)
- `interval` (for intraday, e.g., `5min`)
- `outputsize` (`compact` or `full`)
- `threshold_abs_pct` (default 0.02)
- `zscore_window` (default 50)
- `max_events` (default 8)
- `provider` (`gemini`, `openai`, or omit to auto-pick)

Example JSON-RPC tool invocation payload (schema shape):

```json
{
  "toolName": "get_stock_recommendations",
  "arguments": {
    "symbol": "AAPL",
    "function": "TIME_SERIES_INTRADAY",
    "interval": "5min",
    "outputsize": "compact",
    "threshold_abs_pct": 0.02,
    "zscore_window": 50,
    "max_events": 5,
    "provider": "gemini"
  }
}
```

### Returned fields

The tool returns a JSON object with:

- `symbol`: echo of requested ticker
- `meta`: Alpha Vantage meta-data
- `last_price`: last close value
- `events`: recent rows including
  - `timestamp`, `close`, `return_pct`, `return_zscore`, `momentum_event`, `direction`
- `suggestions`: list of AI suggestions per recent momentum event
- `provider`: which AI provider was used (`gemini`, `openai`, or `none`)

## Rate limits and errors

- Alpha Vantage free tier is rate-limited (5 req/min, 500/day). If you see messages containing `Note` or `Information`, you likely hit the limits. Back off and retry later, or switch to lower-frequency functions (`TIME_SERIES_DAILY`).
- If the AI provider keys are not set, suggestions are generated using a neutral fallback.

## Troubleshooting

- Ensure your `.env` is loaded (we call `dotenv` at startup). You can also export env vars in your shell before starting the server.
- Verify the server starts: `mcp dev stock_mcp_server/server/app.py:main -e .`
- Test a direct Python call (requires real API key):

```bash
python - <<PY
import asyncio, json
from stock_mcp_server.tools.recommend import get_stock_recommendations

async def main():
    res = await get_stock_recommendations(symbol=AAPL, function=TIME_SERIES_DAILY, outputsize=compact, max_events=3)
    print(json.dumps(res, indent=2))

import os
assert os.getenv(ALPHAVANTAGE_API_KEY), Set
