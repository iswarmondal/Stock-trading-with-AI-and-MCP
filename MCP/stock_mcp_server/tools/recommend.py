import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv, find_dotenv

try:
    import google.generativeai as genai  # type: ignore
except Exception:  # pragma: no cover
    genai = None  # type: ignore

try:
    from openai import OpenAI  # type: ignore
except Exception:  # pragma: no cover
    OpenAI = None  # type: ignore


load_dotenv(find_dotenv(usecwd=True))

ALPHAVANTAGE_API_KEY = os.getenv("ALPHAVANTAGE_API_KEY", "")
GEMINI_API_KEY = os.getenv("GOOGLE_GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GOOGLE_GEMINI_MODEL", "gemini-1.5-flash")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

ALPHAVANTAGE_BASE_URL = "https://www.alphavantage.co/query"


@dataclass
class FetchConfig:
    symbol: str
    function: str = "TIME_SERIES_INTRADAY"
    interval: str = "5min"
    outputsize: str = "compact"
    datatype: str = "json"


def fetch_alpha_vantage_http(cfg: FetchConfig) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    params = {
        "function": cfg.function,
        "symbol": cfg.symbol,
        "interval": cfg.interval,
        "outputsize": cfg.outputsize,
        "datatype": cfg.datatype,
        "apikey": ALPHAVANTAGE_API_KEY,
    }
    r = requests.get(ALPHAVANTAGE_BASE_URL, params=params, timeout=30)
    r.raise_for_status()
    payload = r.json()
    if "Error Message" in payload or "Note" in payload or "Information" in payload:
        raise RuntimeError(
            "Alpha Vantage error: %s"
            % (payload.get("Error Message") or payload.get("Note") or payload.get("Information"))
        )

    key_map = {
        "TIME_SERIES_INTRADAY": f"Time Series ({cfg.interval})",
        "TIME_SERIES_DAILY": "Time Series (Daily)",
        "TIME_SERIES_DAILY_ADJUSTED": "Time Series (Daily)",
        "TIME_SERIES_WEEKLY": "Weekly Time Series",
        "TIME_SERIES_MONTHLY": "Monthly Time Series",
    }
    series_key = key_map.get(cfg.function)
    if series_key is None or series_key not in payload:
        # heuristic by key name
        series_key = next((k for k in payload.keys() if "Series" in k or "Time" in k), None)
    if series_key is None or series_key not in payload:
        # robust heuristic by value shape (mapping of timestamps -> ohlc dict)
        for k, v in payload.items():
            if isinstance(v, dict) and v:
                first_val = next(iter(v.values()))
                if isinstance(first_val, dict) and any(
                    key in first_val for key in ("1. open", "4. close", "open", "close")
                ):
                    series_key = k
                    break
    if series_key is None or series_key not in payload:
        raise RuntimeError("Unexpected Alpha Vantage response shape")

    ts = payload[series_key]
    df = pd.DataFrame.from_dict(ts, orient="index")
    rename_map = {
        "1. open": "open",
        "2. high": "high",
        "3. low": "low",
        "4. close": "close",
        "5. volume": "volume",
    }
    df = df.rename(columns=rename_map)
    for col in [c for c in ["open", "high", "low", "close", "volume"] if c in df.columns]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    meta = payload.get("Meta Data", {})
    return df, meta


def detect_momentum_events(
    df: pd.DataFrame,
    price_col: str = "close",
    pct_change_window: int = 1,
    threshold_abs_pct: float = 0.02,
    zscore_window: int = 50,
    use_zscore: bool = True,
) -> pd.DataFrame:
    if price_col not in df.columns:
        raise ValueError("price_col '%s' not in df" % price_col)

    data = df.copy().sort_index()
    data["return_pct"] = data[price_col].pct_change(pct_change_window)
    data["abs_return_pct"] = data["return_pct"].abs()

    event_flag = data["abs_return_pct"] > threshold_abs_pct

    if use_zscore:
        rolling_std = data["return_pct"].rolling(zscore_window, min_periods=max(5, zscore_window // 5)).std()
        rolling_mean = data["return_pct"].rolling(zscore_window, min_periods=max(5, zscore_window // 5)).mean()
        z = (data["return_pct"] - rolling_mean) / (rolling_std.replace(0, np.nan))
        data["return_zscore"] = z
        z_flag = z.abs() > 3
        event_flag = event_flag | z_flag
    else:
        data["return_zscore"] = np.nan

    data["momentum_event"] = event_flag.fillna(False)
    data["direction"] = np.where(data["return_pct"] > 0, "up", np.where(data["return_pct"] < 0, "down", "flat"))

    return data


def suggest_actions_with_gemini(events_df: pd.DataFrame, symbol: str, max_events: int = 10) -> List[Dict[str, Any]]:
    if genai is None or not GEMINI_API_KEY:
        results: List[Dict[str, Any]] = []
        subset = events_df[events_df["momentum_event"].astype(bool)].tail(max_events)
        for ts, row in subset.iterrows():
            results.append({
                "timestamp": ts.isoformat() if hasattr(ts, "isoformat") else str(ts),
                "symbol": symbol,
                "direction": row.get("direction"),
                "return_pct": float(row.get("return_pct", float("nan"))),
                "suggestion": "Review event. Consider risk management; this is not advice.",
            })
        return results

    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel(GEMINI_MODEL)

    outputs: List[Dict[str, Any]] = []
    subset = events_df[events_df["momentum_event"].astype(bool)].tail(max_events)
    for ts, row in subset.iterrows():
        direction = row.get("direction")
        ret = row.get("return_pct")
        abs_ret = abs(ret) if ret is not None else 0.0
        if abs_ret > 0.05:
            strength = "strong"
        elif abs_ret > 0.03:
            strength = "significant"
        else:
            strength = "moderate"
        prompt = (
            f"You are a trading assistant. {symbol} shows {strength} {direction} momentum: {ret*100:.2f}% move.\n"
            "Provide a direct trading recommendation in this format:\n"
            f"- For UP momentum: 'This stock {symbol} is going up, consider buying'\n"
            f"- For DOWN momentum: 'This stock {symbol} is going down, consider selling/shorting'\n"
            f"Add one sentence explaining why based on the {strength} momentum ({ret*100:.2f}% move).\n"
            "Be decisive and actionable. Educational purposes only."
        )
        try:
            response = model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=120,
                    temperature=0.2,
                ),
            )
            suggestion = response.text.strip()
        except Exception as e:  # pragma: no cover
            suggestion = f"Suggestion unavailable ({e})."
        outputs.append({
            "timestamp": ts.isoformat() if hasattr(ts, "isoformat") else str(ts),
            "symbol": symbol,
            "direction": direction,
            "return_pct": float(ret) if ret is not None else None,
            "suggestion": suggestion,
        })
    return outputs


def suggest_actions_with_gpt(events_df: pd.DataFrame, symbol: str, max_events: int = 10) -> List[Dict[str, Any]]:
    if OpenAI is None or not OPENAI_API_KEY:
        results: List[Dict[str, Any]] = []
        subset = events_df[events_df["momentum_event"].astype(bool)].tail(max_events)
        for ts, row in subset.iterrows():
            results.append({
                "timestamp": ts.isoformat() if hasattr(ts, "isoformat") else str(ts),
                "symbol": symbol,
                "direction": row.get("direction"),
                "return_pct": float(row.get("return_pct", float("nan"))),
                "suggestion": "Review event. Consider risk management; this is not advice.",
            })
        return results

    client = OpenAI(api_key=OPENAI_API_KEY)

    outputs: List[Dict[str, Any]] = []
    subset = events_df[events_df["momentum_event"].astype(bool)].tail(max_events)
    for ts, row in subset.iterrows():
        direction = row.get("direction")
        ret = row.get("return_pct")
        prompt = (
            f"You are an assistant for a trader. A sudden momentum event occurred for {symbol}.\n"
            f"Timestamp: {ts} (UTC). Direction: {direction}. One-interval return: {ret:.4f} ({ret*100:.2f}%).\n"
            "Provide a concise 1-2 sentence advisory suggestion with risk management reminders. Do not provide financial advice."
        )
        try:
            resp = client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=120,
                temperature=0.2,
            )
            suggestion = resp.choices[0].message.content.strip()
        except Exception as e:  # pragma: no cover
            suggestion = f"Suggestion unavailable ({e})."
        outputs.append({
            "timestamp": ts.isoformat() if hasattr(ts, "isoformat") else str(ts),
            "symbol": symbol,
            "direction": direction,
            "return_pct": float(ret) if ret is not None else None,
            "suggestion": suggestion,
        })
    return outputs


def _choose_ai_provider(provider: Optional[str]) -> str:
    if provider:
        return provider.lower()
    if GEMINI_API_KEY:
        return "gemini"
    if OPENAI_API_KEY:
        return "openai"
    return "none"


async def get_stock_recommendations(
    symbol: str,
    function: str = "TIME_SERIES_INTRADAY",
    interval: str = "5min",
    outputsize: str = "compact",
    threshold_abs_pct: float = 0.02,
    zscore_window: int = 50,
    max_events: int = 8,
    provider: Optional[str] = None,
) -> Dict[str, Any]:
    """Return stock momentum events and AI recommendations.

    Env:
      - ALPHAVANTAGE_API_KEY (required)
      - GOOGLE_GEMINI_API_KEY, GOOGLE_GEMINI_MODEL (optional)
      - OPENAI_API_KEY, OPENAI_MODEL (optional)
    """
    if not ALPHAVANTAGE_API_KEY:
        raise RuntimeError("ALPHAVANTAGE_API_KEY is not set in environment")

    cfg = FetchConfig(symbol=symbol, function=function, interval=interval, outputsize=outputsize)
    price_df, meta = fetch_alpha_vantage_http(cfg)
    enriched = detect_momentum_events(
        price_df,
        price_col="close",
        threshold_abs_pct=threshold_abs_pct,
        zscore_window=zscore_window,
        use_zscore=True,
    )

    choice = _choose_ai_provider(provider)
    if choice == "gemini":
        suggestions = suggest_actions_with_gemini(enriched, symbol=symbol, max_events=max_events)
    elif choice == "openai":
        suggestions = suggest_actions_with_gpt(enriched, symbol=symbol, max_events=max_events)
    else:
        suggestions = []
        subset = enriched[enriched["momentum_event"].astype(bool)].tail(max_events)
        for ts, row in subset.iterrows():
            suggestions.append({
                "timestamp": ts.isoformat() if hasattr(ts, "isoformat") else str(ts),
                "symbol": symbol,
                "direction": row.get("direction"),
                "return_pct": float(row.get("return_pct", float("nan"))),
                "suggestion": "Review event. Consider risk management; this is not advice.",
            })

    last_price = float(enriched["close"].iloc[-1]) if not enriched.empty else None

    return {
        "symbol": symbol,
        "meta": meta,
        "last_price": last_price,
        "events": [
            {
                "timestamp": idx.isoformat() if hasattr(idx, "isoformat") else str(idx),
                "close": float(row.get("close", float("nan"))),
                "return_pct": float(row.get("return_pct", float("nan"))),
                "return_zscore": (float(row.get("return_zscore")) if pd.notna(row.get("return_zscore")) else None),
                "momentum_event": bool(row.get("momentum_event")),
                "direction": row.get("direction"),
            }
            for idx, row in enriched.tail(200).iterrows()
        ],
        "suggestions": suggestions,
        "provider": choice,
    }


