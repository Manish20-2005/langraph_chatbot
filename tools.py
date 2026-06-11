"""
tools.py
All LangGraph-compatible tools for the AI chatbot.

Tools available:
  - web_search          : DuckDuckGo web search (no API key)
  - get_stock_price     : Real-time stock price + basic stats via yfinance
  - get_company_info    : Company fundamentals via yfinance
  - calculator          : Safe math expression evaluator
  - get_weather         : Current weather via Open-Meteo (no API key)
  - convert_currency    : Currency conversion via exchangerate.host (free)
  - get_news            : Latest news headlines via DuckDuckGo News
  - get_datetime        : Current date/time in any timezone

Install dependencies:
    pip install yfinance duckduckgo-search pytz requests
"""

import ast
import math
import operator
import re
from datetime import datetime
from typing import Optional

import pytz
import requests
import yfinance as yf
from duckduckgo_search import DDGS
from langchain_core.tools import tool


# ─────────────────────────────────────────────────────────────────────────────
# 1. Web Search
# ─────────────────────────────────────────────────────────────────────────────

@tool
def web_search(query: str, max_results: int = 5) -> str:
    """
    Search the web using DuckDuckGo and return top results.

    Use this for: general knowledge, current events, facts, how-to questions,
    anything you are not certain about, or need up-to-date information on.

    Args:
        query:       The search query string.
        max_results: Number of results to return (default 5, max 10).

    Returns:
        Formatted string of search results with title, URL, and snippet.
    """
    max_results = min(max_results, 10)
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))

        if not results:
            return f"No web results found for: '{query}'"

        lines = [f"🔍 Web search results for: **{query}**\n"]
        for i, r in enumerate(results, 1):
            lines.append(f"**{i}. {r.get('title', 'No title')}**")
            lines.append(f"   🔗 {r.get('href', '')}")
            lines.append(f"   {r.get('body', 'No description')}\n")
        return "\n".join(lines)

    except Exception as e:
        return f"Web search failed: {e}"


# ─────────────────────────────────────────────────────────────────────────────
# 2. Stock Price
# ─────────────────────────────────────────────────────────────────────────────

@tool
def get_stock_price(ticker: str) -> str:
    """
    Get the current stock price and key market data for a given ticker symbol.

    Use this for: stock prices, market cap, P/E ratio, 52-week highs/lows,
    trading volume, and daily price changes.

    Args:
        ticker: Stock ticker symbol, e.g. 'AAPL', 'TSLA', 'GOOGL', 'RELIANCE.NS'
                For Indian stocks add .NS (NSE) or .BO (BSE), e.g. 'TCS.NS'

    Returns:
        Formatted string with current price and market statistics.
    """
    try:
        stock = yf.Ticker(ticker.upper().strip())
        info = stock.info

        # Try fast_info first for current price
        try:
            fast = stock.fast_info
            current_price = fast.last_price
        except Exception:
            current_price = info.get("currentPrice") or info.get("regularMarketPrice")

        if current_price is None:
            return f"❌ Could not retrieve price for '{ticker}'. Check the ticker symbol."

        name        = info.get("longName") or info.get("shortName", ticker.upper())
        currency    = info.get("currency", "USD")
        prev_close  = info.get("previousClose") or info.get("regularMarketPreviousClose")
        open_price  = info.get("open") or info.get("regularMarketOpen")
        day_low     = info.get("dayLow") or info.get("regularMarketDayLow")
        day_high    = info.get("dayHigh") or info.get("regularMarketDayHigh")
        week52_low  = info.get("fiftyTwoWeekLow")
        week52_high = info.get("fiftyTwoWeekHigh")
        volume      = info.get("volume") or info.get("regularMarketVolume")
        market_cap  = info.get("marketCap")
        pe_ratio    = info.get("trailingPE")
        dividend    = info.get("dividendYield")

        # Price change
        change_str = ""
        if prev_close and current_price:
            change     = current_price - prev_close
            change_pct = (change / prev_close) * 100
            arrow      = "📈" if change >= 0 else "📉"
            change_str = f"{arrow} {change:+.2f} ({change_pct:+.2f}%)"

        def _fmt_large(n):
            if n is None:
                return "N/A"
            if n >= 1e12:
                return f"{n/1e12:.2f}T"
            if n >= 1e9:
                return f"{n/1e9:.2f}B"
            if n >= 1e6:
                return f"{n/1e6:.2f}M"
            return f"{n:,.0f}"

        lines = [
            f"📊 **{name}** ({ticker.upper()})",
            f"💰 **Current Price:** {current_price:.2f} {currency}  {change_str}",
            f"📅 **Open:** {open_price:.2f}  |  **Prev Close:** {prev_close:.2f}" if open_price and prev_close else "",
            f"📉 **Day Range:** {day_low:.2f} – {day_high:.2f}" if day_low and day_high else "",
            f"📆 **52-Week:** {week52_low:.2f} – {week52_high:.2f}" if week52_low and week52_high else "",
            f"📦 **Volume:** {_fmt_large(volume)}  |  **Market Cap:** {_fmt_large(market_cap)}",
            f"📐 **P/E Ratio:** {pe_ratio:.2f}" if pe_ratio else "",
            f"💵 **Dividend Yield:** {dividend*100:.2f}%" if dividend else "",
        ]
        return "\n".join(l for l in lines if l)

    except Exception as e:
        return f"❌ Error fetching stock data for '{ticker}': {e}"


# ─────────────────────────────────────────────────────────────────────────────
# 3. Company Info
# ─────────────────────────────────────────────────────────────────────────────

@tool
def get_company_info(ticker: str) -> str:
    """
    Get detailed company information and fundamentals for a given stock ticker.

    Use this for: company description, sector, industry, employee count,
    revenue, profit margins, analyst recommendations, and business summary.

    Args:
        ticker: Stock ticker symbol, e.g. 'AAPL', 'MSFT', 'TCS.NS'

    Returns:
        Formatted string with company profile and financial highlights.
    """
    try:
        stock = yf.Ticker(ticker.upper().strip())
        info  = stock.info

        name        = info.get("longName") or info.get("shortName", ticker.upper())
        sector      = info.get("sector", "N/A")
        industry    = info.get("industry", "N/A")
        country     = info.get("country", "N/A")
        employees   = info.get("fullTimeEmployees")
        website     = info.get("website", "N/A")
        summary     = info.get("longBusinessSummary", "No description available.")
        revenue     = info.get("totalRevenue")
        gross_margin= info.get("grossMargins")
        profit_margin=info.get("profitMargins")
        roe         = info.get("returnOnEquity")
        debt_equity = info.get("debtToEquity")
        rec         = info.get("recommendationKey", "N/A")

        def _fmt_large(n):
            if n is None: return "N/A"
            if n >= 1e12: return f"${n/1e12:.2f}T"
            if n >= 1e9:  return f"${n/1e9:.2f}B"
            if n >= 1e6:  return f"${n/1e6:.2f}M"
            return f"${n:,.0f}"

        lines = [
            f"🏢 **{name}** ({ticker.upper()})",
            f"🌐 **Sector:** {sector}  |  **Industry:** {industry}",
            f"🗺️  **Country:** {country}  |  **Website:** {website}",
            f"👥 **Employees:** {employees:,}" if employees else "",
            f"💰 **Revenue:** {_fmt_large(revenue)}",
            f"📊 **Gross Margin:** {gross_margin*100:.1f}%" if gross_margin else "",
            f"💹 **Profit Margin:** {profit_margin*100:.1f}%" if profit_margin else "",
            f"📈 **Return on Equity:** {roe*100:.1f}%" if roe else "",
            f"⚖️  **Debt/Equity:** {debt_equity:.2f}" if debt_equity else "",
            f"🎯 **Analyst Recommendation:** {rec.upper()}" if rec != "N/A" else "",
            f"\n📝 **About:**\n{summary[:500]}{'...' if len(summary) > 500 else ''}",
        ]
        return "\n".join(l for l in lines if l)

    except Exception as e:
        return f"❌ Error fetching company info for '{ticker}': {e}"


# ─────────────────────────────────────────────────────────────────────────────
# 4. Calculator
# ─────────────────────────────────────────────────────────────────────────────

# Whitelist of safe AST node types
_SAFE_NODES = {
    ast.Expression, ast.BinOp, ast.UnaryOp, ast.Constant,
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod,
    ast.Pow, ast.USub, ast.UAdd, ast.Call, ast.Name, ast.Load,
}

_SAFE_FUNCS = {
    "abs": abs, "round": round, "min": min, "max": max, "sum": sum,
    "sqrt": math.sqrt, "log": math.log, "log10": math.log10,
    "log2": math.log2, "exp": math.exp, "ceil": math.ceil,
    "floor": math.floor, "sin": math.sin, "cos": math.cos,
    "tan": math.tan, "pi": math.pi, "e": math.e,
    "factorial": math.factorial, "pow": math.pow,
}


def _safe_eval(expr: str):
    """Evaluate a math expression safely using AST whitelisting."""
    tree = ast.parse(expr.strip(), mode="eval")

    for node in ast.walk(tree):
        if type(node) not in _SAFE_NODES:
            raise ValueError(f"Unsafe operation: {type(node).__name__}")

    return eval(  # noqa: S307
        compile(tree, "<calc>", "eval"),
        {"__builtins__": {}},
        _SAFE_FUNCS,
    )


@tool
def calculator(expression: str) -> str:
    """
    Evaluate a mathematical expression safely.

    Use this for: arithmetic, algebra, unit conversions, financial calculations,
    percentage calculations, scientific math, or any numerical computation.

    Supported: +, -, *, /, **, //, %, sqrt(), log(), sin(), cos(), tan(),
               abs(), round(), ceil(), floor(), factorial(), pi, e, exp()

    Args:
        expression: Math expression as a string, e.g. '2**10', 'sqrt(144)',
                    '(1 + 0.08)**20 * 10000', '15% of 2500' won't work —
                    write it as '2500 * 0.15'

    Returns:
        The computed result as a string.
    """
    try:
        # Clean up common natural language patterns
        expr = expression.strip()
        expr = re.sub(r"(\d+)%\s+of\s+(\d+)", r"\2 * \1 / 100", expr, flags=re.I)
        expr = expr.replace("^", "**")  # allow ^ for power
        expr = expr.replace("×", "*").replace("÷", "/")

        result = _safe_eval(expr)

        # Format result nicely
        if isinstance(result, float):
            if result == int(result) and abs(result) < 1e15:
                result_str = str(int(result))
            else:
                result_str = f"{result:.10g}"
        else:
            result_str = str(result)

        return f"🧮 **{expression}** = **{result_str}**"

    except ZeroDivisionError:
        return "❌ Division by zero."
    except ValueError as e:
        return f"❌ Invalid expression: {e}"
    except Exception as e:
        return f"❌ Calculation error: {e}"


# ─────────────────────────────────────────────────────────────────────────────
# 5. Weather
# ─────────────────────────────────────────────────────────────────────────────

def _geocode(city: str) -> tuple[float, float, str]:
    """Geocode a city name using Open-Meteo's geocoding API."""
    url = "https://geocoding-api.open-meteo.com/v1/search"
    r = requests.get(url, params={"name": city, "count": 1, "language": "en", "format": "json"}, timeout=10)
    r.raise_for_status()
    data = r.json()
    if not data.get("results"):
        raise ValueError(f"City '{city}' not found")
    loc = data["results"][0]
    return loc["latitude"], loc["longitude"], f"{loc['name']}, {loc.get('country', '')}"


@tool
def get_weather(city: str) -> str:
    """
    Get the current weather conditions for any city in the world.

    Use this for: temperature, humidity, wind speed, weather conditions,
    feels-like temperature, UV index, precipitation for any location.

    Args:
        city: City name, e.g. 'Delhi', 'New York', 'Tokyo', 'Mumbai'

    Returns:
        Formatted current weather report.
    """
    try:
        lat, lon, location_name = _geocode(city)

        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": lat, "longitude": lon,
            "current": [
                "temperature_2m", "relative_humidity_2m", "apparent_temperature",
                "precipitation", "weather_code", "wind_speed_10m",
                "wind_direction_10m", "uv_index",
            ],
            "timezone": "auto",
        }
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        data = r.json()["current"]

        # WMO weather code → description
        WMO = {
            0: "☀️ Clear sky", 1: "🌤️ Mainly clear", 2: "⛅ Partly cloudy",
            3: "☁️ Overcast", 45: "🌫️ Foggy", 48: "🌫️ Icy fog",
            51: "🌦️ Light drizzle", 53: "🌦️ Moderate drizzle", 55: "🌧️ Dense drizzle",
            61: "🌧️ Slight rain", 63: "🌧️ Moderate rain", 65: "🌧️ Heavy rain",
            71: "🌨️ Slight snow", 73: "🌨️ Moderate snow", 75: "❄️ Heavy snow",
            80: "🌦️ Slight showers", 81: "🌧️ Moderate showers", 82: "⛈️ Violent showers",
            95: "⛈️ Thunderstorm", 99: "⛈️ Thunderstorm w/ hail",
        }
        code    = data.get("weather_code", 0)
        condition = WMO.get(code, f"Code {code}")

        def _wind_dir(deg):
            dirs = ["N","NE","E","SE","S","SW","W","NW"]
            return dirs[round(deg / 45) % 8]

        temp     = data.get("temperature_2m")
        feels    = data.get("apparent_temperature")
        humidity = data.get("relative_humidity_2m")
        wind_spd = data.get("wind_speed_10m")
        wind_dir = data.get("wind_direction_10m")
        precip   = data.get("precipitation")
        uv       = data.get("uv_index")

        return "\n".join([
            f"🌍 **Weather in {location_name}**",
            f"{condition}",
            f"🌡️  **Temperature:** {temp}°C  (Feels like {feels}°C)",
            f"💧 **Humidity:** {humidity}%",
            f"💨 **Wind:** {wind_spd} km/h {_wind_dir(wind_dir)}" if wind_dir else f"💨 **Wind:** {wind_spd} km/h",
            f"🌧️ **Precipitation:** {precip} mm" if precip else "",
            f"☀️ **UV Index:** {uv}" if uv is not None else "",
        ])

    except Exception as e:
        return f"❌ Weather lookup failed for '{city}': {e}"


# ─────────────────────────────────────────────────────────────────────────────
# 6. Currency Converter
# ─────────────────────────────────────────────────────────────────────────────

@tool
def convert_currency(amount: float, from_currency: str, to_currency: str) -> str:
    """
    Convert an amount from one currency to another using live exchange rates.

    Use this for: currency conversion, forex rates, international money
    calculations, travel budgeting.

    Args:
        amount:        The amount to convert (e.g. 100.0)
        from_currency: Source currency code, e.g. 'USD', 'INR', 'EUR', 'GBP'
        to_currency:   Target currency code, e.g. 'EUR', 'JPY', 'INR'

    Returns:
        Conversion result with exchange rate.
    """
    try:
        base = from_currency.upper().strip()
        target = to_currency.upper().strip()

        url = f"https://open.er-api.com/v6/latest/{base}"
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        data = r.json()

        if data.get("result") != "success":
            return f"❌ Could not fetch exchange rates for {base}."

        rates = data.get("rates", {})
        if target not in rates:
            return f"❌ Unsupported currency code: '{target}'"

        rate   = rates[target]
        result = amount * rate

        return (
            f"💱 **Currency Conversion**\n"
            f"**{amount:,.2f} {base}** = **{result:,.2f} {target}**\n"
            f"📊 Exchange rate: 1 {base} = {rate:.4f} {target}\n"
            f"🕐 Rates updated: {data.get('time_last_update_utc', 'N/A')}"
        )

    except Exception as e:
        return f"❌ Currency conversion failed: {e}"


# ─────────────────────────────────────────────────────────────────────────────
# 7. News Search
# ─────────────────────────────────────────────────────────────────────────────

@tool
def get_news(topic: str, max_results: int = 6) -> str:
    """
    Get the latest news headlines and summaries for any topic.

    Use this for: current events, breaking news, latest updates on companies,
    sports results, technology news, political news, market news.

    Args:
        topic:       Topic or keyword to search news for, e.g. 'NVIDIA AI chips',
                     'India elections', 'Bitcoin price'
        max_results: Number of news articles to return (default 6, max 10).

    Returns:
        Formatted list of recent news articles with title, source, and summary.
    """
    max_results = min(max_results, 10)
    try:
        with DDGS() as ddgs:
            results = list(ddgs.news(topic, max_results=max_results))

        if not results:
            return f"📰 No recent news found for: '{topic}'"

        lines = [f"📰 **Latest news on: {topic}**\n"]
        for i, r in enumerate(results, 1):
            title  = r.get("title", "No title")
            source = r.get("source", "Unknown")
            url    = r.get("url", "")
            body   = r.get("body", "")
            date   = r.get("date", "")

            lines.append(f"**{i}. {title}**")
            if date:
                lines.append(f"   📅 {date}  |  📡 {source}")
            else:
                lines.append(f"   📡 {source}")
            if body:
                lines.append(f"   {body[:200]}{'...' if len(body) > 200 else ''}")
            lines.append(f"   🔗 {url}\n")

        return "\n".join(lines)

    except Exception as e:
        return f"❌ News search failed: {e}"


# ─────────────────────────────────────────────────────────────────────────────
# 8. Date / Time
# ─────────────────────────────────────────────────────────────────────────────

@tool
def get_datetime(timezone: str = "UTC") -> str:
    """
    Get the current date and time for any timezone in the world.

    Use this for: current time, date, day of week in any city or timezone.
    Also useful for calculating time differences.

    Args:
        timezone: Timezone name, e.g. 'UTC', 'Asia/Kolkata', 'America/New_York',
                  'Europe/London', 'Asia/Tokyo', 'Australia/Sydney'
                  City nicknames also work: 'IST', 'EST', 'PST'

    Returns:
        Current date, time, and timezone information.
    """
    # Common abbreviation → IANA timezone map
    ALIASES = {
        "IST": "Asia/Kolkata", "EST": "America/New_York", "PST": "America/Los_Angeles",
        "CST": "America/Chicago", "MST": "America/Denver", "GMT": "UTC",
        "BST": "Europe/London", "CET": "Europe/Paris", "JST": "Asia/Tokyo",
        "AEST": "Australia/Sydney", "SGT": "Asia/Singapore", "HKT": "Asia/Hong_Kong",
        "INDIA": "Asia/Kolkata", "DELHI": "Asia/Kolkata", "MUMBAI": "Asia/Kolkata",
        "LONDON": "Europe/London", "PARIS": "Europe/Paris", "TOKYO": "Asia/Tokyo",
        "NYC": "America/New_York", "LA": "America/Los_Angeles",
    }

    tz_name = ALIASES.get(timezone.upper(), timezone)

    try:
        tz  = pytz.timezone(tz_name)
        now = datetime.now(tz)

        return (
            f"🕐 **Current Date & Time**\n"
            f"📅 **Date:** {now.strftime('%A, %B %d, %Y')}\n"
            f"⏰ **Time:** {now.strftime('%I:%M:%S %p')}\n"
            f"🌍 **Timezone:** {tz_name} (UTC{now.strftime('%z')})\n"
            f"📆 **Week:** {now.strftime('Week %W of %Y')}\n"
            f"🔢 **Day of Year:** {now.strftime('%j')}"
        )

    except pytz.UnknownTimeZoneError:
        available = ", ".join(sorted(ALIASES.keys()))
        return (
            f"❌ Unknown timezone: '{timezone}'\n"
            f"Try: UTC, Asia/Kolkata, America/New_York, Europe/London, etc.\n"
            f"Shortcuts: {available}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Tool registry — import this list in graph.py
# ─────────────────────────────────────────────────────────────────────────────

ALL_TOOLS = [
    web_search,
    get_stock_price,
    get_company_info,
    calculator,
    get_weather,
    convert_currency,
    get_news,
    get_datetime,
]