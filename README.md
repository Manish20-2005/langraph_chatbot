# ✦ AI Chat — LangGraph Powered Chatbot

A production-ready, **ChatGPT-style chatbot** built with **Streamlit**, **LangGraph**, and a **ReAct agent architecture**. It comes with a sleek dark UI, persistent SQLite-backed conversation history, streaming responses, and a powerful suite of real-time tools — web search, stock data, weather, currency conversion, news, and more.

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=streamlit&logoColor=white)
![LangGraph](https://img.shields.io/badge/LangGraph-ReAct%20Agent-1C3C3C?style=flat)
![SQLite](https://img.shields.io/badge/SQLite-Persistent%20Storage-07405E?style=flat&logo=sqlite&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=flat)

---

## ✨ Features

- 🎨 **Modern dark UI** — custom-styled, ChatGPT-inspired Streamlit interface
- 💬 **Multi-conversation support** — create, rename, switch, and delete chat threads from the sidebar
- 💾 **Persistent storage** — conversation metadata stored in `chat.db`, full LangGraph state in `checkpoints.db`
- ⚡ **Streaming responses** — token-by-token output with a live typing indicator
- 🧠 **ReAct agent architecture** — the LLM autonomously decides when to call tools and loops until it has a final answer
- 🔌 **Multi-LLM support** — works with **OpenAI**, **Google Gemini**, or **Mistral** (auto-detected from your `.env`)
- 🛠️ **8 built-in tools** for real-time, grounded responses
- 📤 **Export chats** as `.txt` files
- 🔄 **Automatic retry with exponential backoff** for rate-limited API calls
- 📊 **LangSmith observability** support (optional)
- 📝 **Auto-generated chat titles** from the first message
-    **RAG System** -can take pdf,word file take your queries based on your document
-    **MCP Uses** -use mcp for the better tool use not breaking
---

## 🛠️ Tools

The agent has access to the following tools and decides on its own when to use them:

| Tool | Description | API Key Required |
|---|---|---|
| 🔍 `web_search` | Web search via DuckDuckGo | No |
| 📈 `get_stock_price` | Real-time stock prices & market stats (yfinance) | No |
| 🏢 `get_company_info` | Company fundamentals & business summary (yfinance) | No |
| 🧮 `calculator` | Safe AST-based math expression evaluator | No |
| 🌦️ `get_weather` | Current weather for any city (Open-Meteo) | No |
| 💱 `convert_currency` | Live currency conversion rates | No |
| 📰 `get_news` | Latest news headlines on any topic (DuckDuckGo News) | No |
| 🕐 `get_datetime` | Current date & time in any timezone | No |

---

## 🏗️ Architecture

The chatbot is built around a LangGraph `StateGraph` implementing a classic **ReAct (Reason + Act) loop**:

```
        START
          │
          ▼
   ┌─────────────┐
   │ agent_node  │◄────────────────┐
   └──────┬──────┘                 │
          │                        │
   has tool calls?                 │
          │                        │
   ┌──────┴───────┐                │
   │              │                │
  yes            no                │
   │              │                │
   ▼              ▼                │
┌───────────┐   END         ┌──────────────┐
│ tool_node │───────────────►              │
└───────────┘                └──────────────┘
```

1. **`agent_node`** — invokes the LLM (with all tools bound). The model either replies directly or requests a tool call.
2. **`tool_node`** — a `ToolNode` that automatically executes the requested tool and returns its result.
3. The loop continues until the model produces a final answer with no further tool calls, then the graph reaches `END`.

State is managed via a `ChatState` `TypedDict` with an `add_messages` reducer, ensuring the full message history accumulates correctly across turns.

---

## 📂 Project Structure

```
langraph_chatbot/
├── app.py              # Streamlit UI — chat interface, sidebar, session management
├── graph.py            # LangGraph StateGraph definition (ReAct agent + tools)
├── tools.py            # All tool implementations (search, stocks, weather, etc.)
├── state.py            # ChatState TypedDict schema
├── memory.py           # SQLite-backed LangGraph checkpointer (persistent memory)
├── database.py         # SQLite layer for conversation metadata (chat.db)
├── utils.py            # Helper functions (titles, timestamps, export)
├── requirements.txt    # Python dependencies
├── .env.example        # Sample environment variable file
└── .gitignore
```

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/Manish20-2005/langraph_chatbot.git
cd langraph_chatbot
```

### 2. Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate      # On Windows: .venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> **Note:** The tools module also requires `yfinance`, `duckduckgo-search`, `pytz`, and `requests`:
> ```bash
> pip install yfinance duckduckgo-search pytz requests
> ```

### 4. Configure environment variables

Copy `.env.example` to `.env` and add your API key(s):

```bash
cp .env.example .env
```

```ini
# Choose ONE LLM provider
OPENAI_API_KEY=your_openai_api_key
# or
GOOGLE_API_KEY=your_google_api_key
# or
MISTRAL_API_KEY=your_mistral_api_key

# Optional: model overrides
OPENAI_MODEL=gpt-4o-mini
GOOGLE_MODEL=gemini-1.5-flash
MISTRAL_MODEL=mistral-large-latest

# Optional: LangSmith tracing
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your_langsmith_api_key
LANGCHAIN_PROJECT=ai-chatbot
```

The app automatically picks a provider in this priority order: **OpenAI → Google Gemini → Mistral**.

### 5. Run the app

```bash
streamlit run app.py
```

Open your browser at **http://localhost:8501** 🎉

---

## 💡 Usage

- Click **＋ New Chat** to start a fresh conversation
- Type a message — the assistant will stream its response token-by-token
- The agent automatically uses tools (stocks, weather, search, etc.) when needed
- Switch between conversations from the sidebar — all history is saved in `chat.db`
- Use **⬇ Export as TXT** to download a conversation
- Use **🗑 Clear Chat** to wipe messages from the current thread

### Example prompts

```
What's the current price of TSLA and how has it moved today?
What's the weather like in Tokyo right now?
Convert 500 USD to INR
What's 15% of 2480, and then add the square root of 256?
Give me the latest news on AI chip manufacturing
What time is it in Sydney right now?
```

---

## 🔍 Observability with LangSmith (Optional)

To trace every agent step, tool call, and token in [LangSmith](https://smith.langchain.com), set the following in `.env`:

```ini
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your_langsmith_api_key
LANGCHAIN_PROJECT=ai-chatbot
```

When enabled, the app prints a confirmation banner on startup with a link to your dashboard.

---

## 🗄️ Persistence Model

| File | Purpose |
|---|---|
| `chat.db` | Lightweight conversation metadata (titles, timestamps, message previews) used to render the sidebar |
| `checkpoints.db` | Full LangGraph state snapshots managed by `SqliteSaver`, enabling true conversational memory across sessions |

Both databases use SQLite **WAL mode** for safe concurrent read/write access.

---

## 🧩 Tech Stack

- **[Streamlit](https://streamlit.io/)** — UI framework
- **[LangGraph](https://www.langchain.com/langgraph)** — agent orchestration & state graph
- **[LangChain](https://www.langchain.com/)** — LLM abstractions & tool framework
- **SQLite** — conversation + checkpoint persistence
- **yfinance, DuckDuckGo Search, Open-Meteo, ExchangeRate API** — external data sources

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to open a PR or issue.

## 📄 License

This project is licensed under the MIT License.
