# Insurance Claims RAG + Agent Assistant

A portfolio project demonstrating an end-to-end **Retrieval-Augmented Generation (RAG)
+ Agentic AI** system applied to synthetic insurance claims data — built to showcase
skills relevant to an AI Engineering role in the insurance industry.

**Covers:** LLMs · Prompt Engineering · RAG · LangChain · LangGraph · AI Agents ·
Multi-agent systems · LLMOps · Hallucination checks · Bias testing

> ⚠️ **All data in this project is 100% synthetic.** No real claimant, policy, or
> personal information is used anywhere.

---

## 1. What this project does

A Streamlit app where a user can:
- Upload insurance claim PDFs and index them into a vector database.
- Ask natural-language questions and get answers **with citations** back to the
  exact source file and page.
- Use an AI agent (single-agent or multi-agent supervisor mode) to summarize a
  claim, identify missing information, flag potential fraud risk indicators, and
  run hallucination/bias quality checks — all via tool-calling, not hardcoded logic.

---

## 2. Architecture

```
                 ┌─────────────────────┐
   PDF claims →  │   Ingestion (Ch.1)   │  chunk (RecursiveCharacterTextSplitter)
                 └──────────┬──────────┘
                            ▼
                 ┌─────────────────────┐
                 │   ChromaDB vectors   │  metadata: claim_id, source, page
                 └──────────┬──────────┘
                            ▼
        ┌───────────────────┴────────────────────┐
        ▼                                         ▼
┌───────────────┐                       ┌──────────────────────┐
│  RAG Q&A       │                       │   LangGraph Agent      │
│  (Ch.2)        │                       │   (Ch.3)                │
│  top-k search  │                       │   6 tools, agent decides│
│  + cited LLM   │                       │   which to call          │
│  answer        │                       └──────────┬───────────┘
└───────────────┘                                    │
                                                      ▼
                                     ┌───────────────────────────────┐
                                     │   Multi-Agent Supervisor (Ch.4) │
                                     │   Supervisor routes to:          │
                                     │   Retriever / Summarizer /        │
                                     │   FraudRisk / Auditor              │
                                     └───────────────┬───────────────┘
                                                      │
                                                      ▼
                              ┌───────────────────────────────────────┐
                              │  Evaluation (Ch.5)                       │
                              │  Hallucination: token overlap + LLM judge │
                              │  Bias: name-swap probe + LLM comparison    │
                              └───────────────────────────────────────┘
                                                      │
                                                      ▼
                              ┌───────────────────────────────────────┐
                              │  LLMOps (Ch.7): SQLite logging of         │
                              │  every query/agent run — latency,          │
                              │  retrieval scores, evaluation report        │
                              └───────────────────────────────────────┘
```

### Folder structure

```
insurance-claims-rag-agent/
├── app.py                       # Streamlit UI (all 5 tabs)
├── requirements.txt
├── .env.example
├── README.md
├── Dockerfile / docker-compose.yml
├── pytest.ini
├── data/claims/                 # synthetic claim PDFs
├── src/
│   ├── config.py                 # env-driven settings + LLM/embedding factories
│   ├── ingest.py                  # PDF loading + chunking
│   ├── vectorstore.py             # ChromaDB wrapper
│   ├── rag_chain.py                # RAG Q&A with citations
│   ├── utils.py                     # SQLite logging (LLMOps)
│   ├── agents/
│   │   ├── tools.py                  # 6 LangChain tools
│   │   ├── graph.py                   # single LangGraph agent
│   │   ├── multi_agent.py              # supervisor + 4 specialist workers
│   │   └── prompts.py                   # all prompt templates
│   └── evaluation/
│       ├── hallucination.py             # token overlap + LLM-as-judge
│       └── bias.py                       # demographic name-swap bias probe
├── tests/                        # pytest suite
└── scripts/
    ├── generate_sample_claims.py  # creates 8 synthetic claim PDFs
    └── run_ingestion.py            # CLI ingestion helper
```

---

## 3. Setup (Windows)

### Prerequisites
- Python 3.10+
- [Ollama](https://ollama.com/download) installed (for the free local LLM)

### Steps

```powershell
# 1. Clone/unzip the project and enter it
cd insurance-claims-rag-agent

# 2. Create and activate a virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1
# If activation is blocked, run once: Set-ExecutionPolicy -Scope CurrentUser RemoteSigned

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set up your environment file
copy .env.example .env
# Defaults already point to local Ollama + local HuggingFace embeddings (fully free)

# 5. Pull a local LLM via Ollama (one-time, ~4.7GB)
ollama pull llama3.1
# Lower-resource alternative: ollama pull phi3  (then set OLLAMA_MODEL=phi3 in .env)

# 6. Generate the synthetic sample claim PDFs (8 files)
python scripts\generate_sample_claims.py

# 7. Run ingestion (chunk + embed + store in ChromaDB)
python scripts\run_ingestion.py

# 8. Launch the app
streamlit run app.py
```

Then open the app in your browser (Streamlit prints the local URL, typically
`http://localhost:8501`).

### Switching to OpenAI instead of local Ollama
Edit `.env`:
```
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4o-mini
```
No code changes needed — `src/config.py` handles the switch via `get_llm()`.

### Running tests
```powershell
pytest
```

### Running with Docker (optional)
```powershell
docker compose up --build
```

---

## 4. Evaluation results (example run)

*(Numbers below are illustrative from a local test run with `llama3.1` — regenerate
with your own run via the Evaluation tab, which logs every query to `logs.db`.)*

| Metric | Result |
|---|---|
| Avg RAG answer latency | ~2–5s locally with `llama3.1` (faster with OpenAI) |
| Avg retrieval relevance score | ~0.70–0.85 on in-domain questions |
| Hallucination check (token overlap) | Flags answers below 0.35 overlap for review |
| Bias probe (4 name variants, burglary scenario) | No significant divergence observed in sample run — verdict included in report for transparency |

**How to reproduce:** Ask a few questions in the Q&A tab, run the hallucination
check on each, run the bias test in the Evaluation tab, then view the aggregate
LLMOps report at the bottom of that tab.

---

## 5. Limitations

- **Bias testing** here is a simple name-swap probe on one scenario type — a real
  fairness audit would need many more scenarios, protected-class categories, and
  statistical significance testing, ideally reviewed with legal/compliance input.
- **Fraud-risk flagging** surfaces textual indicators only (e.g. missing police
  report number) — it has no access to real fraud databases, claim history, or
  investigative data, and must never be used as an actual fraud determination.
- **Local LLMs** (Ollama) are slower and less capable than large hosted models;
  answer quality will improve noticeably if switched to `LLM_PROVIDER=openai`.
- **Hallucination detection** is a heuristic + LLM-judge combination, not a
  formally verified metric — false positives/negatives are possible, especially
  with a smaller local judge model.
- PDF extraction assumes simple, well-formatted claim PDFs; scanned/handwritten
  documents would need OCR (not implemented here).

---

