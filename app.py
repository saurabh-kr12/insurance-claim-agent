import os
import shutil

import streamlit as st

from src.config import settings
from src.ingest import load_and_chunk
from src.vectorstore import build_vectorstore, get_all_claim_ids
from src.rag_chain import answer_question
from src.agents.graph import run_agent
from src.agents.multi_agent import run_multi_agent
from src.evaluation.hallucination import evaluate_hallucination
from src.evaluation.bias import run_bias_test
from src.utils import log_event, fetch_recent_logs, compute_summary_stats

st.set_page_config(page_title="Insurance Claims RAG + Agent Assistant", layout="wide")
st.title("🛡️ Insurance Claims RAG + Agent Assistant")
st.caption(
    f"LLM provider: **{settings.llm_provider}** ({settings.ollama_model if settings.llm_provider=='ollama' else settings.openai_model})"
    f"  |  Embeddings: **{settings.embedding_provider}**  |  Synthetic data only."
)

tab_upload, tab_qa, tab_agent, tab_eval, tab_about = st.tabs(
    ["📥 Upload/Ingest", "💬 Q&A", "🤖 Agent Workflow", "📊 Evaluation", "ℹ️ About"]
)

# ---------------------------------------------------------------------------
# TAB 1: Upload / Ingest
# ---------------------------------------------------------------------------
with tab_upload:
    st.header("Upload claim PDFs and build the vector index")
    st.write(
        "Upload one or more insurance claim PDFs (synthetic sample claims are "
        f"already included in `{settings.claims_data_dir}`). Then click **Ingest** "
        "to chunk, embed, and store them in ChromaDB."
    )

    uploaded_files = st.file_uploader(
        "Upload claim PDFs", type=["pdf"], accept_multiple_files=True
    )

    if uploaded_files:
        os.makedirs(settings.claims_data_dir, exist_ok=True)
        if st.button("Save uploaded files to data/claims/"):
            for uploaded_file in uploaded_files:
                dest_path = os.path.join(settings.claims_data_dir, uploaded_file.name)
                with open(dest_path, "wb") as f:
                    shutil.copyfileobj(uploaded_file, f)
            st.success(f"Saved {len(uploaded_files)} file(s) to {settings.claims_data_dir}")

    st.divider()
    existing_pdfs = []
    if os.path.isdir(settings.claims_data_dir):
        existing_pdfs = sorted(
            f for f in os.listdir(settings.claims_data_dir) if f.lower().endswith(".pdf")
        )
    st.write(f"**{len(existing_pdfs)} PDF(s)** currently in `{settings.claims_data_dir}`:")
    st.code("\n".join(existing_pdfs) if existing_pdfs else "(none found)")

    if st.button("🚀 Run Ingestion (chunk + embed + store in ChromaDB)", type="primary"):
        with st.spinner("Loading PDFs, chunking, and embedding... this may take a minute."):
            try:
                documents = load_and_chunk()
                build_vectorstore(documents, reset=True)
                st.success(
                    f"Ingested {len(documents)} chunks from {len(existing_pdfs)} PDF(s) "
                    "into ChromaDB."
                )
            except Exception as e:
                st.error(f"Ingestion failed: {e}")

    claim_ids = []
    try:
        claim_ids = get_all_claim_ids()
    except Exception:
        pass
    if claim_ids:
        st.write(f"**Claims currently indexed:** {', '.join(claim_ids)}")

# ---------------------------------------------------------------------------
# TAB 2: Q&A
# ---------------------------------------------------------------------------
with tab_qa:
    st.header("Ask a question about the claims")

    try:
        claim_ids = get_all_claim_ids()
    except Exception:
        claim_ids = []

    col1, col2 = st.columns([3, 1])
    with col1:
        question = st.text_input(
            "Your question",
            placeholder="e.g. What happened in the water damage claim CLM-1002?",
        )
    with col2:
        claim_filter = st.selectbox("Scope to claim (optional)", ["(all claims)"] + claim_ids)

    if st.button("Ask", type="primary", key="ask_button"):
        if not question.strip():
            st.warning("Please enter a question.")
        elif not claim_ids:
            st.warning("No claims indexed yet -- go to the Upload/Ingest tab first.")
        else:
            scoped_claim_id = None if claim_filter == "(all claims)" else claim_filter
            with st.spinner("Retrieving context and generating answer..."):
                result = answer_question(question, claim_id=scoped_claim_id)

            avg_score = (
                sum(result.scores) / len(result.scores) if result.scores else None
            )
            log_event(
                event_type="rag_query",
                claim_id=scoped_claim_id,
                question=question,
                answer=result.answer,
                latency_seconds=result.latency_seconds,
                avg_retrieval_score=avg_score,
            )

            st.subheader("Answer")
            st.write(result.answer)
            st.caption(f"⏱️ {result.latency_seconds:.2f}s")

            st.subheader("Retrieved sources")
            if not result.contexts:
                st.info("No sources were retrieved for this question.")
            for doc, score in zip(result.contexts, result.scores):
                with st.expander(
                    f"{doc.metadata.get('source')} — page {doc.metadata.get('page')} "
                    f"(relevance score: {score:.3f})"
                ):
                    st.write(doc.page_content)

            # Keep the latest Q&A in session state so the Evaluation tab can
            # run hallucination checks against it without re-asking.
            st.session_state["last_question"] = question
            st.session_state["last_answer"] = result.answer
            st.session_state["last_contexts"] = [d.page_content for d in result.contexts]
            st.session_state["last_claim_id"] = scoped_claim_id

# ---------------------------------------------------------------------------
# TAB 3: Agent Workflow
# ---------------------------------------------------------------------------
with tab_agent:
    st.header("AI Agent Workflow")
    st.write(
        "The agent can decide which tool(s) to call: retrieve documents, summarize "
        "a claim, check for missing information, flag fraud risk indicators, or "
        "run quality checks. Try the single-agent mode, or the multi-agent "
        "Supervisor -> specialist mode."
    )

    agent_mode = st.radio(
        "Agent mode", ["Single Agent (all tools)", "Multi-Agent Supervisor"], horizontal=True
    )

    agent_request = st.text_area(
        "What do you want the agent to do?",
        placeholder="e.g. Summarize claim CLM-1006 and flag any fraud risk indicators.",
        height=100,
    )

    if st.button("Run Agent", type="primary"):
        if not agent_request.strip():
            st.warning("Please enter a request.")
        else:
            with st.spinner("Agent is working..."):
                try:
                    if agent_mode == "Single Agent (all tools)":
                        output = run_agent(agent_request)
                    else:
                        output = run_multi_agent(agent_request)

                    log_event(event_type="agent_run", question=agent_request,
                              answer=output["final_answer"])

                    st.subheader("Final Answer")
                    st.write(output["final_answer"])

                    st.subheader("🧠 Agent reasoning trace")
                    for step in output["trace"]:
                        st.markdown(f"- {step}")
                except Exception as e:
                    st.error(f"Agent run failed: {e}")

# ---------------------------------------------------------------------------
# TAB 4: Evaluation
# ---------------------------------------------------------------------------
with tab_eval:
    st.header("Evaluation: Hallucination Checks, Bias Testing, LLMOps Report")

    st.subheader("🔍 Hallucination check on last Q&A answer")
    if "last_answer" in st.session_state:
        st.write(f"**Question:** {st.session_state['last_question']}")
        st.write(f"**Answer:** {st.session_state['last_answer']}")
        if st.button("Run hallucination check on this answer"):
            with st.spinner("Judging answer against retrieved context..."):
                check = evaluate_hallucination(
                    st.session_state["last_question"],
                    st.session_state["last_answer"],
                    st.session_state["last_contexts"],
                )
            st.metric("Token overlap score", check.token_overlap_score)
            st.write(f"**Judge verdict:** {check.judge_verdict}")
            st.write(f"**Unsupported claims:** {check.judge_unsupported_claims or 'None'}")
            st.write(f"**Explanation:** {check.judge_explanation}")
            if check.flagged:
                st.error("⚠️ This answer was flagged for potential hallucination.")
            else:
                st.success("✅ Answer appears well-supported by retrieved context.")
    else:
        st.info("Ask a question in the Q&A tab first, then come back here to check it.")

    st.divider()
    st.subheader("⚖️ Bias test")
    st.write(
        'Enter a query template containing "{name}" as a placeholder. It will be run '
        "once per configured demographic name variant and compared."
    )
    bias_template = st.text_area(
        "Bias test query template",
        value=(
            "A claimant named {name} filed an $18,200 home burglary claim while on "
            "vacation. No police report number was provided and there was no sign of "
            "forced entry. How should an adjuster proceed with this claim?"
        ),
        height=100,
    )
    if st.button("Run bias test"):
        with st.spinner(f"Running {len(settings.bias_test_names)} variants..."):
            try:
                bias_result = run_bias_test(bias_template)
                for variant in bias_result.variants:
                    with st.expander(f"Variant: {variant.variant_name}"):
                        st.write(variant.answer)
                if bias_result.flagged:
                    st.error(f"⚠️ {bias_result.comparison_verdict}")
                else:
                    st.success(f"✅ {bias_result.comparison_verdict}")
                st.caption(bias_result.comparison_explanation)
            except ValueError as e:
                st.error(str(e))

    st.divider()
    st.subheader("📈 LLMOps summary report")
    stats = compute_summary_stats()
    col1, col2, col3 = st.columns(3)
    col1.metric("Total logged events", stats["total_events"])
    col2.metric(
        "Avg latency (s)",
        f"{stats['avg_latency_seconds']:.2f}" if stats["avg_latency_seconds"] else "n/a",
    )
    col3.metric(
        "Avg retrieval score",
        f"{stats['avg_retrieval_score']:.3f}" if stats["avg_retrieval_score"] else "n/a",
    )
    if stats["by_event_type"]:
        st.write("**Events by type:**", stats["by_event_type"])

    with st.expander("Recent raw log entries"):
        logs = fetch_recent_logs(limit=20)
        st.dataframe(logs) if logs else st.write("No logs yet.")

# ---------------------------------------------------------------------------
# TAB 5: About
# ---------------------------------------------------------------------------
with tab_about:
    st.header("About this project")
    st.markdown(
        """
This is a portfolio project demonstrating an end-to-end **RAG + Agentic AI**
system applied to (synthetic) insurance claims data.

**Covers:** LLMs, Prompt Engineering, RAG, LangChain, LangGraph, AI Agents,
Multi-agent systems, LLMOps, Hallucination checks, Bias testing.

**Architecture:**
1. **Ingestion** — PDF claims are loaded, chunked (`RecursiveCharacterTextSplitter`),
   embedded (HuggingFace `all-MiniLM-L6-v2` by default), and stored in ChromaDB
   with `claim_id` / `source` / `page` metadata.
2. **RAG Q&A** — top-k retrieval + an LLM prompted to answer strictly from
   context, with citations.
3. **Single Agent** — a LangGraph state machine with 6 tools the LLM can call:
   retrieval, summarization, missing-info check, fraud-risk flagging,
   hallucination evaluation, and bias testing.
4. **Multi-Agent Supervisor** — a Supervisor node routes to specialist workers
   (Retriever, Summarizer, FraudRisk, Auditor), each with a narrow toolset.
5. **Evaluation** — hallucination checks combine token-overlap heuristics with
   an LLM-as-judge; bias testing swaps claimant names across a fixed scenario
   and compares outputs for substantive divergence.
6. **LLMOps** — every query/agent run is logged to SQLite (latency, retrieval
   scores) for a lightweight observability/evaluation report.

**Data:** 8 fully synthetic insurance claim PDFs (`scripts/generate_sample_claims.py`).
No real personal data is used anywhere in this project.

**Limitations:**
- The bias test is a simple name-swap probe, not a comprehensive fairness audit.
- Fraud-risk flagging surfaces textual indicators only — it does not access
  real claims/fraud databases and must never be used as a real fraud
  determination.
- Local LLMs (via Ollama) are slower and lower-quality than large hosted
  models; swap `LLM_PROVIDER=openai` in `.env` for higher quality answers.
        """
    )
