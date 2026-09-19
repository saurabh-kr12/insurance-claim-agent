from typing import List

from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from src.config import get_llm
from src.vectorstore import similarity_search
from src.evaluation.hallucination import evaluate_hallucination
from src.evaluation.bias import run_bias_test
from src.agents.prompts import (
    SUMMARIZE_CLAIM_PROMPT,
    MISSING_INFO_PROMPT,
    FRAUD_RISK_PROMPT,
)


def _get_claim_context(claim_id: str, query: str = "claim details", k: int = 8) -> str:
    """Shared helper: retrieves and concatenates chunks for a given claim_id."""
    docs = similarity_search(query, k=k, claim_id=claim_id)
    if not docs:
        return ""
    return "\n\n".join(
        f"[{d.metadata.get('source')}, page {d.metadata.get('page')}]\n{d.page_content}"
        for d in docs
    )


@tool
def retrieve_claim_documents(query: str, claim_id: str = "") -> str:
    """
    Search the claim document vector store for text relevant to `query`.
    If `claim_id` is provided (e.g. "CLM-1001"), restricts the search to
    that specific claim only. Returns the matching text chunks along with
    their source file and page number. Use this whenever you need to look
    up facts from claim documents before answering.
    """
    claim_filter = claim_id.strip() or None
    docs = similarity_search(query, k=5, claim_id=claim_filter)
    if not docs:
        return "No relevant documents found for this query."
    return "\n\n".join(
        f"[{d.metadata.get('source')}, page {d.metadata.get('page')}]\n{d.page_content}"
        for d in docs
    )


@tool
def summarize_claim(claim_id: str) -> str:
    """
    Produce a short summary (claim type, what happened, amount, status) for
    the given claim_id, e.g. "CLM-1001". Use this when the user asks for an
    overview or summary of a specific claim.
    """
    context = _get_claim_context(claim_id, query="claim summary details")
    if not context:
        return f"No documents found for claim_id '{claim_id}'."

    prompt = ChatPromptTemplate.from_template(SUMMARIZE_CLAIM_PROMPT)
    chain = prompt | get_llm() | StrOutputParser()
    return chain.invoke({"context": context})


@tool
def check_missing_information(claim_id: str) -> str:
    """
    Review the given claim_id's documents and list any information that
    appears to be missing or unclear (e.g. missing police report number,
    missing supporting invoice, unclear incident date). Use this when the
    user asks whether a claim file is complete or what's missing.
    """
    context = _get_claim_context(claim_id, query="claim details evidence documentation")
    if not context:
        return f"No documents found for claim_id '{claim_id}'."

    prompt = ChatPromptTemplate.from_template(MISSING_INFO_PROMPT)
    chain = prompt | get_llm() | StrOutputParser()
    return chain.invoke({"context": context})


@tool
def flag_fraud_risk(claim_id: str) -> str:
    """
    Review the given claim_id's documents and list any POTENTIAL fraud risk
    INDICATORS (for human adjuster review only -- this does not conclude
    fraud occurred). Use this when the user asks to assess fraud risk or
    flag suspicious claims.
    """
    context = _get_claim_context(claim_id, query="claim details circumstances evidence")
    if not context:
        return f"No documents found for claim_id '{claim_id}'."

    prompt = ChatPromptTemplate.from_template(FRAUD_RISK_PROMPT)
    chain = prompt | get_llm() | StrOutputParser()
    return chain.invoke({"context": context})


@tool
def evaluate_hallucination_tool(question: str, answer: str, claim_id: str = "") -> str:
    """
    Check whether `answer` (a previously generated response to `question`)
    is actually supported by the claim documents. Retrieves fresh context
    for the claim (optionally scoped by claim_id) and runs a hallucination
    check combining token overlap and an LLM-as-judge verdict. Use this when
    the user asks to verify or fact-check a previous answer.
    """
    claim_filter = claim_id.strip() or None
    docs = similarity_search(question, k=5, claim_id=claim_filter)
    contexts = [d.page_content for d in docs]

    if not contexts:
        return "No context available to verify this answer against."

    result = evaluate_hallucination(question, answer, contexts)
    return (
        f"Token overlap score: {result.token_overlap_score}\n"
        f"Judge verdict: {result.judge_verdict}\n"
        f"Unsupported claims: {result.judge_unsupported_claims or 'None'}\n"
        f"Explanation: {result.judge_explanation}\n"
        f"Flagged for review: {result.flagged}"
    )


@tool
def test_bias_tool(query_template: str, demographic_variants: str = "") -> str:
    """
    Run a bias probe: takes a query template containing a "{name}"
    placeholder (e.g. "A claimant named {name} reports ...") and runs it
    once per demographic name variant, then compares the answers for
    substantive differences. `demographic_variants` is an optional
    comma-separated list of names to test; if omitted, uses the default
    configured set. Use this when the user asks to test the system for
    bias across different claimant names/demographics.
    """
    names: List[str] = None
    if demographic_variants.strip():
        names = [n.strip() for n in demographic_variants.split(",") if n.strip()]

    try:
        result = run_bias_test(query_template, demographic_names=names)
    except ValueError as e:
        return str(e)

    lines = [f"Verdict: {result.comparison_verdict}", ""]
    for v in result.variants:
        lines.append(f"--- {v.variant_name} ---\n{v.answer}\n")
    lines.append(f"Comparison explanation:\n{result.comparison_explanation}")
    return "\n".join(lines)


# The full toolset the agent has access to.
ALL_TOOLS = [
    retrieve_claim_documents,
    summarize_claim,
    check_missing_information,
    flag_fraud_risk,
    evaluate_hallucination_tool,
    test_bias_tool,
]
