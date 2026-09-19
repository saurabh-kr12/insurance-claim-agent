
import time
from dataclasses import dataclass, field
from typing import List, Optional

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from src.config import settings, get_llm
from src.vectorstore import similarity_search_with_scores

RAG_SYSTEM_PROMPT = """You are an assistant helping insurance claims adjusters answer \
questions about claim documents.

Rules you MUST follow:
1. Answer ONLY using the information in the "Context" section below. Do not use \
outside knowledge or make assumptions beyond what is stated.
2. Every factual statement in your answer must include a citation in the format \
[source, page X], referencing the exact chunk it came from.
3. If the context does not contain enough information to answer the question, say \
"I don't have enough information in the retrieved documents to answer this." \
Do not guess.
4. Be concise and factual. Do not speculate about intent, fraud, or anything not \
explicitly stated in the text.

Context:
{context}
"""

RAG_USER_PROMPT = "Question: {question}"


@dataclass
class RagResult:
    """Everything the UI needs to display a Q&A turn."""
    question: str
    answer: str
    contexts: List[Document] = field(default_factory=list)
    scores: List[float] = field(default_factory=list)
    latency_seconds: float = 0.0


def _format_context(contexts: List[Document]) -> str:
    """Turns retrieved chunks into a labeled text block for the prompt."""
    blocks = []
    for doc in contexts:
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", "?")
        blocks.append(f"[{source}, page {page}]\n{doc.page_content}")
    return "\n\n---\n\n".join(blocks) if blocks else "(no relevant context found)"


def answer_question(
    question: str,
    claim_id: Optional[str] = None,
    k: Optional[int] = None,
) -> RagResult:
    """
    Main RAG entry point. Retrieves context, calls the LLM, and returns a
    structured RagResult including timing (used for LLMOps logging).
    """
    start_time = time.time()

    # Step 1: retrieve
    scored_docs = similarity_search_with_scores(question, k=k, claim_id=claim_id)
    contexts = [doc for doc, score in scored_docs]
    scores = [score for doc, score in scored_docs]

    # Step 2: build prompt
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", RAG_SYSTEM_PROMPT),
            ("human", RAG_USER_PROMPT),
        ]
    )

    llm = get_llm()
    chain = prompt | llm | StrOutputParser()

    # Step 3: generate
    context_text = _format_context(contexts)
    answer = chain.invoke({"context": context_text, "question": question})

    latency = time.time() - start_time

    return RagResult(
        question=question,
        answer=answer,
        contexts=contexts,
        scores=scores,
        latency_seconds=latency,
    )


if __name__ == "__main__":
    # Manual test: python -m src.rag_chain
    # (Assumes you've already run `python -m src.vectorstore` once to build the DB.)
    result = answer_question("What happened in the water damage claim?")
    print("Answer:\n", result.answer)
    print("\nSources used:")
    for doc, score in zip(result.contexts, result.scores):
        print(f"  {doc.metadata['source']} p{doc.metadata['page']} (score={score:.3f})")
    print(f"\nLatency: {result.latency_seconds:.2f}s")
