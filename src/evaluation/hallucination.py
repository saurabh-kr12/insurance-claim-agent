import re
from dataclasses import dataclass
from typing import List

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from src.config import get_llm
from src.agents.prompts import HALLUCINATION_JUDGE_PROMPT

# Common English stopwords to exclude from overlap calculation so the score
# reflects *meaningful* word overlap, not just "the/a/is" matching everywhere.
STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "to", "of", "in", "on", "at", "for", "with", "by", "and", "or", "but",
    "this", "that", "these", "those", "it", "its", "as", "from", "not",
    "no", "does", "do", "did", "has", "have", "had", "will", "would",
    "can", "could", "should", "i", "you", "your", "we", "our",
}


def _tokenize(text: str) -> set:
    """Lowercases, strips punctuation, splits into words, removes stopwords."""
    words = re.findall(r"[a-zA-Z0-9]+", text.lower())
    return {w for w in words if w not in STOPWORDS and len(w) > 2}


@dataclass
class HallucinationCheckResult:
    token_overlap_score: float          # 0.0 - 1.0
    judge_verdict: str                  # SUPPORTED / UNSUPPORTED / PARTIALLY_SUPPORTED / ERROR
    judge_unsupported_claims: str
    judge_explanation: str
    flagged: bool                       # convenience boolean for the UI


def compute_token_overlap(answer: str, context: str) -> float:
    """
    Returns the fraction of meaningful words in `answer` that also appear
    somewhere in `context`. This is a coarse heuristic, not a precise metric,
    but is cheap and useful as a first-pass signal.
    """
    answer_words = _tokenize(answer)
    context_words = _tokenize(context)

    if not answer_words:
        return 1.0  # empty/trivial answer -- nothing to hallucinate

    overlap = answer_words.intersection(context_words)
    return len(overlap) / len(answer_words)


def llm_judge_hallucination(question: str, answer: str, context: str) -> dict:
    """Asks the LLM to act as an independent judge of factual support."""
    prompt = ChatPromptTemplate.from_template(HALLUCINATION_JUDGE_PROMPT)
    llm = get_llm()
    chain = prompt | llm | StrOutputParser()

    try:
        raw = chain.invoke({"question": question, "context": context, "answer": answer})
    except Exception as e:
        return {
            "verdict": "ERROR",
            "unsupported_claims": "",
            "explanation": f"Judge call failed: {e}",
        }

    # Parse the structured "VERDICT: / UNSUPPORTED_CLAIMS: / EXPLANATION:" format.
    verdict_match = re.search(r"VERDICT:\s*(.+)", raw)
    claims_match = re.search(r"UNSUPPORTED_CLAIMS:\s*(.+?)(?=EXPLANATION:|$)", raw, re.DOTALL)
    explanation_match = re.search(r"EXPLANATION:\s*(.+)", raw, re.DOTALL)

    return {
        "verdict": verdict_match.group(1).strip() if verdict_match else "UNKNOWN",
        "unsupported_claims": claims_match.group(1).strip() if claims_match else "",
        "explanation": explanation_match.group(1).strip() if explanation_match else raw,
    }


def evaluate_hallucination(
    question: str,
    answer: str,
    contexts: List[str],
    overlap_threshold: float = 0.35,
) -> HallucinationCheckResult:
    """
    Main entry point combining both checks.
    `contexts` is a list of raw context strings (e.g. retrieved chunk text).
    """
    combined_context = "\n\n".join(contexts)

    overlap_score = compute_token_overlap(answer, combined_context)
    judge = llm_judge_hallucination(question, answer, combined_context)

    flagged = (
        overlap_score < overlap_threshold
        or judge["verdict"].upper() in ("UNSUPPORTED", "PARTIALLY_SUPPORTED")
    )

    return HallucinationCheckResult(
        token_overlap_score=round(overlap_score, 3),
        judge_verdict=judge["verdict"],
        judge_unsupported_claims=judge["unsupported_claims"],
        judge_explanation=judge["explanation"],
        flagged=flagged,
    )


if __name__ == "__main__":
    # Manual smoke test (no LLM needed for the overlap half)
    ctx = ["The claim amount is $9,800 for water damage to the kitchen."]
    ans = "The claimed amount is $9,800 due to water damage."
    print("Token overlap:", compute_token_overlap(ans, ctx[0]))
