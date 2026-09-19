"""
test_tools.py
-------------
Tests for the pure/deterministic logic in tools and evaluation modules
(the parts that DON'T require calling a live LLM), so these run fast and
offline in CI.
"""

from src.evaluation.hallucination import compute_token_overlap, _tokenize


def test_tokenize_removes_stopwords_and_punctuation():
    tokens = _tokenize("The claim, is for $9,800 in water damage!")
    assert "the" not in tokens
    assert "is" not in tokens
    assert "for" not in tokens
    assert "water" in tokens
    assert "damage" in tokens


def test_token_overlap_perfect_match():
    context = "The claimed amount is nine thousand eight hundred dollars for water damage."
    answer = "The claimed amount is for water damage."
    score = compute_token_overlap(answer, context)
    assert score == 1.0  # every meaningful word in answer appears in context


def test_token_overlap_no_match():
    context = "Severe windstorm caused a tree branch to fall on the roof."
    answer = "The vehicle was stolen from a shopping center parking lot."
    score = compute_token_overlap(answer, context)
    assert score < 0.3  # essentially no shared meaningful vocabulary


def test_token_overlap_empty_answer_is_trivially_supported():
    assert compute_token_overlap("", "some context here") == 1.0
