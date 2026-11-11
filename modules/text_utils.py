"""Shared text-normalisation helpers.

Real chat text arrives with curly apostrophes (phones, Word), repeated
letters ("sooooo"), odd spacing, etc.  Every rule-based module normalises
through here so patterns such as "haven't" always match.
"""
import re

_QUOTES = {
    "’": "'", "‘": "'", "ʼ": "'", "`": "'", "´": "'",
    "“": '"', "”": '"',
    "–": "-", "—": "-", "…": "...",
    " ": " ",
}
_QUOTE_TABLE = str.maketrans(_QUOTES)

# Idioms that contain emotional / negating words but carry no emotion.
_NEUTRAL_IDIOMS = re.compile(
    r"\bno (?:worries|problem|problems|rush|hurry)\b|\bnot a problem\b|"
    r"\bnot to worry\b|\bdon'?t worry\b|\bnever ?mind\b|"
    r"\b(?:sorry|apologi[sz]e|apologies) for (?:the|any|my|this)(?: \w+)? (?:confusion|inconvenience|mix[- ]?up)\b",
    re.I,
)


def normalize_text(text):
    """Straighten quotes, collapse whitespace, keep case."""
    if not isinstance(text, str):
        return ""
    text = text.translate(_QUOTE_TABLE)
    return re.sub(r"[ \t]+", " ", text).strip()


def collapse_repeats(text):
    """'sooooo' -> 'soo', 'goood' -> 'good' (3+ repeated letters become 2)."""
    return re.sub(r"([a-zA-Z])\1{2,}", r"\1\1", text)


def strip_neutral_idioms(text):
    return _NEUTRAL_IDIOMS.sub(" ", text)
