from __future__ import annotations

import ast
import json
import re
from difflib import SequenceMatcher
from typing import Any


def normalize_text(text: str | None) -> str:
    if text is None:
        return ""
    return re.sub(r"\s+", " ", str(text)).strip().lower()


def extract_choice_letters(text: str | None) -> list[str]:
    if not text:
        return []
    upper = str(text).upper()
    letters = re.findall(r"\b([A-Z])\b", upper)
    filtered = [letter for letter in letters if letter in {"A", "B", "C", "D", "E", "F", "G"}]
    deduped = []
    for letter in filtered:
        if letter not in deduped:
            deduped.append(letter)
    return deduped


def extract_single_choice_answer(
    text: str | None,
    valid_letters: list[str],
    *,
    answer_prefixes: list[str],
) -> str | None:
    if not text or not valid_letters or not answer_prefixes:
        return None

    lines = [line.strip() for line in str(text).splitlines() if line.strip()]
    if not lines:
        return None

    valid_pattern = "".join(re.escape(letter) for letter in valid_letters)
    candidate_lines = [_normalize_choice_line(line) for line in lines[-3:]]
    normalized_prefixes = [_normalize_choice_line(prefix) for prefix in answer_prefixes if prefix]

    for line in reversed(candidate_lines):
        if not line:
            continue
        for prefix in normalized_prefixes:
            if not prefix:
                continue
            match = re.fullmatch(
                rf"{re.escape(prefix)}\s*[:：]?\s*[\[\(（【「『]?\s*([{valid_pattern}])\s*[\]\)）】」』]?\s*[.。．!！?？]*",
                line,
                flags=re.I,
            )
            if match:
                return match.group(1).upper()
            loose_match = re.fullmatch(
                rf"{re.escape(prefix)}\s*[:：]?\s*[\[\(（【「『]?\s*([{valid_pattern}])"
                rf"(?:\s*[\]\)）】」』]|[.。．、,，:：\-]|\s+).+",
                line,
                flags=re.I,
            )
            if loose_match:
                return loose_match.group(1).upper()

    last_line = candidate_lines[-1]
    fallback = re.fullmatch(
        rf"[\[\(（【「『]?\s*([{valid_pattern}])\s*[\]\)）】」』]?\s*[.。．!！?？]*",
        last_line,
        flags=re.I,
    )
    if fallback:
        return fallback.group(1).upper()
    return None


def extract_choice_answer(
    text: str | None,
    valid_letters: list[str],
    *,
    answer_prefixes: list[str],
    allow_multiple: bool = False,
) -> str | None:
    if not allow_multiple:
        return extract_single_choice_answer(text, valid_letters, answer_prefixes=answer_prefixes)
    if not text or not valid_letters or not answer_prefixes:
        return None

    lines = [line.strip() for line in str(text).splitlines() if line.strip()]
    if not lines:
        return None

    valid_pattern = "".join(re.escape(letter) for letter in valid_letters)
    candidate_lines = [_normalize_choice_line(line) for line in lines[-3:]]
    normalized_prefixes = [_normalize_choice_line(prefix) for prefix in answer_prefixes if prefix]

    for line in reversed(candidate_lines):
        if not line:
            continue
        for prefix in normalized_prefixes:
            if not prefix:
                continue
            match = re.fullmatch(
                rf"{re.escape(prefix)}\s*[:：]?\s*[\[\(（【「『]?\s*([{valid_pattern}](?:\s*[,，、/]\s*[{valid_pattern}]|\s*[{valid_pattern}])*)\s*[\]\)）】」』]?\s*[.。？！!?]*",
                line,
                flags=re.I,
            )
            if match:
                return _normalize_choice_answer(match.group(1), valid_letters)

    last_line = candidate_lines[-1]
    fallback = re.fullmatch(
        rf"[\[\(（【「『]?\s*([{valid_pattern}](?:\s*[,，、/]\s*[{valid_pattern}]|\s*[{valid_pattern}])*)\s*[\]\)）】」』]?\s*[.。？！!?]*",
        last_line,
        flags=re.I,
    )
    if fallback:
        return _normalize_choice_answer(fallback.group(1), valid_letters)
    return None


def normalize_choice_answer(value: str | None, valid_letters: list[str]) -> str | None:
    if value in (None, ""):
        return None
    return _normalize_choice_answer(str(value), valid_letters)


def _normalize_choice_answer(value: str, valid_letters: list[str]) -> str | None:
    valid = {letter.upper() for letter in valid_letters}
    letters = [letter for letter in re.findall(r"[A-Z]", value.upper()) if letter in valid]
    if not letters:
        return None
    deduped: list[str] = []
    for letter in letters:
        if letter not in deduped:
            deduped.append(letter)
    return "".join(deduped)


def _normalize_choice_line(text: str) -> str:
    normalized = str(text).translate(
        str.maketrans(
            {
                "：": ":",
                "（": "(",
                "）": ")",
                "【": "[",
                "】": "]",
                "［": "[",
                "］": "]",
                "「": "[",
                "」": "]",
                "『": "[",
                "』": "]",
                "。": ".",
                "．": ".",
                "，": ",",
            }
        )
    )
    normalized = re.sub(r"[*_`#>\-]+", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized.strip()


def parse_score_value(value: Any) -> tuple[float | None, float | None]:
    if value is None:
        return None, None
    text = str(value).strip()
    if not text:
        return None, None
    if "/" in text:
        left, right = text.split("/", 1)
        try:
            score = float(left.strip())
            total = float(right.strip())
            return score, total
        except ValueError:
            return None, None
    try:
        score = float(text)
        return score, None
    except ValueError:
        return None, None


def text_similarity(left: str | None, right: str | None) -> float:
    a = normalize_text(left)
    b = normalize_text(right)
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return SequenceMatcher(a=a, b=b).ratio()


def extract_json_object(text: str | None) -> dict[str, Any] | None:
    if not text:
        return None
    candidate = str(text).strip()
    fenced = re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", candidate, flags=re.S | re.I)
    if fenced:
        candidate = fenced[0]
    else:
        match = re.search(r"\{.*\}", candidate, flags=re.S)
        if match:
            candidate = match.group(0)
    for parser in (_try_json, _try_literal_eval):
        parsed = parser(candidate)
        if isinstance(parsed, dict):
            return parsed
    return None


def extract_required_json_keys(prompt: str) -> list[str]:
    matches = re.findall(r'["\']([^"\']{1,80})["\']', prompt)
    keys: list[str] = []
    for match in matches:
        normalized = match.strip()
        if not normalized:
            continue
        if "\n" in normalized:
            continue
        if len(normalized) > 60:
            continue
        if normalized not in keys:
            keys.append(normalized)
    return keys


def _try_json(text: str) -> dict[str, Any] | None:
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else None
    except Exception:
        return None


def _try_literal_eval(text: str) -> dict[str, Any] | None:
    try:
        parsed = ast.literal_eval(text)
        return parsed if isinstance(parsed, dict) else None
    except Exception:
        return None
