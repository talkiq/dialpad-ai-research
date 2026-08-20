"""Compute When2Call binary accuracy by detecting whether the model
emitted a tool call.

For each ``when2call`` / ``When2call`` CSV under every model directory in
``model_responses/`` (at the repo root, as written by the inference
scripts), this script inspects the model output column (``response`` or
``model_response``) and labels each row as either ``tool_call`` or
``no_tool_call``. Gold labels are collapsed the same way (anything other
than ``tool_call`` becomes ``no_tool_call``).
"""

from __future__ import annotations

import ast
import json
import os
import re
from typing import Optional

import pandas as pd

RESPONSES_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "model_responses",
)


GOLD_CLASSES = {"tool_call", "no_tool_call"}

# Regexes used to detect tool-call output shapes.
_CODE_FENCE_RE = re.compile(r"```(?:json|JSON)?\s*(.*?)```", re.DOTALL)
_TYPE_FIELD_RE = re.compile(
    r"""["']\s*Type\s*["']\s*:\s*["']([^"'\n]+)["']""",
    re.IGNORECASE,
)
_LEADING_LABEL_RE = re.compile(
    r"^\s*(?:TYPE\s*[:=]\s*)?([A-Z_][A-Z_]{2,})\b",
)
_TOOL_CALL_LIST_RE = re.compile(r'^\s*\[\s*\{\s*["\']id["\']\s*:')
_NAME_ARGS_RE = re.compile(
    r'["\']name["\']\s*:\s*["\'][^"\']+["\']\s*,\s*'
    r'["\'](?:arguments|args_json|args|parameters)["\']\s*:'
)
_PLAIN_TYPE_TOOL_CALL_RE = re.compile(
    r"^\s*[\"']?Type[\"']?\s*[:=]\s*[\"']?TOOL[_-]?CALL\b",
    re.IGNORECASE,
)
_TOOL_CALL_LABELS = {"TOOL_CALL", "TOOLCALL"}
_NON_TOOL_CALL_LABELS = {
    "FOLLOW_UP_QUESTION",
    "FOLLOWUP_QUESTION",
    "FOLLOW_QUESTION",
    "FOLLOWUP",
    "REQUEST_FOR_INFO",
    "CLARIFY",
    "CLARIFICATION",
    "CANNOT_ANSWER",
    "CANT_ANSWER",
    "NO_ANSWER",
    "REFUSE",
    "DIRECT_ANSWER",
    "DIRECT",
    "ANSWER",
}


def _label_is_tool_call(raw: str) -> bool:
    """Return ``True`` if a label string denotes a tool call.

    Recognises explicit ``TOOL_CALL`` labels and tool-name-like identifiers
    (e.g. ``Music_3_LookupMusic``). Known non-tool-call control labels
    (``FOLLOW_UP_QUESTION``, ``REQUEST_FOR_INFO``, ``CANNOT_ANSWER``, ...)
    are short-circuited to ``False`` so their underscores don't trip the
    tool-name heuristic.
    """
    if not raw:
        return False
    key = re.sub(r"\s+", "_", raw.strip()).upper()
    if key in _TOOL_CALL_LABELS:
        return True
    if key in _NON_TOOL_CALL_LABELS:
        return False
    if re.search(r"[A-Z0-9]+[._][A-Z0-9]", raw) or "." in raw:
        return True
    if raw.isupper() and "_" in raw:
        return True
    return False


def _strip_code_fences(text: str) -> str:
    """Return inner content of a fenced JSON block when present."""
    match = _CODE_FENCE_RE.search(text)
    return match.group(1) if match else text


def _try_json_like(text: str) -> Optional[dict]:
    """Attempt to parse ``text`` as JSON, with a Python-literal fallback."""
    candidate = text.strip()
    if not candidate:
        return None
    first = candidate.find("{")
    last = candidate.rfind("}")
    if first != -1 and last > first:
        candidate = candidate[first : last + 1]
    for loader in (json.loads, ast.literal_eval):
        try:
            value = loader(candidate)
            if isinstance(value, dict):
                return value
        except Exception:
            continue
    return None


def parse_response_type(raw: object) -> str:
    """Return ``"tool_call"`` if the raw model output is a tool call,
    otherwise ``"no_tool_call"``.

    A response counts as a tool call when any of the following is true:
      - It is a list of call dicts (``[{"id": ..., "name": ...}]``).
      - It contains a Qwen ``<tool_call>`` tag.
      - It has a ``Type`` field whose value is ``TOOL_CALL`` (or a
        tool-name-like identifier).
      - It parses as a function-call dict (``{"name": ..., "arguments": ...}``).
      - It starts with a ``TOOL_CALL`` / tool-name leading label.
      - A ``{"name": ..., "arguments": ...}`` shape appears anywhere in body.

    Empty / NaN / ``"error"`` outputs are treated as ``no_tool_call``.
    """
    if raw is None:
        return "no_tool_call"
    text = str(raw).strip()
    if not text or text.lower() == "nan" or text.lower() == "error":
        return "no_tool_call"

    # Unwrap Qwen-style ``['...']`` list-of-strings wrappers.
    if text.startswith("['") or text.startswith('["'):
        try:
            unwrapped = ast.literal_eval(text)
            if isinstance(unwrapped, list) and unwrapped:
                text = str(unwrapped[0]).strip()
        except Exception:
            pass

    if _TOOL_CALL_LIST_RE.match(text):
        return "tool_call"
    if "<tool_call>" in text.lower():
        return "tool_call"

    inner = _strip_code_fences(text)

    if _PLAIN_TYPE_TOOL_CALL_RE.match(inner):
        return "tool_call"

    type_match = _TYPE_FIELD_RE.search(inner)
    if type_match and _label_is_tool_call(type_match.group(1)):
        return "tool_call"

    parsed = _try_json_like(inner)
    if isinstance(parsed, dict):
        for key in ("Type", "type", "TYPE"):
            value = parsed.get(key)
            if isinstance(value, str) and _label_is_tool_call(value):
                return "tool_call"
        if "name" in parsed and (
            "arguments" in parsed
            or "parameters" in parsed
            or "args_json" in parsed
            or "args" in parsed
        ):
            return "tool_call"

    label_match = _LEADING_LABEL_RE.match(inner)
    if label_match and _label_is_tool_call(label_match.group(1)):
        return "tool_call"

    if _NAME_ARGS_RE.search(inner):
        return "tool_call"

    return "no_tool_call"


def _label_to_class(raw: object) -> str:
    """Collapse a pipeline intent label to ``tool_call`` / ``no_tool_call``.

    Recognises ``tool_call`` / ``TOOL_CALL`` (case-insensitive) as positive
    and everything else (``text``, ``FOLLOW_UP_QUESTION``, ``CANNOT_ANSWER``,
    ``DIRECT_ANSWER``, NaN, etc.) as ``no_tool_call``.
    """
    if raw is None:
        return "no_tool_call"
    key = str(raw).strip().lower()
    if not key or key == "nan":
        return "no_tool_call"
    return "tool_call" if key == "tool_call" else "no_tool_call"


def _normalize_gold(raw: object) -> Optional[str]:
    """Collapse a gold label to ``tool_call`` / ``no_tool_call``.

    Accepts either a plain string label (``tool_call`` /
    ``cannot_answer`` / ``request_for_info`` / ...) or a JSON object with
    a ``correct_answer`` field (used by the ASR-pipeline CSVs).

    Returns ``None`` for missing / unrecognised gold values.
    """
    if raw is None:
        return None
    text = str(raw).strip()
    if not text or text.lower() == "nan":
        return None
    if text.startswith("{"):
        try:
            obj = json.loads(text)
            if isinstance(obj, dict) and "correct_answer" in obj:
                text = str(obj["correct_answer"]).strip()
        except Exception:
            pass
    text = text.lower()
    if not text or text == "nan":
        return None
    return "tool_call" if text == "tool_call" else "no_tool_call"


def _iter_model_dirs(base: str) -> list[tuple[str, str]]:
    """Yield ``(model_name, directory_path)`` for every subdirectory of
    ``model_responses/``. The inference scripts name their output folders
    ``<model>_responses``, so no per-model pattern list is needed — a new
    model is picked up automatically."""
    entries: list[tuple[str, str]] = []
    if not os.path.isdir(base):
        return entries
    for name in sorted(os.listdir(base)):
        full = os.path.join(base, name)
        if not os.path.isdir(full):
            continue
        model = name.replace("_responses", "")
        entries.append((model, full))
    return entries


def _pick_columns(df: pd.DataFrame) -> tuple[Optional[str], str, str]:
    """Return ``(label_column, response_column, gold_column)`` for the given
    dataframe.

    ``label_column`` holds a pre-classified intent (``tool_call`` / other)
    produced by the response-generation pipeline. When present, scoring
    should trust it instead of re-parsing the raw response text. Lookup
    order: ``final_response_type`` -> ``response_type`` -> ``model_intent``.

    ``response_column`` is the raw model output and is only used as a
    fallback when no label column is available.
    """
    label_col: Optional[str] = None
    for cand in ("final_response_type", "response_type", "model_intent"):
        if cand in df.columns:
            label_col = cand
            break
    response_col = (
        "model_response"
        if "model_response" in df.columns
        else "response"
        if "response" in df.columns
        else "final_response"
    )
    gold_col = (
        "gold_answer" if "gold_answer" in df.columns else "correct_answer"
    )
    return label_col, response_col, gold_col


def _tts_fields(filename: str, model_name: str) -> tuple[str, str]:
    """Extract the TTS model and voice type from a When2Call filename.

    Filenames follow two patterns::

        when2call_<model>_final_<tts-model>_<voice>.csv
        When2call_<model>_<tts-model>_<voice>.csv
    """
    stem = os.path.splitext(filename)[0]
    parts = stem.split("_")
    voice = parts[-1]
    tts = parts[-2]
    # Some phi filenames use a double-underscore (``microsoft__Phi-...``)
    # which produces an empty string when split.
    if not tts and len(parts) >= 3:
        tts = parts[-3]
    return tts, voice


def evaluate_directory(
    model_name: str, directory: str
) -> list[dict]:
    """Compute parsed accuracy for each When2Call CSV in ``directory``.

    :param model_name: Short model identifier (directory name minus suffix).
    :type model_name: str
    :param directory: Absolute path to the model's response directory.
    :type directory: str
    :returns: List of per-file result dicts.
    :rtype: list[dict]
    """
    results: list[dict] = []
    for filename in sorted(os.listdir(directory)):
        if not filename.lower().endswith(".csv"):
            continue
        if "when2call" not in filename.lower():
            continue
        filepath = os.path.join(directory, filename)
        try:
            df = pd.read_csv(filepath)
        except Exception as exc:
            print(f"[skip] {filepath}: {exc}")
            continue

        label_col, response_col, gold_col = _pick_columns(df)
        if gold_col not in df.columns:
            print(f"[skip] {filepath}: missing gold column")
            continue
        if label_col is None and response_col not in df.columns:
            print(f"[skip] {filepath}: missing label and response columns")
            continue

        correct = 0
        total = 0
        tp = 0  # predicted tool_call & gold tool_call
        fp = 0  # predicted tool_call & gold no_tool_call
        fn = 0  # predicted no_tool_call & gold tool_call
        tn = 0  # predicted no_tool_call & gold no_tool_call
        for _, row in df.iterrows():
            gold = _normalize_gold(row[gold_col])
            if gold is None:
                continue
            if label_col is not None:
                parsed = _label_to_class(row[label_col])
            else:
                parsed = parse_response_type(row[response_col])
            total += 1
            if parsed == gold:
                correct += 1
            if parsed == "tool_call" and gold == "tool_call":
                tp += 1
            elif parsed == "tool_call" and gold == "no_tool_call":
                fp += 1
            elif parsed == "no_tool_call" and gold == "tool_call":
                fn += 1
            else:
                tn += 1

        tts, voice = _tts_fields(filename, model_name)
        accuracy = (100.0 * correct / total) if total else 0.0
        precision = (100.0 * tp / (tp + fp)) if (tp + fp) else 0.0
        recall = (100.0 * tp / (tp + fn)) if (tp + fn) else 0.0
        f1 = (
            (2 * precision * recall / (precision + recall))
            if (precision + recall)
            else 0.0
        )
        no_tool_precision = (
            (100.0 * tn / (tn + fn)) if (tn + fn) else 0.0
        )
        specificity = (100.0 * tn / (tn + fp)) if (tn + fp) else 0.0
        results.append(
            {
                "model": model_name,
                "tts": tts,
                "voice": voice,
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "no_tool_precision": no_tool_precision,
                "specificity": specificity,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tn": tn,
                "total": total,
                "filename": filename,
            }
        )
    return results


OUTPUT_CSV = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "when2call_parsed_accuracy.csv",
)


def main() -> None:
    """Entry point: print and persist parsed accuracies per model / TTS."""
    print("When2Call Benchmark - parsed from raw model output")
    header = (
        f"{'Model':30} | {'TTS Model':24} | {'Voice':8} | "
        f"{'Acc':>7} | {'Prec':>7} | {'Rec':>7} | {'F1':>7} | "
        f"{'NoToolP':>7} | {'Spec':>7} | {'N':>4}"
    )
    print(header)
    print("-" * len(header))
    rows: list[dict] = []
    for model_name, directory in _iter_model_dirs(RESPONSES_DIR):
        for result in evaluate_directory(model_name, directory):
            rows.append(result)
            print(
                f"{result['model']:30} | {result['tts']:24} | "
                f"{result['voice']:8} | "
                f"{result['accuracy']:6.2f}% | "
                f"{result['precision']:6.2f}% | "
                f"{result['recall']:6.2f}% | "
                f"{result['f1']:6.2f}% | "
                f"{result['no_tool_precision']:6.2f}% | "
                f"{result['specificity']:6.2f}% | "
                f"{result['total']:4d}"
            )

    if rows:
        out_df = pd.DataFrame(
            rows,
            columns=[
                "model",
                "tts",
                "voice",
                "accuracy",
                "precision",
                "recall",
                "f1",
                "no_tool_precision",
                "specificity",
                "tp",
                "fp",
                "fn",
                "tn",
                "total",
                "filename",
            ],
        )
        for col in (
            "accuracy",
            "precision",
            "recall",
            "f1",
            "no_tool_precision",
            "specificity",
        ):
            out_df[col] = out_df[col].round(2)
        out_df.to_csv(OUTPUT_CSV, index=False)
        print(f"\nSaved detailed results to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
