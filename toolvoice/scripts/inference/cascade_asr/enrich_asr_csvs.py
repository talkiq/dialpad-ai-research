#!/usr/bin/env python3
"""
Enrich the ASR CSVs in this directory with context, tool, and gold_answer columns
by joining each ASR row to its source-dataset row via `source_audio_index`.

- Confetti  -> ../../../data/confetti/confetti_benchmark.csv
    columns used: context, tools, gold_answer
- When2Call -> ../../../data/when2call/when2call_benchmark.csv
    columns used: tools (-> tool), gold_answer + target_tool (-> gold_answer JSON)

Writes results back to the ASR CSVs in place.
"""

import json
import os
import pandas as pd

ASR_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.normpath(os.path.join(ASR_DIR, "..", "..", "..", "data"))

BFCL_SRC = os.path.join(DATA_DIR, "confetti", "confetti_benchmark.csv")
W2C_SRC = os.path.join(DATA_DIR, "when2call", "when2call_benchmark.csv")

CONFETTI_CSV = os.path.join(ASR_DIR, "GPT-4o-Mini-STT-Confetti-GPT.csv")
W2C_CSV = os.path.join(ASR_DIR, "GPT-4o-Mini-STT-When2Call-GPT.csv")


def enrich_confetti() -> None:
    bfcl = pd.read_csv(BFCL_SRC)
    asr = pd.read_csv(CONFETTI_CSV)

    idx = asr["source_audio_index"].astype(int)
    if idx.max() >= len(bfcl):
        raise RuntimeError(
            f"source_audio_index out of range: max={idx.max()} but BFCL has {len(bfcl)} rows"
        )

    asr["context"] = idx.map(lambda i: bfcl.iloc[i]["context"])
    asr["tool"] = idx.map(lambda i: bfcl.iloc[i]["tools"])
    asr["gold_answer"] = idx.map(lambda i: bfcl.iloc[i]["gold_answer"])

    asr.to_csv(CONFETTI_CSV, index=False)
    print(f"[Confetti] enriched {len(asr)} rows -> {CONFETTI_CSV}")


def enrich_when2call() -> None:
    src = pd.read_csv(W2C_SRC)
    asr = pd.read_csv(W2C_CSV)

    idx = asr["source_audio_index"].astype(int)
    if idx.max() >= len(src):
        raise RuntimeError(
            f"source_audio_index out of range: max={idx.max()} but When2Call source has {len(src)} rows"
        )

    def gold_for(i: int) -> str:
        row = src.iloc[i]
        gold = {
            "correct_answer": None if pd.isna(row.get("gold_answer")) else row["gold_answer"],
            "target_tool": None if pd.isna(row.get("target_tool")) else row["target_tool"],
        }
        return json.dumps(gold, ensure_ascii=False)

    asr["context"] = ""
    asr["tool"] = idx.map(lambda i: src.iloc[i]["tools"])
    asr["gold_answer"] = idx.map(gold_for)

    asr.to_csv(W2C_CSV, index=False)
    print(f"[When2Call] enriched {len(asr)} rows -> {W2C_CSV}")


if __name__ == "__main__":
    enrich_confetti()
    enrich_when2call()
