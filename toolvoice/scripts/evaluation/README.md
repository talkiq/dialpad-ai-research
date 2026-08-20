# Evaluation

Scores the model-response CSVs produced by the scripts in `../inference/`. Nothing here generates model outputs — run inference first, then run the scorer for the benchmark you evaluated.

## Where the inputs come from

All inference scripts write per-model CSVs to `model_responses/<model>_responses/` at the repo root, in two column flavors:

1. **Realtime/live models** (`../inference/openai_realtime_infer.py`, `gemini_live_infer.py`) carry the benchmark columns through and add `assistant_text`/`tool_calls`/`model_intent` (Confetti) or `response_type`/`final_response` (When2Call). Nemotron (`vllm_nemotron_infer.py`) also writes a `tool_calls` column in the same format.
2. **Local vLLM models** (`qwen_omni_infer.py`, `vllm_omni_infer.py`) write a `model_response` + `gold_answer` schema.

Both scorers auto-detect the flavor per file, so no flags are needed for the standard pipeline.

For the ASR-cascade baseline: `../audio_generation/transcribe_audio.py` transcribes the audio, `../inference/cascade_asr/enrich_asr_csvs.py` joins in `context`/`tool`/`gold_answer` from the benchmark CSVs, then `asr_cascade_{openai,gemini}_confetti.py` run a text LLM over the transcripts.

## Confetti — `evaluate_confetti.py`

Deterministic AST scoring, the canonical scorer that produced the paper's numbers. Parses predicted and gold calls into ASTs and matches function names, parameter names, and parameter values.

```bash
python evaluate_confetti.py --eval-all                  # batch: every Confetti CSV under model_responses/ (AST Soft, the paper's default)
python evaluate_confetti.py --eval-all --binary          # batch, exact-match binary scoring instead
python evaluate_confetti.py -i <responses.csv> -o <scored.csv>   # single file
```

Batch mode writes `ast_eval_results_{binary|soft}.csv` next to this script, one row per (model, TTS, voice). The response column is auto-detected per file (`tool_calls` if present, else `model_response`); override with `--response-column` if needed.

AST Soft (the default) scores string-valued arguments with AlignScore, which requires the AlignScore package and the `AlignScore-base.ckpt` checkpoint (~1.8GB) — neither ships with this repo; see `requirements.txt` for install notes and the checkpoint URL. Without AlignScore installed, it falls back to fuzzy string matching with a warning. Pass `--binary` for exact-match binary scoring of string arguments instead.

## When2Call — `evaluate_when2call.py`

Deterministic binary scoring, the canonical scorer. Run with no args — it scans every model folder under `model_responses/`, classifies each response as tool_call / no_tool_call (preferring a pre-labeled `final_response_type`/`response_type`/`model_intent` column, else re-parsing the raw text), and writes `when2call_parsed_accuracy.csv` with accuracy/precision/recall/F1 per (model, TTS, voice).
