# ASR→text cascade baseline

Runs the cascade condition from the paper (§4.3.3): instead of feeding audio to an
omni-modal model, the audio is first transcribed with `gpt-4o-mini-transcribe`, and a
text-only LLM does the tool calling over the transcript. Comparing this against the
direct-voice results isolates whether transcription plus text reasoning recovers more
of the clean-text performance than end-to-end audio inference.

The track is a three-step chain — each step depends on the previous one's output:

## 1. Transcribe the audio

```bash
python ../../audio_generation/transcribe_audio.py
```

Scans `data/<benchmark>/{audio,audio_noisy}` and writes two CSVs
(`GPT-4o-Mini-STT-Confetti-GPT.csv`, `GPT-4o-Mini-STT-When2Call-GPT.csv`) with one row
per clip: TTS model, voice, clean/noisy condition, SNR, and `transcription_text`.
The CSVs are written next to that script — **copy them into this folder** before step 2
(the scripts here read them from their own directory).

## 2. Join in the benchmark columns

```bash
python enrich_asr_csvs.py
```

The transcription CSVs carry only audio metadata. This joins each row back to its
source benchmark row (via `source_audio_index`), adding the `context`, `tool`, and
`gold_answer` columns the cascade runners and the evaluation scripts need. Edits the
CSVs in place; safe to re-run.

## 3. Run a text LLM over the transcripts

```bash
export OPENAI_API_KEY=...
python asr_cascade_openai_confetti.py      # OpenAI Realtime in text mode

export GEMINI_API_KEY=...
python asr_cascade_gemini_confetti.py      # Gemini Live in text mode (GEMINI_LIVE_MODEL overrides the default)

python asr_cascade_qwen_confetti.py        # local Qwen3-Omni in text mode, via in-process vLLM
```

All three filter to clean-audio rows (`audio_condition == "clean"`), send
`transcription_text` as the user turn along with the row's context and tools, and write
`<input>_<model>_text_only.csv` with the same `tool_calls`/`model_intent` (or
`model_response`, for the Qwen runner) columns the direct-voice inference scripts
produce — so `../../evaluation/evaluate_confetti.py` scores them the same way (place them
under `model_responses/<model>_responses/` or pass them with `-i`).
