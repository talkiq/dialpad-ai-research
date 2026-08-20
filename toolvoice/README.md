# From Text to Voice: A Reproducible and Verifiable Framework for Evaluating Tool Calling LLM Agents

**Paper:** [arXiv:2605.15104](https://arxiv.org/abs/2605.15104)

**Authors:** Md Tahmid Rahman Laskar, Xue-Yong Fu, Seyyed Saeed Sarfjoo, Quinten McNamara, Jonas Robertson, Shashi Bhushan TN

---

## Overview

This repository contains the datasets and evaluation scripts for our framework that converts existing text-based tool-calling benchmarks into controlled audio evaluations — without re-annotation. By applying text-to-speech synthesis, speaker variation, and environmental noise injection, we enable reproducible voice-based evaluation of omni-modal LLMs while preserving original tool schemas and gold labels.

We evaluate 7 omni-modal models across two benchmarks and find that:
- Text-to-voice performance gaps range from **1.8 to 4.8 points** across models
- **Argument-value misunderstandings in speech** are the primary failure mode (39.5%–57.2% of errors)
- Neither cascade nor native omni architectures uniformly dominate
- Open-source **Qwen3 models (≥8B)** achieve 80%+ agreement with proprietary LLM judges

---

## Framework Pipeline

```
Text Benchmark  →  TTS Synthesis  →  Speaker & Noise Augmentation  →  Omni-Modal Inference  →  Evaluation
```

1. **TTS Conversion** — `scripts/audio_generation/generate_tts_audio.py`: queries synthesized with Gemini-2.5-Flash-TTS, Gemini-2.5-Pro-TTS, and GPT-4o-Mini-TTS into `data/<benchmark>/audio/<tts_model>/<voice>/`
2. **Speaker Variation** — Female (Kore/Coral) and male (Orus/Ash) voice personas
3. **Noise Injection** — DEMAND dataset noise (cars, buses, traffic, cafés, kitchens, meeting rooms, living rooms) mixed at SNR 5/10/15/20 dB, as described in the paper (script not included in this release)
4. **Audio Format** — 16 kHz mono 16-bit PCM WAV
5. **Human Validation** — 97.7% clean and 94.3% noisy samples confirmed content-faithful

## Quickstart

```bash
pip install -r requirements.txt

# 1. Synthesize audio for a benchmark
python scripts/audio_generation/generate_tts_audio.py --provider openai --benchmark confetti

# 2. Run a model over the audio (writes model_responses/<model>_responses/*.csv)
python scripts/inference/openai_realtime_infer.py --benchmark confetti --realtime-version 1.5

# 3. Score the responses
python scripts/evaluation/evaluate_confetti.py --eval-all           # Confetti
python scripts/evaluation/evaluate_when2call.py                     # When2Call
```

---

## Datasets

We release audio-converted versions of two tool-calling benchmarks:

### Confetti
- 313 examples requiring explicit tool calls
- Multi-turn conversational context with tool/API documentation
- Evaluates function selection and parameter extraction

### When2Call
- 300 instances (non-MCQ subset)
- Tests whether tool invocation is necessary
- Assesses model decision-making about tool usage

Both datasets preserve the original text annotations; only the input modality changes. Audio is not checked into the repo — regenerate it with the scripts above (see `data/README.md` for the layout).

---

## Evaluation

### Metrics

| Benchmark | Metric | Script |
|-----------|--------|--------|
| Confetti | AST-based soft accuracy (exact match for function names & non-string args; AlignScore for string values) | `scripts/evaluation/evaluate_confetti.py` |
| When2Call | F1-score (tool-call vs. no-tool-call decisions) | `scripts/evaluation/evaluate_when2call.py` |

### Models Evaluated (paper)

- GPT-4o-Realtime, GPT-Realtime-1.5, GPT-Realtime-Mini (`scripts/inference/openai_realtime_infer.py`)
- Gemini-2.5-Flash-Live, Gemini-3.1-Flash-Live (`scripts/inference/gemini_live_infer.py`)
- Qwen3-Omni-30B-A3B-Instruct (`scripts/inference/qwen_omni_infer.py` or `vllm_omni_infer.py`)
- Phi-4-Multimodal (inference script not included in this release; run via its HuggingFace checkpoint as described in the paper)

The repo additionally includes runners for Qwen2.5-Omni and Nemotron-3-Nano-Omni (not part of the paper's evaluation), and an ASR→text cascade baseline for Confetti (`scripts/inference/cascade_asr/`).

### Not included in this release

To keep the release focused on the core, verifiable pipeline, the following are described in the paper but do not ship with scripts here: the DEMAND noise-injection step, the LLM-as-judge evaluation (reference-aware/reference-free, including the open Qwen3 judges), the text-only scaling analysis, the ambiguous-query reformulation stress test, and the TTS quality assessment (UTMOS/WER).

---

## Results Summary

| Model | Confetti (AST Soft Accuracy) | When2Call (F1) |
|-------|------------------------------|----------------|
| Gemini-3.1-Flash-Live | **70.4** | 63.4 |
| GPT-Realtime-1.5 | 59.2 | **71.9** |

Model rankings shift between benchmarks, confirming task-dependent performance. The text-to-voice gap varies substantially by model, indicating that voice robustness is a model-specific property. See the paper for full per-model, per-TTS, per-voice results.

---

## Repository Structure

```
toolvoice/
├── data/
│   ├── confetti/          # Text benchmark CSV (audio regenerated locally, not committed)
│   └── when2call/         # Text benchmark CSV
├── scripts/
│   ├── audio_generation/  # generate_tts_audio.py, transcribe_audio.py (ASR for cascade)
│   ├── inference/         # Omni-modal model inference (+ cascade_asr/ baseline)
│   └── evaluation/        # evaluate_confetti.py (AST accuracy), evaluate_when2call.py (F1)
├── model_responses/       # Created by inference scripts; consumed by evaluation scripts
├── requirements.txt
└── README.md
```

---

## Citation

```bibtex
@article{laskar2026toolvoice,
  title     = {From Text to Voice: A Reproducible and Verifiable Framework for Evaluating Tool Calling LLM Agents},
  author    = {Laskar, Md Tahmid Rahman and Fu, Xue-Yong and Sarfjoo, Seyyed Saeed and McNamara, Quinten and Robertson, Jonas and {Bhushan TN}, Shashi},
  journal   = {arXiv preprint arXiv:2605.15104},
  year      = {2026}
}
```

---

## License

This project is licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) — see [LICENSE](LICENSE).
