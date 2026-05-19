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
Text Benchmark  →  TTS Synthesis  →  Speaker & Noise Augmentation  →  Omni-Modal Inference  →  Evaluation & Error Analysis
```

1. **TTS Conversion** — Queries synthesized with Gemini-2.5-Flash-TTS, Gemini-2.5-Pro-TTS, and GPT-4o-Mini-TTS
2. **Speaker Variation** — Female (Kore/Coral) and male (Orus/Ash) voice personas
3. **Noise Injection** — DEMAND dataset noise (cars, buses, cafés, kitchens, meeting rooms) at SNR 5/10/15/20 dB
4. **Audio Format** — 16 kHz mono 16-bit PCM WAV
5. **Human Validation** — 97.7% clean and 94.3% noisy samples confirmed content-faithful

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

Both datasets preserve the original text annotations; only the input modality changes.

---

## Evaluation

### Metrics

| Benchmark | Metric |
|-----------|--------|
| Confetti | AST-based soft accuracy (exact match for function names & non-string args; AlignScore for string values) |
| When2Call | F1-score (tool-call vs. no-tool-call decisions) |

LLM-as-judge scoring is also provided in both reference-aware and reference-free variants.

### Models Evaluated

- Gemini-Flash-Live series models (2.5 and 3.1)
- GPT-Realtime series models (4o, 1.5, etc.)
- Qwen3-Omni-30B-A3B
- *(+ cascade baselines)*

---

## Key Results Summary

| Model | Confetti (Accuracy) | When2Call (F1) |
|-------|-------------------|----------------|
| Gemini-3.1-Flash-Live | **70.4** | 63.4 |
| GPT-Realtime-1.5 | 59.2 | **71.9** |
| Qwen3-Omni-30B-A3B | 60.4 | 60.4 |

Model rankings shift between benchmarks, confirming task-dependent performance. The text-to-voice gap varies substantially by model, indicating that voice robustness is a model-specific property.

---

## Repository Structure

```
toolvoice/
├── data/
│   ├── confetti/          # Audio-converted Confetti benchmark
│   └── when2call/         # Audio-converted When2Call benchmark
├── scripts/
│   ├── audio_generation/  # TTS and noise injection pipeline
│   ├── inference/         # Omni-modal model inference
│   └── evaluation/        # Scoring scripts (AST accuracy, F1, LLM judge)
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

This project is licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
