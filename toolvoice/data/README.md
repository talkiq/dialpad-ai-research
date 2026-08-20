# Data

`confetti_benchmark.csv` and `when2call_benchmark.csv` are the full text-form benchmarks (313 and 300 examples respectively) used as the source for TTS conversion.

No audio is checked into this repo (the full converted corpus — clean + all noise/SNR conditions, all voices — is several GB). Generate it locally with the scripts in `../scripts/audio_generation/`, run against the benchmark CSVs here.

## Generated audio layout

The pipeline scripts read and write audio using this layout (row index `<idx>` refers to the benchmark CSV's row order):

```
data/<benchmark>/audio/<tts_model>/<voice>/output_audio<idx>.wav                       # clean (generate_tts_audio.py)
data/<benchmark>/audio_noisy/<tts_model>/<voice>/noisy<k>_SNRdb_<snr>_output_audio<idx>.wav   # noisy (naming convention; generation script not included)
```

e.g. `data/confetti/audio/gpt-4o-mini-tts/ash/output_audio0.wav`. Every inference script and the ASR-cascade transcriber (`transcribe_audio.py`) resolve audio through these paths, so no path configuration is needed after generation.
