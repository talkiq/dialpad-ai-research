"""Synthesize benchmark questions to speech via Gemini TTS or OpenAI TTS.

Writes clean audio to data/<benchmark>/audio/<tts_model>/<voice>/output_audio<row_idx>.wav,
which is the layout every inference script reads from.

Usage:
    export OPENAI_API_KEY=...          # --provider openai
    # Gemini TTS auth is via application-default credentials (gcloud auth login)

    python generate_tts_audio.py --provider gemini --benchmark confetti
    python generate_tts_audio.py --provider openai --benchmark when2call
"""
import argparse
import os
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[2] / "data"

BENCHMARK_CSV = {
    "confetti": DATA_DIR / "confetti" / "confetti_benchmark.csv",
    "when2call": DATA_DIR / "when2call" / "when2call_benchmark.csv",
}

PROVIDER_CONFIG = {
    "gemini": {
        "voice_types": ["kore", "orus"],
        "models": {"confetti": ["gemini-2.5-flash-tts"], "when2call": ["gemini-2.5-pro-tts", "gemini-2.5-flash-tts"]},
    },
    "openai": {
        "voice_types": ["ash", "coral"],
        "models": {"confetti": ["gpt-4o-mini-tts"], "when2call": ["gpt-4o-mini-tts"]},
    },
}


def synth_gemini(client, model, voice, text):
    from google.cloud import texttospeech_v1beta1 as texttospeech

    response = client.synthesize_speech(
        input=texttospeech.SynthesisInput(text=text, prompt=""),
        voice=texttospeech.VoiceSelectionParams(name=voice, language_code="en-us", model_name=model),
        audio_config=texttospeech.AudioConfig(audio_encoding=texttospeech.AudioEncoding.MP3),
    )
    return response.audio_content


def synth_openai(client, model, voice, text, out_path):
    with client.audio.speech.with_streaming_response.create(
        model=model, voice=voice, input=text, instructions=""
    ) as response:
        response.stream_to_file(out_path)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--provider", choices=["gemini", "openai"], required=True)
    ap.add_argument("--benchmark", choices=["confetti", "when2call"], required=True)
    ap.add_argument("--output-dir", default=None, help="Defaults to data/<benchmark>/audio/.")
    args = ap.parse_args()

    cfg = PROVIDER_CONFIG[args.provider]
    out_root = args.output_dir or str(DATA_DIR / args.benchmark / "audio")
    df = pd.read_csv(BENCHMARK_CSV[args.benchmark])

    if args.provider == "gemini":
        from google.cloud import texttospeech_v1beta1 as texttospeech
        from google.api_core.client_options import ClientOptions

        client = texttospeech.TextToSpeechClient(
            client_options=ClientOptions(api_endpoint="texttospeech.googleapis.com")
        )
    else:
        from openai import OpenAI

        client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    os.makedirs(out_root, exist_ok=True)
    for model in cfg["models"][args.benchmark]:
        model_dir = os.path.join(out_root, model)
        for voice in cfg["voice_types"]:
            voice_dir = os.path.join(model_dir, voice)
            os.makedirs(voice_dir, exist_ok=True)
            existing = set(os.listdir(voice_dir))

            for idx, row in df.iterrows():
                filename = f"output_audio{idx}.wav"
                if filename in existing:
                    continue
                out_path = os.path.join(voice_dir, filename)
                content = row["question"]
                try:
                    if args.provider == "gemini":
                        audio = synth_gemini(client, model, voice, content)
                        with open(out_path, "wb") as f:
                            f.write(audio)
                    else:
                        synth_openai(client, model, voice, content, out_path)
                    print(f'Generated speech saved to "{out_path}"')
                except Exception as e:
                    print(e)
                    continue


if __name__ == "__main__":
    main()
