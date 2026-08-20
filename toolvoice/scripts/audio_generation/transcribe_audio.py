import argparse
import csv
import io
import os
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from openai import (
    APIConnectionError,
    APITimeoutError,
    BadRequestError,
    InternalServerError,
    OpenAI,
    RateLimitError,
)


MODEL_NAME = "gpt-4o-mini-transcribe"
ROOT = Path(__file__).resolve().parent
DATA_DIR = Path(__file__).resolve().parents[2] / "data"

# Audio locations follow the canonical layout written by generate_tts_audio.py:
#   data/<benchmark>/audio/<tts_model>/<voice>/output_audio<idx>.wav
#   data/<benchmark>/audio_noisy/<tts_model>/<voice>/noisy<k>_SNRdb_<snr>_output_audio<idx>.wav
DATASET_CONFIG = {
    "confetti": {
        "source_dirs": [
            DATA_DIR / "confetti" / "audio",
            DATA_DIR / "confetti" / "audio_noisy",
        ],
        "output_csv": ROOT / "GPT-4o-Mini-STT-Confetti-GPT.csv",
    },
    "when2call": {
        "source_dirs": [
            DATA_DIR / "when2call" / "audio",
            DATA_DIR / "when2call" / "audio_noisy",
        ],
        "output_csv": ROOT / "GPT-4o-Mini-STT-When2Call-GPT.csv",
    },
}

CSV_COLUMNS = [
    "dataset",
    "source_root",
    "source_dir",
    "audio_condition",
    "tts_model",
    "voice",
    "relative_path",
    "filename",
    "original_extension",
    "detected_audio_format",
    "upload_filename",
    "file_size_bytes",
    "source_audio_index",
    "noisy_sample_index",
    "snr_db",
    "transcription_model",
    "transcription_status",
    "transcription_text",
    "error_message",
    "processed_at_utc",
]

_THREAD_LOCAL = threading.local()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Transcribe the clean and noisy benchmark audio under data/<benchmark>/{audio,audio_noisy} using gpt-4o-mini-transcribe."
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=sorted(DATASET_CONFIG),
        default=sorted(DATASET_CONFIG),
        help="Datasets to process.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=6,
        help="Maximum number of concurrent transcription requests.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Recreate output CSVs instead of resuming from existing rows.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional per-dataset file limit for testing.",
    )
    return parser.parse_args()


def get_client() -> OpenAI:
    client = getattr(_THREAD_LOCAL, "client", None)
    if client is None:
        client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        _THREAD_LOCAL.client = client
    return client


def sniff_audio_format(data: bytes, fallback_extension: str) -> str:
    extension = fallback_extension.lower().lstrip(".")
    if len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WAVE":
        return "wav"
    if data.startswith(b"ID3"):
        return "mp3"
    if len(data) >= 2 and data[0] == 0xFF and (data[1] & 0xE0) == 0xE0:
        return "mp3"
    if data.startswith(b"OggS"):
        return "ogg"
    if len(data) >= 8 and data[4:8] == b"ftyp":
        return "m4a" if extension in {"m4a", "aac"} else "mp4"
    if data.startswith(b"\x1A\x45\xDF\xA3"):
        return "webm"
    return extension or "wav"


def parse_filename_fields(filename: str) -> Dict[str, Optional[str]]:
    stem = Path(filename).stem
    base = {
        "audio_condition": "clean",
        "source_audio_index": None,
        "noisy_sample_index": None,
        "snr_db": None,
    }
    if stem.startswith("output_audio"):
        base["source_audio_index"] = stem.replace("output_audio", "", 1) or None
        return base
    if stem.startswith("noisy") and "_SNRdb_" in stem and "_output_audio" in stem:
        noisy_part, remainder = stem.split("_SNRdb_", 1)
        snr_part, output_part = remainder.split("_output_audio", 1)
        base["audio_condition"] = "noisy"
        base["noisy_sample_index"] = noisy_part.replace("noisy", "", 1) or None
        base["snr_db"] = snr_part or None
        base["source_audio_index"] = output_part or None
        return base
    base["audio_condition"] = "unknown"
    return base


def build_record(dataset: str, file_path: Path) -> Dict[str, str]:
    # parts = (<benchmark>, audio|audio_noisy, <tts_model>, <voice>, <filename>)
    relative_path = file_path.relative_to(DATA_DIR)
    parts = relative_path.parts
    filename_fields = parse_filename_fields(file_path.name)
    detected_extension = file_path.suffix.lower().lstrip(".")
    upload_filename = f"{file_path.stem}.{detected_extension}" if detected_extension else file_path.name
    return {
        "dataset": dataset,
        "source_root": parts[0],
        "source_dir": parts[1],
        "audio_condition": filename_fields["audio_condition"] or "",
        "tts_model": parts[2],
        "voice": parts[3],
        "relative_path": str(relative_path),
        "filename": file_path.name,
        "original_extension": file_path.suffix.lower(),
        "detected_audio_format": detected_extension,
        "upload_filename": upload_filename,
        "file_size_bytes": str(file_path.stat().st_size),
        "source_audio_index": filename_fields["source_audio_index"] or "",
        "noisy_sample_index": filename_fields["noisy_sample_index"] or "",
        "snr_db": filename_fields["snr_db"] or "",
        "transcription_model": MODEL_NAME,
        "transcription_status": "",
        "transcription_text": "",
        "error_message": "",
        "processed_at_utc": "",
    }


def list_audio_records(dataset: str, limit: Optional[int]) -> List[Dict[str, str]]:
    records: List[Dict[str, str]] = []
    for source_dir in DATASET_CONFIG[dataset]["source_dirs"]:
        for file_path in sorted(source_dir.rglob("*.wav")):
            records.append(build_record(dataset, file_path))
    if limit is not None:
        return records[:limit]
    return records


def ensure_csv(csv_path: Path, overwrite: bool) -> None:
    if overwrite or not csv_path.exists():
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
            writer.writeheader()


def load_processed_rows(csv_path: Path) -> Dict[str, Dict[str, str]]:
    if not csv_path.exists():
        return {}
    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return {
            row["relative_path"]: row
            for row in reader
            if row.get("relative_path")
        }


def with_inferred_upload_name(file_path: Path) -> Dict[str, object]:
    data = file_path.read_bytes()
    detected_format = sniff_audio_format(data, file_path.suffix)
    return {
        "data": data,
        "detected_format": detected_format,
        "upload_filename": f"{file_path.stem}.{detected_format}",
    }


def transcribe_record(record: Dict[str, str], max_attempts: int = 6) -> Dict[str, str]:
    file_path = DATA_DIR / record["relative_path"]
    upload_payload = with_inferred_upload_name(file_path)
    audio_bytes = bytes(upload_payload["data"])
    record = dict(record)
    record["detected_audio_format"] = str(upload_payload["detected_format"])
    record["upload_filename"] = str(upload_payload["upload_filename"])

    for attempt in range(1, max_attempts + 1):
        buffer = io.BytesIO(audio_bytes)
        buffer.name = record["upload_filename"]
        try:
            result = get_client().audio.transcriptions.create(
                model=MODEL_NAME,
                file=buffer,
                response_format="text",
            )
            text = result if isinstance(result, str) else getattr(result, "text", "")
            record["transcription_status"] = "success"
            record["transcription_text"] = text.strip()
            record["error_message"] = ""
            record["processed_at_utc"] = datetime.now(timezone.utc).isoformat()
            return record
        except (RateLimitError, APIConnectionError, APITimeoutError, InternalServerError) as exc:
            if attempt == max_attempts:
                record["transcription_status"] = "error"
                record["transcription_text"] = ""
                record["error_message"] = f"{type(exc).__name__}: {exc}"
                record["processed_at_utc"] = datetime.now(timezone.utc).isoformat()
                return record
            sleep_seconds = min(30.0, (2 ** (attempt - 1)) + random.random())
            time.sleep(sleep_seconds)
        except BadRequestError as exc:
            record["transcription_status"] = "error"
            record["transcription_text"] = ""
            record["error_message"] = f"{type(exc).__name__}: {exc}"
            record["processed_at_utc"] = datetime.now(timezone.utc).isoformat()
            return record

    raise RuntimeError("Unreachable retry state encountered")


def append_rows(csv_path: Path, rows: Iterable[Dict[str, str]]) -> None:
    with csv_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        for row in rows:
            writer.writerow(row)
        handle.flush()


def sort_csv(csv_path: Path) -> None:
    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        deduped: Dict[str, Dict[str, str]] = {}
        for row in csv.DictReader(handle):
            relative_path = row["relative_path"]
            previous = deduped.get(relative_path)
            if previous is None or should_replace(previous, row):
                deduped[relative_path] = row
        rows = list(deduped.values())

    def sort_key(row: Dict[str, str]) -> tuple:
        return (
            row["source_root"],
            row["source_dir"],
            row["tts_model"],
            row["voice"],
            row["audio_condition"],
            numeric_sort_value(row["snr_db"]),
            numeric_sort_value(row["noisy_sample_index"]),
            numeric_sort_value(row["source_audio_index"]),
            row["filename"],
        )

    rows.sort(key=sort_key)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def should_replace(previous: Dict[str, str], candidate: Dict[str, str]) -> bool:
    previous_success = previous.get("transcription_status") == "success"
    candidate_success = candidate.get("transcription_status") == "success"
    if previous_success != candidate_success:
        return candidate_success
    return candidate.get("processed_at_utc", "") >= previous.get("processed_at_utc", "")


def numeric_sort_value(raw: str) -> tuple:
    if raw in ("", None):
        return (1, float("inf"))
    try:
        return (0, float(raw))
    except ValueError:
        return (0, raw)


def process_dataset(dataset: str, max_workers: int, overwrite: bool, limit: Optional[int]) -> None:
    csv_path = DATASET_CONFIG[dataset]["output_csv"]
    ensure_csv(csv_path, overwrite)
    existing = {} if overwrite else load_processed_rows(csv_path)
    records = list_audio_records(dataset, limit=limit)
    pending = [
        record
        for record in records
        if existing.get(record["relative_path"], {}).get("transcription_status") != "success"
    ]

    print(
        f"{dataset}: total={len(records)} processed={len(existing)} pending={len(pending)} output={csv_path.name}",
        flush=True,
    )
    if not pending:
        sort_csv(csv_path)
        return

    completed = 0
    batch: List[Dict[str, str]] = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_map = {executor.submit(transcribe_record, record): record for record in pending}
        for future in as_completed(future_map):
            row = future.result()
            batch.append(row)
            completed += 1
            if len(batch) >= 20:
                append_rows(csv_path, batch)
                batch.clear()
            if completed % 25 == 0 or completed == len(pending):
                print(f"{dataset}: completed {completed}/{len(pending)}", flush=True)

    if batch:
        append_rows(csv_path, batch)
    sort_csv(csv_path)


def main() -> None:
    args = parse_args()
    for dataset in args.datasets:
        process_dataset(
            dataset=dataset,
            max_workers=max(1, args.max_workers),
            overwrite=args.overwrite,
            limit=args.limit,
        )


if __name__ == "__main__":
    main()
