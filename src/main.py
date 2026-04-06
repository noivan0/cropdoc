"""
CropDoc — CLI Entry Point & Batch Tester
=========================================
Usage examples:

  # Single image diagnosis
  python main.py --image leaf.jpg --language sw

  # Image + audio diagnosis
  python main.py --image leaf.jpg --audio farmer_voice.wav --language hi

  # Batch test on PlantVillage samples
  python main.py --batch --n-samples 10 --output results/batch_test.json

  # Download sample images
  python main.py --download-samples

  # List supported languages
  python main.py --list-languages
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from datetime import datetime

# ---------------------------------------------------------------------------
# Allow running from project root or src/
# ---------------------------------------------------------------------------
SRC_DIR = Path(__file__).parent
PROJECT_DIR = SRC_DIR.parent
sys.path.insert(0, str(SRC_DIR))

from model import CropDoctorModel, SUPPORTED_LANGUAGES
from pipeline import (
    download_sample_images,
    create_dummy_sample,
    load_local_samples,
    evaluate_batch,
    save_results,
    parse_diagnosis,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

OUTPUT_DIR = PROJECT_DIR / "results"


# ---------------------------------------------------------------------------
# CLI Argument Parser
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="CropDoc — AI Crop Disease Diagnosis CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Single-image mode
    p.add_argument("--image", type=str, help="Path to crop image for diagnosis.")
    p.add_argument(
        "--audio", type=str, default=None,
        help="(Optional) Path to farmer voice audio file. Enables E4B multimodal mode.",
    )
    p.add_argument(
        "--language", type=str, default="en",
        help="Language code for response (default: en). Use --list-languages to see all.",
    )

    # Batch test mode
    p.add_argument(
        "--batch", action="store_true",
        help="Run batch test on local sample images.",
    )
    p.add_argument(
        "--n-samples", type=int, default=5,
        help="Number of samples for batch test (default: 5).",
    )

    # Utilities
    p.add_argument(
        "--download-samples", action="store_true",
        help="Download PlantVillage sample images to samples/ directory.",
    )
    p.add_argument(
        "--list-languages", action="store_true",
        help="Print supported language codes and names.",
    )
    p.add_argument(
        "--output", type=str, default=None,
        help="Path to save results JSON (default: results/YYYY-MM-DD_HH-MM-SS.json).",
    )
    p.add_argument(
        "--api-key", type=str, default=None,
        help="Google AI Studio API key (overrides GOOGLE_API_KEY env var).",
    )
    p.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable verbose logging.",
    )

    return p


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------
def print_result(result: dict, verbose: bool = False) -> None:
    """Pretty-print a single diagnosis result to stdout."""
    print("\n" + "=" * 60)
    print(f"  🌾 CropDoc Diagnosis")
    print("=" * 60)

    if result.get("error"):
        print(f"  ❌ ERROR: {result['error']}")
        return

    lang = result.get("language_name", result.get("language", "?"))
    audio_flag = "✅" if result.get("audio_included") else "❌"

    print(f"  Language   : {lang}")
    print(f"  Audio used : {audio_flag}")
    print(f"  Disease    : {result.get('disease', '—')[:80]}")
    print(f"  Severity   : {result.get('severity', '—')}")
    print("-" * 60)
    print(result.get("diagnosis", ""))
    print("=" * 60 + "\n")


def compute_accuracy(results: list) -> dict:
    """
    Compute simple accuracy metrics for batch results.

    Since we use free-form text responses, we check if the ground truth
    disease label appears anywhere in the diagnosis text.

    Returns:
        Dict with hit_rate, n_total, n_hits, severity_distribution.
    """
    n_total = len(results)
    n_hits = 0
    severity_counts = {"CRITICAL": 0, "MODERATE": 0, "MONITOR": 0, "UNKNOWN": 0}

    for r in results:
        label = (r.get("ground_truth") or "").lower().replace("_", " ")
        diagnosis = (r.get("diagnosis") or "").lower()

        # Partial match: check if any word from the ground truth appears in diagnosis
        label_words = [w for w in label.split() if len(w) > 4]  # skip short words
        if label_words and any(w in diagnosis for w in label_words):
            n_hits += 1

        sev = r.get("severity", "UNKNOWN")
        if sev in severity_counts:
            severity_counts[sev] += 1
        else:
            severity_counts["UNKNOWN"] += 1

    return {
        "n_total": n_total,
        "n_hits": n_hits,
        "hit_rate": round(n_hits / n_total, 3) if n_total else 0.0,
        "severity_distribution": severity_counts,
    }


# ---------------------------------------------------------------------------
# Main commands
# ---------------------------------------------------------------------------

def cmd_list_languages() -> None:
    print("\n🌍 Supported Languages (ISO 639-1 codes):\n")
    for code, name in SUPPORTED_LANGUAGES.items():
        print(f"  {code:6s}  {name}")
    print()


def cmd_download_samples() -> None:
    print("📥 Downloading PlantVillage sample images...\n")
    paths = download_sample_images()
    if paths:
        print(f"\n✅ Downloaded {len(paths)} sample images:")
        for p in paths:
            print(f"   {p}")
    else:
        print("⚠️  Download failed. Creating synthetic dummy image...")
        dummy = create_dummy_sample()
        print(f"   Created: {dummy}")


def cmd_single_image(args, model: CropDoctorModel) -> dict:
    """Run diagnosis on a single image and return the result."""
    if not args.image:
        print("❌ --image is required for single-image mode.")
        sys.exit(1)

    print(f"\n🔍 Analysing: {args.image}")
    print(f"   Language : {SUPPORTED_LANGUAGES.get(args.language, args.language)}")

    if args.audio:
        print(f"   Audio    : {args.audio}")
        result = model.analyze_with_audio(
            args.image, args.audio, language=args.language
        )
    else:
        result = model.analyze_image(args.image, language=args.language)

    print_result(result, verbose=args.verbose)
    return result


def cmd_batch_test(args, model: CropDoctorModel) -> list:
    """Run batch diagnosis on local sample images."""
    sample_dir = PROJECT_DIR / "samples"
    samples = load_local_samples(str(sample_dir))

    if not samples:
        print("⚠️  No local samples found. Downloading...")
        cmd_download_samples()
        samples = load_local_samples(str(sample_dir))

    if not samples:
        print("❌ Could not load any samples. Exiting.")
        sys.exit(1)

    # Limit to requested count
    samples = samples[: args.n_samples]
    print(f"\n🧪 Running batch test on {len(samples)} samples | Language: {args.language}\n")

    results = []
    for i, sample in enumerate(samples, 1):
        img_path = sample.get("path", "")
        label = sample.get("label_name", "unknown")
        print(f"  [{i}/{len(samples)}] {label} ({Path(img_path).name})")

        try:
            r = model.analyze_image(img_path, language=args.language)
            r["ground_truth"] = label
            r["source_path"] = img_path
            results.append(r)

            print(f"         → Severity: {r['severity']} | Error: {r.get('error') or 'None'}")

        except Exception as exc:  # noqa: BLE001
            logger.error("Failed on sample %d: %s", i, exc)
            results.append({"error": str(exc), "ground_truth": label})

        time.sleep(0.5)  # Gentle rate-limiting

    # Accuracy summary
    metrics = compute_accuracy(results)
    print("\n" + "=" * 60)
    print("  📊 Batch Test Summary")
    print("=" * 60)
    print(f"  Total samples  : {metrics['n_total']}")
    print(f"  Keyword hits   : {metrics['n_hits']} / {metrics['n_total']}")
    print(f"  Hit rate       : {metrics['hit_rate'] * 100:.1f}%")
    print(f"  Severity dist  : {metrics['severity_distribution']}")
    print("=" * 60 + "\n")

    # Attach metrics to results
    for r in results:
        r["_batch_metrics"] = metrics

    return results


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # ── Utility commands (no model needed) ──────────────────────────────
    if args.list_languages:
        cmd_list_languages()
        return

    if args.download_samples:
        cmd_download_samples()
        return

    # ── Model init ───────────────────────────────────────────────────────
    api_key = args.api_key or os.environ.get("GOOGLE_API_KEY", "")
    if not api_key:
        print("❌ GOOGLE_API_KEY is not set.")
        print("   Export it: export GOOGLE_API_KEY='your-key'")
        print("   Or pass:   --api-key 'your-key'")
        sys.exit(1)

    try:
        model = CropDoctorModel(api_key=api_key)
    except Exception as exc:
        print(f"❌ Model init failed: {exc}")
        sys.exit(1)

    # ── Execute command ──────────────────────────────────────────────────
    results = []
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    if args.batch:
        results = cmd_batch_test(args, model)
    elif args.image:
        r = cmd_single_image(args, model)
        results = [r]
    else:
        parser.print_help()
        print("\n💡 Tip: Use --image <path> to diagnose a crop image.")
        sys.exit(0)

    # ── Save results ─────────────────────────────────────────────────────
    output_path = args.output or str(OUTPUT_DIR / f"cropdoc_{timestamp}.json")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    save_results(results, output_path)
    print(f"💾 Results saved to: {output_path}")


if __name__ == "__main__":
    main()
