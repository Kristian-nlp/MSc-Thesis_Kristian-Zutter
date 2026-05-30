"""
run_features_batch.py

Master's thesis: Learning the Levers — Which Creator-Controllable
Features Predict Post Visibility on TikTok, Instagram, and LinkedIn
from a Swiss German-Language Perspective
Kristian Zutter | HSLU MSc Applied Information and Data Science | 2026

Purpose:
    Batch feature-engineering orchestrator. Runs all feature
    extraction pipelines sequentially in dependency order (text,
    topic embeddings, audio, visual, temporal), each as its own
    subprocess so memory is fully released between stages. Decoupled
    from the hourly scrape so the small VM never has to run heavy
    NLP and vision models concurrently with Selenium.

Inputs:
    01_config/settings.py             DB_PATH, LOG_DIR
    04_database/scraper.db            captures (read by each stage)

Outputs:
    04_database/scraper.db            features_text, features_topic,
                                      features_audio, features_visual,
                                      features_temporal tables
    DATA_DIR/logs/features_batch.log  run log
    /tmp/features_batch.lock          process lock file

Usage:
    python 03_features/run_features_batch.py
    python 03_features/run_features_batch.py --dry-run
    python 03_features/run_features_batch.py --only text temporal
    python 03_features/run_features_batch.py --skip visual
    python 03_features/run_features_batch.py --db /data/scraper.db
"""

import argparse
import fcntl
import logging
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger("features_batch")


# ===================================================================
# Configuration
# ===================================================================

DB_PATH = Path(__file__).resolve().parent.parent / "04_database" / "scraper.db"

# Directory containing the individual extraction scripts
FEATURES_DIR = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# Pipeline registry
# ---------------------------------------------------------------------------
# Each entry: (key, script_filename, description, dependencies)
# Order matters: pipelines run top to bottom.
# "dependencies" lists keys that should run first.  The orchestrator
# does not enforce this at runtime (it relies on fixed ordering) but
# documents it for clarity.

PIPELINES = [
    {
        "key": "text",
        "script": "extract_text_features.py",
        "description": "Text features: caption length, hashtags, emoji, CTA, language",
        "stage": 1,
        "dependencies": [],
    },
    {
        "key": "embeddings",
        "script": "extract_topic_embeddings.py",
        "description": "Semantic topic embeddings on captions",
        "stage": 2,
        "dependencies": ["text"],
    },
    {
        "key": "style",
        "script": "extract_style_features.py",
        "description": "Style features: readability, punctuation, capitalization, formatting",
        "stage": 3,
        "dependencies": ["text"],
    },
    {
        "key": "audio",
        "script": "extract_audio_features.py",
        "description": "Audio features: presence flag, trending sound",
        "stage": 4,
        "dependencies": [],
    },
    {
        "key": "visual",
        "script": "extract_visual_features.py",
        "description": "Visual features: brightness, contrast, colourfulness, face detection, OCR",
        "stage": 5,
        "dependencies": [],
    },
    {
        "key": "temporal",
        "script": "extract_temporal_features.py",
        "description": "Temporal features: local hour, weekday, weekend, post age",
        "stage": 6,
        "dependencies": [],
    },
]

# Timeout per pipeline (seconds).  The embeddings and visual pipelines
# can take a long time on a full corpus; others are fast.
PIPELINE_TIMEOUTS = {
    "text": 1800,        # 30 min
    "embeddings": 7200,  # 2 h  (sentence-transformers on CPU)
    "style": 1800,       # 30 min (NLTK + textstat, pure computation)
    "audio": 900,        # 15 min
    "visual": 7200,      # 2 h  (HTTP fetches + OpenCV + Tesseract, single-pass)
    "temporal": 600,     # 10 min (pure computation, no I/O)
}

LOCK_FILE = Path("/tmp/features_batch.lock")

DEFAULT_TIMEOUT = 3600  # 1 h fallback


# ===================================================================
# Pipeline execution
# ===================================================================

def run_pipeline(
    pipeline: dict,
    db_path: Path,
    dry_run: bool = False,
    log_level: str = "INFO",
) -> dict:
    """
    Run a single feature extraction pipeline as a subprocess.

    Using subprocess keeps memory isolated: when the child process
    exits, all its allocations (model weights, image buffers) are
    freed by the OS.  This is critical for the embeddings pipeline
    which loads ~500 MB of model weights.

    Args:
        pipeline: Dict from the PIPELINES registry.
        db_path: Path to the SQLite database.
        dry_run: If True, pass --dry-run to the child script.
        log_level: Logging level for the child.

    Returns:
        Dict with keys: key, task, success, duration_sec,
        return_code, error.
    """
    key = pipeline["key"]
    script = FEATURES_DIR / pipeline["script"]
    timeout = PIPELINE_TIMEOUTS.get(key, DEFAULT_TIMEOUT)

    if not script.exists():
        logger.error(
            "--- [%s] Script not found: %s ---", key.upper(), script,
        )
        return {
            "key": key,
            "stage": pipeline["stage"],
            "success": False,
            "duration_sec": 0,
            "return_code": -1,
            "error": f"Script not found: {script}",
        }

    cmd = [
        sys.executable,
        str(script),
        "--db", str(db_path),
        "--log-level", log_level,
    ]
    if dry_run:
        cmd.append("--dry-run")

    logger.info(
        "--- [%s] Starting: %s ---",
        key.upper(), pipeline["description"],
    )
    logger.debug("Command: %s", " ".join(cmd))

    start = time.monotonic()

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        duration = time.monotonic() - start
        success = result.returncode == 0

        # Log child stdout (contains the pipeline's own summary)
        if result.stdout.strip():
            for line in result.stdout.strip().splitlines():
                logger.info("  [%s] %s", key, line)

        if success:
            logger.info(
                "--- [%s] Completed in %.1f s ---", key.upper(), duration,
            )
        else:
            logger.error(
                "--- [%s] FAILED (exit %d) in %.1f s ---",
                key.upper(), result.returncode, duration,
            )
            if result.stderr.strip():
                for line in result.stderr.strip().splitlines()[-10:]:
                    logger.error("  [%s] stderr: %s", key, line)

        return {
            "key": key,
            "stage": pipeline["stage"],
            "success": success,
            "duration_sec": round(duration, 1),
            "return_code": result.returncode,
            "error": result.stderr[-500:] if not success and result.stderr else "",
        }

    except subprocess.TimeoutExpired:
        duration = time.monotonic() - start
        logger.error(
            "--- [%s] TIMED OUT after %.0f s (limit: %d s) ---",
            key.upper(), duration, timeout,
        )
        return {
            "key": key,
            "stage": pipeline["stage"],
            "success": False,
            "duration_sec": round(duration, 1),
            "return_code": -1,
            "error": f"Subprocess timed out ({timeout} s limit)",
        }

    except Exception as e:
        duration = time.monotonic() - start
        logger.exception(
            "--- [%s] Unexpected error: %s ---", key.upper(), e,
        )
        return {
            "key": key,
            "stage": pipeline["stage"],
            "success": False,
            "duration_sec": round(duration, 1),
            "return_code": -1,
            "error": str(e),
        }


# ===================================================================
# Orchestration
# ===================================================================

def resolve_pipelines(
    only: list[str] | None = None,
    skip: list[str] | None = None,
) -> list[dict]:
    """
    Filter and validate the pipeline list based on --only / --skip.

    Args:
        only: If set, run only these pipeline keys.
        skip: If set, exclude these pipeline keys.

    Returns:
        Filtered list of pipeline dicts in execution order.

    Raises:
        SystemExit if an invalid key is supplied.
    """
    valid_keys = {p["key"] for p in PIPELINES}

    if only:
        invalid = set(only) - valid_keys
        if invalid:
            logger.error(
                "Unknown pipeline key(s) in --only: %s. "
                "Valid keys: %s",
                ", ".join(sorted(invalid)),
                ", ".join(sorted(valid_keys)),
            )
            sys.exit(1)

        # Warn about missing dependencies
        selected = set(only)
        for p in PIPELINES:
            if p["key"] in selected:
                missing_deps = [d for d in p["dependencies"] if d not in selected]
                if missing_deps:
                    logger.warning(
                        "Pipeline '%s' depends on %s which are not included. "
                        "Results may be incomplete if those haven't run before.",
                        p["key"], missing_deps,
                    )

        return [p for p in PIPELINES if p["key"] in only]

    if skip:
        invalid = set(skip) - valid_keys
        if invalid:
            logger.error(
                "Unknown pipeline key(s) in --skip: %s. "
                "Valid keys: %s",
                ", ".join(sorted(invalid)),
                ", ".join(sorted(valid_keys)),
            )
            sys.exit(1)
        return [p for p in PIPELINES if p["key"] not in skip]

    return list(PIPELINES)


def run_all(
    db_path: Path | str | None = None,
    only: list[str] | None = None,
    skip: list[str] | None = None,
    dry_run: bool = False,
    log_level: str = "INFO",
    use_lock: bool = True,
) -> list[dict]:
    """
    Run all (or selected) feature extraction pipelines sequentially.

    Args:
        db_path: Path to SQLite database.
        only: If set, run only these pipeline keys.
        skip: If set, skip these pipeline keys.
        dry_run: Pass --dry-run to each child script.
        log_level: Logging level for child scripts.

    Returns:
        List of result dicts, one per pipeline.
    """
    path = Path(db_path) if db_path else DB_PATH
    pipelines = resolve_pipelines(only=only, skip=skip)

    # Acquire an exclusive lock to prevent concurrent batch runs.
    # This is a Python-level safety net in addition to the flock in the
    # cron entry -- protects against manual runs overlapping.
    # --no-lock skips this for the inline hourly caller (fast, no visual).
    lock_fd = None
    if use_lock:
        try:
            lock_fd = open(LOCK_FILE, "w")
            try:
                fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError:
                logger.error(
                    "Another features_batch instance is already running "
                    "(lock file: %s). Exiting.", LOCK_FILE,
                )
                return []
        except OSError as e:
            logger.warning(
                "Could not open lock file %s: %s. Continuing without lock.",
                LOCK_FILE, e,
            )
            lock_fd = None
    else:
        logger.debug("Lock acquisition skipped (--no-lock)")

    now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    pipeline_keys = ", ".join(p["key"] for p in pipelines)

    logger.info("=" * 65)
    logger.info(
        "FEATURE BATCH | %s | Pipelines: %s | Dry run: %s",
        now_utc, pipeline_keys, dry_run,
    )
    logger.info("=" * 65)

    results = []
    try:
        for pipeline in pipelines:
            result = run_pipeline(
                pipeline,
                db_path=path,
                dry_run=dry_run,
                log_level=log_level,
            )
            results.append(result)
    finally:
        # Release the lock
        if lock_fd is not None:
            try:
                fcntl.flock(lock_fd, fcntl.LOCK_UN)
                lock_fd.close()
            except OSError:
                pass

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    succeeded = sum(1 for r in results if r["success"])
    failed = len(results) - succeeded
    total_dur = sum(r["duration_sec"] for r in results)

    logger.info("=" * 65)
    logger.info(
        "BATCH SUMMARY | ok=%d fail=%d | total=%.1f s (%.1f min)",
        succeeded, failed, total_dur, total_dur / 60,
    )

    for r in results:
        status = "OK" if r["success"] else "FAIL"
        logger.info(
            "  [%s] Stage %d  %s  %.1f s",
            r["key"], r["stage"], status, r["duration_sec"],
        )

    if failed > 0:
        logger.error(
            "%d pipeline(s) failed. Details above.", failed,
        )

    logger.info("=" * 65)

    return results


# ===================================================================
# CLI
# ===================================================================

def main():
    all_keys = [p["key"] for p in PIPELINES]

    parser = argparse.ArgumentParser(
        description=(
            "Batch feature-engineering orchestrator. "
            "Runs all extraction pipelines sequentially, decoupled "
            "from the hourly scraping cycle."
        ),
        epilog=(
            "Pipeline keys: " + ", ".join(all_keys) + ". "
            "Recommended cron: 30 2,14 * * * flock -n /tmp/features_batch.lock "
            "nice -n 10 python run_features_batch.py"
        ),
    )
    parser.add_argument(
        "--db", type=str, default=str(DB_PATH),
        help=f"Path to SQLite database (default: {DB_PATH})",
    )
    parser.add_argument(
        "--only", nargs="+", choices=all_keys, default=None,
        metavar="KEY",
        help="Run only these pipelines (space-separated keys)",
    )
    parser.add_argument(
        "--skip", nargs="+", choices=all_keys, default=None,
        metavar="KEY",
        help="Skip these pipelines (space-separated keys)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Pass --dry-run to each pipeline (compute but do not write)",
    )
    parser.add_argument(
        "--log-level", type=str, default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)",
    )
    parser.add_argument(
        "--no-lock", action="store_true",
        help="Skip lock acquisition (for inline callers like run_hourly.py)",
    )

    args = parser.parse_args()

    if args.only and args.skip:
        parser.error("--only and --skip are mutually exclusive")

    # Set up logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    results = run_all(
        db_path=args.db,
        only=args.only,
        skip=args.skip,
        dry_run=args.dry_run,
        log_level=args.log_level,
        use_lock=not args.no_lock,
    )

    # Exit code: 0 if all ok, 1 if any failed
    any_failed = any(not r["success"] for r in results)
    sys.exit(1 if any_failed else 0)


if __name__ == "__main__":
    main()
