"""
run_hourly.py

Master's thesis: Learning the Levers — Which Creator-Controllable
Features Predict Post Visibility on TikTok, Instagram, and LinkedIn
from a Swiss German-Language Perspective
Kristian Zutter | HSLU MSc Applied Information and Data Science | 2026

Purpose:
    Hourly scrape orchestrator. Alternates accounts (fresh at even
    hours, light-seeded at odd hours, with cross-platform offset),
    inserts randomised jitter between platforms, performs cookie
    health checks, isolates each platform in its own subprocess,
    writes JSON fallback envelopes on database errors, and triggers
    feature extraction after a successful cycle.

Inputs:
    01_config/settings.py             ACCOUNTS, PLATFORMS, paths
    01_config/cookies/*.json          per-account session cookies
    .env                              proxy + Telegram credentials

Outputs:
    DATA_DIR/scraper.db               new snapshot + post rows
    DATA_DIR/logs/hourly.log          orchestrator log
    DATA_DIR/fallback/*.json          fallback envelopes on DB errors
    Telegram alerts                   on failure or auth issue

Usage:
    python 02_scraper/run_hourly.py
    python 02_scraper/run_hourly.py --dry-run
    python 02_scraper/run_hourly.py --platforms tiktok instagram
    python 02_scraper/run_hourly.py --hour 14    # override UTC hour
"""

import argparse
import logging
import os
import random
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCRAPER_DIR = Path(__file__).resolve().parent  # 02_scraper/
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "01_config"))
sys.path.insert(0, str(PROJECT_ROOT / "02_scraper" / "01_core"))
sys.path.insert(0, str(PROJECT_ROOT / "02_scraper" / "02_platforms"))
sys.path.insert(0, str(PROJECT_ROOT / "02_scraper" / "03_parsers"))

from settings_loader import load_settings
from logging_config import setup_logging
from auth import load_cookies_from_file, check_cookie_expiry
from alerting import (
    alert_auth_failure,
    alert_cookie_expiry,
    alert_hourly_summary,
    alert_scrape_failure,
    alert_db_failure,
    send_alert,
)
from fallback import write_fallback, count_pending

logger = logging.getLogger("orchestrator")


# ---------------------------------------------------------------------------
# Alternation schedule
# ---------------------------------------------------------------------------
# Each entry: (account_key_at_even_hour, account_key_at_odd_hour)
# The cross-platform offset ensures that when TikTok uses Fresh,
# Instagram uses Light-seeded, and vice versa. LinkedIn mirrors TikTok.
ALTERNATION = {
    "tiktok":    ("tiktok_fresh",    "tiktok_seeded"),
    # Instagram fresh paused 2026-03-22: sufficient data. Seeded every hour (catch-up).
    "instagram": ("instagram_seeded", "instagram_seeded"),
    # LinkedIn: Stefanie Keller (fresh) + Nadine Zimmermann (seeded), since 2026-03-22.
    "linkedin":  ("linkedin_fresh",   "linkedin_seeded"),
}

# Order in which platforms are scraped each hour
PLATFORM_ORDER = ["tiktok", "instagram", "linkedin"]

# Runner scripts (relative to 02_scraper/)
RUNNER_SCRIPTS = {
    "tiktok":    "run_tiktok.py",
    "instagram": "run_instagram.py",
    "linkedin":  "run_linkedin_api.py",
}

# Jitter range between platform scrapes (seconds)
JITTER_MIN = 20
JITTER_MAX = 40

# Extra jitter for LinkedIn to avoid predictable hourly pattern (seconds)
# LinkedIn sessions were dying after ~7-8h of clockwork API calls.
LINKEDIN_EXTRA_JITTER_MIN = 60
LINKEDIN_EXTRA_JITTER_MAX = 900  # up to 15 minutes

# Known auth failure patterns in subprocess stderr
AUTH_FAILURE_PATTERNS = [
    "redirected to login",
    "session cookies may have expired",
    "authwall",
    "/accounts/login",
    "/challenge/",
    "/checkpoint",
    "login_required",
    "LOGIN_REQUIRED",
]


# ---------------------------------------------------------------------------
# Account selection
# ---------------------------------------------------------------------------

def get_account_for_hour(platform: str, hour: int) -> str | None:
    """
    Return the correct account key for a platform at a given hour.

    Even hours -> first element of the tuple
    Odd hours  -> second element of the tuple

    Args:
        platform: 'tiktok', 'instagram', or 'linkedin'.
        hour: Hour of day (0-23).

    Returns:
        Account key, e.g. 'tiktok_fresh'.
    """
    even_account, odd_account = ALTERNATION[platform]
    return even_account if hour % 2 == 0 else odd_account


# ---------------------------------------------------------------------------
# Cookie health check
# ---------------------------------------------------------------------------

def check_cookies_before_scrape(
    platform: str,
    account_key: str,
    settings,
) -> None:
    """
    Load and validate session cookies before launching a platform scrape.

    Checks for expired or soon-to-expire cookies and sends a Telegram
    alert if anything is wrong. This runs inside the orchestrator (not
    the subprocess) so alerts fire even if the scraper itself crashes.

    Args:
        platform: Platform name.
        account_key: Account key from settings.ACCOUNTS.
        settings: Loaded settings module.
    """
    account_cfg = settings.ACCOUNTS.get(account_key)
    if not account_cfg:
        logger.warning("No account config for %s, skipping cookie check", account_key)
        return

    cookie_path = Path(account_cfg["cookie_file"])

    if not cookie_path.exists():
        msg = f"Cookie file missing for {account_key}: {cookie_path}"
        logger.error(msg)
        alert_auth_failure(platform, account_key, msg)
        return

    try:
        cookies = load_cookies_from_file(cookie_path, platform=platform)
        warnings = check_cookie_expiry(cookies, warn_days=3)

        if warnings:
            for w in warnings:
                logger.warning("Cookie health (%s): %s", account_key, w)
            alert_cookie_expiry(platform, warnings)
        else:
            logger.debug("Cookie health (%s): all OK", account_key)

    except Exception as e:
        logger.error("Cookie check failed for %s: %s", account_key, e)


# ---------------------------------------------------------------------------
# Auth failure detection from subprocess output
# ---------------------------------------------------------------------------

def detect_auth_failure(stderr: str) -> bool:
    """
    Scan subprocess stderr for known authentication failure patterns.

    Returns True if an auth failure was detected.
    """
    if not stderr:
        return False

    stderr_lower = stderr.lower()
    return any(pattern.lower() in stderr_lower for pattern in AUTH_FAILURE_PATTERNS)


# ---------------------------------------------------------------------------
# Platform scrape via subprocess
# ---------------------------------------------------------------------------

def run_platform_scrape(
    platform: str,
    account_key: str,
    dry_run: bool = False,
) -> dict:
    """
    Run a single platform scrape as a subprocess.

    Each runner script (run_tiktok.py, run_instagram.py, run_linkedin.py)
    is invoked as a separate process. This keeps browser sessions fully
    isolated and avoids any import/path conflicts between platforms.

    Each platform runs in its own subprocess with try/except so one
    failure never cascades to the entire hourly run.

    Args:
        platform: Platform name.
        account_key: Account key from settings.ACCOUNTS.
        dry_run: If True, pass --dry-run to the runner.

    Returns:
        Dict with keys: platform, account_key, success, duration_sec,
        return_code, error, auth_failure.
    """
    script = SCRAPER_DIR / RUNNER_SCRIPTS[platform]
    cmd = [
        sys.executable,   # same Python interpreter (poetry venv)
        str(script),
        "--account", account_key,
    ]
    if dry_run:
        cmd.append("--dry-run")

    logger.info("Launching: %s", " ".join(cmd))
    start = time.monotonic()

    try:
        # Start in a new process group so we can kill Chrome children on timeout
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=str(PROJECT_ROOT),
            start_new_session=True,  # new process group
        )
        try:
            stdout, stderr = proc.communicate(timeout=300)  # 5-min safety timeout
        except subprocess.TimeoutExpired:
            # Kill the entire process group (Python + Chrome + ChromeDriver)
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except OSError:
                proc.kill()
            proc.wait()
            raise

        # Build a CompletedProcess-like result for compatibility
        class _Result:
            pass
        result = _Result()
        result.returncode = proc.returncode
        result.stdout = stdout
        result.stderr = stderr

        duration = time.monotonic() - start
        success = result.returncode == 0
        stderr_tail = result.stderr[-500:] if result.stderr else ""
        auth_failure = detect_auth_failure(result.stderr)

        if success:
            logger.info(
                "%s (%s) completed in %.1f s",
                platform, account_key, duration,
            )
        else:
            logger.error(
                "%s (%s) FAILED (exit %d) in %.1f s\nstderr: %s",
                platform, account_key, result.returncode,
                duration, stderr_tail,
            )

            # Send targeted alerts
            if auth_failure:
                alert_auth_failure(platform, account_key, stderr_tail)
            else:
                alert_scrape_failure(platform, account_key, stderr_tail)

        return {
            "platform": platform,
            "account_key": account_key,
            "success": success,
            "duration_sec": round(duration, 1),
            "return_code": result.returncode,
            "error": stderr_tail if not success else "",
            "auth_failure": auth_failure,
        }

    except subprocess.TimeoutExpired:
        duration = time.monotonic() - start
        error_msg = "Subprocess timed out (300 s limit)"
        logger.error(
            "%s (%s) TIMED OUT after %.0f s",
            platform, account_key, duration,
        )
        alert_scrape_failure(platform, account_key, error_msg)

        return {
            "platform": platform,
            "account_key": account_key,
            "success": False,
            "duration_sec": round(duration, 1),
            "return_code": -1,
            "error": error_msg,
            "auth_failure": False,
        }

    except Exception as e:
        duration = time.monotonic() - start
        error_msg = str(e)
        logger.exception(
            "%s (%s) raised unexpected error: %s",
            platform, account_key, e,
        )
        alert_scrape_failure(platform, account_key, error_msg)

        # Write fallback JSON if this looks like a DB error
        if "database" in error_msg.lower() or "sqlite" in error_msg.lower():
            write_fallback(
                data={"platform": platform, "account_key": account_key,
                      "error": error_msg, "hour": datetime.now(timezone.utc).hour},
                context=f"{platform}_{account_key}_dberror",
            )
            alert_db_failure(f"{platform} ({account_key})", error_msg)

        return {
            "platform": platform,
            "account_key": account_key,
            "success": False,
            "duration_sec": round(duration, 1),
            "return_code": -1,
            "error": error_msg,
            "auth_failure": False,
        }


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------

def run_hourly(
    hour: int | None = None,
    platforms: list[str] | None = None,
    dry_run: bool = False,
) -> list[dict]:
    """
    Run the full hourly scrape cycle with account alternation and jitter.

    Steps per platform:
        1. Check cookie health and alert if expiring
        2. Determine the correct account (fresh or seeded) for this hour
        3. Run the platform scraper as a subprocess (isolated)
        4. Detect auth failures from stderr and alert
        5. Sleep with random jitter before the next platform

    Args:
        hour: Override UTC hour (0-23). Defaults to current UTC hour.
        platforms: Subset of platforms to scrape. Defaults to all three.
        dry_run: Pass --dry-run to each platform runner.

    Returns:
        List of result dicts, one per platform.
    """
    if hour is None:
        hour = datetime.now(timezone.utc).hour

    if platforms is None:
        platforms = PLATFORM_ORDER
    else:
        # Preserve the standard order for consistency
        platforms = [p for p in PLATFORM_ORDER if p in platforms]

    settings = load_settings()

    # Pre-flight: verify XDG_RUNTIME_DIR exists (snap chromium needs it)
    runtime_dir = f"/run/user/{os.getuid()}"
    if not os.path.isdir(runtime_dir):
        logger.error(
            "XDG_RUNTIME_DIR %s does not exist. Chromium snap requires this. "
            "Fix: sudo loginctl enable-linger %s",
            runtime_dir, os.environ.get("USER", "unknown"),
        )

    logger.info(
        "=" * 60 + "\n"
        "HOURLY SCRAPE | UTC hour: %02d | Platforms: %s | Dry run: %s\n"
        + "=" * 60,
        hour, ", ".join(platforms), dry_run,
    )

    # Log the alternation plan for this hour
    for p in platforms:
        acct = get_account_for_hour(p, hour)
        logger.info("  %s -> %s", p, acct)

    # Attempt to recover any pending fallback JSON files before scraping
    pending_fallbacks = count_pending()
    if pending_fallbacks > 0:
        logger.info(
            "%d pending fallback file(s) found -- attempting recovery",
            pending_fallbacks,
        )
        try:
            sys.path.insert(0, str(PROJECT_ROOT / "04_database"))
            from db import recover_from_json
            recovery = recover_from_json()
            if recovery["replayed"] > 0:
                logger.info(
                    "Recovered %d fallback file(s) into SQLite",
                    recovery["replayed"],
                )
            if recovery["failed"] > 0:
                logger.warning(
                    "%d fallback file(s) still failed to recover",
                    recovery["failed"],
                )
        except Exception as e:
            logger.warning("Fallback recovery failed: %s", e)

    results = []
    for i, platform in enumerate(platforms):
        account_key = get_account_for_hour(platform, hour)

        # Skip platform if no account scheduled for this hour (e.g. paused)
        if account_key is None:
            logger.info("Skipping %s at hour %d (no account scheduled)", platform, hour)
            continue

        # Check cookie health before scrape
        check_cookies_before_scrape(platform, account_key, settings)

        # Extra jitter for LinkedIn to break the predictable hourly pattern
        if platform == "linkedin":
            li_jitter = random.uniform(
                LINKEDIN_EXTRA_JITTER_MIN, LINKEDIN_EXTRA_JITTER_MAX
            )
            logger.info(
                "LinkedIn jitter: sleeping %.0f s (%.1f min) before scrape",
                li_jitter, li_jitter / 60,
            )
            time.sleep(li_jitter)

        # Run scrape in isolated subprocess
        result = run_platform_scrape(platform, account_key, dry_run)

        # Single retry for transient (non-auth) failures to avoid losing
        # a full hour of data. Auth failures are not retried because retrying
        # the same credentials won't help and may burn the account.
        if not result["success"] and not result["auth_failure"]:
            stderr_lower = (result.get("error") or "").lower()
            is_auth_like = any(
                kw in stderr_lower
                for kw in ("auth failure", "login", "429")
            )
            if not is_auth_like:
                logger.info("Retrying %s after transient failure...", platform)
                time.sleep(45)
                result = run_platform_scrape(platform, account_key, dry_run)

        results.append(result)

        # Jitter between platforms (not after the last one)
        if i < len(platforms) - 1:
            jitter = random.uniform(JITTER_MIN, JITTER_MAX)
            logger.info("Jitter: sleeping %.1f s before next platform", jitter)
            time.sleep(jitter)

    # ---------------------------------------------------------------------------
    # Summary
    # ---------------------------------------------------------------------------
    succeeded = sum(1 for r in results if r["success"])
    failed = len(results) - succeeded
    total_dur = sum(r["duration_sec"] for r in results)

    logger.info(
        "HOURLY SUMMARY | hour=%02d | ok=%d fail=%d | total=%.1f s",
        hour, succeeded, failed, total_dur,
    )

    if failed > 0:
        failure_details = [r for r in results if not r["success"]]
        for r in failure_details:
            logger.error(
                "  FAILED: %s (%s) -- %s",
                r["platform"], r["account_key"], r["error"][:200],
            )

        # Send summary alert via Telegram
        alert_hourly_summary(
            hour=hour,
            succeeded=succeeded,
            failed=failed,
            total_duration=total_dur,
            failures=failure_details,
        )

    # ---------------------------------------------------------------------------
    # Feature engineering at ingest (skip visual — too slow for hourly runs)
    # ---------------------------------------------------------------------------
    if succeeded > 0 and not dry_run:
        logger.info("Running feature extraction on newly ingested posts...")
        features_script = PROJECT_ROOT / "03_features" / "run_features_batch.py"
        try:
            feat_result = subprocess.run(
                [sys.executable, str(features_script), "--skip", "visual", "--no-lock"],
                capture_output=True,
                text=True,
                timeout=180,  # 3-minute cap (text/style/audio/temporal are fast)
                cwd=str(PROJECT_ROOT),
            )
            if feat_result.returncode == 0:
                logger.info("Feature extraction completed successfully")
            else:
                logger.warning(
                    "Feature extraction failed (exit %d): %s",
                    feat_result.returncode, feat_result.stderr[-300:],
                )
        except subprocess.TimeoutExpired:
            logger.warning("Feature extraction timed out (180s limit), will retry at next batch")
        except Exception as e:
            logger.warning("Feature extraction error: %s", e)

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Hourly scrape orchestrator with account alternation, "
            "inter-platform jitter, cookie health checks, "
            "and graceful failure handling."
        ),
    )
    parser.add_argument(
        "--hour",
        type=int,
        default=None,
        help=(
            "Override UTC hour (0-23). Defaults to current UTC hour. "
            "Useful for manual reruns, e.g. --hour 14."
        ),
    )
    parser.add_argument(
        "--platforms",
        nargs="+",
        choices=["tiktok", "instagram", "linkedin"],
        default=None,
        help="Subset of platforms to scrape. Defaults to all three.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Pass --dry-run to each platform runner (skip DB writes).",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Console log level (default: INFO).",
    )
    args = parser.parse_args()

    # Logging
    setup_logging(platform="hourly", level=args.log_level)

    # Validate hour
    if args.hour is not None and not (0 <= args.hour <= 23):
        logger.error("--hour must be between 0 and 23, got %d", args.hour)
        sys.exit(1)

    # Top-level try/except so nothing crashes silently
    try:
        results = run_hourly(
            hour=args.hour,
            platforms=args.platforms,
            dry_run=args.dry_run,
        )
    except Exception as e:
        logger.critical("FATAL: Hourly orchestrator crashed: %s", e, exc_info=True)
        send_alert(
            f"FATAL: Hourly orchestrator crashed!\n"
            f"Error: {str(e)[:400]}\n\n"
            f"The entire hourly run was lost. Check /data/logs/hourly*.log."
        )
        sys.exit(2)

    # Exit code: 0 if all succeeded, 1 if any failed
    any_failed = any(not r["success"] for r in results)
    sys.exit(1 if any_failed else 0)


if __name__ == "__main__":
    main()
