#!/usr/bin/env python3
"""
run_revisits.py

Master's thesis: Learning the Levers — Which Creator-Controllable
Features Predict Post Visibility on TikTok, Instagram, and LinkedIn
from a Swiss German-Language Perspective
Kristian Zutter | HSLU MSc Applied Information and Data Science | 2026

Purpose:
    Orchestrator for engagement velocity revisits (T+24h and T+72h).
    Queries the database for posts due for revisit, groups by platform,
    creates one browser session (or API session for LinkedIn) per
    platform, visits each permalink to extract updated engagement
    counters, and writes results back to the database.

Inputs:
    01_config/settings.py             ACCOUNTS, PROXY_URL
    01_config/cookies/*.json          per-platform session cookies
    04_database/scraper.db            posts table (queries pending revisits)

Outputs:
    04_database/scraper.db            revisit_counters + revisit_log rows
    DATA_DIR/logs/scraper.log         run log

Usage:
    python 02_scraper/run_revisits.py --revisit-type t24
    python 02_scraper/run_revisits.py --revisit-type t72
    python 02_scraper/run_revisits.py --revisit-type t24 --platform tiktok
    python 02_scraper/run_revisits.py --revisit-type t24 --dry-run
"""

import argparse
import json
import logging
import random
import sys
import time
from datetime import datetime, timezone
from itertools import groupby
from operator import attrgetter
from pathlib import Path

# ---------------------------------------------------------------------------
# Path setup (numbered folders cannot be imported directly)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "01_config"))
sys.path.insert(0, str(PROJECT_ROOT / "02_scraper" / "01_core"))
sys.path.insert(0, str(PROJECT_ROOT / "02_scraper" / "02_platforms"))
sys.path.insert(0, str(PROJECT_ROOT / "02_scraper" / "03_parsers"))
sys.path.insert(0, str(PROJECT_ROOT / "04_database"))

from settings_loader import load_settings
from logging_config import setup_logging
from browser import create_browser, cleanup_browser
from auth import load_cookies_from_file, inject_cookies, check_cookie_expiry
from alerting import alert_auth_failure
from db import get_connection
from revisit_db import (
    PendingRevisit,
    RevisitCounters,
    RevisitResult,
    ensure_revisit_log_table,
    get_posts_due_for_revisit,
    write_revisit_counters,
    write_revisit_log,
)
from revisit_scraper import (
    extract_counters,
    is_post_missing,
    check_revisit_auth,
    AuthExpiredError,
    build_linkedin_revisit_session,
    extract_linkedin_counters_api,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger = logging.getLogger("run_revisits")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
# Jitter between individual post visits (seconds).
# Keeps browsing pattern human-like and avoids rate limits.
VISIT_JITTER_MIN = 2
VISIT_JITTER_MAX = 5

# Maximum posts to revisit per platform per run.
# Safety cap to prevent runaway batches that exceed the 1-hour cron window.
MAX_PER_PLATFORM = 500

# Page load timeout for individual post visits (seconds).
PAGE_TIMEOUT = 15

# Maximum total batch duration (seconds). If exceeded, stop processing
# remaining platforms to avoid overlapping with the next cron trigger.
BATCH_TIMEOUT = 50 * 60  # 50 minutes


# ---------------------------------------------------------------------------
# Cookie mapping: which account cookies to use per platform
# ---------------------------------------------------------------------------

def _get_cookie_path(platform: str, settings) -> Path:
    """
    Return the cookie file for the seeded account on this platform.

    Revisits use the light-seeded account's cookies because:
        1. It is logged in, so private/restricted posts are accessible
        2. Revisiting an already-seen post does not bias the fresh
           account's algorithm (revisits are not discovery-surface views)
    """
    account_key = f"{platform}_seeded"
    account_cfg = settings.ACCOUNTS.get(account_key)
    if account_cfg:
        return Path(account_cfg["cookie_file"])
    # Fallback to fresh if seeded not configured
    return Path(settings.ACCOUNTS[f"{platform}_fresh"]["cookie_file"])


# ---------------------------------------------------------------------------
# Core revisit logic
# ---------------------------------------------------------------------------

def run_revisit_batch(
    revisit_type: str,
    settings,
    platform: str | None = None,
    dry_run: bool = False,
    db_path: Path | str | None = None,
) -> dict:
    """
    Run a complete revisit batch for one revisit_type.

    1. Query all posts due for this revisit_type
    2. Group by platform
    3. For each platform:
       a. Create browser + inject cookies
       b. Visit each permalink
       c. Extract counters or detect missing post
       d. Write results to DB
       e. Log batch summary
    4. Return aggregate statistics

    Args:
        revisit_type: "t24" or "t72"
        settings: Loaded settings module.
        platform: Optional single platform to process.
        dry_run: If True, query and log but do not visit pages.
        db_path: Override database path.

    Returns:
        Dict with keys: total_due, total_success, total_missing,
        total_failed, duration_sec.
    """
    batch_start = time.time()
    started_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    # Ensure revisit_log table exists
    ensure_revisit_log_table(db_path)

    # Query pending revisits
    pending = get_posts_due_for_revisit(
        revisit_type=revisit_type,
        platform=platform,
        db_path=db_path,
    )

    if not pending:
        logger.info("No posts due for %s revisit. Done.", revisit_type)
        return {
            "total_due": 0,
            "total_success": 0,
            "total_missing": 0,
            "total_failed": 0,
            "duration_sec": 0.0,
        }

    logger.info(
        "=== %s revisit batch: %d posts due ===",
        revisit_type.upper(), len(pending),
    )

    if dry_run:
        logger.info("DRY RUN -- listing posts without visiting:")
        for p in pending[:20]:
            logger.info(
                "  %s | %s | %.1fh since T0 | %s",
                p.platform, p.post_id, p.hours_since_t0, p.permalink,
            )
        if len(pending) > 20:
            logger.info("  ... and %d more", len(pending) - 20)
        return {
            "total_due": len(pending),
            "total_success": 0,
            "total_missing": 0,
            "total_failed": 0,
            "duration_sec": 0.0,
        }

    # Group by platform for batched browser sessions
    pending.sort(key=attrgetter("platform"))
    grouped = {
        plat: list(posts)
        for plat, posts in groupby(pending, key=attrgetter("platform"))
    }

    stats = {
        "total_due": len(pending),
        "total_success": 0,
        "total_missing": 0,
        "total_failed": 0,
    }

    for plat, posts in grouped.items():
        # Check batch timeout before starting a new platform
        elapsed = time.time() - batch_start
        if elapsed > BATCH_TIMEOUT:
            remaining = sum(len(v) for k, v in grouped.items() if k >= plat)
            logger.warning(
                "Batch timeout (%.0fs > %ds). Skipping %d remaining posts.",
                elapsed, BATCH_TIMEOUT, remaining,
            )
            break

        # Cap per-platform volume
        if len(posts) > MAX_PER_PLATFORM:
            logger.warning(
                "%s: capping from %d to %d posts",
                plat, len(posts), MAX_PER_PLATFORM,
            )
            posts = posts[:MAX_PER_PLATFORM]

        plat_start = time.time()

        logger.info(
            "--- %s: processing %d %s revisits ---",
            plat, len(posts), revisit_type,
        )

        # ----- LinkedIn: API path (no browser, fast) -----
        if plat == "linkedin":
            plat_success, plat_missing, plat_failed = _run_linkedin_api_revisits(
                posts=posts,
                revisit_type=revisit_type,
                settings=settings,
                db_path=db_path,
            )
        else:
            # ----- TikTok / Instagram: browser path -----
            plat_success, plat_missing, plat_failed = _run_browser_revisits(
                plat=plat,
                posts=posts,
                revisit_type=revisit_type,
                settings=settings,
                db_path=db_path,
            )

        plat_duration = time.time() - plat_start

        # Write platform-level log
        write_revisit_log(
            revisit_type=revisit_type,
            platform=plat,
            total_due=len(posts),
            total_success=plat_success,
            total_missing=plat_missing,
            total_failed=plat_failed,
            duration_sec=plat_duration,
            started_at_utc=started_at,
            db_path=db_path,
        )

        stats["total_success"] += plat_success
        stats["total_missing"] += plat_missing
        stats["total_failed"] += plat_failed

        logger.info(
            "%s %s batch done: success=%d, missing=%d, failed=%d (%.1fs)",
            plat, revisit_type,
            plat_success, plat_missing, plat_failed, plat_duration,
        )

    stats["duration_sec"] = round(time.time() - batch_start, 2)

    logger.info(
        "=== %s revisit batch complete: due=%d, success=%d, "
        "missing=%d, failed=%d (%.1fs) ===",
        revisit_type.upper(),
        stats["total_due"],
        stats["total_success"],
        stats["total_missing"],
        stats["total_failed"],
        stats["duration_sec"],
    )

    return stats


def _run_linkedin_api_revisits(
    posts: list,
    revisit_type: str,
    settings,
    db_path,
) -> tuple[int, int, int]:
    """
    Process LinkedIn revisits via Voyager API (no browser needed).

    Builds a requests.Session with LinkedIn auth cookies and calls
    the Voyager API for each post permalink. ~20x faster than the
    browser-based approach and avoids PerimeterX blocking.

    Returns:
        Tuple of (success_count, missing_count, failed_count).
    """
    plat_success = 0
    plat_missing = 0
    plat_failed = 0
    db_conn = None

    try:
        cookie_path = _get_cookie_path("linkedin", settings)
        if not cookie_path.exists():
            logger.error("LinkedIn cookie file not found: %s", cookie_path)
            return 0, 0, len(posts)

        # Check for expiring cookies
        cookies = load_cookies_from_file(cookie_path, platform="linkedin")
        warnings = check_cookie_expiry(cookies)
        for w in warnings:
            logger.warning("Cookie health (linkedin): %s", w)

        # Build API session (proxy from settings)
        proxy_url = settings.PROXY_URL if hasattr(settings, "PROXY_URL") else ""
        session = build_linkedin_revisit_session(cookie_path, proxy_url=proxy_url)
        logger.info("LinkedIn API session built for revisits")

        db_conn = get_connection(db_path)

        for i, post in enumerate(posts, 1):
            result = RevisitResult(
                post_id=post.post_id,
                platform=post.platform,
                permalink=post.permalink,
                revisit_type=revisit_type,
            )

            visit_start = time.time()
            try:
                counters = extract_linkedin_counters_api(session, post.permalink)

                has_counters = (
                    counters.likes is not None or counters.comments is not None
                )
                if has_counters:
                    result.counters = counters
                    result.success = True
                else:
                    result.error_message = "No counters from API"

            except AuthExpiredError as e:
                remaining = len(posts) - (plat_success + plat_missing + plat_failed)
                logger.error(
                    "linkedin: auth expired after %d posts (%d remaining): %s",
                    i - 1, remaining, e,
                )
                plat_failed += remaining
                alert_auth_failure("linkedin", "linkedin_seeded", str(e))
                break

            except Exception as e:
                result.error_message = str(e)
                logger.error(
                    "Error with LinkedIn API revisit %s: %s",
                    post.post_id, e,
                )

            result.duration_seconds = time.time() - visit_start
            write_revisit_counters(result, db_path, conn=db_conn)

            if result.success:
                plat_success += 1
            else:
                plat_failed += 1

            # Progress logging every 50 posts
            if i % 50 == 0:
                logger.info(
                    "linkedin: %d/%d done (success=%d, missing=%d, failed=%d)",
                    i, len(posts), plat_success, plat_missing, plat_failed,
                )

            # Light jitter between API calls (1-2s)
            if i < len(posts):
                time.sleep(random.uniform(1.0, 2.0))

    except Exception as e:
        logger.exception("linkedin: API session error: %s", e)
        plat_failed += len(posts) - (plat_success + plat_missing + plat_failed)

    finally:
        if db_conn:
            db_conn.close()

    return plat_success, plat_missing, plat_failed


def _run_browser_revisits(
    plat: str,
    posts: list,
    revisit_type: str,
    settings,
    db_path,
) -> tuple[int, int, int]:
    """
    Process TikTok/Instagram revisits via headless browser.

    Creates a browser with CDP enabled, injects cookies, and visits
    each permalink to extract counters.

    Returns:
        Tuple of (success_count, missing_count, failed_count).
    """
    plat_success = 0
    plat_missing = 0
    plat_failed = 0
    driver = None
    db_conn = None

    try:
        # Create browser with CDP enabled (needed for Instagram CDP strategy).
        # Instagram revisits need images enabled so React SPA fully hydrates
        # counter elements. TikTok revisits use data-e2e selectors (no images needed).
        block_images = (plat != "instagram")
        driver = create_browser(headless=True, enable_cdp=True, block_images=block_images)

        cookie_path = _get_cookie_path(plat, settings)
        if cookie_path.exists():
            cookies = load_cookies_from_file(cookie_path, platform=plat)

            # Check for expiring cookies
            warnings = check_cookie_expiry(cookies)
            for w in warnings:
                logger.warning("Cookie health (%s): %s", plat, w)

            # Navigate to platform domain first (required for cookie injection)
            domain_urls = {
                "tiktok": "https://www.tiktok.com",
                "instagram": "https://www.instagram.com",
            }
            inject_cookies(driver, cookies, domain_urls[plat])
            logger.info("%s: cookies injected from %s", plat, cookie_path)
        else:
            logger.warning(
                "%s: cookie file not found at %s, proceeding without auth",
                plat, cookie_path,
            )

        # Open one DB connection for the entire platform batch
        db_conn = get_connection(db_path)

        # Visit each post
        for i, post in enumerate(posts, 1):
            result = _visit_single_post(driver, post, revisit_type)

            # Write to database (reuse connection)
            write_revisit_counters(result, db_path, conn=db_conn)

            if result.post_missing:
                plat_missing += 1
            elif result.success:
                plat_success += 1
            else:
                plat_failed += 1

            # Progress logging every 50 posts
            if i % 50 == 0:
                logger.info(
                    "%s: %d/%d done (success=%d, missing=%d, failed=%d)",
                    plat, i, len(posts),
                    plat_success, plat_missing, plat_failed,
                )

            # Human-like jitter between visits
            if i < len(posts):
                jitter = random.uniform(VISIT_JITTER_MIN, VISIT_JITTER_MAX)
                time.sleep(jitter)

    except AuthExpiredError as e:
        remaining = len(posts) - (plat_success + plat_missing + plat_failed)
        logger.error(
            "%s: auth expired after %d posts (%d remaining): %s",
            plat, plat_success + plat_missing + plat_failed, remaining, e,
        )
        plat_failed += remaining
        alert_auth_failure(plat, f"{plat}_seeded", str(e))

    except Exception as e:
        logger.exception("%s: browser session error: %s", plat, e)
        plat_failed += len(posts) - (plat_success + plat_missing + plat_failed)

    finally:
        if db_conn:
            db_conn.close()
        if driver:
            cleanup_browser(driver)

    return plat_success, plat_missing, plat_failed


def _visit_single_post(
    driver,
    post: PendingRevisit,
    revisit_type: str,
) -> RevisitResult:
    """
    Visit a single post permalink and extract counters.

    Raises AuthExpiredError if the session has expired (propagates
    to the platform loop for immediate abort).

    Args:
        driver: Active Selenium WebDriver.
        post: PendingRevisit with permalink and metadata.
        revisit_type: "t24" or "t72".

    Returns:
        RevisitResult with success/failure status and counters.
    """
    result = RevisitResult(
        post_id=post.post_id,
        platform=post.platform,
        permalink=post.permalink,
        revisit_type=revisit_type,
    )

    visit_start = time.time()

    try:
        # Navigate and extract
        counters = extract_counters(
            driver=driver,
            platform=post.platform,
            url=post.permalink,
            timeout=PAGE_TIMEOUT,
        )

        # Check for expired session BEFORE checking missing post.
        # An expired session shows login pages that must not be
        # misclassified as deleted posts.
        # NOTE: AuthExpiredError is NOT caught here — it propagates
        # to the platform loop for immediate abort + alert.
        check_revisit_auth(driver, post.platform)

        # If we already extracted valid counters, the post exists —
        # skip is_post_missing() which can false-positive when the
        # counter was found via a fallback selector not checked by
        # the missing-post heuristic.
        has_counters = counters.likes is not None or counters.comments is not None

        # Check if the post is missing/deleted (only when no counters found)
        if not has_counters and is_post_missing(driver, post.platform):
            result.post_missing = True
            result.duration_seconds = time.time() - visit_start
            logger.info(
                "Post missing: %s (%s)", post.post_id, post.permalink,
            )
            return result

        # Check if we got any usable data
        if not has_counters:
            result.error_message = "No counters extracted"
            result.duration_seconds = time.time() - visit_start
            logger.warning(
                "No counters for %s: %s", post.post_id, post.permalink,
            )
            return result

        result.counters = counters
        result.success = True
        result.duration_seconds = time.time() - visit_start

    except AuthExpiredError:
        # Re-raise so the platform loop can handle it
        raise

    except Exception as e:
        result.error_message = str(e)
        result.duration_seconds = time.time() - visit_start
        logger.error(
            "Error visiting %s (%s): %s", post.post_id, post.permalink, e,
        )

    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Run engagement velocity revisits (T24/T72).",
    )
    parser.add_argument(
        "--revisit-type",
        required=True,
        choices=["t24", "t72"],
        help="Which revisit window to process.",
    )
    parser.add_argument(
        "--platform",
        choices=["tiktok", "instagram", "linkedin"],
        default=None,
        help="Optionally limit to a single platform.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Query and log pending revisits without visiting pages.",
    )
    parser.add_argument(
        "--db-path",
        type=str,
        default=None,
        help="Override database path (for testing).",
    )

    args = parser.parse_args()

    # Initialise settings and logging here (not at module level)
    settings = load_settings()
    setup_logging()

    logger.info(
        "Starting revisit run: type=%s, platform=%s, dry_run=%s",
        args.revisit_type,
        args.platform or "all",
        args.dry_run,
    )

    stats = run_revisit_batch(
        revisit_type=args.revisit_type,
        settings=settings,
        platform=args.platform,
        dry_run=args.dry_run,
        db_path=args.db_path,
    )

    # Print summary to stdout for cron log capture
    print(json.dumps(stats, indent=2))

    # Exit code: 0 if no failures, 1 if any failed
    sys.exit(0 if stats["total_failed"] == 0 else 1)


if __name__ == "__main__":
    main()
