"""
run_tiktok.py

Master's thesis: Learning the Levers — Which Creator-Controllable
Features Predict Post Visibility on TikTok, Instagram, and LinkedIn
from a Swiss German-Language Perspective
Kristian Zutter | HSLU MSc Applied Information and Data Science | 2026

Purpose:
    Entry point for a single TikTok Explore scrape. Creates a fresh
    proxied browser, injects session cookies, validates the session,
    runs the TikTok scraper, writes the snapshot to SQLite, and
    emits a structured snapshot summary log line.

Inputs:
    01_config/settings.py             ACCOUNTS, PLATFORMS, proxy
    01_config/cookies/*.json          TikTok session cookies

Outputs:
    04_database/scraper.db            snapshot + post + scrape_log rows
    DATA_DIR/logs/tiktok.log          run log
    DATA_DIR/screenshots/*.png        debug screenshot on failure

Usage:
    python 02_scraper/run_tiktok.py --account tiktok_fresh
    python 02_scraper/run_tiktok.py --account tiktok_seeded --no-headless
    python 02_scraper/run_tiktok.py --account tiktok_fresh --dry-run
"""

import argparse
import json
import logging
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Path setup (numbered folders cannot be imported directly)
# ---------------------------------------------------------------------------
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "01_config"))
sys.path.insert(0, str(project_root / "02_scraper" / "01_core"))
sys.path.insert(0, str(project_root / "02_scraper" / "02_platforms"))
sys.path.insert(0, str(project_root / "02_scraper" / "03_parsers"))
sys.path.insert(0, str(project_root / "04_database"))

from settings_loader import load_settings
from browser import create_browser, cleanup_browser
from auth import load_cookies_from_file, inject_cookies, check_cookie_expiry, verify_session
from tiktok import TikTokScraper

# Centralised logging
from logging_config import setup_logging, log_snapshot_summary, log_scrape_start

# SQLite persistence
from db import write_snapshot


def run_scrape(account_key: str, headless: bool = True, dry_run: bool = False) -> dict:
    """
    Execute a single TikTok scrape for the given account.

    Args:
        account_key: Key from settings.ACCOUNTS (e.g. "tiktok_fresh").
        headless: Run browser in headless mode.
        dry_run: If True, run scraper but skip database write.

    Returns:
        Dictionary summary of the scrape result.
    """
    settings = load_settings()
    logger = logging.getLogger("run_tiktok")

    # Validate account key
    if account_key not in settings.ACCOUNTS:
        logger.error(
            "Unknown account key: %s. Valid keys: %s",
            account_key,
            list(settings.ACCOUNTS.keys()),
        )
        return {"success": False, "error": f"Unknown account: {account_key}"}

    account = settings.ACCOUNTS[account_key]
    if account["platform"] != "tiktok":
        logger.error("Account %s is not a TikTok account", account_key)
        return {"success": False, "error": "Not a TikTok account"}

    # Structured scrape start log
    log_scrape_start(
        platform="tiktok",
        account_type=account["account_type"],
        account_key=account_key,
    )

    driver = None
    try:
        # Step 1: Create fresh browser (enable_cdp=True activates Network.enable)
        logger.info("Creating browser instance...")
        driver = create_browser(headless=headless, enable_cdp=True)

        # Step 2: Load and inject cookies
        cookie_file = Path(account["cookie_file"])
        logger.info("Loading cookies from: %s", cookie_file)

        cookies = load_cookies_from_file(cookie_file)

        # Check for expiring cookies
        warnings = check_cookie_expiry(cookies)
        for w in warnings:
            logger.warning("Cookie health: %s", w)

        # Inject cookies (navigates to tiktok.com first)
        injected = inject_cookies(driver, cookies, "https://www.tiktok.com")
        if injected < len(cookies) * 0.5:
            raise RuntimeError(
                f"Cookie injection failed: only {injected}/{len(cookies)} injected. "
                f"Re-export cookies from your browser and save to: {cookie_file}"
            )

        # Step 4: Navigate to discovery surface and verify session
        discovery_url = settings.PLATFORMS["tiktok"]["discovery_url"]
        driver.get(discovery_url)
        if not verify_session(driver, "tiktok"):
            raise RuntimeError(
                "Session validation failed for TikTok — cookies may have expired. "
                f"Re-export cookies and save to: {cookie_file}"
            )

        # Step 5: Run the scraper
        scraper = TikTokScraper(
            driver=driver,
            account_type=account["account_type"],
            account_key=account_key,
        )

        result = scraper.scrape()

        # Step 5: Write to SQLite
        if dry_run:
            logger.info(
                "DRY RUN: skipping database write. "
                "Top=%d, Baseline=%d, Success=%s",
                len(result.top_posts),
                len(result.baseline_posts),
                result.success,
            )
            db_success = True
        else:
            db_success = write_snapshot(result)
            if not db_success:
                logger.warning("SQLite write failed, data saved to temp JSON")

        # Step 6: Build summary
        summary = {
            "success": result.success,
            "snapshot_id": result.snapshot_id,
            "platform": result.platform,
            "account_type": result.account_type,
            "captured_at": result.captured_at_utc,
            "top_posts_count": len(result.top_posts),
            "baseline_posts_count": len(result.baseline_posts),
            "duration_seconds": round(result.duration_seconds, 1),
            "db_write": db_success,
            "error": result.error_message,
            "screenshot": result.screenshot_path,
        }

        # Step 7: Log snapshot summary
        log_snapshot_summary(
            snapshot_id=result.snapshot_id,
            platform=result.platform,
            account_type=result.account_type,
            success=result.success,
            top_count=len(result.top_posts),
            baseline_count=len(result.baseline_posts),
            duration=result.duration_seconds,
            error=result.error_message,
        )

        if result.success:
            # Print sample posts for verification
            logger.info("Sample Top posts:")
            for post in result.top_posts[:5]:
                logger.info(
                    "  Rank %d: %s (ID: %s, likes: %s)",
                    post.rank_observed,
                    post.permalink[:60] if post.permalink else "no-link",
                    post.post_id,
                    post.likes,
                )

        return summary

    except Exception as e:
        logger.exception("Fatal error during scrape: %s", e)
        return {"success": False, "error": str(e)}

    finally:
        if driver:
            cleanup_browser(driver)
            logger.info("Browser closed")


def main():
    parser = argparse.ArgumentParser(
        description="Run a single TikTok Explore scrape."
    )
    parser.add_argument(
        "--account",
        required=True,
        choices=["tiktok_fresh", "tiktok_seeded"],
        help="Account key from settings.ACCOUNTS",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run scraper but skip database write",
    )
    parser.add_argument(
        "--no-headless",
        action="store_true",
        help="Run with visible browser (for debugging)",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)",
    )

    args = parser.parse_args()

    # Centralised logging
    setup_logging(platform="tiktok", level=args.log_level)

    result = run_scrape(
        account_key=args.account,
        headless=not args.no_headless,
        dry_run=args.dry_run,
    )

    # Print JSON summary
    print("\n" + "=" * 60)
    print("RESULT SUMMARY:")
    print(json.dumps(result, indent=2))
    print("=" * 60)

    sys.exit(0 if result.get("success") else 1)


if __name__ == "__main__":
    main()
