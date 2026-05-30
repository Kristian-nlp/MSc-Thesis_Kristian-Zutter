"""
run_instagram.py

Master's thesis: Learning the Levers — Which Creator-Controllable
Features Predict Post Visibility on TikTok, Instagram, and LinkedIn
from a Swiss German-Language Perspective
Kristian Zutter | HSLU MSc Applied Information and Data Science | 2026

Purpose:
    Entry point for a single Instagram Explore scrape. Creates a fresh
    proxied browser, injects session cookies, detects login/challenge
    redirects, runs the Instagram scraper (Top + scroll baseline), and
    writes the snapshot to SQLite with a structured summary log line.

Inputs:
    01_config/settings.py             ACCOUNTS, PLATFORMS, proxy
    01_config/cookies/insta_*.json    Instagram session cookies

Outputs:
    04_database/scraper.db            snapshot + post + scrape_log rows
    DATA_DIR/logs/instagram.log       run log
    DATA_DIR/screenshots/*.png        debug screenshot on failure

Usage:
    python 02_scraper/run_instagram.py --account instagram_fresh
    python 02_scraper/run_instagram.py --account instagram_seeded
    python 02_scraper/run_instagram.py --account instagram_fresh --dry-run
"""

import argparse
import json
import logging
import sys
from pathlib import Path

from selenium.common.exceptions import TimeoutException

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
from browser import create_browser, cleanup_browser
from auth import load_cookies_from_file, inject_cookies, check_cookie_expiry, verify_session
from instagram import InstagramScraper
from logging_config import setup_logging, log_snapshot_summary, log_scrape_start
from db import write_snapshot


def run_scrape(account_key: str, headless: bool = True, dry_run: bool = False) -> dict:
    """
    Execute a single Instagram scrape for the given account.

    Args:
        account_key: Key from settings.ACCOUNTS (e.g. "instagram_fresh").
        headless: Run browser in headless mode.
        dry_run: If True, run scraper but skip database write.

    Returns:
        Dictionary summary of the scrape result.
    """
    settings = load_settings()
    logger = logging.getLogger("run_instagram")

    # Validate account key
    if account_key not in settings.ACCOUNTS:
        logger.error(
            "Unknown account key: %s. Valid keys: %s",
            account_key,
            list(settings.ACCOUNTS.keys()),
        )
        return {"success": False, "error": f"Unknown account: {account_key}"}

    account = settings.ACCOUNTS[account_key]
    if account["platform"] != "instagram":
        logger.error("Account %s is not an Instagram account", account_key)
        return {"success": False, "error": "Not an Instagram account"}

    log_scrape_start(
        platform="instagram",
        account_type=account["account_type"],
        account_key=account_key,
    )

    driver = None
    try:
        # Step 1: Create fresh browser
        logger.info("Creating browser instance...")
        driver = create_browser(headless=headless, enable_cdp=True)

        # Step 2: Load and inject cookies
        cookie_path = Path(account["cookie_file"])
        logger.info("Loading cookies from: %s", cookie_path)

        cookies = load_cookies_from_file(cookie_path)

        warnings = check_cookie_expiry(cookies, warn_days=3)
        for w in warnings:
            logger.warning("Cookie health: %s", w)

        injected = inject_cookies(driver, cookies, "https://www.instagram.com")
        if injected < len(cookies) * 0.5:
            raise RuntimeError(
                f"Cookie injection failed: only {injected}/{len(cookies)} injected. "
                f"Re-export cookies from your browser and save to: {cookie_path}"
            )

        # Step 3: Navigate to discovery surface with challenge detection
        discovery_url = settings.PLATFORMS["instagram"]["discovery_url"]
        challenge_indicators = ("/challenge/", "/accounts/login", "/checkpoint", "/consent/")
        original_timeout = driver.timeouts.page_load  # already in seconds (Selenium 4)
        driver.set_page_load_timeout(30)
        try:
            try:
                driver.get(discovery_url)
            except TimeoutException:
                current = driver.current_url or ""
                if any(ind in current for ind in challenge_indicators):
                    raise RuntimeError(
                        f"Instagram login/challenge page detected: {current} — "
                        f"cookies are invalid. Re-export and save to: {cookie_path}"
                    )
                # Non-challenge timeout — retry once
                logger.warning("Page load timeout (non-challenge), retrying once...")
                try:
                    driver.get(discovery_url)
                except TimeoutException:
                    raise RuntimeError(
                        f"Instagram page load timed out twice (30s each). URL: {driver.current_url}"
                    )
        finally:
            driver.set_page_load_timeout(original_timeout)

        if not verify_session(driver, "instagram"):
            raise RuntimeError(
                "Session validation failed for Instagram — cookies may have expired. "
                f"Re-export cookies and save to: {cookie_path}"
            )

        # Step 4: Run the scraper
        scraper = InstagramScraper(
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
        description="Run a single Instagram Explore scrape."
    )
    parser.add_argument(
        "--account",
        required=True,
        choices=["instagram_fresh", "instagram_seeded"],
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

    setup_logging(platform="instagram", level=args.log_level)

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
