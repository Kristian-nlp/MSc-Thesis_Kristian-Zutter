"""
settings.py

Master's thesis: Learning the Levers — Which Creator-Controllable
Features Predict Post Visibility on TikTok, Instagram, and LinkedIn
from a Swiss German-Language Perspective
Kristian Zutter | HSLU MSc Applied Information and Data Science | 2026

Purpose:
    Centralised configuration for the scraper pipeline. All tuneable
    settings (paths, proxy, browser, platforms, accounts, timing,
    logging) are exposed here so nothing is hard-coded in the scraping
    logic. Secrets are loaded from a .env file at the repo root.

Inputs:
    .env                                proxy and Telegram credentials,
                                        path overrides, log level

Outputs:
    (none — configuration module; exposes constants and dictionaries
    consumed by every other pipeline module)

Usage:
    from settings_loader import load_settings
    settings = load_settings()
    db_path = settings.DB_PATH
"""

import logging
import os
from pathlib import Path
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Load .env file (lives at repo root, never committed to Git)
# ---------------------------------------------------------------------------
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.getenv("DATA_DIR", "/data"))  # Mounted data disk on VM
DB_PATH = DATA_DIR / "scraper.db"                # SQLite database
COOKIE_DIR = PROJECT_ROOT / "01_config" / "cookies"  # Cookie JSON files
LOG_DIR = DATA_DIR / "logs"
SCREENSHOT_DIR = DATA_DIR / "screenshots"        # Debug screenshots
FALLBACK_DIR = DATA_DIR / "fallback"             # JSON fallback when DB fails

# Ensure directories exist
# COOKIE_DIR lives inside the repo — always safe to create.
# DATA_DIR children (/data/logs, /data/screenshots) only exist on the VM
# where /data is a mounted disk.  Skip them locally to avoid errors.
COOKIE_DIR.mkdir(parents=True, exist_ok=True)
if DATA_DIR.exists():
    for d in [LOG_DIR, SCREENSHOT_DIR, FALLBACK_DIR]:
        d.mkdir(parents=True, exist_ok=True)
else:
    logger.warning(
        "DATA_DIR %s does not exist (expected a mounted disk on the VM). "
        "Set DATA_DIR env var for local development.", DATA_DIR,
    )

# ---------------------------------------------------------------------------
# Proxy (Evomi residential, Swiss IPs)
# ---------------------------------------------------------------------------
PROXY_HOST = os.getenv("PROXY_HOST", "rp.evomi.com")
PROXY_PORT = os.getenv("PROXY_PORT", "1000")
PROXY_USER = os.getenv("PROXY_USER", "")
PROXY_PASS = os.getenv("PROXY_PASS", "")

# Full proxy URL for Selenium (HTTP format)
if PROXY_USER and PROXY_PASS:
    PROXY_URL = f"http://{PROXY_USER}:{PROXY_PASS}@{PROXY_HOST}:{PROXY_PORT}"
else:
    PROXY_URL = ""
    logger.warning(
        "Proxy credentials missing (PROXY_USER / PROXY_PASS). "
        "Scraper will connect directly — responses may differ from Swiss IPs."
    )

# ---------------------------------------------------------------------------
# Browser configuration
# Each account uses a fixed user agent, viewport size, and locale and
# language set to Switzerland with German (see thesis methodology).
# ---------------------------------------------------------------------------
# Update this to match the current stable Chrome version before starting
# the 30-day collection.  Override via env var without code changes.
USER_AGENT = os.getenv(
    "USER_AGENT",
    "Mozilla/5.0 (X11; Linux x86_64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/145.0.7632.109 Safari/537.36",
)
VIEWPORT_WIDTH = 1920
VIEWPORT_HEIGHT = 1080
HEADLESS = os.getenv("HEADLESS", "true").lower() == "true"
LOCALE = "de_CH"
LANGUAGE = "de"
TIMEZONE = "Europe/Zurich"

# ChromeDriver path (Chromium installed via apt on VM)
CHROMEDRIVER_PATH = os.getenv("CHROMEDRIVER_PATH", "/usr/bin/chromedriver")

# ---------------------------------------------------------------------------
# Platform URLs (discovery surfaces)
# ---------------------------------------------------------------------------
PLATFORMS = {
    "tiktok": {
        "name": "TikTok",
        "discovery_url": "https://www.tiktok.com/explore",
        "surface": "explore",
        "top_n": 20,
        "baseline_start": 51,
        "baseline_end": 100,
        "max_scroll_attempts": 25,     # Max stale-scroll retries before giving up
    },
    "instagram": {
        "name": "Instagram",
        "discovery_url": "https://www.instagram.com/explore/",
        "surface": "explore",
        "top_n": 20,
        "baseline_start": 51,
        "baseline_end": 100,
        "max_scroll_attempts": 30,     # Higher: IG loads content slower
    },
    "linkedin": {
        "name": "LinkedIn",
        "discovery_url": "https://www.linkedin.com/feed/",
        "surface": "top_feed",
        "top_n": 20,
        "baseline_start": 21,
        "baseline_end": 70,
        "max_scroll_attempts": 30,
    },
}

# ---------------------------------------------------------------------------
# Account configuration
# Two personas per platform: one fresh (no follows), one light-seeded
# (follows a curated list of accounts). See the thesis methodology for
# the persona design and rationale.
# NOTE: account_type uses "light_seeded" (not just "seeded") throughout
# the codebase, DB schema, and analysis to distinguish from a hypothetically
# heavier seeding strategy.  Keep this consistent everywhere.
# ---------------------------------------------------------------------------
ACCOUNTS = {
    "tiktok_fresh": {
        "platform": "tiktok",
        "account_type": "fresh",
        "persona": "sandra_mueller",
        "cookie_file": COOKIE_DIR / "tiktok_sandra_fresh.json",
    },
    "tiktok_seeded": {
        "platform": "tiktok",
        "account_type": "light_seeded",
        "persona": "laura_berger",
        "cookie_file": COOKIE_DIR / "tiktok_laura_seed.json",
    },
    "instagram_fresh": {
        "platform": "instagram",
        "account_type": "fresh",
        "persona": "sandra_mueller",
        "cookie_file": COOKIE_DIR / "insta_sandra_fresh.json",
    },
    # Andrea Häberli — light_seeded Instagram account (follows curated accounts).
    # Replaces Laura Berger (disabled by Instagram on 6 Mar 2026, selfie verification).
    "instagram_seeded": {
        "platform": "instagram",
        "account_type": "light_seeded",
        "persona": "andrea_haeberli",
        "cookie_file": COOKIE_DIR / "insta_andrea_seeded.json",
    },
    # Melanie Gerber — fresh LinkedIn account (no follows, no activity).
    # Replaces Fabienne Steiner (Mar 23 2026).
    "linkedin_fresh": {
        "platform": "linkedin",
        "account_type": "fresh",
        "persona": "melanie_gerber",
        "cookie_file": COOKIE_DIR / "linkedin_melanie_fresh.json",
    },
    # Simone Frei — light_seeded (follows curated accounts).
    # Replaces Corinne Brunner (Mar 23 2026).
    "linkedin_seeded": {
        "platform": "linkedin",
        "account_type": "light_seeded",
        "persona": "simone_frei",
        "cookie_file": COOKIE_DIR / "linkedin_simone_seeded.json",
    },
}

# Warn early if cookie files are missing
for _acct_key, _acct in ACCOUNTS.items():
    if not _acct["cookie_file"].exists():
        logger.warning(
            "Cookie file missing for %s: %s",
            _acct_key, _acct["cookie_file"],
        )

# ---------------------------------------------------------------------------
# Scraping parameters
# ---------------------------------------------------------------------------
PAGE_LOAD_TIMEOUT = int(os.getenv("PAGE_LOAD_TIMEOUT", "120"))
RENDER_WAIT = int(os.getenv("RENDER_WAIT", "5"))
SCROLL_WAIT = int(os.getenv("SCROLL_WAIT", "3"))
JITTER_MIN = int(os.getenv("JITTER_MIN", "20"))
JITTER_MAX = int(os.getenv("JITTER_MAX", "40"))
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "2"))

# ---------------------------------------------------------------------------
# Telegram alerting (used by 01_config/alerting.py)
# ---------------------------------------------------------------------------
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FORMAT = "%(asctime)s | %(name)s | %(levelname)s | %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
